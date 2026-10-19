"""Discrete-event bus simulation over a grid of intersecting city corridors."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
import simpy

from src.city_pipeline import (
    CityCalibration,
    stop_coordinates,
    usable_distances,
)

MINUTES_PER_SECOND = 1.0 / 60.0

# Highway BPR defaults (HCM family). Cities override: NYC Midtown uses α=0.85,
# β=2.0 (higher delay floor at modest V/C; flatter once already congested).
BPR_ALPHA_DEFAULT = 0.15
BPR_BETA_DEFAULT = 4.0

# Direction pairs along each axis.
AXIS_DIRECTIONS = {
    "NS": ("N", "S"),  # forward = increasing position = northbound
    "EW": ("E", "W"),  # forward = increasing position = eastbound
}


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """
    All times are minutes unless the field name says seconds.

    Door times are per passenger and belong in seconds: a cash fare takes a
    few seconds to pay, not a few minutes, and at minute-scale a single
    boarding would stall a bus longer than a whole inter-stop leg.
    """

    sim_minutes: float = 200.0

    num_buses: int = 12
    bus_capacity: int = 60
    n_ns_routes: int = 2
    n_ew_routes: int = 2

    fare_price: float = 3.00  # MTA local base (Jan 2026)
    # "cash" | "card" | "mixed" (cash+card by card_share) | "free" (no fare, fast board)
    # NYC paid default is mixed: almost all *payers* use OMNY/card; cash is rare.
    fare_type: str = "mixed"
    # P(card|pays). Among paying riders only (evaders drawn separately).
    # Target ~3% cash of *all* boardings with evasion=0.44:
    #   cash = (1 − 0.44) × (1 − card_share) ≈ 0.03 → card_share ≈ 0.946.
    # Implies roughly: 44% evade, 53% card, 3% cash.
    card_share: float = 0.946
    # Bus fare evasion ≈ 44% (MTA Q1 2025).
    evasion_rate: float = 0.44
    # Paid-world evasion used to weight fare elasticity. Free-fare runs set
    # `evasion_rate=0` for doors/revenue, but former evaders already faced ~$0
    # so they should not get the APTA ridership uplift. Defaults to evasion_rate.
    reference_evasion_rate: float | None = None

    # Fare → ridership (APTA / Pham & Linsalata 1991).
    # All-hours averages: large urbanized ≥1M ≈ -0.36; smaller ≈ -0.43;
    # all-systems ≈ -0.40 (Simpson–Curtin rule-of-thumb was -0.33).
    # Peak vs off-peak (same study; large-city defaults below):
    #   peak ≈ -0.18 (commuters less fare-sensitive),
    #   off-peak ≈ -0.39 (~2× peak). Smaller cities: -0.27 / -0.46.
    ridership_fare_elasticity: float = -0.36  # all-hours fallback
    ridership_fare_elasticity_peak: float = -0.18
    ridership_fare_elasticity_offpeak: float = -0.39
    # Peak windows for elasticity = AM/PM crest ± scale×Gaussian width.
    peak_elasticity_width_scale: float = 1.35
    # Paid baseline against which free / counterfactual fares are measured.
    reference_fare: float = 3.00  # MTA local base (Jan 2026)
    # Of riders induced by a fare cut, where did they come from?
    # Empirical anchors (not a fitted model):
    # - TCRP Synthesis 101 (Volinski 2012): of *additional* fare-free trips, only
    #   ~5–30% switched from motorized modes; most otherwise walked/biked or
    #   would not have traveled (https://www.trb.org/Publications/Blurbs/167607.aspx).
    # - MTA NYC fare-free bus pilot survey of riders *new to the route*: prior mode
    #   ~27% walk, ~22% drive/taxi, remainder other / other MTA / no trip
    #   (MTA Fare-Free Bus Pilot 1-Year Evaluation, June 2025).
    # - Albany CBD fare-free: ridership gain “almost entirely” walk→bus; traffic
    #   / air-quality impacts minimal (UMTA demonstration).
    # NYC Midtown defaults sit near the MTA new-rider mix, with walk still the
    # largest single source and car diversion modest (bus ≈ inferior for drivers).
    fare_new_riders_from_car: float = 0.22
    fare_new_riders_from_walk: float = 0.40
    # If set, replaces fare-elasticity demand scaling (used by GC fixed-point
    # iteration: arrivals × ridership_scale). None → fare_demand_multiplier().
    ridership_scale: float | None = None

    # --- BPR congestion: T = T0 × (1 + α (κ · V/C)^β) ---
    # T0 = free-flow time (25 mph rolling + lights). V = auto volume on the
    # study streets; C = capacity (same units). κ ∈ [0, 1] is how much of auto
    # congestion buses feel (1 = mixed traffic; <1 ≈ dedicated bus lane).
    # Fare-induced bus riders who formerly drove subtract from V directly.
    # NYC Midtown defaults: N–S screenline from DOT lane capacity (see
    # city_pipeline.nyc_midtown_screenline_capacity_per_min).
    bpr_alpha: float = BPR_ALPHA_DEFAULT
    bpr_beta: float = BPR_BETA_DEFAULT
    bpr_kappa: float = 1.0  # bus exposure to auto V/C; 1 = no bus-lane insulation
    auto_volume: float = 100.0  # veh/min across Midtown N–S screenline
    auto_capacity: float = 137.5  # veh/min screenline capacity
    # Network bus boarding rate at paid baseline (trips/min). Set in simulate()
    # from stop weights if left at 0.
    bus_flow_ref_per_min: float = 0.0

    # Legacy fields kept for older city calibrations / conditions overlays.
    # Travel time no longer uses the 1–5 scale when BPR is active (always on).
    traffic_base: float = 4.0
    traffic_amplitude: float = 0.0

    # Smooth weekday rhythm (not step blocks): midday ≈ 1.0, mild AM/PM crests,
    # soft overnight dip. NYC overrides amps from ATR (see city_pipeline.NYC_ATR_*).
    # Drives demand and auto volume.
    diurnal_profile: bool = False
    day_start_hour: float = 6.0  # clock hour at measurement t=0
    am_peak_hour: float = 8.0
    am_peak_width_hours: float = 1.15
    am_peak_amp: float = 0.30  # generic prior; NYC uses ATR ≈ 0.08
    pm_peak_hour: float = 17.75
    pm_peak_width_hours: float = 1.45
    pm_peak_amp: float = 0.55  # generic prior; NYC uses ATR ≈ 0.13
    night_dip_hour: float = 3.0
    night_dip_width_hours: float = 2.2
    night_dip_amp: float = 0.40  # generic prior; NYC uses ATR ≈ 0.68

    # Per physical stop (not per platform). Midtown mean ~51/hour (0.85/min)
    # with POI weights lifting hubs; paired with overlap-aware fleet sizing.
    arrival_rate_per_min: float = 0.85
    # Clear-road rolling speed (NYC default = 25 mph posted limit).
    free_flow_speed_kmh: float = 40.23
    # Lights along the path: n ≈ distance / spacing; each adds mean delay
    # (red wait + a lump for stop/start — not full accel curves).
    signal_spacing_m: float = 80.0
    signal_mean_delay_sec: float = 14.0

    # Size the fleet so each corridor's cycle time / buses ≈ this headway.
    # Dense Midtown common-line service makes effective waits shorter than H/2.
    target_headway_min: float = 12.0
    min_buses_per_route: int = 2
    # Optional per-route midday headway targets (minutes), e.g. GTFS/MTA
    # schedules for clipped Midtown stubs. Overrides target_headway_min by name/id.
    route_target_headway_min: dict | None = None
    # Destination choice: P(d) ∝ exp(-euclid / od_gravity_meters).
    od_gravity_meters: float = 700.0
    # Hold at terminals so departures stay on an even headway (anti-bunching).
    hold_for_headway: bool = True
    # Random ±minutes on terminal holds / travel to keep similar-cycle routes
    # from phase-locking into a multi-route waiting wave.
    schedule_jitter_min: float = 0.55
    travel_noise: float = 0.10  # ±fraction on each leg
    # Extra random walk time on transfers so a whole deboarding group does not
    # re-enter the waiting stock as one synchronized pulse.
    transfer_walk_jitter: float = 0.6  # ±fraction on transfer_walk_seconds
    # Seed each stop with ~λ·(headway/2) waiting passengers at t=0 so the
    # run starts near steady state instead of climbing from an empty network.
    # Scale >1 folds in typical transfer-queue occupancy in the waiting metric.
    warm_start: bool = True
    warm_start_scale: float = 2.0
    # Extra minutes run before the measurement window so transfer queues also
    # settle; results/plots are rebased to t=0 at the end of warmup.
    # With diurnal_profile, warmup sits on the clock just before day_start_hour.
    warmup_minutes: float = 120.0

    board_seconds_cash: float = 4.5
    board_seconds_card: float = 2.0
    board_seconds_evade: float = 1.5
    board_seconds_free: float = 1.0  # all-door / proof-of-payment tap-off feel
    alight_seconds: float = 1.7  # TCQSM typical single-door alighting (1.7–2.0 s/pax)
    # TCQSM (TCRP 100 / Report 165): +0.5 s/pax boarding when standees are
    # present. Same additive drag on alighting — manual notes both movements
    # slow once the aisle has standees. seated_capacity=None → ~2/3 of bus_capacity
    # (≈40 seats on a 60-pax NYC local).
    crowd_door_penalty_seconds: float = 0.5
    seated_capacity: int | None = None
    pull_in_seconds: float = 10.0
    pull_out_seconds: float = 10.0
    transfer_walk_seconds: float = 60.0  # walk between platforms at a hub

    seed: int | None = None

    @classmethod
    def from_calibration(cls, calibration: CityCalibration, conditions=None, **overrides) -> "SimConfig":
        # Prefer explicit BPR volume/capacity; else map legacy 1–5 traffic_base
        # to a V/C-style index (traffic=1 → V≈0, traffic=5 → V≈2×C).
        if getattr(calibration, "auto_volume", None) is not None:
            auto_volume = float(calibration.auto_volume)
            auto_capacity = float(getattr(calibration, "auto_capacity", 50.0) or 50.0)
        else:
            auto_capacity = 50.0
            # traffic_base 3.2 → V/C ≈ 1.1 → V ≈ 55
            vc = max(0.0, (float(calibration.traffic_base) - 1.0) / 2.0)
            auto_volume = vc * auto_capacity

        params = dict(
            num_buses=calibration.num_buses,
            fare_price=calibration.fare_price,
            fare_type=getattr(calibration, "fare_type", "mixed"),
            card_share=float(getattr(calibration, "card_share", 0.946)),
            evasion_rate=calibration.evasion_rate,
            traffic_base=calibration.traffic_base,
            traffic_amplitude=calibration.traffic_amplitude,
            arrival_rate_per_min=calibration.arrival_rate_per_min,
            n_ns_routes=calibration.n_ns_routes,
            n_ew_routes=calibration.n_ew_routes,
            free_flow_speed_kmh=getattr(calibration, "free_flow_speed_kmh", 40.23),
            signal_spacing_m=getattr(calibration, "signal_spacing_m", 80.0),
            signal_mean_delay_sec=getattr(calibration, "signal_mean_delay_sec", 14.0),
            bpr_alpha=getattr(calibration, "bpr_alpha", BPR_ALPHA_DEFAULT),
            bpr_beta=getattr(calibration, "bpr_beta", BPR_BETA_DEFAULT),
            bpr_kappa=float(getattr(calibration, "bpr_kappa", 1.0) or 1.0),
            auto_volume=auto_volume,
            auto_capacity=auto_capacity,
            bus_flow_ref_per_min=getattr(calibration, "bus_flow_ref_per_min", 0.0) or 0.0,
            target_headway_min=getattr(calibration, "target_headway_min", 7.0),
            min_buses_per_route=getattr(calibration, "min_buses_per_route", 2),
            route_target_headway_min=(
                dict(getattr(calibration, "route_target_headway_min", None) or {})
                or None
            ),
            od_gravity_meters=getattr(calibration, "od_gravity_meters", 450.0),
            diurnal_profile=getattr(calibration, "diurnal_profile", False),
            ridership_fare_elasticity=getattr(
                calibration, "ridership_fare_elasticity", -0.36
            ),
            ridership_fare_elasticity_peak=getattr(
                calibration, "ridership_fare_elasticity_peak", -0.18
            ),
            ridership_fare_elasticity_offpeak=getattr(
                calibration, "ridership_fare_elasticity_offpeak", -0.39
            ),
            reference_fare=getattr(calibration, "reference_fare", None)
            or getattr(calibration, "fare_price", 3.00),
            fare_new_riders_from_car=getattr(
                calibration, "fare_new_riders_from_car", 0.22
            ),
            fare_new_riders_from_walk=getattr(
                calibration, "fare_new_riders_from_walk", 0.40
            ),
        )
        for _diurnal_key in (
            "am_peak_hour",
            "am_peak_amp",
            "pm_peak_hour",
            "pm_peak_amp",
            "night_dip_amp",
        ):
            _val = getattr(calibration, _diurnal_key, None)
            if _val is not None:
                params[_diurnal_key] = float(_val)
        if conditions is not None:
            if conditions.arrival_rate_per_min is not None:
                params["arrival_rate_per_min"] = conditions.arrival_rate_per_min
            params["arrival_rate_per_min"] *= conditions.arrival_rate_scale
            if conditions.fare_type is not None:
                params["fare_type"] = conditions.fare_type
            if getattr(conditions, "card_share", None) is not None:
                params["card_share"] = float(conditions.card_share)
            if getattr(conditions, "target_headway_min", None) is not None:
                params["target_headway_min"] = conditions.target_headway_min
            if getattr(conditions, "min_buses_per_route", None) is not None:
                params["min_buses_per_route"] = conditions.min_buses_per_route
            if getattr(conditions, "od_gravity_meters", None) is not None:
                params["od_gravity_meters"] = conditions.od_gravity_meters
            if getattr(conditions, "diurnal_profile", None) is not None:
                params["diurnal_profile"] = conditions.diurnal_profile
            if getattr(conditions, "day_start_hour", None) is not None:
                params["day_start_hour"] = conditions.day_start_hour
            if getattr(conditions, "ridership_fare_elasticity", None) is not None:
                params["ridership_fare_elasticity"] = conditions.ridership_fare_elasticity
            if getattr(conditions, "ridership_fare_elasticity_peak", None) is not None:
                params["ridership_fare_elasticity_peak"] = (
                    conditions.ridership_fare_elasticity_peak
                )
            if getattr(conditions, "ridership_fare_elasticity_offpeak", None) is not None:
                params["ridership_fare_elasticity_offpeak"] = (
                    conditions.ridership_fare_elasticity_offpeak
                )
            if getattr(conditions, "fare_new_riders_from_car", None) is not None:
                params["fare_new_riders_from_car"] = conditions.fare_new_riders_from_car
            if getattr(conditions, "fare_new_riders_from_walk", None) is not None:
                params["fare_new_riders_from_walk"] = conditions.fare_new_riders_from_walk
            if getattr(conditions, "auto_volume", None) is not None:
                params["auto_volume"] = conditions.auto_volume
            if getattr(conditions, "auto_capacity", None) is not None:
                params["auto_capacity"] = conditions.auto_capacity
            if getattr(conditions, "bpr_alpha", None) is not None:
                params["bpr_alpha"] = conditions.bpr_alpha
            if getattr(conditions, "bpr_beta", None) is not None:
                params["bpr_beta"] = conditions.bpr_beta
            if getattr(conditions, "bpr_kappa", None) is not None:
                params["bpr_kappa"] = float(conditions.bpr_kappa)
            for _diurnal_key in (
                "am_peak_hour",
                "am_peak_amp",
                "pm_peak_hour",
                "pm_peak_amp",
                "night_dip_amp",
            ):
                _val = getattr(conditions, _diurnal_key, None)
                if _val is not None:
                    params[_diurnal_key] = float(_val)
        params.update(overrides)
        # Freeze paid-world evasion before free fare zeros the live rate.
        if params.get("reference_evasion_rate") is None:
            params["reference_evasion_rate"] = float(params.get("evasion_rate") or 0.0)
        if params.get("fare_type") == "free":
            params["fare_price"] = 0.0
            params["evasion_rate"] = 0.0
        if params.get("reference_fare") is None:
            params["reference_fare"] = params.get("fare_price") or 3.00
        return cls(**params)

    def board_seconds(self, evading: bool, pay_with_card: bool = False) -> float:
        if self.fare_type == "free":
            return self.board_seconds_free
        if evading:
            return self.board_seconds_evade
        if self.fare_type == "card" or (self.fare_type == "mixed" and pay_with_card):
            return self.board_seconds_card
        return self.board_seconds_cash

    def seated_cap(self) -> int:
        if self.seated_capacity is not None:
            return max(1, int(self.seated_capacity))
        return max(1, int(round(0.67 * self.bus_capacity)))

    def door_service_seconds(self, base_seconds: float, onboard: int) -> float:
        """
        Passenger door time with TCQSM standee penalty.

            t = t0 + 0.5 · 1{n > n_seat}

        Source: TCRP Report 100 / TCQSM 2nd ed. Part 4 — “Add 0.5 s/p to
        boarding times when standees are present.” Applied to alighting too
        (TCRP 165: board/alight both slower with in-vehicle circulation friction).
        """
        penalty = float(self.crowd_door_penalty_seconds)
        if penalty <= 0:
            return float(base_seconds)
        if int(onboard) > self.seated_cap():
            return float(base_seconds) + penalty
        return float(base_seconds)

    def choose_pay_with_card(self, rng: random.Random) -> bool:
        if self.fare_type == "card":
            return True
        if self.fare_type == "mixed":
            return rng.random() < self.card_share
        return False


def _gauss_bump(hour: float, center: float, width: float, amp: float) -> float:
    if width <= 0 or amp == 0:
        return 0.0
    # Shortest circular distance on a 24h clock (overnight wrap).
    d = abs(hour - center) % 24.0
    d = min(d, 24.0 - d)
    return float(amp * math.exp(-0.5 * (d / width) ** 2))


def diurnal_factor(cfg: SimConfig, hour: float) -> float:
    """
    Continuous weekday activity multiplier.

    Midday baseline ≈ 1.0. Morning crest is modest; evening crest is the day's
    highest. Overnight dips smoothly — no step changes between "periods".
    """
    if not cfg.diurnal_profile:
        return 1.0
    h = float(hour) % 24.0
    factor = (
        1.0
        + _gauss_bump(h, cfg.am_peak_hour, cfg.am_peak_width_hours, cfg.am_peak_amp)
        + _gauss_bump(h, cfg.pm_peak_hour, cfg.pm_peak_width_hours, cfg.pm_peak_amp)
        - _gauss_bump(h, cfg.night_dip_hour, cfg.night_dip_width_hours, cfg.night_dip_amp)
    )
    return float(max(0.15, factor))


def clock_hour(cfg: SimConfig, now: float) -> float:
    """
    Wall-clock hour for absolute sim time `now` (includes warmup).

    Measurement t=0 (after rebase) is `day_start_hour`. Warmup occupies the
    clock minutes immediately before that, so the network settles on the
    early-morning ramp rather than a discontinuous jump into the day.
    """
    warmup = float(cfg.warmup_minutes) if cfg.warm_start else 0.0
    return float(cfg.day_start_hour + (now - warmup) / 60.0)


def effective_fare(cfg: SimConfig) -> float:
    """Charged fare for elasticity; free boarding is a full fare elimination."""
    if cfg.fare_type == "free":
        return 0.0
    return float(max(cfg.fare_price, 0.0))


def _hour_near_crest(hour: float, center: float, halfwidth: float) -> bool:
    if halfwidth <= 0:
        return False
    d = abs(float(hour) - float(center)) % 24.0
    d = min(d, 24.0 - d)
    return d <= halfwidth


def is_peak_period(cfg: SimConfig, hour: float) -> bool:
    """
    APTA peak vs off-peak window for fare elasticity.

    Peak = near the AM or PM diurnal crests (crest ± scale×width). Everything
    else — including midday — uses the more elastic off-peak figure.
    """
    scale = float(max(cfg.peak_elasticity_width_scale, 0.0))
    am_hw = scale * float(cfg.am_peak_width_hours)
    pm_hw = scale * float(cfg.pm_peak_width_hours)
    return _hour_near_crest(hour, cfg.am_peak_hour, am_hw) or _hour_near_crest(
        hour, cfg.pm_peak_hour, pm_hw
    )


def fare_elasticity_at(cfg: SimConfig, now: float | None = None) -> float:
    """
    APTA bus fare elasticity for the current clock time.

    With a diurnal profile and a sim time, use peak vs off-peak. Otherwise fall
    back to the all-hours average (`ridership_fare_elasticity`).
    """
    if now is None or not cfg.diurnal_profile:
        return float(cfg.ridership_fare_elasticity)
    hour = clock_hour(cfg, now)
    if is_peak_period(cfg, hour):
        return float(cfg.ridership_fare_elasticity_peak)
    return float(cfg.ridership_fare_elasticity_offpeak)


def _fare_demand_from_elasticity(cfg: SimConfig, e: float) -> float:
    """APTA-style constant elasticity Q1/Q0 for elasticity `e`."""
    p0 = float(max(cfg.reference_fare, 1e-6))
    p1 = effective_fare(cfg)
    if abs(p1 - p0) < 1e-9:
        return 1.0
    if p1 <= 0.0:
        # Full elimination: %ΔP = -100% → %ΔQ ≈ -ε.
        return float(max(0.5, 1.0 - e))
    # Midpoint arc elasticity inverted for Q1/Q0 given P0, P1, e.
    dp = (p1 - p0) / (p1 + p0)
    denom = 1.0 - e * dp
    if abs(denom) < 1e-9:
        return 1.0
    return float(max(0.5, (1.0 + e * dp) / denom))


def fare_demand_multiplier(cfg: SimConfig, now: float | None = None) -> float:
    """
    Scale baseline ridership for the current fare vs `reference_fare`.

    APTA ε applies only to riders who face the fare. Evaders already ride at
    ~$0 effective price, so a posted-fare cut does not multiply their demand:

        Q1/Q0 = (1 − e_ref) · m(ε) + e_ref · 1

    where m(ε) is the payer-only elasticity factor and e_ref is paid-world
    evasion (`reference_evasion_rate`). Peak/off-peak ε from APTA when `now`
    is set under a diurnal profile. Same paid fare → 1.0 (door time separate).

    If `cfg.ridership_scale` is set (GC demand fixed-point), that constant
    replaces the fare formula for both arrivals and auto diversion.
    """
    if cfg.ridership_scale is not None:
        return float(max(1e-9, cfg.ridership_scale))
    raw = _fare_demand_from_elasticity(cfg, fare_elasticity_at(cfg, now))
    if abs(raw - 1.0) < 1e-12:
        return 1.0
    ref_evade = cfg.reference_evasion_rate
    if ref_evade is None:
        ref_evade = cfg.evasion_rate
    evade_share = float(np.clip(ref_evade, 0.0, 1.0))
    pay_share = 1.0 - evade_share
    return float(pay_share * raw + evade_share * 1.0)


def fare_mode_shift_shares(cfg: SimConfig) -> dict[str, float]:
    """Fraction of *new* (fare-induced) riders by prior mode; sums ≤ 1."""
    car = float(np.clip(cfg.fare_new_riders_from_car, 0.0, 1.0))
    walk = float(np.clip(cfg.fare_new_riders_from_walk, 0.0, 1.0 - car))
    other = max(0.0, 1.0 - car - walk)
    return {"car": car, "walk": walk, "other": other}


def diverted_auto_flow(cfg: SimConfig, now: float | None = None) -> float:
    """
    Auto trips/min removed from (or added to) the road by the fare policy.

    diverted = (demand_mult − 1) × bus_flow_ref × share_from_cars

    Positive → cars leave the road (fare cut). Same units as auto_volume.
    """
    dem = fare_demand_multiplier(cfg, now)
    bus_flow = float(cfg.bus_flow_ref_per_min)
    car = fare_mode_shift_shares(cfg)["car"]
    return float((dem - 1.0) * bus_flow * car)


def auto_volume_now(cfg: SimConfig, now: float | None = None,
                    apply_fare_diversion: bool = True) -> float:
    """Auto volume V(t) for BPR, optionally after fare-driven diversion."""
    v = float(cfg.auto_volume)
    if cfg.diurnal_profile and now is not None:
        v *= diurnal_factor(cfg, clock_hour(cfg, now))
    if apply_fare_diversion:
        v = max(0.0, v - diverted_auto_flow(cfg, now))
    return float(v)


def bpr_factor(cfg: SimConfig, now: float | None = None,
              apply_fare_diversion: bool = True) -> float:
    """T/T0 = 1 + α (κ · V/C)^β. κ=1 → full auto congestion on the bus."""
    v = auto_volume_now(cfg, now, apply_fare_diversion=apply_fare_diversion)
    c = float(max(cfg.auto_capacity, 1e-9))
    kappa = float(max(0.0, min(cfg.bpr_kappa, 1.0)))
    return float(1.0 + float(cfg.bpr_alpha) * (kappa * v / c) ** float(cfg.bpr_beta))


def traffic_level(cfg: SimConfig, now: float, horizon: float | None = None) -> float:
    """
    Congestion index for plots / legacy callers.

    Returns the BPR multiplier T/T0 (≥ 1). Free flow with no cars ⇒ 1.0.
    """
    del horizon  # BPR path does not use the legacy mid-run swell.
    return bpr_factor(cfg, now)


def demand_rate(cfg: SimConfig, base_rate: float, now: float) -> float:
    """Poisson rate: baseline × diurnal × fare-elasticity ridership scale."""
    return float(
        max(
            1e-9,
            base_rate
            * diurnal_factor(cfg, clock_hour(cfg, now))
            * fare_demand_multiplier(cfg, now),
        )
    )


def free_flow_minutes(cfg: SimConfig, distance_m: float) -> float:
    """
    T0: travel time with *no cars* — posted-limit rolling + signal delays.

        T0 = distance / speed_limit + (# lights) × delay_per_light
    """
    dist = float(max(distance_m, 0.0))
    cruise_m_per_min = float(cfg.free_flow_speed_kmh) * 1000.0 / 60.0
    rolling = dist / cruise_m_per_min
    spacing = float(max(cfg.signal_spacing_m, 1.0))
    signals = (dist / spacing) * float(cfg.signal_mean_delay_sec) * MINUTES_PER_SECOND
    return float(rolling + signals)


def travel_minutes_breakdown(
    cfg: SimConfig,
    distance_m: float,
    now: float | None = None,
    *,
    apply_fare_diversion: bool = True,
    volume: float | None = None,
) -> dict[str, float]:
    """
    Inter-stop time via BPR:

        T = T0 × (1 + α (κ · V/C)^β)

    κ (`bpr_kappa`) is bus exposure to auto congestion (1 = mixed traffic).
    If `volume` is given it overrides the live V(t). Door dwell is separate.
    """
    dist = float(max(distance_m, 0.0))
    t0 = free_flow_minutes(cfg, dist)
    kappa = float(max(0.0, min(cfg.bpr_kappa, 1.0)))
    if volume is None:
        v = auto_volume_now(cfg, now, apply_fare_diversion=apply_fare_diversion)
        factor = bpr_factor(cfg, now, apply_fare_diversion=apply_fare_diversion)
    else:
        v = float(max(volume, 0.0))
        c = float(max(cfg.auto_capacity, 1e-9))
        factor = float(
            1.0 + float(cfg.bpr_alpha) * (kappa * v / c) ** float(cfg.bpr_beta)
        )
    total = t0 * factor
    spacing = float(max(cfg.signal_spacing_m, 1.0))
    return {
        "distance_m": dist,
        "t0_min": float(t0),
        "bpr_factor": float(factor),
        "bpr_kappa": float(kappa),
        "auto_volume": float(v),
        "auto_capacity": float(cfg.auto_capacity),
        "v_over_c": float(v / max(cfg.auto_capacity, 1e-9)),
        "n_signals": float(dist / spacing),
        "total_min": float(total),
        "effective_kmh": (
            (dist / 1000.0) / (total / 60.0) if total > 0 else float("nan")
        ),
    }


def travel_minutes(cfg: SimConfig, distance_m: float, now: float,
                   horizon: float | None = None) -> float:
    """Door-to-door running time between stops (excludes dwell)."""
    del horizon
    return travel_minutes_breakdown(cfg, distance_m, now)["total_min"]


# ---------------------------------------------------------------------------
# 2. Entities
# ---------------------------------------------------------------------------

@dataclass
class TripLeg:
    route_id: str
    board_stop: int  # global stop index into city stop_nodes
    alight_stop: int
    direction: str


@dataclass
class Passenger:
    id: int
    origin_stop: int
    destination_stop: int
    itinerary: list[TripLeg]
    arrival_time: float
    evade: bool
    board_seconds: float
    current_leg: int = 0
    board_time: float | None = None  # first boarding of the trip
    deboard_time: float | None = None  # final alight at destination
    transfers: int = 0
    bus_id: int | None = None
    # Platforms this passenger is currently standing in (common-line).
    queued_platforms: list = field(default_factory=list, repr=False)

    @property
    def active_leg(self) -> TripLeg | None:
        if self.current_leg >= len(self.itinerary):
            return None
        return self.itinerary[self.current_leg]

    @property
    def wait_minutes(self) -> float | None:
        if self.board_time is None:
            return None
        return self.board_time - self.arrival_time

    @property
    def trip_minutes(self) -> float | None:
        if self.deboard_time is None:
            return None
        # Walk-only trips never board a bus; clock the whole journey from spawn.
        start = self.arrival_time if self.board_time is None else self.board_time
        return self.deboard_time - start


class Platform:
    """One direction of travel on one route at a physical stop.

    The berth (`dock`) is shared across routes at the same (stop, direction) so
    a dwelling M42 blocks a following M34 at that curb — Midtown reality and
    the channel through which demand shocks bunch service.
    """

    def __init__(self, env: simpy.Environment, route_id: str, position: int,
                 direction: str, stop_index: int, dock: simpy.Resource):
        self.route_id = route_id
        self.position = position
        self.direction = direction
        self.stop_index = stop_index
        self.passengers: list[Passenger] = []
        self.dock = dock


class Bus:
    def __init__(self, bus_id: int, route_id: str, position: int,
                 direction: str, capacity: int):
        self.id = bus_id
        self.route_id = route_id
        self.position = position
        self.direction = direction
        self.capacity = capacity
        self.passengers: list[Passenger] = []
        self.revenue = 0.0
        self.boardings = 0
        self.evaded_boardings = 0
        self.denied_boardings = 0
        self.skipped_stops = 0


class Corridor:
    """One bidirectional linear route through a subset of the stop cloud."""

    def __init__(
        self,
        env: simpy.Environment,
        route: dict,
        distances: np.ndarray,
        berth_for,
    ):
        self.id = route["id"]
        self.axis = route["axis"]
        self.name = route.get("name", route["id"])
        self.stop_indices = list(route["stop_indices"])
        self.path_nodes = list(route.get("path_nodes", []))
        self.path_xy = route.get("path_xy")
        self.leg_meters = list(route.get("leg_meters") or [])
        self.distances = distances
        forward, reverse = AXIS_DIRECTIONS[self.axis]

        self.platforms = {
            forward: [
                Platform(env, self.id, pos, forward, stop, berth_for(stop, forward))
                for pos, stop in enumerate(self.stop_indices)
            ],
            reverse: [
                Platform(env, self.id, pos, reverse, stop, berth_for(stop, reverse))
                for pos, stop in enumerate(self.stop_indices)
            ],
        }
        self.position_of = {stop: pos for pos, stop in enumerate(self.stop_indices)}

    @property
    def num_stops(self) -> int:
        return len(self.stop_indices)

    def platform(self, position: int, direction: str) -> Platform:
        return self.platforms[direction][position]

    def direction_toward(self, board_stop: int, alight_stop: int) -> str:
        forward, reverse = AXIS_DIRECTIONS[self.axis]
        return forward if self.position_of[alight_stop] > self.position_of[board_stop] else reverse

    def terminal_direction(self, position: int, direction: str) -> str:
        forward, reverse = AXIS_DIRECTIONS[self.axis]
        if direction == forward and position == self.num_stops - 1:
            return reverse
        if direction == reverse and position == 0:
            return forward
        return direction

    def next_position(self, position: int, direction: str) -> int:
        forward, _ = AXIS_DIRECTIONS[self.axis]
        return position + 1 if direction == forward else position - 1

    def length_m(self) -> float:
        if self.leg_meters:
            return float(sum(self.leg_meters))
        return float(
            sum(self.distances[a, b] for a, b in zip(self.stop_indices, self.stop_indices[1:]))
        )

    def hop_distance(self, a: int, b: int) -> float:
        """Network distance along this corridor between two of its stops."""
        i, j = self.position_of[a], self.position_of[b]
        if i > j:
            i, j = j, i
        if self.leg_meters:
            return float(sum(self.leg_meters[k] for k in range(i, j)))
        return float(sum(
            self.distances[self.stop_indices[k], self.stop_indices[k + 1]]
            for k in range(i, j)
        ))


class TransitNetwork:
    """Intersecting N/S and E/W corridors over the same set of physical stops."""

    def __init__(self, env: simpy.Environment, city_data: dict, routes: list[dict]):
        self.stop_nodes = list(city_data["stop_nodes"])
        if city_data.get("stop_coords") is not None:
            self.coords = np.asarray(city_data["stop_coords"], dtype=float)
        else:
            self.coords = stop_coordinates(city_data["graph"], self.stop_nodes)
        self.distances = usable_distances(city_data["dist_matrix"], self.coords)
        # Shared curb berths: one Resource per (physical stop, direction).
        self.berths: dict[tuple[int, str], simpy.Resource] = {}

        def berth_for(stop_index: int, direction: str) -> simpy.Resource:
            key = (int(stop_index), direction)
            dock = self.berths.get(key)
            if dock is None:
                dock = simpy.Resource(env, capacity=1)
                self.berths[key] = dock
            return dock

        self.corridors = {
            route["id"]: Corridor(env, route, self.distances, berth_for)
            for route in routes
        }
        self.routes_at_stop: dict[int, list[str]] = {i: [] for i in range(len(self.stop_nodes))}
        for corridor in self.corridors.values():
            for stop in corridor.stop_indices:
                self.routes_at_stop[stop].append(corridor.id)

    @property
    def num_stops(self) -> int:
        return len(self.stop_nodes)

    @property
    def routes(self) -> list[Corridor]:
        return list(self.corridors.values())

    def total_route_length_m(self) -> float:
        return sum(c.length_m() for c in self.routes)

    def plan_trip(self, origin: int, destination: int) -> list[TripLeg]:
        """
        Direct ride if origin and destination share a corridor; otherwise one
        transfer. Prefer a shared hub stop; if two corridors never meet, walk
        between their nearest stop pair.
        """
        if origin == destination:
            raise ValueError("origin and destination must differ")

        shared = [
            rid for rid in self.routes_at_stop[origin]
            if rid in self.routes_at_stop[destination]
        ]
        if shared:
            route_id = min(
                shared,
                key=lambda rid: self.corridors[rid].hop_distance(origin, destination),
            )
            corridor = self.corridors[route_id]
            return [TripLeg(
                route_id=route_id,
                board_stop=origin,
                alight_stop=destination,
                direction=corridor.direction_toward(origin, destination),
            )]

        candidates = []
        for first_id in self.routes_at_stop[origin]:
            first = self.corridors[first_id]
            for second_id in self.routes_at_stop[destination]:
                if first_id == second_id:
                    continue
                second = self.corridors[second_id]
                hubs = set(first.stop_indices) & set(second.stop_indices)
                for hub in hubs:
                    if hub == origin:
                        # Already at a stop on the second corridor — one leg.
                        candidates.append((
                            second.hop_distance(origin, destination),
                            [TripLeg(
                                route_id=second_id,
                                board_stop=origin,
                                alight_stop=destination,
                                direction=second.direction_toward(origin, destination),
                            )],
                        ))
                        continue
                    if hub == destination:
                        candidates.append((
                            first.hop_distance(origin, destination),
                            [TripLeg(
                                route_id=first_id,
                                board_stop=origin,
                                alight_stop=destination,
                                direction=first.direction_toward(origin, destination),
                            )],
                        ))
                        continue
                    cost = first.hop_distance(origin, hub) + second.hop_distance(hub, destination)
                    candidates.append((cost, [
                        TripLeg(
                            route_id=first_id,
                            board_stop=origin,
                            alight_stop=hub,
                            direction=first.direction_toward(origin, hub),
                        ),
                        TripLeg(
                            route_id=second_id,
                            board_stop=hub,
                            alight_stop=destination,
                            direction=second.direction_toward(hub, destination),
                        ),
                    ]))

                if hubs:
                    continue

                # Corridors don't share a stop: ride to the nearest gateway on
                # the first corridor, walk to the nearest gateway on the second,
                # then ride to the destination.
                hub_a, hub_b = min(
                    ((a, b) for a in first.stop_indices for b in second.stop_indices),
                    key=lambda pair: float(np.linalg.norm(self.coords[pair[0]] - self.coords[pair[1]])),
                )
                walk = float(np.linalg.norm(self.coords[hub_a] - self.coords[hub_b]))
                cost = first.hop_distance(origin, hub_a) + walk + second.hop_distance(hub_b, destination)
                legs = []
                if hub_a != origin:
                    legs.append(TripLeg(
                        route_id=first_id,
                        board_stop=origin,
                        alight_stop=hub_a,
                        direction=first.direction_toward(origin, hub_a),
                    ))
                if hub_b != destination:
                    legs.append(TripLeg(
                        route_id=second_id,
                        board_stop=hub_b,
                        alight_stop=destination,
                        direction=second.direction_toward(hub_b, destination),
                    ))
                if legs:
                    candidates.append((cost, legs))
                else:
                    # Origin and destination are already each other's nearest
                    # gateways — the trip is a walk, not a bus ride.
                    candidates.append((walk, []))

        if not candidates:
            # Last resort: walk-only trip (no bus legs).
            walk = float(np.linalg.norm(self.coords[origin] - self.coords[destination]))
            return []

        return min(candidates, key=lambda row: row[0])[1]

    def common_line_legs(self, leg: TripLeg) -> list[TripLeg]:
        """
        All corridors that can carry this passenger from board_stop to alight_stop
        without transferring. Used so overlapping Midtown avenues share a queue.
        """
        options = []
        for rid in self.routes_at_stop.get(leg.board_stop, []):
            if rid not in self.routes_at_stop.get(leg.alight_stop, []):
                continue
            corridor = self.corridors[rid]
            options.append(TripLeg(
                route_id=rid,
                board_stop=leg.board_stop,
                alight_stop=leg.alight_stop,
                direction=corridor.direction_toward(leg.board_stop, leg.alight_stop),
            ))
        return options or [leg]


# ---------------------------------------------------------------------------
# 3. Processes
# ---------------------------------------------------------------------------

def _log(log: list, env: simpy.Environment, event: str, **fields):
    log.append({"time": env.now, "event": event, **fields})


def _queue_for_leg(network: TransitNetwork, passenger: Passenger):
    leg = passenger.active_leg
    corridor = network.corridors[leg.route_id]
    position = corridor.position_of[leg.board_stop]
    return corridor.platform(position, leg.direction)


def _enqueue_common_line(network: TransitNetwork, passenger: Passenger):
    """Stand in every platform that can serve the active board→alight pair."""
    leg = passenger.active_leg
    if leg is None:
        return
    options = network.common_line_legs(leg)
    passenger.itinerary[passenger.current_leg] = options[0]
    passenger.queued_platforms = []
    seen = set()
    for opt in options:
        corridor = network.corridors[opt.route_id]
        position = corridor.position_of[opt.board_stop]
        platform = corridor.platform(position, opt.direction)
        key = (opt.route_id, position, opt.direction)
        if key in seen:
            continue
        seen.add(key)
        platform.passengers.append(passenger)
        passenger.queued_platforms.append(platform)


def _leave_all_queues(passenger: Passenger):
    for platform in passenger.queued_platforms:
        try:
            platform.passengers.remove(passenger)
        except ValueError:
            pass
    passenger.queued_platforms = []


def _bind_leg_to_bus(network: TransitNetwork, passenger: Passenger, bus: Bus):
    """After boarding, lock the active leg to the bus's route (common-line)."""
    leg = passenger.active_leg
    if leg is None:
        return
    corridor = network.corridors[bus.route_id]
    if leg.board_stop not in corridor.position_of or leg.alight_stop not in corridor.position_of:
        return
    passenger.itinerary[passenger.current_leg] = TripLeg(
        route_id=bus.route_id,
        board_stop=leg.board_stop,
        alight_stop=leg.alight_stop,
        direction=corridor.direction_toward(leg.board_stop, leg.alight_stop),
    )


def _transfer_to_next_leg(
    env,
    passenger: Passenger,
    network: TransitNetwork,
    cfg: SimConfig,
    log: list,
    nprng: np.random.Generator | None = None,
):
    """Walk to the next platform without holding the bus at the berth."""
    walk = cfg.transfer_walk_seconds * MINUTES_PER_SECOND
    if nprng is not None and cfg.transfer_walk_jitter > 0:
        j = float(cfg.transfer_walk_jitter)
        walk *= float(nprng.uniform(max(0.1, 1.0 - j), 1.0 + j))
    yield env.timeout(walk)
    _enqueue_common_line(network, passenger)
    leg = passenger.active_leg
    _log(
        log, env, "transfer",
        passenger=passenger.id,
        stop=leg.board_stop if leg else None,
        next_route=leg.route_id if leg else None,
    )


def bus_process(env, bus: Bus, network: TransitNetwork, cfg: SimConfig, log: list,
                schedule: dict | None = None, horizon: float | None = None,
                startup_delay: float = 0.0, nprng: np.random.Generator | None = None):
    corridor = network.corridors[bus.route_id]
    if startup_delay > 0:
        yield env.timeout(startup_delay)
    while True:
        prev_direction = bus.direction
        bus.direction = corridor.terminal_direction(bus.position, bus.direction)
        turned = bus.direction != prev_direction
        platform = corridor.platform(bus.position, bus.direction)

        # One berth per platform direction. If occupied, queue behind (Midtown
        # curb behavior) so heavy dwell / demand shocks can propagate into
        # bunching. Terminal anti-bunching holds still re-space at the ends.
        with platform.dock.request() as request:
            berth_arrive = env.now
            yield request
            berth_wait = float(env.now - berth_arrive)
            if berth_wait > 1e-9:
                _log(
                    log, env, "berth_wait",
                    bus=bus.id, route=bus.route_id, stop=platform.stop_index,
                    direction=bus.direction, wait_min=berth_wait,
                )
            yield env.timeout(cfg.pull_in_seconds * MINUTES_PER_SECOND)
            _log(
                log, env, "arrive",
                bus=bus.id, route=bus.route_id, stop=platform.stop_index,
                direction=bus.direction, onboard=len(bus.passengers),
            )

            transferring = []
            for passenger in list(bus.passengers):
                leg = passenger.active_leg
                if leg is None or leg.alight_stop != platform.stop_index:
                    continue
                # Onboard count includes this passenger (aisle still occupied).
                alight_s = cfg.door_service_seconds(
                    cfg.alight_seconds, len(bus.passengers)
                )
                yield env.timeout(alight_s * MINUTES_PER_SECOND)
                bus.passengers.remove(passenger)
                passenger.current_leg += 1
                _log(
                    log, env, "deboard",
                    bus=bus.id, route=bus.route_id, stop=platform.stop_index,
                    passenger=passenger.id, onboard=len(bus.passengers),
                    door_s=alight_s,
                )
                if passenger.active_leg is None:
                    passenger.deboard_time = env.now
                    _log(log, env, "arrive_dest", passenger=passenger.id,
                         stop=platform.stop_index)
                else:
                    transferring.append(passenger)

            for passenger in transferring:
                passenger.transfers += 1
                env.process(
                    _transfer_to_next_leg(
                        env, passenger, network, cfg, log, nprng=nprng
                    )
                )

            while platform.passengers:
                if len(bus.passengers) >= bus.capacity:
                    bus.denied_boardings += len(platform.passengers)
                    _log(
                        log, env, "full",
                        bus=bus.id, route=bus.route_id, stop=platform.stop_index,
                        left_behind=len(platform.passengers),
                    )
                    break

                passenger = platform.passengers[0]
                _leave_all_queues(passenger)
                _bind_leg_to_bus(network, passenger, bus)
                board_s = cfg.door_service_seconds(
                    passenger.board_seconds, len(bus.passengers)
                )
                yield env.timeout(board_s * MINUTES_PER_SECOND)
                first_board = passenger.board_time is None
                if first_board:
                    passenger.board_time = env.now
                    if passenger.evade:
                        bus.evaded_boardings += 1
                    else:
                        bus.revenue += cfg.fare_price
                passenger.bus_id = bus.id
                bus.passengers.append(passenger)
                bus.boardings += 1
                _log(
                    log, env, "board",
                    bus=bus.id, route=bus.route_id, stop=platform.stop_index,
                    passenger=passenger.id, evade=passenger.evade,
                    onboard=len(bus.passengers), leg=passenger.current_leg,
                    first_board=first_board, door_s=board_s,
                )

            yield env.timeout(cfg.pull_out_seconds * MINUTES_PER_SECOND)

        # Terminal headway hold: claim an evenly spaced departure slot so buses
        # on the same corridor do not leave in a pack after variable dwell.
        at_terminal = bus.position in (0, corridor.num_stops - 1)
        if (
            cfg.hold_for_headway
            and schedule is not None
            and (turned or at_terminal)
        ):
            hold = _claim_headway_slot(schedule, env.now, bus.direction, nprng=nprng)
            if hold > 0:
                _log(
                    log, env, "hold",
                    bus=bus.id, route=bus.route_id, stop=platform.stop_index,
                    direction=bus.direction, hold_min=hold,
                )
                yield env.timeout(hold)

        next_position = corridor.next_position(bus.position, bus.direction)
        if corridor.leg_meters:
            i, j = sorted((bus.position, next_position))
            dist = float(sum(corridor.leg_meters[k] for k in range(i, j)))
        else:
            a = corridor.stop_indices[bus.position]
            b = corridor.stop_indices[next_position]
            dist = float(network.distances[a, b])
        leg_min = travel_minutes(cfg, dist, env.now, horizon=horizon)
        if nprng is not None and cfg.travel_noise > 0:
            noise = float(nprng.uniform(1.0 - cfg.travel_noise, 1.0 + cfg.travel_noise))
            leg_min *= noise
        yield env.timeout(leg_min)
        bus.position = next_position


def _claim_headway_slot(
    schedule: dict,
    now: float,
    direction: str,
    nprng: np.random.Generator | None = None,
) -> float:
    """Reserve the next departure slot for this direction; return hold minutes."""
    headway = float(schedule["headway"])
    last = schedule["last_depart"].get(direction)
    jitter = 0.0
    jmax = float(schedule.get("jitter_min", 0.0) or 0.0)
    if nprng is not None and jmax > 0:
        jitter = float(nprng.uniform(-jmax, jmax))
    if last is None:
        slot = now
    else:
        slot = max(now, float(last) + headway + jitter)
    schedule["last_depart"][direction] = slot
    return max(0.0, slot - now)


def _join_first_platform(env, passenger: Passenger, network: TransitNetwork,
                         cfg: SimConfig, log: list):
    """Walk to a gateway if needed, then join common-line boarding queues."""
    if not passenger.itinerary:
        yield env.timeout(cfg.transfer_walk_seconds * MINUTES_PER_SECOND)
        passenger.deboard_time = env.now
        _log(log, env, "walk_only", passenger=passenger.id,
             stop=passenger.origin_stop, destination=passenger.destination_stop)
        return

    if passenger.active_leg.board_stop != passenger.origin_stop:
        yield env.timeout(cfg.transfer_walk_seconds * MINUTES_PER_SECOND)
    _enqueue_common_line(network, passenger)


def _choose_destination(nprng: np.random.Generator, origin: int, network: TransitNetwork,
                        gravity_m: float) -> int:
    """
    Gravity-model destination with a same-corridor bonus so Midtown trips more
    often stay on one avenue/street (longer productive rides, fewer empty buses).
    """
    coords = network.coords
    n = len(coords)
    others = np.array([i for i in range(n) if i != origin], dtype=int)
    dist = np.linalg.norm(coords[others] - coords[origin], axis=1)
    weights = np.exp(-np.maximum(dist, 80.0) / max(gravity_m, 1.0))

    origin_routes = set(network.routes_at_stop.get(origin, []))
    if origin_routes:
        bonus = np.ones(len(others), dtype=float)
        for i, dest in enumerate(others):
            shared = origin_routes.intersection(network.routes_at_stop.get(int(dest), []))
            if shared:
                bonus[i] = 3.0
        weights = weights * bonus

    weights = weights / weights.sum()
    return int(nprng.choice(others, p=weights))


def passenger_arrivals(env, stop_index: int, network: TransitNetwork, cfg: SimConfig,
                       passengers: list, log: list, rng: random.Random,
                       nprng: np.random.Generator, arrival_rate: float):
    """Non-homogeneous Poisson arrivals; rate tracks the diurnal curve."""
    while True:
        rate = demand_rate(cfg, arrival_rate, env.now)
        yield env.timeout(float(nprng.exponential(1.0 / rate)))

        destination = _choose_destination(
            nprng, stop_index, network, cfg.od_gravity_meters
        )
        itinerary = network.plan_trip(stop_index, destination)
        evade = False if cfg.fare_type == "free" else (rng.random() < cfg.evasion_rate)
        pay_card = cfg.choose_pay_with_card(rng)
        passenger = Passenger(
            id=len(passengers) + 1,
            origin_stop=stop_index,
            destination_stop=destination,
            itinerary=itinerary,
            arrival_time=env.now,
            evade=evade,
            board_seconds=cfg.board_seconds(evade, pay_with_card=pay_card),
        )
        passengers.append(passenger)
        _log(
            log, env, "spawn",
            passenger=passenger.id, stop=stop_index,
            destination=destination, legs=len(itinerary), evade=passenger.evade,
            arrival_rate=rate,
            clock_hour=clock_hour(cfg, env.now),
            diurnal=diurnal_factor(cfg, clock_hour(cfg, env.now)),
            fare_demand=fare_demand_multiplier(cfg, env.now),
        )
        env.process(_join_first_platform(env, passenger, network, cfg, log))


# ---------------------------------------------------------------------------
# 4. Running
# ---------------------------------------------------------------------------

@dataclass
class SimResults:
    city_key: str
    config: SimConfig
    network: TransitNetwork
    buses: list[Bus]
    passengers: list[Passenger]
    log: list = field(repr=False)
    warmup_minutes: float = 0.0
    initial_waiting: float = 0.0
    initial_onboard: float = 0.0

    @property
    def revenue(self) -> float:
        return sum(bus.revenue for bus in self.buses)

    @property
    def boarded(self) -> list[Passenger]:
        # Measurement window only (times already rebased so t=0 is post-warmup).
        return [
            p for p in self.passengers
            if p.board_time is not None and p.board_time >= 0.0
        ]

    @property
    def completed(self) -> list[Passenger]:
        return [
            p for p in self.passengers
            if p.deboard_time is not None and p.deboard_time >= 0.0
        ]

    def summary(self) -> dict:
        waits = [p.wait_minutes for p in self.boarded]
        trips = [p.trip_minutes for p in self.completed if p.trip_minutes is not None]
        # Boarding stats from the measurement log so warmup activity is excluded.
        boardings = sum(1 for e in self.log if e["event"] == "board")
        first_boards = [
            e for e in self.log
            if e["event"] == "board" and e.get("first_board")
        ]
        evaded = sum(1 for e in first_boards if e.get("evade"))
        transferred = sum(1 for p in self.passengers if p.transfers > 0 and (
            p.board_time is not None and p.board_time >= 0.0
        ))
        spawned = sum(1 for e in self.log if e["event"] == "spawn")
        multi_leg = sum(
            1 for p in self.passengers
            if len(p.itinerary) > 1 and p.arrival_time >= 0.0
        )

        return {
            "city": self.city_key,
            "sim_minutes": self.config.sim_minutes,
            "warmup_minutes": self.warmup_minutes,
            "initial_waiting": self.initial_waiting,
            "stops": self.network.num_stops,
            "routes": len(self.network.routes),
            "buses": len(self.buses),
            "route_length_km": self.network.total_route_length_m() / 1000.0,
            "passengers_spawned": spawned,
            "passengers_boarded": len(self.boarded),
            "trips_completed": len(self.completed),
            "walk_only_trips": sum(
                1 for e in self.log if e["event"] == "walk_only"
            ),
            "still_waiting": sum(
                1 for p in self.passengers
                if p.board_time is None and p.deboard_time is None
            ),
            "multi_leg_trips": multi_leg,
            "passengers_who_transferred": transferred,
            "boardings": boardings,
            "evaded_boardings": evaded,
            "realized_evasion_rate": (
                evaded / len(first_boards) if first_boards else 0.0
            ),
            "denied_boardings": sum(
                int(e.get("left_behind", 0)) for e in self.log if e["event"] == "full"
            ),
            "skipped_stops": sum(1 for e in self.log if e["event"] == "skip"),
            "revenue": len(first_boards) * self.config.fare_price - evaded * self.config.fare_price,
            "revenue_lost_to_evasion": evaded * self.config.fare_price,
            "mean_wait_min": float(np.mean(waits)) if waits else float("nan"),
            "p90_wait_min": float(np.percentile(waits, 90)) if waits else float("nan"),
            "mean_trip_min": float(np.mean(trips)) if trips else float("nan"),
        }


def _estimate_cycle_minutes(cfg: SimConfig, corridor: Corridor) -> float:
    """
    Round-trip time for *fleet sizing* only.

    Uses midday baseline auto volume, no fare diversion, and κ=1 so fleet is
    not a policy variable when comparing fare or bus-lane (κ) scenarios. Live
    travel still uses the scenario's full BPR path.
    """
    from dataclasses import replace

    sizing_cfg = replace(cfg, bpr_kappa=1.0)
    one_way = travel_minutes_breakdown(
        sizing_cfg, corridor.length_m(), now=None, apply_fare_diversion=False
    )["total_min"]
    # ~20s average stop dwell (doors + berth) each visit; both directions.
    dwell = corridor.num_stops * 2 * (20.0 / 60.0)
    return 2.0 * one_way + dwell


def _stop_overlap(a: Corridor, b: Corridor) -> float:
    sa, sb = set(a.stop_indices), set(b.stop_indices)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(min(len(sa), len(sb)))


def _spine_coordinate(corridor: Corridor, network: TransitNetwork) -> float:
    """Median easting (N/S) or northing (E/W) — identifies the street the route rides."""
    if corridor.path_xy is not None and len(corridor.path_xy) >= 2:
        xy = np.asarray(corridor.path_xy, dtype=float)
    else:
        xy = network.coords[corridor.stop_indices]
    if corridor.axis == "NS":
        return float(np.median(xy[:, 0]))
    return float(np.median(xy[:, 1]))


def _corridor_target_headway(cfg: SimConfig, corridor: Corridor) -> float:
    """Per-route midday headway if tabulated; else the network-wide target."""
    table = getattr(cfg, "route_target_headway_min", None) or {}
    for key in (corridor.name, corridor.id):
        if key in table:
            return max(float(table[key]), 1.0)
    return max(cfg.target_headway_min, 1.0)


def _allocate_buses(
    cfg: SimConfig,
    network: TransitNetwork,
    stop_weights: np.ndarray | None = None,
) -> list[tuple[str, int, str]]:
    """
    Demand-weighted fleet on each distinct street spine.

    Same-street duplicate route IDs (e.g. M5/M7 snapped to one avenue) keep a
    single residual bus — common-line boarding already pools those passengers.
    Tiny clipped stubs get no buses. When `route_target_headway_min` is set
    (NYC Midtown), each primary is sized to its scheduled midday headway so the
    clipped bbox does not get an over-frequent garage. Remaining budget, if any,
    follows stop POI demand on primary trunks.
    """
    corridors = network.routes
    n_routes = len(corridors)
    if n_routes == 0:
        return []

    if stop_weights is None or len(stop_weights) != network.num_stops:
        stop_weights = np.ones(network.num_stops, dtype=float)
    else:
        stop_weights = np.asarray(stop_weights, dtype=float)

    lengths = np.array([max(c.length_m(), 1.0) for c in corridors], dtype=float)
    cycles = np.array([_estimate_cycle_minutes(cfg, c) for c in corridors], dtype=float)
    spines = np.array([_spine_coordinate(c, network) for c in corridors], dtype=float)
    demand = np.array([
        float(sum(stop_weights[i] for i in c.stop_indices)) * lengths[i]
        for i, c in enumerate(corridors)
    ], dtype=float)
    headways = np.array(
        [_corridor_target_headway(cfg, c) for c in corridors], dtype=float
    )

    spine_tol_m = 35.0
    order = list(np.argsort(-lengths))

    primary = np.ones(n_routes, dtype=bool)
    chosen_primaries: list[int] = []
    for idx in order:
        if lengths[idx] < 500.0:
            primary[idx] = False
            continue
        if any(
            corridors[idx].axis == corridors[p].axis
            and abs(spines[idx] - spines[p]) <= spine_tol_m
            for p in chosen_primaries
        ):
            primary[idx] = False
        else:
            chosen_primaries.append(idx)

    # Base need from headway on primaries; one residual bus on same-street copies.
    needed = np.zeros(n_routes, dtype=int)
    for idx in range(n_routes):
        if lengths[idx] < 500.0:
            needed[idx] = 0
        elif primary[idx]:
            # round() (not ceil) so short stubs don't pick up an extra bus from
            # a fractional cycle/H (e.g. 25.8/8 → 3, not 4).
            n = int(np.round(cycles[idx] / headways[idx]))
            needed[idx] = max(cfg.min_buses_per_route, max(1, n))
            needed[idx] = min(needed[idx], 8)
        else:
            # Same-street duplicate: common-line boarding uses the primary fleet.
            needed[idx] = 0

    total_needed = int(needed.sum())
    # With per-route GTFS headways, size exactly to round(cycle/H). Padding up to
    # num_buses over-frequents short Midtown stubs once dwell thins cycles.
    use_route_headways = bool(getattr(cfg, "route_target_headway_min", None))
    if use_route_headways:
        budget = max(total_needed, 1)
    else:
        budget = max(cfg.num_buses, total_needed)
    counts = needed.astype(int).copy()
    buses_left = budget - int(counts.sum())

    # Extras follow demand on primary trunks only (uniform-target cities).
    primary_demand = demand * primary.astype(float)
    if buses_left > 0 and primary_demand.sum() > 0 and not use_route_headways:
        shares = primary_demand / primary_demand.sum()
        # Greedy: always give the next bus to the most under-served vs share.
        while buses_left > 0:
            target_counts = shares * budget
            deficits = target_counts - counts
            deficits[~primary] = -np.inf
            deficits[counts >= 10] = -np.inf
            idx = int(np.argmax(deficits))
            if not np.isfinite(deficits[idx]) or deficits[idx] <= -1e9:
                break
            counts[idx] += 1
            buses_left -= 1

    placements = []
    for corridor, count in zip(corridors, counts):
        if count <= 0:
            continue
        forward, reverse = AXIS_DIRECTIONS[corridor.axis]
        for index in range(int(count)):
            position = round(index * (corridor.num_stops - 1) / max(count - 1, 1))
            direction = forward if index % 2 == 0 else reverse
            placements.append((corridor.id, int(position), direction))
    return placements


def _seed_steady_state_waiters(
    env: simpy.Environment,
    network: TransitNetwork,
    cfg: SimConfig,
    weights: np.ndarray,
    passengers: list,
    log: list,
    rng: random.Random,
    nprng: np.random.Generator,
) -> int:
    """
    Put an equilibrium waiting load on each stop before the clock starts.

    Under random passenger arrivals and ~regular headways, mean wait ≈ H/2, so
    Little's law gives E[queue at stop i] = λ_i · H/2. Arrival times are spread
    over the past headway so first boardings already look mid-regime.
    """
    headway = max(cfg.target_headway_min, 1.0)
    seeded = 0
    # Seed at absolute t=0 (pre-measurement clock — early-morning off-peak for
    # the usual 6:00 start + warmup).
    fare_scale = fare_demand_multiplier(cfg, 0.0)
    for stop_index in range(network.num_stops):
        rate = cfg.arrival_rate_per_min * float(weights[stop_index]) * fare_scale
        expected = rate * (headway / 2.0) * max(cfg.warm_start_scale, 0.0)
        n_wait = int(nprng.poisson(expected))
        for _ in range(n_wait):
            # Uniform age in (0, H): residual wait to the next bus under a
            # deterministic headway with random passenger arrival.
            age = float(nprng.uniform(0.0, headway))
            destination = _choose_destination(
                nprng, stop_index, network, cfg.od_gravity_meters
            )
            itinerary = network.plan_trip(stop_index, destination)
            evade = False if cfg.fare_type == "free" else (rng.random() < cfg.evasion_rate)
            pay_card = cfg.choose_pay_with_card(rng)
            passenger = Passenger(
                id=len(passengers) + 1,
                origin_stop=stop_index,
                destination_stop=destination,
                itinerary=itinerary,
                arrival_time=-age,
                evade=evade,
                board_seconds=cfg.board_seconds(evade, pay_with_card=pay_card),
            )
            passengers.append(passenger)
            _log(
                log, env, "spawn",
                passenger=passenger.id, stop=stop_index,
                destination=destination, legs=len(itinerary),
                evade=passenger.evade, warm_start=True,
            )
            if not itinerary:
                passenger.deboard_time = 0.0
                _log(
                    log, env, "walk_only",
                    passenger=passenger.id, stop=stop_index,
                    destination=destination, warm_start=True,
                )
            else:
                _enqueue_common_line(network, passenger)
                seeded += 1
    return seeded


def simulate(city_data: dict, cfg: SimConfig | None = None,
             routes: list[dict] | None = None, seed: int | None = None) -> SimResults:
    if cfg is None:
        cfg = SimConfig.from_calibration(city_data["calibration"])
    if seed is not None:
        cfg.seed = seed

    rng = random.Random(cfg.seed)
    nprng = np.random.default_rng(cfg.seed)

    env = simpy.Environment()
    if routes is None:
        routes = city_data.get("routes")
    if not routes:
        from src.city_pipeline import build_grid_routes
        routes = build_grid_routes(
            city_data["graph"],
            city_data["stop_nodes"],
            city_data["dist_matrix"],
            n_ns=cfg.n_ns_routes,
            n_ew=cfg.n_ew_routes,
        )

    # Provisional network over all corridors so fleet allocation can see spines.
    draft = TransitNetwork(env, city_data, routes)
    weights = np.asarray(
        city_data.get("arrival_weights", np.ones(draft.num_stops)),
        dtype=float,
    )
    if len(weights) != draft.num_stops:
        weights = np.ones(draft.num_stops)

    # Paid-baseline bus flow (trips/min) so fare diversion ΔV is in the same
    # units as auto_volume. Exclude fare demand multiplier by construction.
    if cfg.bus_flow_ref_per_min <= 0.0:
        cfg.bus_flow_ref_per_min = float(np.sum(weights) * cfg.arrival_rate_per_min)

    placements = _allocate_buses(cfg, draft, stop_weights=weights)
    active_ids = {rid for rid, _, _ in placements}
    active_routes = [r for r in routes if r["id"] in active_ids]
    if not active_routes:
        active_routes = routes
        placements = _allocate_buses(cfg, draft, stop_weights=weights)

    # Planning / boarding only use corridors that actually have buses.
    network = TransitNetwork(env, city_data, active_routes)

    passengers: list[Passenger] = []
    log: list[dict] = []

    buses = []
    for bus_id, (route_id, position, direction) in enumerate(placements, start=1):
        if route_id not in network.corridors:
            continue
        buses.append(Bus(bus_id, route_id, position, direction, cfg.bus_capacity))

    # Per-corridor even-headway schedules (anti-bunching holds at terminals).
    # Random phase offsets so similar cycle times (M1–M5 etc.) do not lock
    # into one system-wide waiting wave.
    from collections import Counter

    route_counts = Counter(b.route_id for b in buses)
    schedules: dict[str, dict] = {}
    for corridor in network.routes:
        n = max(int(route_counts.get(corridor.id, 0)), 1)
        cycle = _estimate_cycle_minutes(cfg, corridor)
        # Never run more often than the timetable when the clipped stub's cycle
        # is shorter than H (1 bus on M20/M55): hold out the rest at the terminal.
        headway = max(cycle / n, _corridor_target_headway(cfg, corridor))
        phase = float(nprng.uniform(0.0, headway)) if nprng is not None else 0.0
        forward, reverse = AXIS_DIRECTIONS[corridor.axis]
        schedules[corridor.id] = {
            "headway": headway,
            "jitter_min": cfg.schedule_jitter_min,
            "last_depart": {
                forward: -phase,
                reverse: -phase - 0.5 * headway,
            },
        }

    # Startup stagger within each route so spatially co-located buses do not
    # all claim the same first terminal slot.
    startup_delay = {}
    by_route: dict[str, list] = {}
    for bus in buses:
        by_route.setdefault(bus.route_id, []).append(bus)
    for route_id, group in by_route.items():
        headway = float(schedules[route_id]["headway"])
        group.sort(key=lambda b: (b.direction, b.position))
        for i, bus in enumerate(group):
            startup_delay[bus.id] = (i / max(len(group), 1)) * headway

    horizon = float(cfg.warmup_minutes if cfg.warm_start else 0.0) + cfg.sim_minutes
    for bus in buses:
        env.process(
            bus_process(
                env, bus, network, cfg, log,
                schedule=schedules[bus.route_id],
                horizon=horizon,
                startup_delay=startup_delay.get(bus.id, 0.0),
                nprng=nprng,
            )
        )

    initial_waiting = 0.0
    initial_onboard = 0.0
    if cfg.warm_start:
        n_seed = _seed_steady_state_waiters(
            env, network, cfg, weights, passengers, log, rng, nprng
        )
        _log(log, env, "warm_start", seeded=n_seed)
        initial_waiting = float(n_seed)

    for stop_index in range(network.num_stops):
        rate = cfg.arrival_rate_per_min * float(weights[stop_index])
        env.process(
            passenger_arrivals(
                env, stop_index, network, cfg, passengers, log, rng, nprng, rate
            )
        )

    warmup = float(cfg.warmup_minutes) if cfg.warm_start else 0.0
    env.run(until=warmup + cfg.sim_minutes)

    if warmup > 0:
        initial_waiting, initial_onboard, log = _rebase_measurement_window(
            passengers, buses, log, warmup
        )
    elif cfg.warm_start:
        # No clock rebase, but the waiting chart should start at the seeded load.
        log = [
            e for e in log
            if not (e["event"] == "spawn" and e.get("warm_start"))
            and not (e["event"] == "walk_only" and e.get("warm_start"))
        ]

    return SimResults(
        city_key=city_data["city_key"],
        config=cfg,
        network=network,
        buses=buses,
        passengers=passengers,
        log=log,
        warmup_minutes=warmup,
        initial_waiting=initial_waiting,
        initial_onboard=initial_onboard,
    )


def _waiting_delta(event: dict) -> int | None:
    if event["event"] == "spawn":
        return 1
    if event["event"] == "board" and event.get("first_board"):
        return -1
    if event["event"] == "transfer":
        return 1
    if event["event"] == "board" and not event.get("first_board"):
        return -1
    if event["event"] == "walk_only":
        return -1
    return None


def _rebase_measurement_window(
    passengers: list[Passenger],
    buses: list[Bus],
    log: list,
    warmup: float,
) -> tuple[float, float, list]:
    """
    Drop the warmup period from the event log and shift clocks so measurement
    starts at t=0 with the empirically settled waiting / onboard loads.
    """
    waiting = 0
    onboard = 0
    for event in log:
        if event["time"] > warmup + 1e-12:
            break
        # Warm-start spawn/walk events establish the initial stock; count them.
        if event["event"] == "warm_start":
            continue
        delta = _waiting_delta(event)
        if delta is not None:
            waiting += delta
        if event["event"] == "board":
            onboard += 1
        elif event["event"] == "deboard":
            onboard -= 1

    onboard = float(max(onboard, 0))

    meas_log = [{
        "time": 0.0,
        "event": "warm_start",
        "seeded": waiting,
        "onboard": onboard,
        "after_warmup": True,
    }]
    for event in log:
        if event["time"] < warmup - 1e-12:
            continue
        if event["event"] in ("warm_start",):
            continue
        shifted = dict(event)
        shifted["time"] = float(event["time"] - warmup)
        # Spawns that occurred exactly at the boundary keep their meaning.
        if shifted.get("warm_start") and shifted["time"] <= 0:
            continue
        meas_log.append(shifted)

    for passenger in passengers:
        passenger.arrival_time -= warmup
        if passenger.board_time is not None:
            passenger.board_time -= warmup
        if passenger.deboard_time is not None:
            passenger.deboard_time -= warmup

    # Reset cumulative bus counters; measurement summary uses the rebased log.
    for bus in buses:
        bus.revenue = 0.0
        bus.boardings = 0
        bus.evaded_boardings = 0
        bus.denied_boardings = 0
        bus.skipped_stops = 0

    return float(max(waiting, 0)), onboard, meas_log


def print_summary(results: SimResults):
    summary = results.summary()
    width = max(len(key) for key in summary)
    print(f"\n--- {summary['city']} simulation ---")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key:<{width}}  {value:,.2f}")
        else:
            print(f"  {key:<{width}}  {value}")
