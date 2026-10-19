"""City operating conditions for cross-city / counterfactual policy runs."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from src.city_pipeline import CityCalibration


@dataclass
class CityConditions:
    """
    Knobs that change how a fixed GTFS template *feels* to operate.

    Use these to ask counterfactuals like: "what if Midtown had SF-like stop
    spacing, demand, and fare evasion — would the same fare policy still work?"

    Geometry comes from GTFS; conditions only thin/scale it.
    """

    label: str = "baseline"

    # Network thinning
    stop_keep_fraction: float = 1.0  # 1.0 = keep every GTFS stop on chosen routes
    min_stop_spacing_m: float | None = None  # drop intermediate stops closer than this

    # Demand / operations (None = leave calibration / SimConfig defaults)
    arrival_rate_per_min: float | None = None
    arrival_rate_scale: float = 1.0  # multiply whatever base rate is in use
    num_buses_scale: float = 1.0
    target_headway_min: float | None = None
    min_buses_per_route: int | None = None
    od_gravity_meters: float | None = None
    evasion_rate: float | None = None
    fare_price: float | None = None
    fare_type: str | None = None
    card_share: float | None = None  # P(card|pays) when fare_type=mixed
    traffic_base: float | None = None
    traffic_amplitude: float | None = None
    fare_type: str | None = None
    diurnal_profile: bool | None = None
    day_start_hour: float | None = None
    # APTA-style fare elasticity + mode-shift composition (see SimConfig).
    ridership_fare_elasticity: float | None = None
    ridership_fare_elasticity_peak: float | None = None
    ridership_fare_elasticity_offpeak: float | None = None
    fare_new_riders_from_car: float | None = None
    fare_new_riders_from_walk: float | None = None
    # BPR auto volume / capacity (trips/min on study streets).
    auto_volume: float | None = None
    auto_capacity: float | None = None
    bpr_alpha: float | None = None
    bpr_beta: float | None = None
    # Bus exposure to auto V/C in BPR (1 = mixed traffic; <1 ≈ bus lane).
    bpr_kappa: float | None = None
    # Diurnal auto-volume crests (NYC: ATR-calibrated; see city_pipeline.NYC_ATR_*).
    am_peak_hour: float | None = None
    am_peak_amp: float | None = None
    pm_peak_hour: float | None = None
    pm_peak_amp: float | None = None
    night_dip_amp: float | None = None


# Midtown daytime mean boardings at shared hubs: 0.85/min ≈ 51/hour
# (busy hubs higher via POI weights). Prior from 2016 MTA weekday route
# ridership (City Limits) × Midtown-in-box fractions × ~0.58 post-COVID
# scale ÷ ~28 hubs ÷ ~13 daytime hours → ~1.2/min paid-only order of
# magnitude; we temper to 0.85 so the thinned ~13-bus clip stays below
# crush capacity (λ≈1.0 begins denials). Fleet is demand-weighted on
# distinct street spines; same-street GTFS duplicates share via common-
# line boarding. Diurnal: ATR Midtown (~+8% AM / +13% PM vs midday).
# Fare elasticity: APTA large-city peak/off-peak (−0.18 / −0.39).
CONDITIONS = {
    "nyc_midtown": CityConditions(
        label="nyc_midtown",
        stop_keep_fraction=1.0,
        # MTA Midtown locals ≈ 200–230 m; placement already targets this, but
        # per-corridor thinning drops residual near-duplicates after sharing.
        min_stop_spacing_m=220.0,
        arrival_rate_per_min=0.85,
        arrival_rate_scale=1.0,
        num_buses_scale=1.0,
        diurnal_profile=True,
        day_start_hour=6.0,
        fare_type="mixed",
        card_share=0.946,
        evasion_rate=0.44,
        ridership_fare_elasticity=-0.36,
        ridership_fare_elasticity_peak=-0.18,
        ridership_fare_elasticity_offpeak=-0.39,
        # MTA fare-free pilot new-to-route prior mode ≈ 22% drive/taxi, 27% walk;
        # remainder other transit / induced. Walk still largest single source.
        fare_new_riders_from_car=0.22,
        fare_new_riders_from_walk=0.40,
        # N–S screenline: C = 5×3×550 vph (DOT), V ≈ 5×1200 vph.
        # Urban Midtown BPR (not highway α=0.15, β=4): higher α → delay floor even
        # at modest V/C; lower β → flatter once already congested.
        auto_volume=100.0,
        auto_capacity=137.5,
        bpr_alpha=0.85,
        bpr_beta=2.0,
        bpr_kappa=1.0,  # Section A: mixed traffic (buses feel full auto V/C)
        # NYC Open Data ATR pooled Midtown avenues (vs midday 11–14h).
        am_peak_hour=8.0,
        am_peak_amp=0.08,
        pm_peak_hour=19.0,
        pm_peak_amp=0.13,
        night_dip_amp=0.68,
    ),
    # Same Midtown priors as nyc_midtown, but buses only feel a fraction of
    # auto congestion (dedicated / enforced bus lane). κ=0.4 is a judgment
    # prior: signals still in T0; residual delay from intersections, turns
    # across the lane, and lane violations.
    "nyc_midtown_bus_lane": CityConditions(
        label="nyc_midtown_bus_lane",
        stop_keep_fraction=1.0,
        min_stop_spacing_m=220.0,
        arrival_rate_per_min=0.85,
        arrival_rate_scale=1.0,
        num_buses_scale=1.0,
        diurnal_profile=True,
        day_start_hour=6.0,
        fare_type="mixed",
        card_share=0.946,
        evasion_rate=0.44,
        ridership_fare_elasticity=-0.36,
        ridership_fare_elasticity_peak=-0.18,
        ridership_fare_elasticity_offpeak=-0.39,
        fare_new_riders_from_car=0.22,
        fare_new_riders_from_walk=0.40,
        auto_volume=100.0,
        auto_capacity=137.5,
        bpr_alpha=0.85,
        bpr_beta=2.0,
        bpr_kappa=0.4,  # Section B: bus-lane insulation
        am_peak_hour=8.0,
        am_peak_amp=0.08,
        pm_peak_hour=19.0,
        pm_peak_amp=0.13,
        night_dip_amp=0.68,
    ),
    # Same Midtown GTFS skeleton, thinned toward SF stop spacing (~320 m) but
    # with downtown-Muni demand: APC studies show many SF local hubs at
    # ~200–500+ weekday boardings (busy corridors much higher). Target ~17/hour
    # midday mean ≈ 0.28/min → ~250 boardings/stop over 6–19 with diurnal.
    # Lean fleet + longer OD gravity so loads aren't empty on short clipped routes.
    "sf_like_from_nyc": CityConditions(
        label="sf_like_from_nyc",
        stop_keep_fraction=0.6,
        min_stop_spacing_m=320.0,
        arrival_rate_per_min=0.28,
        arrival_rate_scale=1.0,
        num_buses_scale=0.33,  # ~8 bus floor
        target_headway_min=15.0,
        min_buses_per_route=1,
        od_gravity_meters=1100.0,
        evasion_rate=0.30,
        fare_price=2.75,
        traffic_base=1.8,
        traffic_amplitude=2.0,
        diurnal_profile=True,
        day_start_hour=6.0,
        ridership_fare_elasticity=-0.43,
        ridership_fare_elasticity_peak=-0.27,
        ridership_fare_elasticity_offpeak=-0.46,
        fare_new_riders_from_car=0.35,
        fare_new_riders_from_walk=0.45,
        auto_volume=40.0,
        auto_capacity=50.0,
        bpr_alpha=0.15,
        bpr_beta=4.0,
    ),
    # Real OSM canal-belt street graph (not thinned NYC). GVB/Centrum priors:
    # OV-chip / OVpay, Autoluwe traffic, induced riders mostly walk/bike.
    "amsterdam_centrum": CityConditions(
        label="amsterdam_centrum",
        stop_keep_fraction=1.0,
        # Placement already targets ~250 m; no second Euclidean thin pass.
        min_stop_spacing_m=None,
        arrival_rate_per_min=0.40,  # ~24/h mean; hubs higher via POI weights
        arrival_rate_scale=1.0,
        num_buses_scale=1.0,
        target_headway_min=8.0,  # GVB daytime tram ~5–10 min
        min_buses_per_route=2,
        od_gravity_meters=550.0,
        evasion_rate=0.03,  # GVB check rates (open tram/bus blend)
        fare_price=3.40,  # GVB 1-hour 2025/26
        fare_type="card",
        card_share=1.0,
        traffic_base=1.5,
        traffic_amplitude=0.9,
        diurnal_profile=True,
        day_start_hour=6.0,
        # APTA large-city (same family as NYC) for cross-city fare experiments.
        ridership_fare_elasticity=-0.36,
        ridership_fare_elasticity_peak=-0.18,
        ridership_fare_elasticity_offpeak=-0.39,
        # Amsterdam mode share: very low car; walk+bike dominate new riders.
        fare_new_riders_from_car=0.08,
        fare_new_riders_from_walk=0.70,
        # Autoluwe screenline: V≈16.7, C≈37.5 → V/C≈0.45 (see city_pipeline).
        auto_volume=16.7,
        auto_capacity=37.5,
        bpr_alpha=0.85,
        bpr_beta=2.0,
        bpr_kappa=1.0,
        am_peak_hour=8.5,
        am_peak_amp=0.12,
        pm_peak_hour=17.5,
        pm_peak_amp=0.18,
        night_dip_amp=0.55,
    ),
}


def apply_conditions_to_calibration(
    calib: CityCalibration, conditions: CityConditions
) -> CityCalibration:
    """Overlay condition knobs onto a city's static calibration."""
    updates = {}
    if conditions.evasion_rate is not None:
        updates["evasion_rate"] = conditions.evasion_rate
    if conditions.fare_price is not None:
        updates["fare_price"] = conditions.fare_price
    if conditions.fare_type is not None:
        updates["fare_type"] = conditions.fare_type
    if conditions.card_share is not None:
        updates["card_share"] = conditions.card_share
    if conditions.traffic_base is not None:
        updates["traffic_base"] = conditions.traffic_base
    if conditions.traffic_amplitude is not None:
        updates["traffic_amplitude"] = conditions.traffic_amplitude
    if conditions.auto_volume is not None:
        updates["auto_volume"] = conditions.auto_volume
    if conditions.auto_capacity is not None:
        updates["auto_capacity"] = conditions.auto_capacity
    if conditions.bpr_alpha is not None:
        updates["bpr_alpha"] = conditions.bpr_alpha
    if conditions.bpr_beta is not None:
        updates["bpr_beta"] = conditions.bpr_beta
    if conditions.bpr_kappa is not None:
        updates["bpr_kappa"] = conditions.bpr_kappa
    if conditions.am_peak_hour is not None:
        updates["am_peak_hour"] = conditions.am_peak_hour
    if conditions.am_peak_amp is not None:
        updates["am_peak_amp"] = conditions.am_peak_amp
    if conditions.pm_peak_hour is not None:
        updates["pm_peak_hour"] = conditions.pm_peak_hour
    if conditions.pm_peak_amp is not None:
        updates["pm_peak_amp"] = conditions.pm_peak_amp
    if conditions.night_dip_amp is not None:
        updates["night_dip_amp"] = conditions.night_dip_amp
    if conditions.diurnal_profile is not None:
        updates["diurnal_profile"] = conditions.diurnal_profile
    if conditions.num_buses_scale != 1.0:
        updates["num_buses"] = max(1, int(round(calib.num_buses * conditions.num_buses_scale)))
    if conditions.target_headway_min is not None:
        updates["target_headway_min"] = conditions.target_headway_min
    if conditions.min_buses_per_route is not None:
        updates["min_buses_per_route"] = conditions.min_buses_per_route
    if conditions.od_gravity_meters is not None:
        updates["od_gravity_meters"] = conditions.od_gravity_meters
    return replace(calib, **updates) if updates else calib


def thin_routes(routes: list[dict], stop_coords: np.ndarray,
                conditions: CityConditions) -> tuple[list[dict], list[int]]:
    """
    Drop stops under spacing / keep-fraction rules.

    Returns (new_routes, surviving_old_indices) where surviving indices refer
    to the original stop_coords rows, in the order of the new global stop list.
    """
    keep_frac = float(np.clip(conditions.stop_keep_fraction, 0.15, 1.0))
    min_spacing = conditions.min_stop_spacing_m

    # Prefer multi-corridor hubs when several candidates sit in the same gap.
    serve_count: dict[int, int] = {}
    for route in routes:
        for idx in route["stop_indices"]:
            serve_count[idx] = serve_count.get(idx, 0) + 1

    survivors: list[int] = []
    seen = set()
    new_routes = []

    for route in routes:
        idxs = list(route["stop_indices"])
        if len(idxs) < 2:
            continue

        # Always keep terminals; thin the middle with a hard spacing floor,
        # preferring transfer hubs inside each admissible window.
        kept = [idxs[0]]
        for idx in idxs[1:-1]:
            prev = kept[-1]
            gap = float(np.linalg.norm(stop_coords[idx] - stop_coords[prev]))
            if min_spacing is not None and gap < min_spacing:
                # Upgrade last intermediate if this hub is more shared and still
                # clears spacing from the stop before it.
                if (
                    len(kept) > 1
                    and serve_count.get(idx, 0) > serve_count.get(kept[-1], 0)
                ):
                    before = kept[-2]
                    alt_gap = float(
                        np.linalg.norm(stop_coords[idx] - stop_coords[before])
                    )
                    if alt_gap >= float(min_spacing):
                        kept[-1] = idx
                continue
            kept.append(idx)
        if idxs[-1] != kept[-1]:
            # If the terminal is too close to the last intermediate, drop that
            # intermediate (unless it is the other terminal).
            if (
                min_spacing is not None
                and len(kept) > 1
                and float(np.linalg.norm(stop_coords[idxs[-1]] - stop_coords[kept[-1]]))
                < min_spacing
            ):
                kept.pop()
            kept.append(idxs[-1])

        # Fractional thinning of intermediate stops (evenly).
        if keep_frac < 1.0 and len(kept) > 2:
            mids = kept[1:-1]
            n_keep = max(0, int(round(len(mids) * keep_frac)))
            if n_keep == 0:
                kept = [kept[0], kept[-1]]
            elif n_keep < len(mids):
                pick = np.linspace(0, len(mids) - 1, n_keep).round().astype(int)
                kept = [kept[0]] + [mids[i] for i in sorted(set(pick))] + [kept[-1]]

        if len(kept) < 2:
            continue

        for idx in kept:
            if idx not in seen:
                seen.add(idx)
                survivors.append(idx)

        # Remap later once survivors are ordered.
        new_routes.append({**route, "stop_indices": kept})

    # Stable order: first-seen across routes.
    old_to_new = {old: new for new, old in enumerate(survivors)}
    remapped = []
    for route in new_routes:
        new_idxs = [old_to_new[i] for i in route["stop_indices"] if i in old_to_new]
        # Dedup consecutive.
        compact = [new_idxs[0]]
        for i in new_idxs[1:]:
            if i != compact[-1]:
                compact.append(i)
        if len(compact) < 2:
            continue
        coords = stop_coords[survivors]
        legs = [
            float(np.linalg.norm(coords[b] - coords[a]))
            for a, b in zip(compact, compact[1:])
        ]
        remapped.append({**route, "stop_indices": compact, "leg_meters": legs})

    return remapped, survivors
