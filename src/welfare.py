"""Generalized-cost demand and consumer surplus (rule of half)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from src.simulation import SimConfig, SimResults, simulate


# Priors for the Midtown article welfare ledger.
VOT_USD_PER_HOUR = 22.50  # ≈ half a Midtown-ish wage
MU_WAIT = 2.0  # waiting valued 2× in-vehicle
EPS_GC = -0.36  # constant elasticity of trips w.r.t. GC (APTA all-hours magnitude)
M_MIN, M_MAX = 0.50, 2.50
MAX_ITERS = 12
TOL_REL_GC = 0.005
DAMP = 0.55  # under-relax m updates to limit oscillation


@dataclass
class GCBreakdown:
    gc: float
    f_eff: float
    t_wait: float
    t_ride: float
    s: float  # standee-spot fill ∈ [0, 1]
    mu_c: float  # 1 + s ∈ [1, 2]
    boarded: int
    denied: int


def effective_fare(cfg: SimConfig) -> float:
    """Mean money cost per boarding attempt (evaders pay 0)."""
    if cfg.fare_type == "free" or cfg.fare_price <= 0:
        return 0.0
    evade = float(cfg.evasion_rate or 0.0)
    return float(cfg.fare_price) * (1.0 - evade)


def standee_spot_fill(results: SimResults) -> float:
    """
    Mean fraction of standee spots occupied at boarding.

        s = clip((n - n_seat) / (n_cap - n_seat), 0, 1)
    """
    cfg = results.config
    seated = float(cfg.seated_cap())
    cap = float(max(cfg.bus_capacity, seated + 1))
    slots = max(cap - seated, 1.0)
    fills = []
    for e in results.log:
        if e["event"] != "board" or "onboard" not in e:
            continue
        n = float(e["onboard"])
        fills.append(float(np.clip((n - seated) / slots, 0.0, 1.0)))
    if not fills:
        return 0.0
    return float(np.mean(fills))


def mean_wait_ride(results: SimResults) -> tuple[float, float]:
    waits, rides = [], []
    for p in results.completed:
        if p.board_time is None or p.deboard_time is None:
            continue
        if p.wait_minutes is not None:
            waits.append(float(p.wait_minutes))
        rides.append(float(p.deboard_time - p.board_time))
    if not waits:
        # Fall back to boarded wait + summary ride
        waits = [
            float(p.wait_minutes)
            for p in results.boarded
            if p.wait_minutes is not None
        ]
    w = float(np.mean(waits)) if waits else 0.0
    r = float(np.mean(rides)) if rides else float(results.summary().get("mean_trip_min") or 0.0)
    return w, r


def generalized_cost(
    results: SimResults,
    *,
    vot: float = VOT_USD_PER_HOUR,
    mu_wait: float = MU_WAIT,
) -> GCBreakdown:
    """
    GC = f_eff + VOT/60 * (μ_w * t_wait + μ_c * t_ride)

    μ_c = 1 + s, s = mean standee-spot fill at boarding.
    Denied clock time is already inside t_wait for eventual boarders.
    """
    cfg = results.config
    summary = results.summary()
    t_wait, t_ride = mean_wait_ride(results)
    s = standee_spot_fill(results)
    mu_c = 1.0 + s
    f_eff = effective_fare(cfg)
    minutes = mu_wait * t_wait + mu_c * t_ride
    gc = f_eff + (vot / 60.0) * minutes
    return GCBreakdown(
        gc=float(gc),
        f_eff=f_eff,
        t_wait=t_wait,
        t_ride=t_ride,
        s=s,
        mu_c=mu_c,
        boarded=int(summary["passengers_boarded"]),
        denied=int(summary["denied_boardings"]),
    )


def gc_demand_multiplier(gc: float, gc0: float, eps: float = EPS_GC) -> float:
    """Q/Q0 = (GC/GC0)^ε_GC."""
    if gc0 <= 1e-9:
        return 1.0
    ratio = max(gc, 1e-9) / gc0
    m = float(ratio ** float(eps))
    return float(np.clip(m, M_MIN, M_MAX))


def delta_cs(gc0: float, q0: float, gc1: float, q1: float) -> float:
    """Rule of half: ΔCS ≈ ½(Q0+Q1)(GC0−GC1)."""
    return 0.5 * (q0 + q1) * (gc0 - gc1)


def solve_gc_equilibrium(
    city_data: dict,
    cfg: SimConfig,
    *,
    gc0: float,
    m0: float = 1.0,
    eps: float = EPS_GC,
    vot: float = VOT_USD_PER_HOUR,
    mu_wait: float = MU_WAIT,
    max_iters: int = MAX_ITERS,
    tol: float = TOL_REL_GC,
    damp: float = DAMP,
    verbose: bool = True,
) -> tuple[SimResults, GCBreakdown, list[dict[str, Any]]]:
    """
    Outer fixed point: run sim at ridership_scale m → GC → m' = (GC/GC0)^ε.

    `gc0` is the reference generalized cost (usually base-fare, m=1).
    """
    m = float(np.clip(m0, M_MIN, M_MAX))
    history: list[dict[str, Any]] = []
    results = None
    br = None
    for it in range(max_iters):
        run_cfg = replace(cfg, ridership_scale=m)
        results = simulate(city_data, cfg=run_cfg)
        br = generalized_cost(results, vot=vot, mu_wait=mu_wait)
        m_target = gc_demand_multiplier(br.gc, gc0, eps=eps)
        m_next = (1.0 - damp) * m + damp * m_target
        m_next = float(np.clip(m_next, M_MIN, M_MAX))
        rel = abs(br.gc - gc0) / gc0 if it == 0 else abs(m_next - m) / max(m, 1e-9)
        # Convergence on GC stability across iters
        row = {
            "iter": it,
            "m": m,
            "m_target": m_target,
            "m_next": m_next,
            "gc": br.gc,
            "boarded": br.boarded,
            "t_wait": br.t_wait,
            "t_ride": br.t_ride,
            "s": br.s,
            "denied": br.denied,
        }
        history.append(row)
        if verbose:
            print(
                f"    iter {it}: m={m:.3f} → target {m_target:.3f}  "
                f"GC=${br.gc:.2f}  Q={br.boarded}  "
                f"wait={br.t_wait:.2f} ride={br.t_ride:.2f} s={br.s:.3f}"
            )
        if it > 0:
            gc_rel = abs(history[-1]["gc"] - history[-2]["gc"]) / max(
                history[-2]["gc"], 1e-9
            )
            m_rel = abs(m_next - m) / max(abs(m), 1e-9)
            if gc_rel < tol and m_rel < tol:
                m = m_next
                break
        m = m_next
    assert results is not None and br is not None
    # One last run at the settled scale so reported Q matches m.
    if abs((results.config.ridership_scale or 0.0) - m) > 1e-6:
        results = simulate(city_data, cfg=replace(cfg, ridership_scale=m))
        br = generalized_cost(results, vot=vot, mu_wait=mu_wait)
        history.append({
            "iter": len(history),
            "m": m,
            "m_target": gc_demand_multiplier(br.gc, gc0, eps=eps),
            "m_next": m,
            "gc": br.gc,
            "boarded": br.boarded,
            "t_wait": br.t_wait,
            "t_ride": br.t_ride,
            "s": br.s,
            "denied": br.denied,
        })
        if verbose:
            print(
                f"    final: m={m:.3f}  GC=${br.gc:.2f}  Q={br.boarded}  "
                f"wait={br.t_wait:.2f} ride={br.t_ride:.2f} s={br.s:.3f}"
            )
    return results, br, history


def calibrate_gc0(
    city_data: dict,
    cfg: SimConfig,
    *,
    vot: float = VOT_USD_PER_HOUR,
    mu_wait: float = MU_WAIT,
) -> tuple[float, GCBreakdown, SimResults]:
    """Reference GC at baseline ridership (m=1), usually base fare / κ=1."""
    run_cfg = replace(cfg, ridership_scale=1.0)
    results = simulate(city_data, cfg=run_cfg)
    br = generalized_cost(results, vot=vot, mu_wait=mu_wait)
    return br.gc, br, results
