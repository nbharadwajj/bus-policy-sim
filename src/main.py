"""Build GTFS-based NYC Midtown network and compare baseline vs SF-like conditions."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.city_pipeline import build_city
from src.simulation import SimConfig, print_summary, simulate
from src.viz import plot_route_map, plot_sim_metrics

OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"
SEED = 42
# 6:00 → 19:00 wall clock so both the mild AM crest and the heavier PM crest
# fall inside one measurement window (1 sim minute = 1 clock minute).
DAY_SPAN_MINUTES = 13.0 * 60.0


def run_scenario(conditions_key: str | None, tag: str):
    nyc = build_city("nyc_times_square", seed=SEED, conditions=conditions_key)
    cond = nyc.get("conditions")
    print(
        f"[{tag}] {len(nyc['stop_nodes'])} stops, "
        f"arrival weights {nyc['arrival_weights'].min():.2f}–{nyc['arrival_weights'].max():.2f}"
    )

    cfg = SimConfig.from_calibration(
        nyc["calibration"],
        conditions=cond,
        sim_minutes=DAY_SPAN_MINUTES,
        warmup_minutes=120.0,
        seed=SEED,
    )
    print(
        f"[{tag}] arrival_rate_per_min={cfg.arrival_rate_per_min:.3f}, "
        f"buses_budget={cfg.num_buses}, headway_target={cfg.target_headway_min:.1f}m, "
        f"evasion={cfg.evasion_rate:.2f}, fare=${cfg.fare_price:.2f}, "
        f"diurnal={cfg.diurnal_profile} (day starts {cfg.day_start_hour:g}:00)"
    )
    results = simulate(nyc, cfg=cfg)
    print_summary(results)

    OUTPUTS.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    plot_route_map(nyc, results, ax=ax)
    route_path = OUTPUTS / f"{tag}_route_map.png"
    fig.savefig(route_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plot_sim_metrics(results)
    metrics_path = OUTPUTS / f"{tag}_sim_metrics.png"
    fig.savefig(metrics_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {route_path}")
    print(f"Saved {metrics_path}")
    return results


def main():
    # Baseline Midtown GTFS + NYC operating conditions.
    run_scenario("nyc_midtown", "nyc_midtown")

    # Same Midtown GTFS skeleton, thinned/scaled toward SF-like service so
    # fare-policy experiments can be compared under different city conditions
    # without swapping the underlying street geography.
    run_scenario("sf_like_from_nyc", "sf_like_from_nyc")


if __name__ == "__main__":
    main()
