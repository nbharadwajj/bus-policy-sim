"""Plots for the city network and simulation output."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox

from src.simulation import (
    clock_hour,
    diurnal_factor,
    traffic_level,
)

# Okabe–Ito cycling palette + dash patterns. NS routes use solid/dotted;
# EW routes use dashed/dash-dot — readable without relying on hue alone.
_NS_STYLES = [
    {"color": "#0072B2", "linestyle": "-", "linewidth": 2.4},
    {"color": "#000000", "linestyle": (0, (1, 1.2)), "linewidth": 2.4},
    {"color": "#009E73", "linestyle": "-", "linewidth": 2.4},
    {"color": "#56B4E9", "linestyle": (0, (1, 1.2)), "linewidth": 2.4},
]
_EW_STYLES = [
    {"color": "#E69F00", "linestyle": "--", "linewidth": 2.4},
    {"color": "#CC79A7", "linestyle": "-.", "linewidth": 2.4},
    {"color": "#D55E00", "linestyle": "--", "linewidth": 2.4},
    {"color": "#F0E442", "linestyle": "-.", "linewidth": 2.4},
]


def _style_for(corridor, ns_i: list[int], ew_i: list[int]):
    if corridor.axis == "NS":
        style = _NS_STYLES[ns_i[0] % len(_NS_STYLES)]
        ns_i[0] += 1
    else:
        style = _EW_STYLES[ew_i[0] % len(_EW_STYLES)]
        ew_i[0] += 1
    return style


def plot_route_map(city_data: dict, results, ax=None, show_pois: bool = True):
    """Street basemap with GTFS (or synthetic) corridors and stops."""
    graph = city_data["graph"]
    network = results.network
    weights = np.asarray(city_data.get("arrival_weights", np.ones(network.num_stops)))
    conditions = city_data.get("conditions")
    cond_label = f" [{conditions.label}]" if conditions else ""

    fig, ax = ox.plot_graph(
        graph,
        ax=ax,
        show=False,
        close=False,
        node_size=0,
        edge_color="#cccccc",
        edge_linewidth=0.5,
        bgcolor="white",
    )

    if show_pois:
        pois = city_data["pois"].to_crs(graph.graph["crs"]).geometry.centroid
        ax.scatter(pois.x, pois.y, c="#999999", s=2, alpha=0.2, zorder=2, label="POIs")

    ns_i, ew_i = [0], [0]
    # Draw routes first; only label a subset in the legend to avoid clutter.
    legend_budget = 10
    labeled = 0
    active_stops: set[int] = set()
    for corridor in network.routes:
        style = _style_for(corridor, ns_i, ew_i)
        axis_tag = "N/S" if corridor.axis == "NS" else "E/W"
        active_stops.update(corridor.stop_indices)

        # Prefer stop-ordered polylines — OSM snaps can still jog a block when a
        # curb node sits on a crossing avenue inside the path buffer.
        pts = network.coords[corridor.stop_indices]
        if len(pts) >= 2:
            xs, ys = pts[:, 0], pts[:, 1]
        elif corridor.path_xy is not None and len(corridor.path_xy) >= 2:
            xs = corridor.path_xy[:, 0]
            ys = corridor.path_xy[:, 1]
        elif corridor.path_nodes and len(corridor.path_nodes) >= 2:
            xs = [graph.nodes[n]["x"] for n in corridor.path_nodes]
            ys = [graph.nodes[n]["y"] for n in corridor.path_nodes]
        else:
            continue

        show_label = labeled < legend_budget
        ax.plot(
            xs, ys,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            solid_capstyle="round",
            alpha=0.85,
            zorder=3,
            label=(f"{corridor.name} ({axis_tag})" if show_label else None),
        )
        if show_label:
            labeled += 1

    coords = network.coords
    if active_stops:
        stop_list = sorted(active_stops)
        coords = coords[stop_list]
        weights = weights[stop_list] if len(weights) == network.num_stops else np.ones(len(stop_list))
        annotate_ids = stop_list
    else:
        annotate_ids = list(range(len(coords)))
    sizes = 35 + 55 * (weights / max(float(np.max(weights)), 1e-9))
    ax.scatter(
        coords[:, 0], coords[:, 1],
        c="white", edgecolors="black", linewidths=0.9, s=sizes, zorder=5, label="Bus stops",
    )
    # Number stops only when the map isn't too dense.
    if len(annotate_ids) <= 100:
        for stop_index, (x, y) in zip(annotate_ids, coords):
            ax.annotate(
                str(stop_index), (x, y), color="black", fontsize=5.5, fontweight="bold",
                ha="center", va="center", zorder=6,
            )

    extra = "" if labeled >= len(network.routes) else f" (+{len(network.routes) - labeled} more routes)"
    ax.set_title(
        f"{results.city_key}{cond_label} — {len(annotate_ids)} stops, "
        f"{len(network.routes)} routes, "
        f"{network.total_route_length_m() / 1000:.1f} km{extra}\n"
        f"solid/dotted = N/S · dashed/dash-dot = E/W · stop size ∝ POI demand"
    )
    ax.legend(loc="lower left", fontsize=7, framealpha=0.95)
    return ax


def plot_study_network(city_data: dict, results, ax=None, label_streets: bool = True):
    """
    Reader-facing Midtown figure: one spine per street, stop-ordered lines,
    only stops that buses actually serve.
    """
    from src.city_pipeline import street_grid_axes

    graph = city_data["graph"]
    network = results.network
    along_ns, along_ew = street_grid_axes(graph)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.5, 8.5), facecolor="white")
    ax.set_facecolor("white")

    for u, v, _ in graph.edges(data=True):
        ax.plot(
            [graph.nodes[u]["x"], graph.nodes[v]["x"]],
            [graph.nodes[u]["y"], graph.nodes[v]["y"]],
            color="#e5e5e5", lw=0.5, zorder=1,
        )

    def _is_linear(corridor, max_secondary_m=120.0, max_secondary_frac=0.25):
        coords = network.coords[corridor.stop_indices]
        ns = float(np.ptp(coords @ along_ns))
        ew = float(np.ptp(coords @ along_ew))
        primary, secondary = (ns, ew) if corridor.axis == "NS" else (ew, ns)
        if primary < 200:
            return False
        return secondary <= max_secondary_m or secondary / primary <= max_secondary_frac

    def _spine_key(corridor):
        coords = network.coords[corridor.stop_indices]
        across = along_ew if corridor.axis == "NS" else along_ns
        return corridor.axis, round(float(np.median(coords @ across)) / 40.0) * 40.0

    by_spine: dict = {}
    for corridor in network.routes:
        if not _is_linear(corridor):
            continue
        coords = network.coords[corridor.stop_indices]
        along = along_ns if corridor.axis == "NS" else along_ew
        length = float(np.ptp(coords @ along))
        key = _spine_key(corridor)
        if key not in by_spine or length > by_spine[key][0]:
            by_spine[key] = (length, corridor)

    ns_lab = ew_lab = False
    shown: set[int] = set()
    for _, corridor in by_spine.values():
        coords = network.coords[list(corridor.stop_indices)]
        along = along_ns if corridor.axis == "NS" else along_ew
        xy = coords[np.argsort(coords @ along)]
        shown.update(corridor.stop_indices)
        if corridor.axis == "NS":
            style = dict(color="#2B6CB0", ls="-", lw=2.4)
            label = "N/S corridor" if not ns_lab else None
            ns_lab = True
        else:
            style = dict(color="#C05621", ls="--", lw=2.4)
            label = "E/W corridor" if not ew_lab else None
            ew_lab = True
        ax.plot(xy[:, 0], xy[:, 1], zorder=3, alpha=0.95, label=label, **style)

    if shown:
        pts = network.coords[sorted(shown)]
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=30, c="white", edgecolors="#111111", linewidths=0.9, zorder=5, label="Stop",
        )

    if label_streets:
        _annotate_midtown_streets(ax, graph, along_ns, along_ew)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Midtown study network\n"
        "≈ 34th–50th St · 9th Ave–Lexington Ave",
        fontsize=12, pad=12,
    )
    ax.legend(loc="lower left", frameon=True, fancybox=False, edgecolor="#cccccc", fontsize=9)
    return ax


def _annotate_midtown_streets(ax, graph, along_ns, along_ew):
    """Place short labels on major avenues and a few cross-streets."""
    from collections import defaultdict

    # Short display names for the Midtown box; skip minor / duplicate aliases.
    avenue_labels = {
        "9th Avenue": "9th Ave",
        "8th Avenue": "8th Ave",
        "7th Avenue": "7th Ave",
        "6th Avenue": "6th Ave",
        "5th Avenue": "5th Ave",
        "Madison Avenue": "Madison",
        "Park Avenue": "Park Ave",
        "Lexington Avenue": "Lexington",
        "Broadway": "Broadway",
    }
    street_labels = {
        "West 34th Street": "34th St",
        "East 34th Street": "34th St",
        "West 42nd Street": "42nd St",
        "East 42nd Street": "42nd St",
        "West 50th Street": "50th St",
        "East 50th Street": "50th St",
    }

    def _edge_names(data):
        n = data.get("name")
        if n is None:
            return []
        if isinstance(n, list):
            return list(n)
        return [n]

    buckets: dict[str, list] = defaultdict(list)
    for u, v, data in graph.edges(data=True):
        x0, y0 = graph.nodes[u]["x"], graph.nodes[u]["y"]
        x1, y1 = graph.nodes[v]["x"], graph.nodes[v]["y"]
        mid = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)], dtype=float)
        for name in _edge_names(data):
            if name in avenue_labels or name in street_labels:
                buckets[name].append(mid)

    # One label per display name; for W/E 34th etc., merge to the same text.
    placed_text: dict[str, list] = defaultdict(list)
    for name, mids in buckets.items():
        if name in avenue_labels:
            text = avenue_labels[name]
            axis = "NS"
        else:
            text = street_labels[name]
            axis = "EW"
        pts = np.asarray(mids, dtype=float)
        # Put avenue labels near the south edge of their spine; streets near west.
        if axis == "NS":
            order = np.argsort(pts @ along_ns)
            anchor = pts[order[max(0, len(order) // 8)]]
            # Angle of the avenue in plot coordinates.
            span = pts[order[-1]] - pts[order[0]]
            angle = float(np.degrees(np.arctan2(span[1], span[0])))
        else:
            order = np.argsort(pts @ along_ew)
            anchor = pts[order[max(0, len(order) // 10)]]
            span = pts[order[-1]] - pts[order[0]]
            angle = float(np.degrees(np.arctan2(span[1], span[0])))
        placed_text[text].append((anchor, angle, axis, float(np.linalg.norm(span))))

    for text, entries in placed_text.items():
        # Prefer the longest named segment if W/E duplicates merge.
        anchor, angle, axis, _ = max(entries, key=lambda e: e[3])
        # Keep text roughly upright.
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        ax.text(
            anchor[0], anchor[1], text,
            fontsize=7.5, color="#555555", ha="center", va="center",
            rotation=angle, rotation_mode="anchor",
            zorder=6, clip_on=True,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
        )


def _series_over_time(log: list, events: dict[str, float]):
    times, values = [0.0], [0.0]
    total = 0.0
    for entry in log:
        delta = events.get(entry["event"])
        if delta is None:
            continue
        total += delta
        times.append(entry["time"])
        values.append(total)
    return np.array(times), np.array(values)


def _clock_label(hour: float) -> str:
    h = hour % 24.0
    hh = int(h) % 24
    mm = int(round((h - int(h)) * 60)) % 60
    return f"{hh}:{mm:02d}"


def _time_axis(ax, cfg, t_max: float):
    """Label x-axis in wall-clock hours when diurnal profile is on."""
    if not getattr(cfg, "diurnal_profile", False):
        ax.set_xlabel("minute")
        return
    # After rebase, sim minute 0 == day_start_hour.
    start = float(cfg.day_start_hour)
    end = start + t_max / 60.0
    # Tick on whole hours.
    first = int(np.ceil(start - 1e-9))
    last = int(np.floor(end + 1e-9))
    hours = list(range(first, last + 1))
    if len(hours) > 14:
        hours = hours[::2]
    ticks = [(h - start) * 60.0 for h in hours]
    ax.set_xticks(ticks)
    ax.set_xticklabels([_clock_label(h) for h in hours], fontsize=8)
    ax.set_xlabel("clock time")


def _shade_peaks(ax, cfg, t_max: float):
    """Faint AM / PM crest bands on time-series panels."""
    if not getattr(cfg, "diurnal_profile", False):
        return
    start = float(cfg.day_start_hour)

    def band(center, width, color, label):
        lo = (center - width - start) * 60.0
        hi = (center + width - start) * 60.0
        if hi < 0 or lo > t_max:
            return
        ax.axvspan(max(0.0, lo), min(t_max, hi), color=color, alpha=0.08, zorder=0, label=label)

    band(cfg.am_peak_hour, cfg.am_peak_width_hours, "#56B4E9", "AM crest")
    band(cfg.pm_peak_hour, cfg.pm_peak_width_hours, "#D55E00", "PM crest")


def plot_sim_metrics(results, fig=None):
    """Wait, journey time, diurnal passenger times, and BPR congestion."""
    if fig is None:
        fig = plt.figure(figsize=(12, 8))
    axes = fig.subplots(2, 2)
    summary = results.summary()
    cfg = results.config
    t_max = float(summary["sim_minutes"])
    warmup = float(cfg.warmup_minutes) if cfg.warm_start else 0.0

    waits = [p.wait_minutes for p in results.boarded if p.wait_minutes is not None]
    # Door-to-door: arrive at stop → final alight (includes first wait + ride).
    journeys = [
        p.deboard_time - p.arrival_time
        for p in results.completed
        if p.deboard_time is not None and p.board_time is not None
    ]
    # First board → final alight (excludes initial wait).
    rides = [
        p.trip_minutes
        for p in results.completed
        if p.trip_minutes is not None and p.board_time is not None
    ]

    ax = axes[0][0]
    if waits:
        ax.hist(waits, bins=25, color="#0072B2", edgecolor="white")
        ax.axvline(np.mean(waits), color="#D55E00", linestyle="--",
                   label=f"mean {np.mean(waits):.1f} min")
        ax.axvline(np.percentile(waits, 90), color="#CC79A7", linestyle=":",
                   label=f"p90 {np.percentile(waits, 90):.1f} min")
        ax.legend(fontsize=8)
    ax.set_title("Wait to first board")
    ax.set_xlabel("minutes")
    ax.set_ylabel("passengers")

    ax = axes[0][1]
    if journeys:
        ax.hist(journeys, bins=25, color="#009E73", edgecolor="white", alpha=0.9)
        ax.axvline(np.mean(journeys), color="#D55E00", linestyle="--",
                   label=f"door-to-door mean {np.mean(journeys):.1f}")
        if rides:
            ax.axvline(np.mean(rides), color="#0072B2", linestyle=":",
                       label=f"ride mean {np.mean(rides):.1f}")
        ax.axvline(np.percentile(journeys, 90), color="#CC79A7", linestyle="-.",
                   label=f"door-to-door p90 {np.percentile(journeys, 90):.1f}")
        ax.legend(fontsize=7)
    ax.set_title("Journey time (arrive → final alight)")
    ax.set_xlabel("minutes")
    ax.set_ylabel("passengers")

    ax = axes[1][0]
    _shade_peaks(ax, cfg, t_max)
    # Hourly mean wait / door-to-door through the measurement window.
    completed = [
        p for p in results.completed
        if p.deboard_time is not None and p.board_time is not None
    ]
    if completed:
        # Bin by clock hour of trip start (arrival).
        hours = np.arange(int(np.floor(cfg.day_start_hour)),
                          int(np.ceil(cfg.day_start_hour + t_max / 60.0)))
        wait_by_h, journey_by_h, centers = [], [], []
        for h in hours:
            in_bin = []
            for p in completed:
                # arrival_time is absolute sim time; convert to clock hour.
                ch = clock_hour(cfg, p.arrival_time)
                if h <= ch < h + 1:
                    in_bin.append(p)
            if len(in_bin) < 20:
                continue
            centers.append(h + 0.5)
            wait_by_h.append(np.mean([p.wait_minutes for p in in_bin
                                      if p.wait_minutes is not None]))
            journey_by_h.append(np.mean([
                p.deboard_time - p.arrival_time for p in in_bin
            ]))
        # Plot against measurement-window minutes for shared time axis.
        xs = [(c - cfg.day_start_hour) * 60.0 for c in centers]
        if xs:
            ax.plot(xs, wait_by_h, color="#0072B2", marker="o", linewidth=2.0,
                    label="mean wait")
            ax.plot(xs, journey_by_h, color="#009E73", marker="o", linewidth=2.0,
                    label="mean door-to-door")
            ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Hourly mean passenger times")
    _time_axis(ax, cfg, t_max)
    ax.set_ylabel("minutes")

    ax = axes[1][1]
    _shade_peaks(ax, cfg, t_max)
    grid = np.linspace(0.0, t_max, max(int(t_max) + 1, 50))
    congestion = np.array([traffic_level(cfg, warmup + t) for t in grid])
    ax.plot(grid, congestion, color="#D55E00", linewidth=2.2, label="BPR T/T₀")
    ax.axhline(float(np.mean(congestion)), color="#666666", linestyle="--",
               linewidth=1.0, label=f"run mean {np.mean(congestion):.2f}")
    if cfg.diurnal_profile:
        ax2 = ax.twinx()
        activity = np.array([
            diurnal_factor(cfg, clock_hour(cfg, warmup + t)) for t in grid
        ])
        ax2.plot(grid, activity, color="#0072B2", alpha=0.5, linewidth=1.4,
                 label="demand factor")
        ax2.set_ylabel("demand factor", color="#0072B2", fontsize=8)
        ax2.tick_params(axis="y", labelsize=7, colors="#0072B2")
        ax2.set_ylim(0.5, max(1.8, float(activity.max()) + 0.1))
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, fontsize=7, loc="upper left")
    else:
        ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Street congestion on buses (BPR)")
    ax.set_ylabel("T / T₀")
    ax.set_ylim(1.0, max(1.8, float(np.nanmax(congestion)) + 0.15))
    _time_axis(ax, cfg, t_max)

    fare = float(cfg.fare_price)
    fare_bit = "free" if cfg.fare_type == "free" or fare <= 0 else f"${fare:.2f}"
    span = ""
    if cfg.diurnal_profile:
        end_h = cfg.day_start_hour + t_max / 60.0
        span = f" · {_clock_label(cfg.day_start_hour)}–{_clock_label(end_h)}"
    j_mean = float(np.mean(journeys)) if journeys else float("nan")
    fig.suptitle(
        f"{results.city_key} · fare {fare_bit} · "
        f"{summary['passengers_boarded']:,} boarded · "
        f"wait {summary['mean_wait_min']:.1f} / journey {j_mean:.1f} min · "
        f"T/T₀ {float(np.mean(congestion)):.2f}{span}"
    )
    fig.tight_layout()
    return fig


_EXPERIMENT_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]


def _passenger_times(results):
    waits = [p.wait_minutes for p in results.boarded if p.wait_minutes is not None]
    journeys = [
        p.deboard_time - p.arrival_time
        for p in results.completed
        if p.deboard_time is not None and p.board_time is not None
    ]
    return waits, journeys


def _mean_bpr(results) -> float:
    cfg = results.config
    t_max = float(results.config.sim_minutes)
    warmup = float(cfg.warmup_minutes) if cfg.warm_start else 0.0
    grid = np.linspace(0.0, t_max, max(int(t_max) // 5 + 1, 40))
    return float(np.mean([traffic_level(cfg, warmup + t) for t in grid]))


def _cdf_xy(values, grid=None):
    vals = np.sort(np.asarray(values, dtype=float))
    if len(vals) == 0:
        return np.array([]), np.array([])
    if grid is None:
        grid = np.linspace(vals.min(), vals.max(), 200)
    cdf = np.searchsorted(vals, grid, side="right") / len(vals)
    return grid, cdf


def plot_fare_experiments(results_by_label: dict, fig=None):
    """
    One line per fare experiment — CDFs + summary means. No diurnal time series.

    `results_by_label`: ordered dict like {"free": results, "−25%": ..., "base": ...}.
    """
    if fig is None:
        fig = plt.figure(figsize=(11, 7.5))
    axes = fig.subplots(2, 2)

    # Shared CDF grids so lines are comparable.
    all_waits = []
    all_journeys = []
    means = []  # (label, wait, journey, boarded, tt0, fare)
    for label, res in results_by_label.items():
        waits, journeys = _passenger_times(res)
        all_waits.extend(waits)
        all_journeys.extend(journeys)
        fare = float(res.config.fare_price)
        means.append({
            "label": label,
            "fare": fare,
            "wait": float(np.mean(waits)) if waits else float("nan"),
            "journey": float(np.mean(journeys)) if journeys else float("nan"),
            "boarded": int(res.summary()["passengers_boarded"]),
            "tt0": _mean_bpr(res),
        })

    wait_grid = np.linspace(0.0, np.percentile(all_waits, 99) if all_waits else 1.0, 200)
    journey_grid = np.linspace(
        0.0, np.percentile(all_journeys, 99) if all_journeys else 1.0, 200
    )

    ax = axes[0][0]
    for i, (label, res) in enumerate(results_by_label.items()):
        waits, _ = _passenger_times(res)
        x, y = _cdf_xy(waits, wait_grid)
        ax.plot(x, y, color=_EXPERIMENT_COLORS[i % len(_EXPERIMENT_COLORS)],
                linewidth=2.2, label=label)
    ax.set_title("Wait to first board (CDF)")
    ax.set_xlabel("minutes")
    ax.set_ylabel("fraction of passengers")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[0][1]
    for i, (label, res) in enumerate(results_by_label.items()):
        _, journeys = _passenger_times(res)
        x, y = _cdf_xy(journeys, journey_grid)
        ax.plot(x, y, color=_EXPERIMENT_COLORS[i % len(_EXPERIMENT_COLORS)],
                linewidth=2.2, label=label)
    ax.set_title("Door-to-door journey (CDF)")
    ax.set_xlabel("minutes")
    ax.set_ylabel("fraction of passengers")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1][0]
    xs = np.arange(len(means))
    ax.plot(xs, [m["wait"] for m in means], color="#0072B2", marker="o",
            linewidth=2.2, label="mean wait")
    ax.plot(xs, [m["journey"] for m in means], color="#009E73", marker="o",
            linewidth=2.2, label="mean door-to-door")
    ax.set_xticks(xs)
    ax.set_xticklabels([m["label"] for m in means])
    ax.set_title("Mean passenger times by fare")
    ax.set_ylabel("minutes")
    ax.legend(fontsize=8, loc="best")

    ax = axes[1][1]
    ax.plot(xs, [m["tt0"] for m in means], color="#D55E00", marker="o",
            linewidth=2.2, label="mean T/T₀")
    ax.set_xticks(xs)
    ax.set_xticklabels([m["label"] for m in means])
    ax.set_ylabel("T / T₀", color="#D55E00")
    ax.tick_params(axis="y", colors="#D55E00")
    ax2 = ax.twinx()
    ax2.plot(xs, [m["boarded"] / 1000.0 for m in means], color="#0072B2",
             marker="s", linewidth=2.0, linestyle="--", label="boarded (k)")
    ax2.set_ylabel("boarded (thousands)", color="#0072B2")
    ax2.tick_params(axis="y", colors="#0072B2")
    ax.set_title("Congestion vs ridership")
    lines, labs = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labs + labs2, fontsize=8, loc="best")

    # Tighten T/T0 axis so tiny fare effects are visible without lying.
    tt0s = [m["tt0"] for m in means]
    lo, hi = min(tt0s), max(tt0s)
    pad = max(0.01, (hi - lo) * 0.4)
    ax.set_ylim(lo - pad, hi + pad)

    fig.suptitle("Midtown fare experiments · same stops & fleet · κ = 1")
    fig.tight_layout()
    return fig
