"""Load real agency GTFS feeds and build Midtown-scale corridor networks."""

from __future__ import annotations

import csv
import io
import math
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import networkx as nx

CACHE_DIR = Path(__file__).resolve().parents[1] / "cache" / "gtfs"

GTFS_FEEDS = {
    "nyc_manhattan": "https://rrgtfsfeeds.s3.amazonaws.com/gtfs_m.zip",
    "sf_muni": "https://gtfs.sfmta.com/transitdata/google_transit.zip",
}


def _download_gtfs(feed_key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / f"{feed_key}.zip"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = GTFS_FEEDS[feed_key]
    print(f"[gtfs] downloading {feed_key} ...")
    urlretrieve(url, dest)
    return dest


def _read_csv(zf: zipfile.ZipFile, name: str) -> list[dict]:
    with zf.open(name) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
        return list(csv.DictReader(text))


def _in_bbox(lat: float, lon: float, bbox: tuple[float, float, float, float]) -> bool:
    north, south, east, west = bbox
    return south <= lat <= north and west <= lon <= east


def _project_local(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    x = (lon - lon0) * 111_320.0 * math.cos(math.radians(lat0))
    y = (lat - lat0) * 110_540.0
    return x, y


def _polyline_length(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def _classify_axis(stop_xy: np.ndarray, route_name: str | None = None) -> str:
    """
    Infer N/S vs E/W from path geometry; use route name only as a tiebreaker.

    Name heuristics alone misfire when a bbox clips a "crosstown" route onto an
    avenue segment (e.g. M34A inside Midtown runs mostly N/S).
    """
    name = (route_name or "").upper().replace(" ", "")
    name_guess = None
    for token in ("M34", "M42", "M50", "M66", "M72", "M79", "M86", "M96", "M106", "M116"):
        if name.startswith(token):
            name_guess = "EW"
            break
    if name_guess is None:
        for token in (
            "M1", "M2", "M3", "M4", "M5", "M7", "M10", "M11", "M15", "M20", "M55",
            "M101", "M102", "M103", "M104",
        ):
            if name == token or name.startswith(token + "-"):
                name_guess = "NS"
                break

    if len(stop_xy) < 2:
        return name_guess or "NS"

    # Midtown street grid is ~29° off map north; score travel on that basis.
    # (Matches city_pipeline.street_grid_axes for this study box.)
    along_ns = np.array([0.481, 0.877])
    along_ew = np.array([0.877, -0.481])
    d = np.diff(np.asarray(stop_xy, dtype=float), axis=0)
    ns = float(np.abs(d @ along_ns).sum())
    ew = float(np.abs(d @ along_ew).sum())
    if ns >= ew * 1.15:
        return "NS"
    if ew >= ns * 1.15:
        return "EW"
    return name_guess or ("NS" if ns >= ew else "EW")


def classify_axis_from_path(path_xy: np.ndarray, along_ns, along_ew,
                            route_name: str | None = None) -> str:
    """Reclassify a projected path using the city's street-grid unit vectors."""
    name = (route_name or "").upper().replace(" ", "")
    name_guess = None
    for token in ("M34", "M42", "M50", "M66", "M72", "M79", "M86", "M96", "M106", "M116"):
        if name.startswith(token):
            name_guess = "EW"
            break
    if name_guess is None:
        for token in (
            "M1", "M2", "M3", "M4", "M5", "M7", "M10", "M11", "M15", "M20", "M55",
            "M101", "M102", "M103", "M104",
        ):
            if name == token or name.startswith(token + "-"):
                name_guess = "NS"
                break

    xy = np.asarray(path_xy, dtype=float)
    if len(xy) < 2:
        return name_guess or "NS"
    d = np.diff(xy, axis=0)
    ns = float(np.abs(d @ np.asarray(along_ns, dtype=float)).sum())
    ew = float(np.abs(d @ np.asarray(along_ew, dtype=float)).sum())
    if ns >= ew * 1.15:
        return "NS"
    if ew >= ns * 1.15:
        return "EW"
    return name_guess or ("NS" if ns >= ew else "EW")


def place_stops_at_path_corners(
    G,
    routes: list[dict],
    target_n: int = 80,
    path_buffer_m: float = 40.0,
    min_spacing_m: float = 70.0,
) -> dict:
    """
    Ignore GTFS curb stops. Keep GTFS route *paths*, and place simulation stops
    only at OSM street corners that lie on those paths.

    `min_spacing_m` is a hard floor for the *pre-thin* corner set (typically
    ~block scale). Midtown agency gaps (~200–230 m) are applied afterward by
    `thin_routes` via CityConditions.min_stop_spacing_m. `target_n` is only an
    upper cap — spacing is never relaxed to hit it.
    """
    Gu = nx.Graph(G)
    # True corners: junctions where 3+ street links meet (not mid-block degree-2).
    corner_nodes = [n for n in Gu.nodes() if Gu.degree(n) >= 3]
    if len(corner_nodes) < 4:
        corner_nodes = list(Gu.nodes())

    corner_xy = {
        n: np.array([G.nodes[n]["x"], G.nodes[n]["y"]], dtype=float)
        for n in corner_nodes
    }

    # Build a quick spatial index via brute force (graphs are small Midtown clips).
    def near_path(node_xy: np.ndarray, path_xy: np.ndarray, buffer: float) -> bool:
        if path_xy is None or len(path_xy) < 2:
            return False
        # Distance to nearest polyline vertex is enough at Midtown resolution.
        return float(np.linalg.norm(path_xy - node_xy, axis=1).min()) <= buffer

    # Score corners by how many routes they serve (transfer value) and keep
    # those within the path buffer.
    route_paths = []
    for route in routes:
        path_xy = route.get("path_xy")
        if path_xy is None or len(path_xy) < 2:
            continue
        route_paths.append((route, np.asarray(path_xy, dtype=float)))

    candidates = []
    for n, xy in corner_xy.items():
        hits = [r["id"] for r, path in route_paths if near_path(xy, path, path_buffer_m)]
        if hits:
            candidates.append({
                "node": n,
                "xy": xy,
                "n_routes": len(hits),
                "route_ids": hits,
            })

    if not candidates:
        raise RuntimeError("no OSM corners found near GTFS paths — check CRS alignment")

    # Prefer transfer corners, then spread with a hard min-spacing floor.
    # Do NOT relax below min_spacing_m or pad to target_n — denser packing was
    # inventing mid-block dwells once spacing fell to ~35 m.
    candidates.sort(key=lambda c: (-c["n_routes"], c["xy"][0], c["xy"][1]))
    chosen = []
    for cand in candidates:
        if any(float(np.linalg.norm(cand["xy"] - c["xy"])) < min_spacing_m for c in chosen):
            continue
        chosen.append(cand)
        if len(chosen) >= target_n:
            break
    spacing = float(min_spacing_m)

    stop_coords = np.array([c["xy"] for c in chosen], dtype=float)
    stop_nodes = [c["node"] for c in chosen]
    stop_ids = [str(n) for n in stop_nodes]
    stop_names = [f"corner:{n}" for n in stop_nodes]
    index_of = {n: i for i, n in enumerate(stop_nodes)}

    from src.city_pipeline import axis_preferring_path, street_grid_axes

    along_ns, along_ew = street_grid_axes(G)

    def distance_along(path_xy: np.ndarray, point: np.ndarray) -> float:
        """Approximate arc length at the closest vertex on the polyline."""
        dists = np.linalg.norm(path_xy - point, axis=1)
        i = int(np.argmin(dists))
        if i == 0:
            return 0.0
        return float(np.linalg.norm(np.diff(path_xy[: i + 1], axis=0), axis=1).sum())

    new_routes = []
    for route, path_xy in route_paths:
        on_route = []
        for c in chosen:
            if near_path(c["xy"], path_xy, path_buffer_m):
                on_route.append((distance_along(path_xy, c["xy"]), index_of[c["node"]]))
        on_route.sort()
        idxs = [i for _, i in on_route]
        # Dedup consecutive.
        compact = []
        for i in idxs:
            if not compact or compact[-1] != i:
                compact.append(i)
        if len(compact) < 2:
            continue
        # Keep stops that ride the corridor spine. A curb node on a crossing
        # avenue can sit within the path buffer yet force a block-long jog
        # (M42's eastern "box" was one of these).
        along = along_ns if route.get("axis") == "NS" else along_ew
        across = along_ew if route.get("axis") == "NS" else along_ns
        along = np.asarray(along, dtype=float)
        along = along / np.linalg.norm(along)

        def _has_axis_edge(node, min_align: float = 0.7) -> bool:
            for _, v, _data in G.edges(node, data=True):
                dx = G.nodes[v]["x"] - G.nodes[node]["x"]
                dy = G.nodes[v]["y"] - G.nodes[node]["y"]
                norm = float(np.hypot(dx, dy))
                if norm < 1e-6:
                    continue
                if abs(float(np.dot([dx / norm, dy / norm], along))) >= min_align:
                    return True
            return False

        if len(compact) >= 2:
            across_vals = stop_coords[compact] @ across
            spine = float(np.median(across_vals))
            spine_tol_m = 25.0
            compact = [
                i for i in compact
                if abs(float(stop_coords[i] @ across) - spine) <= spine_tol_m
                and _has_axis_edge(stop_nodes[i])
            ]
        if len(compact) < 2:
            continue

        def _edge_length(u, v) -> float:
            data = G.get_edge_data(u, v)
            if data is None:
                data = G.get_edge_data(v, u)
            if data is None:
                return float(
                    np.hypot(
                        G.nodes[v]["x"] - G.nodes[u]["x"],
                        G.nodes[v]["y"] - G.nodes[u]["y"],
                    )
                )
            if 0 in data and isinstance(data[0], dict):
                return float(data[0].get("length", 0.0))
            return float(min(d.get("length", 1e9) for d in data.values()))

        def _route_segment(na, nb, chord: float):
            """Prefer axis-aligned OSM path; fall back to straight if it jogs."""
            try:
                segment = axis_preferring_path(G, na, nb, along, turn_penalty=8.0)
            except Exception:
                try:
                    segment = nx.shortest_path(Gu, na, nb, weight="length")
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    return [na, nb]
            if len(segment) < 2:
                return [na, nb]
            path_len = sum(
                _edge_length(u, v) for u, v in zip(segment, segment[1:])
            )
            # Missing mid-block / wrong-corner nodes make Dijkstra loop a block.
            if path_len > max(chord * 1.35, chord + 80.0):
                return [na, nb]
            return segment

        legs = [
            float(np.linalg.norm(stop_coords[b] - stop_coords[a]))
            for a, b in zip(compact, compact[1:])
        ]

        osm_path_nodes = []
        for a, b, chord in zip(compact, compact[1:], legs):
            segment = _route_segment(stop_nodes[a], stop_nodes[b], chord)
            if osm_path_nodes and segment and osm_path_nodes[-1] == segment[0]:
                segment = segment[1:]
            osm_path_nodes.extend(segment)
        if osm_path_nodes:
            osm_path_xy = np.array(
                [[G.nodes[n]["x"], G.nodes[n]["y"]] for n in osm_path_nodes],
                dtype=float,
            )
        else:
            osm_path_xy = stop_coords[compact]

        new_routes.append({
            **route,
            "stop_indices": compact,
            "leg_meters": legs,
            "path_xy": osm_path_xy,
            "path_nodes": osm_path_nodes,
        })

    return {
        "stop_coords": stop_coords,
        "stop_nodes": stop_nodes,
        "stop_ids": stop_ids,
        "stop_names": stop_names,
        "routes": new_routes,
        "n_candidates": len(candidates),
        "min_spacing_m": spacing if len(chosen) < target_n else min_spacing_m,
    }


def collapse_stops_to_corners(
    stop_coords: np.ndarray,
    routes: list[dict],
    stop_ids: list[str] | None = None,
    stop_names: list[str] | None = None,
    target_n: int = 80,
    min_radius_m: float = 30.0,
    max_radius_m: float = 120.0,
) -> dict:
    """
    Merge GTFS curb-side duplicates into ~one hub per corner.

    Opposite-direction stops at the same intersection are usually 20–50 m apart
    in Midtown GTFS. Clustering within a radius collapses those piles so the
    map reads as one stop per corner, while aiming for about `target_n` hubs.
    """
    coords = np.asarray(stop_coords, dtype=float)
    n = len(coords)
    if n <= target_n:
        return {
            "stop_coords": coords,
            "routes": routes,
            "stop_ids": stop_ids,
            "stop_names": stop_names,
            "old_to_new": {i: i for i in range(n)},
            "merge_radius_m": 0.0,
        }

    def cluster(radius: float) -> list[list[int]]:
        """Greedy: each unassigned stop seeds a cluster of everything within radius."""
        remaining = set(range(n))
        groups: list[list[int]] = []
        # Seed from a spatial sort so clusters grow from a stable sweep.
        order = list(np.lexsort((coords[:, 0], coords[:, 1])))
        for seed in order:
            if seed not in remaining:
                continue
            members = [seed]
            remaining.remove(seed)
            # Absorb nearby remaining stops (one-pass around the seed; then
            # expand once more from new members for mild chaining at a corner).
            frontier = [seed]
            while frontier:
                nxt = []
                for i in frontier:
                    for j in list(remaining):
                        if float(np.linalg.norm(coords[i] - coords[j])) <= radius:
                            remaining.remove(j)
                            members.append(j)
                            nxt.append(j)
                frontier = nxt
            groups.append(members)
        return groups

    # Binary-search a merge radius that lands near target_n hubs.
    lo, hi = min_radius_m, max_radius_m
    best_groups = cluster(min_radius_m)
    best_radius = min_radius_m

    def score(groups, radius):
        # Prefer counts at or just above the target; penalize undershooting more.
        n_hubs = len(groups)
        return (abs(n_hubs - target_n) + (0 if n_hubs >= target_n else 5), radius)

    best_key = score(best_groups, best_radius)
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        groups = cluster(mid)
        key = score(groups, mid)
        if key < best_key:
            best_groups, best_radius, best_key = groups, mid, key
        if len(groups) > target_n:
            lo = mid  # still too many hubs → merge more aggressively
        else:
            hi = mid

    groups = best_groups
    radius = best_radius

    # Only force-merge further if we are still clearly above the target.
    hubs = [coords[g].mean(axis=0) for g in groups]
    while len(groups) > target_n + 2:
        best_pair = None
        best_d = float("inf")
        for i in range(len(hubs)):
            for j in range(i + 1, len(hubs)):
                d = float(np.linalg.norm(hubs[i] - hubs[j]))
                if d < best_d:
                    best_d = d
                    best_pair = (i, j)
        i, j = best_pair
        groups[i] = groups[i] + groups[j]
        hubs[i] = coords[groups[i]].mean(axis=0)
        del groups[j]
        del hubs[j]

    # Representative = member closest to the cluster centroid.
    survivors = []
    old_to_new: dict[int, int] = {}
    new_ids = []
    new_names = []
    for new_i, members in enumerate(groups):
        centroid = coords[members].mean(axis=0)
        rep = min(members, key=lambda k: float(np.linalg.norm(coords[k] - centroid)))
        survivors.append(rep)
        for m in members:
            old_to_new[m] = new_i
        if stop_ids is not None:
            new_ids.append(stop_ids[rep])
        if stop_names is not None:
            new_names.append(stop_names[rep])

    new_coords = coords[survivors]
    new_routes = []
    for route in routes:
        remapped = []
        for idx in route["stop_indices"]:
            hub = old_to_new[idx]
            if not remapped or remapped[-1] != hub:
                remapped.append(hub)
        if len(remapped) < 2:
            continue
        legs = [
            float(np.linalg.norm(new_coords[b] - new_coords[a]))
            for a, b in zip(remapped, remapped[1:])
        ]
        new_routes.append({**route, "stop_indices": remapped, "leg_meters": legs})

    return {
        "stop_coords": new_coords,
        "routes": new_routes,
        "stop_ids": new_ids if stop_ids is not None else None,
        "stop_names": new_names if stop_names is not None else None,
        "old_to_new": old_to_new,
        "merge_radius_m": radius,
        "n_before": n,
        "n_after": len(survivors),
    }


def load_gtfs_routes_in_bbox(
    feed_key: str,
    bbox: tuple[float, float, float, float],
    n_ns: int = 2,
    n_ew: int = 2,
    preferred_routes: list[str] | None = None,
    include_all: bool = False,
) -> dict:
    """
    Build a corridor network from a real GTFS feed clipped to `bbox`.

    include_all=True keeps every route with ≥2 stops in the box and every
    GTFS stop inside the box. Otherwise picks up to `n_ns` + `n_ew` routes.
    """
    zpath = _download_gtfs(feed_key)
    north, south, east, west = bbox
    lat0 = 0.5 * (north + south)
    lon0 = 0.5 * (east + west)

    with zipfile.ZipFile(zpath) as zf:
        stops_rows = _read_csv(zf, "stops.txt")
        routes_rows = _read_csv(zf, "routes.txt")
        trips_rows = _read_csv(zf, "trips.txt")
        stop_times_rows = _read_csv(zf, "stop_times.txt")
        shapes_rows = _read_csv(zf, "shapes.txt") if "shapes.txt" in zf.namelist() else []

    route_name = {
        r["route_id"]: (r.get("route_short_name") or r.get("route_id"))
        for r in routes_rows
    }

    stops_in_box = {}
    for row in stops_rows:
        try:
            lat, lon = float(row["stop_lat"]), float(row["stop_lon"])
        except (KeyError, ValueError):
            continue
        if _in_bbox(lat, lon, bbox):
            stops_in_box[row["stop_id"]] = {
                "stop_id": row["stop_id"],
                "name": row.get("stop_name", row["stop_id"]),
                "lat": lat,
                "lon": lon,
                "xy": _project_local(lat, lon, lat0, lon0),
            }

    if len(stops_in_box) < 4:
        raise RuntimeError(f"fewer than 4 GTFS stops inside bbox for {feed_key}")

    trip_meta = {
        t["trip_id"]: {"route_id": t["route_id"], "shape_id": t.get("shape_id", "")}
        for t in trips_rows
    }

    trip_seq_sorted: dict[str, list[tuple[int, str]]] = {}
    for row in stop_times_rows:
        if row["stop_id"] not in stops_in_box:
            continue
        try:
            seq_n = int(float(row["stop_sequence"]))
        except ValueError:
            continue
        trip_seq_sorted.setdefault(row["trip_id"], []).append((seq_n, row["stop_id"]))

    best_by_route: dict[str, list[str]] = {}
    shape_by_route: dict[str, str] = {}
    for trip_id, pairs in trip_seq_sorted.items():
        meta = trip_meta.get(trip_id)
        if meta is None:
            continue
        pairs = sorted(pairs)
        deduped = []
        for _, sid in pairs:
            if not deduped or deduped[-1] != sid:
                deduped.append(sid)
        if len(deduped) < 2:
            continue
        route_id = meta["route_id"]
        if len(deduped) > len(best_by_route.get(route_id, [])):
            best_by_route[route_id] = deduped
            shape_by_route[route_id] = meta.get("shape_id") or ""

    shapes: dict[str, list[tuple[int, float, float]]] = {}
    for row in shapes_rows:
        try:
            shapes.setdefault(row["shape_id"], []).append(
                (
                    int(float(row["shape_pt_sequence"])),
                    float(row["shape_pt_lat"]),
                    float(row["shape_pt_lon"]),
                )
            )
        except (KeyError, ValueError):
            continue

    candidates = []
    for route_id, seq in best_by_route.items():
        xy = np.array([stops_in_box[s]["xy"] for s in seq])
        axis = _classify_axis(xy, route_name.get(route_id, route_id))
        candidates.append({
            "route_id": route_id,
            "name": route_name.get(route_id, route_id),
            "axis": axis,
            "seq": seq,
            "n_stops": len(seq),
            "length_m": _polyline_length(xy),
        })

    preferred = set(preferred_routes or [])
    if include_all:
        chosen = sorted(candidates, key=lambda c: (-c["n_stops"], c["name"]))
    else:
        ns = [c for c in candidates if c["axis"] == "NS"]
        ew = [c for c in candidates if c["axis"] == "EW"]

        def rank(c):
            hit = c["route_id"] in preferred or c["name"] in preferred
            return (hit, c["n_stops"], c["length_m"])

        ns.sort(key=rank, reverse=True)
        ew.sort(key=rank, reverse=True)
        chosen = ns[:n_ns] + ew[:n_ew]
        if len(chosen) < 2:
            candidates.sort(key=lambda c: c["n_stops"], reverse=True)
            chosen = candidates[: max(2, n_ns + n_ew)]

    stop_ids: list[str] = []
    seen: set[str] = set()
    for c in chosen:
        for sid in c["seq"]:
            if sid not in seen:
                seen.add(sid)
                stop_ids.append(sid)
    if include_all:
        for sid in stops_in_box:
            if sid not in seen:
                seen.add(sid)
                stop_ids.append(sid)

    stop_lons = np.array([stops_in_box[s]["lon"] for s in stop_ids], dtype=float)
    stop_lats = np.array([stops_in_box[s]["lat"] for s in stop_ids], dtype=float)
    stop_coords = np.array([stops_in_box[s]["xy"] for s in stop_ids], dtype=float)
    stop_names = [stops_in_box[s]["name"] for s in stop_ids]
    index_of = {sid: i for i, sid in enumerate(stop_ids)}

    routes = []
    used_ids: set[str] = set()
    for c in chosen:
        indices = [index_of[s] for s in c["seq"] if s in index_of]
        if len(indices) < 2:
            continue
        compact = [indices[0]]
        for idx in indices[1:]:
            if idx != compact[-1]:
                compact.append(idx)
        if len(compact) < 2:
            continue

        leg_meters = [
            float(np.linalg.norm(stop_coords[b] - stop_coords[a]))
            for a, b in zip(compact, compact[1:])
        ]

        shape_id = shape_by_route.get(c["route_id"], "")
        if shape_id and shape_id in shapes:
            pts = sorted(shapes[shape_id])
            full_xy = np.array([_project_local(lat, lon, lat0, lon0) for _, lat, lon in pts])
            full_ll = np.array([(lon, lat) for _, lat, lon in pts])
            lo = stop_coords[compact].min(axis=0) - 150.0
            hi = stop_coords[compact].max(axis=0) + 150.0
            mask = (
                (full_xy[:, 0] >= lo[0]) & (full_xy[:, 0] <= hi[0])
                & (full_xy[:, 1] >= lo[1]) & (full_xy[:, 1] <= hi[1])
            )
            if mask.any():
                path_xy, path_ll = full_xy[mask], full_ll[mask]
            else:
                path_xy = stop_coords[compact]
                path_ll = np.column_stack([stop_lons[compact], stop_lats[compact]])
        else:
            path_xy = stop_coords[compact]
            path_ll = np.column_stack([stop_lons[compact], stop_lats[compact]])

        base = str(c["name"]).replace(" ", "") or c["route_id"]
        route_key = base
        suffix = 2
        while route_key in used_ids:
            route_key = f"{base}_{suffix}"
            suffix += 1
        used_ids.add(route_key)

        routes.append({
            "id": route_key,
            "axis": c["axis"],
            "gtfs_route_id": c["route_id"],
            "name": c["name"],
            "stop_indices": compact,
            "leg_meters": leg_meters,
            "path_xy": path_xy,
            "path_ll": path_ll,
            "path_nodes": [],
        })

    n = len(stop_ids)
    dist = np.full((n, n), np.inf)
    np.fill_diagonal(dist, 0.0)
    for route in routes:
        idxs = route["stop_indices"]
        legs = route["leg_meters"]
        prefix = [0.0]
        for leg in legs:
            prefix.append(prefix[-1] + leg)
        for i in range(len(idxs)):
            for j in range(len(idxs)):
                d = abs(prefix[j] - prefix[i])
                a, b = idxs[i], idxs[j]
                if d < dist[a, b]:
                    dist[a, b] = d
    for i in range(n):
        for j in range(n):
            if not np.isfinite(dist[i, j]):
                dist[i, j] = float(np.linalg.norm(stop_coords[i] - stop_coords[j]))

    return {
        "feed_key": feed_key,
        "stop_ids": stop_ids,
        "stop_names": stop_names,
        "stop_lats": stop_lats,
        "stop_lons": stop_lons,
        "stop_coords": stop_coords,
        "routes": routes,
        "dist_matrix": dist,
        "lat0": lat0,
        "lon0": lon0,
        "n_stops_in_bbox": len(stops_in_box),
        "n_routes_in_bbox": len(candidates),
    }
