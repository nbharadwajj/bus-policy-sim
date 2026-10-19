"""OSM city graph loading, POI-weighted stop placement, and calibration."""

from __future__ import annotations

import random
from dataclasses import dataclass

import networkx as nx
import numpy as np
import osmnx as ox

# ---------------------------------------------------------------------------
# 1. City graph loading
# ---------------------------------------------------------------------------

CITY_CONFIGS = {
    # bbox format for graph_from_bbox: (north, south, east, west) stored;
    # osmnx expects (west, south, east, north).
    # ~1.4 km N–S × ~1.7 km E–W (roughly 34th–50th St, 9th–5th Ave).
    # Sized so 15 stops at min_spacing_m=200 can pack without failing.
    "nyc_times_square": dict(
        bbox=(40.7615, 40.7485, -73.9750, -73.9950),
        network_type="drive",
    ),
    "sf_financial": dict(
        place="Financial District, San Francisco, CA, USA",
        network_type="drive",
    ),
    # Canal-belt / Dam core (~1.7 km N–S × ~1.7 km E–W): Centraal–Leidseplein,
    # Jordaan–Nieuwmarkt. Real OSM drive graph; corridors traced on those streets
    # (no GTFS — GVB is tram-heavy in the core).
    "amsterdam_centrum": dict(
        bbox=(52.3795, 52.3640, 4.9050, 4.8790),
        network_type="drive",
    ),
    "small_town": dict(
        place="Woodstock, Vermont, USA",
        network_type="drive",
    ),
}


# ---------------------------------------------------------------------------
# Midtown auto screenline capacity (for BPR C)
# ---------------------------------------------------------------------------
# Study box ≈ 34th–50th St × 9th–5th Ave → five one-way avenues (9,8,7,6,5).
# NYC DOT Midtown: one travel lane accommodates ~500–600 veh/hr
# (7th Ave 34–42 St corridor study). Use 550 as midpoint.
# Typical Midtown avenue: ~3 usable through lanes (parking / bus lane aside).
#
# N–S screenline capacity:
#   5 avenues × 3 lanes × 550 veh/hr/lane = 8,250 veh/hr ≈ 137.5 veh/min
#
# Midday volume prior (DOT Midtown avenue peaks often ~1,000–1,400 veh/hr):
#   5 × 1,200 = 6,000 veh/hr ≈ 100 veh/min → V/C ≈ 0.73
NYC_MIDTOWN_NS_AVENUES = 5
NYC_MIDTOWN_LANES_PER_AVENUE = 3
NYC_DOT_LANE_CAPACITY_VPH = 550.0
NYC_MIDTOWN_AVENUE_VOLUME_VPH = 1200.0  # representative midday thru volume

# Weekday diurnal crest amps from NYC Open Data ATR (Automated Traffic Volume
# Counts), pooled Midtown avenues vs midday (11–14h = 1.0):
#   9 Ave @ 34–35 St (2023), 7 Ave Broadway line (2017), 8 Ave line (2017),
#   6 Ave / Avenue of the Americas (2017). Midtown daytime is relatively flat
#   (commercial fills midday); overnight drops to ~0.32× midday.
NYC_ATR_AM_PEAK_HOUR = 8.0
NYC_ATR_AM_PEAK_AMP = 0.08  # crest ≈ 1.08× midday
NYC_ATR_PM_PEAK_HOUR = 19.0
NYC_ATR_PM_PEAK_AMP = 0.13  # crest ≈ 1.13× midday
NYC_ATR_NIGHT_DIP_AMP = 0.68  # 3:00 ≈ 0.32× midday


def nyc_midtown_screenline_capacity_per_min() -> float:
    """Aggregate N–S auto throughput capacity across the study box (veh/min)."""
    vph = (
        NYC_MIDTOWN_NS_AVENUES
        * NYC_MIDTOWN_LANES_PER_AVENUE
        * NYC_DOT_LANE_CAPACITY_VPH
    )
    return float(vph / 60.0)


def nyc_midtown_screenline_volume_per_min() -> float:
    """Representative N–S auto volume across the study box (veh/min)."""
    vph = NYC_MIDTOWN_NS_AVENUES * NYC_MIDTOWN_AVENUE_VOLUME_VPH
    return float(vph / 60.0)


# ---------------------------------------------------------------------------
# Amsterdam canal-belt auto screenline (for BPR C / V)
# ---------------------------------------------------------------------------
# Study box ≈ Centraal–Leidseplein × Jordaan–Nieuwmarkt (~1.7 km).
# Agenda Autoluw: cars are guests; many canal streets are 1 effective through
# lane at 30 km/h with bike friction → ~400–500 vph capacity (use 450).
# Midday through volumes on those streets are modest (~150–250 vph), not
# Midtown avenue flows.
#
# N–S screenline:
#   C ≈ 5 streets × 1 lane × 450 vph = 2,250 vph ≈ 37.5 /min
#   V ≈ 5 × 200 vph = 1,000 vph ≈ 16.7 /min → V/C ≈ 0.45
AMS_CENTRUM_NS_STREETS = 5
AMS_CENTRUM_LANES_PER_STREET = 1
AMS_LANE_CAPACITY_VPH = 450.0
AMS_CENTRUM_STREET_VOLUME_VPH = 200.0

# Centrum weekday demand is flatter than US CBD commuting (tourists + retail
# fill midday). Mild crests vs midday; overnight still drops hard.
AMS_DIURNAL_AM_PEAK_HOUR = 8.5
AMS_DIURNAL_AM_PEAK_AMP = 0.12
AMS_DIURNAL_PM_PEAK_HOUR = 17.5
AMS_DIURNAL_PM_PEAK_AMP = 0.18
AMS_DIURNAL_NIGHT_DIP_AMP = 0.55


def ams_centrum_screenline_capacity_per_min() -> float:
    """Aggregate N–S auto throughput capacity across the canal-belt box."""
    vph = (
        AMS_CENTRUM_NS_STREETS
        * AMS_CENTRUM_LANES_PER_STREET
        * AMS_LANE_CAPACITY_VPH
    )
    return float(vph / 60.0)


def ams_centrum_screenline_volume_per_min() -> float:
    """Representative N–S auto volume across the canal-belt box (veh/min)."""
    vph = AMS_CENTRUM_NS_STREETS * AMS_CENTRUM_STREET_VOLUME_VPH
    return float(vph / 60.0)


def load_city_graph_raw(config):
    """Pull a real street network from OSM, unprojected (lat/lon)."""
    if "bbox" in config:
        north, south, east, west = config["bbox"]
        G = ox.graph_from_bbox(
            (west, south, east, north), network_type=config["network_type"]
        )
    else:
        G = ox.graph_from_place(config["place"], network_type=config["network_type"])
    return G


def load_city_graph(config):
    """Pull and project a real street network from OSM (meters)."""
    G = load_city_graph_raw(config)
    return ox.project_graph(G)


# ---------------------------------------------------------------------------
# 2. Points of interest (real activity signal, replaces geometric centroid)
# ---------------------------------------------------------------------------

DEFAULT_POI_TAGS = {"amenity": True, "shop": True, "office": True, "tourism": True}


def get_pois(G, tags=None):
    """
    Pull points of interest (shops, offices, amenities, tourism sites) within
    the graph's boundary. Used as a real activity-density signal for stop
    placement -- OSM has no live traffic/pedestrian counts, but POI density
    is a well-established proxy for foot traffic and commercial activity.
    """
    if tags is None:
        tags = DEFAULT_POI_TAGS

    nodes_gdf = ox.graph_to_gdfs(G, edges=False)
    boundary = nodes_gdf.geometry.union_all().convex_hull

    pois = ox.features_from_polygon(boundary, tags=tags)
    return pois


def poi_weighted_stop_placement(
    G, pois, num_stops, radius_m=150, min_spacing_m=250, seed=None
):
    """
    Legacy helper: choose stops by POI density.

    Prefer `build_street_corridors` for the main pipeline — stops should sit on
    bus paths, while POIs drive arrival rates via `poi_arrival_weights`.
    """
    if seed is not None:
        random.seed(seed)

    # Project POIs into the graph CRS so distance checks are in meters.
    pois_projected = pois.to_crs(G.graph["crs"])
    nodes_gdf = ox.graph_to_gdfs(G, edges=False)
    poi_points = pois_projected.geometry.centroid

    node_ids = list(G.nodes())
    weights = []
    for n in node_ids:
        node_point = nodes_gdf.loc[n].geometry
        nearby = poi_points[poi_points.distance(node_point) < radius_m]
        weights.append(len(nearby) + 1)  # +1 so zero-POI nodes still have nonzero chance

    chosen = []
    chosen_points = []
    attempts = 0
    max_attempts = num_stops * 200  # spacing rejects far more candidates than uniqueness alone

    while len(chosen) < num_stops and attempts < max_attempts:
        pick = random.choices(node_ids, weights=weights, k=1)[0]
        pick_point = nodes_gdf.loc[pick].geometry

        too_close = any(pick_point.distance(cp) < min_spacing_m for cp in chosen_points)

        if pick not in chosen and not too_close:
            chosen.append(pick)
            chosen_points.append(pick_point)

        attempts += 1

    if len(chosen) < num_stops:
        print(
            f"Warning: only placed {len(chosen)}/{num_stops} stops given "
            f"min_spacing_m={min_spacing_m}. Consider lowering min_spacing_m "
            f"or num_stops for this city's area."
        )

    return chosen


def poi_arrival_weights(G, stop_nodes, pois, radius_m=120):
    """
    Relative passenger arrival rates from nearby POI density.

    Returns an array of length len(stop_nodes) with mean 1.0, so
    `arrival_rate_per_min * weight[i]` is the Poisson rate at stop i.
    """
    pois_projected = pois.to_crs(G.graph["crs"])
    poi_points = pois_projected.geometry.centroid
    nodes_gdf = ox.graph_to_gdfs(G, edges=False)

    raw = []
    for node in stop_nodes:
        nearby = poi_points[poi_points.distance(nodes_gdf.loc[node].geometry) < radius_m]
        raw.append(len(nearby) + 1.0)

    weights = np.asarray(raw, dtype=float)
    return weights / weights.mean()


def _corridor_path(G, along_axis, along_coord, across_coord, target_across,
                   half_width_m=70.0, turn_penalty=12.0):
    """
    Path from the low-along extreme to the high-along extreme of a spine,
    staying on streets near `target_across` and preferring axis-aligned edges.

    Uses an undirected view of the graph so one-way streets don't leave the
    two spine endpoints disconnected.
    """
    Gu = nx.Graph(G)
    candidates = [
        n for n in Gu.nodes()
        if abs(across_coord[n] - target_across) <= half_width_m
    ]
    width = half_width_m
    while len(candidates) < 2 and width < half_width_m * 8:
        width *= 1.5
        candidates = [
            n for n in Gu.nodes()
            if abs(across_coord[n] - target_across) <= width
        ]
    if len(candidates) < 2:
        raise RuntimeError("not enough nodes near corridor spine")

    # Anchor endpoints on nodes that are both extreme along the corridor and
    # close to the spine, so we don't start on a parallel street a block away.
    start = min(
        candidates,
        key=lambda n: (along_coord[n], abs(across_coord[n] - target_across)),
    )
    end = max(
        candidates,
        key=lambda n: (along_coord[n], -abs(across_coord[n] - target_across)),
    )
    # Prefer the candidate nearest the spine among the most-extreme 20%.
    along_vals = np.array([along_coord[n] for n in candidates])
    lo_cut = np.percentile(along_vals, 15)
    hi_cut = np.percentile(along_vals, 85)
    south = [n for n in candidates if along_coord[n] <= lo_cut]
    north = [n for n in candidates if along_coord[n] >= hi_cut]
    if south:
        start = min(south, key=lambda n: abs(across_coord[n] - target_across))
    if north:
        end = min(north, key=lambda n: abs(across_coord[n] - target_across))

    along = np.asarray(along_axis, dtype=float)
    along = along / np.linalg.norm(along)

    def weight(u, v, _data):
        length = _edge_length(G, u, v)
        dx = G.nodes[v]["x"] - G.nodes[u]["x"]
        dy = G.nodes[v]["y"] - G.nodes[u]["y"]
        norm = float(np.hypot(dx, dy))
        alignment = abs(np.dot([dx / norm, dy / norm], along)) if norm else 0.0
        mid_across = 0.5 * (across_coord[u] + across_coord[v])
        lateral = abs(mid_across - target_across) / max(half_width_m, 1.0)
        # Strongly punish leaving the spine or turning onto the other axis.
        return length * (1.0 + turn_penalty * (1.0 - alignment) ** 2) * (1.0 + 3.0 * lateral)

    try:
        return nx.shortest_path(Gu, start, end, weight=weight)
    except nx.NetworkXNoPath as exc:
        raise RuntimeError(
            f"no street path near spine across={target_across:.1f}"
        ) from exc


def _edge_length(G, u, v):
    data = G.get_edge_data(u, v)
    if data is None:
        data = G.get_edge_data(v, u)
    if data is None:
        return float(np.hypot(G.nodes[v]["x"] - G.nodes[u]["x"],
                              G.nodes[v]["y"] - G.nodes[u]["y"]))
    # MultiGraph: take the shortest parallel edge.
    return min(float(attr.get("length", 0.0)) or 1.0 for attr in data.values())


def _sample_path_stops(G, path, min_spacing_m):
    """Take the path endpoints plus nodes at least `min_spacing_m` apart."""
    if len(path) < 2:
        return list(path)
    stops = [path[0]]
    traveled = 0.0
    for u, v in zip(path, path[1:]):
        traveled += _edge_length(G, u, v)
        if traveled >= min_spacing_m:
            stops.append(v)
            traveled = 0.0
    if stops[-1] != path[-1]:
        # Keep the terminal even if the last gap is short, unless it's almost
        # on top of the previous stop.
        if _path_distance(G, path, stops[-1], path[-1]) >= 0.5 * min_spacing_m:
            stops.append(path[-1])
    return stops


def _path_distance(G, path, a, b):
    """Distance along `path` between two nodes that appear on it."""
    i, j = path.index(a), path.index(b)
    if i > j:
        i, j = j, i
    return sum(_edge_length(G, path[k], path[k + 1]) for k in range(i, j))


def _merge_nearby_stops(G, stop_nodes, merge_radius_m=45.0):
    """
    Collapse stops within `merge_radius_m` so intersecting corridors share a
    transfer hub instead of two curb nodes a few metres apart.
    """
    coords = {n: np.array([G.nodes[n]["x"], G.nodes[n]["y"]]) for n in stop_nodes}
    surviving = []
    remap = {}
    for node in stop_nodes:
        matched = None
        for kept in surviving:
            if np.linalg.norm(coords[node] - coords[kept]) <= merge_radius_m:
                matched = kept
                break
        if matched is None:
            surviving.append(node)
            remap[node] = node
        else:
            remap[node] = matched
    return surviving, remap


def build_street_corridors(G, n_ns=2, n_ew=2, min_spacing_m=200.0,
                           corridor_half_width_m=70.0):
    """
    Build bus corridors on real streets, then place stops on those paths.

    1. Detect the local street-grid orientation (Midtown avenues are not true N).
    2. Pick `n_ns` avenue spines and `n_ew` cross-street spines.
    3. Trace each spine as a path through the OSM graph (axis-preferring).
    4. Drop stops along each path at `min_spacing_m`.
    5. Merge stops that land near the same intersection so transfers share a hub.

    Returns (stop_nodes, routes) where each route is
        {id, axis, stop_indices, path_nodes}.
    """
    along_ns, along_ew = street_grid_axes(G)
    along_coord = {
        n: float(np.dot([G.nodes[n]["x"], G.nodes[n]["y"]], along_ns))
        for n in G.nodes()
    }
    across_coord = {
        n: float(np.dot([G.nodes[n]["x"], G.nodes[n]["y"]], along_ew))
        for n in G.nodes()
    }

    t_vals = np.array(list(across_coord.values()))
    s_vals = np.array(list(along_coord.values()))

    def spine_targets(values, n):
        lo, hi = float(values.min()), float(values.max())
        if n <= 1 or hi <= lo:
            return [0.5 * (lo + hi)]
        # Keep spines off the extreme edges of the bbox.
        return list(np.linspace(lo, hi, n + 2)[1:-1])

    raw_routes = []
    tentative_stops = []

    for i, target in enumerate(spine_targets(t_vals, n_ns)):
        path = _corridor_path(
            G, along_ns, along_coord, across_coord, target,
            half_width_m=corridor_half_width_m,
        )
        sampled = _sample_path_stops(G, path, min_spacing_m)
        raw_routes.append({"id": f"NS{i}", "axis": "NS", "path_nodes": path, "sampled": sampled})
        tentative_stops.extend(sampled)

    for i, target in enumerate(spine_targets(s_vals, n_ew)):
        path = _corridor_path(
            G, along_ew, across_coord, along_coord, target,
            half_width_m=corridor_half_width_m,
        )
        sampled = _sample_path_stops(G, path, min_spacing_m)
        raw_routes.append({"id": f"EW{i}", "axis": "EW", "path_nodes": path, "sampled": sampled})
        tentative_stops.extend(sampled)

    # Preserve order while uniquing before the merge pass.
    ordered_unique = list(dict.fromkeys(tentative_stops))
    stop_nodes, remap = _merge_nearby_stops(G, ordered_unique)

    routes = []
    for raw in raw_routes:
        indices = []
        seen = set()
        for node in raw["sampled"]:
            stop = remap[node]
            if stop in seen:
                continue
            if stop not in stop_nodes:
                continue
            indices.append(stop_nodes.index(stop))
            seen.add(stop)
        if len(indices) < 2:
            continue
        routes.append({
            "id": raw["id"],
            "axis": raw["axis"],
            "stop_indices": indices,
            "path_nodes": raw["path_nodes"],
        })

    return stop_nodes, routes


# ---------------------------------------------------------------------------
# 4. Distance / travel-time matrix between stops (precomputed once)
# ---------------------------------------------------------------------------

def build_distance_matrix(G, stop_nodes):
    """
    Real shortest-path distance (meters) between every pair of stops, using
    the actual street network -- not straight-line distance. Computed once
    up front so the simulation itself never calls shortest_path per-step.
    """
    n = len(stop_nodes)
    matrix = np.full((n, n), np.inf)

    for i, origin in enumerate(stop_nodes):
        lengths = nx.single_source_dijkstra_path_length(G, origin, weight="length")
        for j, dest in enumerate(stop_nodes):
            if i == j:
                matrix[i, j] = 0.0
            elif dest in lengths:
                matrix[i, j] = lengths[dest]

    return matrix


def build_route_paths(G, stop_nodes):
    """
    Actual node-by-node paths between every pair of stops, for plotting
    routes on the real street network later. Returns a dict keyed by
    (i, j) stop index pairs -> list of graph nodes.
    """
    n = len(stop_nodes)
    paths = {}
    for i, origin in enumerate(stop_nodes):
        _, path_dict = nx.single_source_dijkstra(G, origin, weight="length")
        for j, dest in enumerate(stop_nodes):
            if i != j and dest in path_dict:
                paths[(i, j)] = path_dict[dest]
    return paths


def stop_coordinates(G, stop_nodes):
    """(n, 2) array of projected x/y metres for each stop node."""
    return np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in stop_nodes])


def usable_distances(dist_matrix, coords):
    """
    Distance matrix with every entry finite.

    One-way streets can leave a directed pair with no path at all, which would
    make a route leg take infinite time. Fall back to the opposite direction,
    then to straight-line distance scaled up for detour.
    """
    d = np.asarray(dist_matrix, dtype=float)
    euclidean = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    d = np.where(np.isfinite(d), d, d.T)
    return np.where(np.isfinite(d), d, euclidean * 1.3)


def _path_length(order, distances):
    return sum(distances[a, b] for a, b in zip(order, order[1:]))


def _two_opt(order, distances):
    """
    Shorten a stop ordering by repeatedly reversing sub-paths.

    A route that crosses itself is always longer than the same route with the
    crossing undone, so this removes the doubling-back that makes a corridor
    look nothing like a real bus line.
    """
    order = list(order)
    improved = True
    while improved:
        improved = False
        for i in range(len(order) - 1):
            for j in range(i + 2, len(order)):
                candidate = order[:i + 1] + order[i + 1:j + 1][::-1] + order[j + 1:]
                if _path_length(candidate, distances) < _path_length(order, distances) - 1e-9:
                    order = candidate
                    improved = True
    return order


def order_stops_into_route(G, stop_nodes, dist_matrix):
    """
    Order scattered stops into a single corridor a bus can run end to end.

    POI-weighted placement returns a cloud of stops, not a line. Stops are
    first swept along the cloud's principal axis -- the direction the stops
    actually spread in, which is the corridor a planner would draw -- then
    refined so the route does not double back on itself.

    Returns indices into `stop_nodes`, ordered south to north.
    """
    coords = stop_coordinates(G, stop_nodes)
    distances = usable_distances(dist_matrix, coords)
    distances = np.minimum(distances, distances.T)

    centered = coords - coords.mean(axis=0)
    principal_axis = np.linalg.svd(centered, full_matrices=False)[2][0]
    order = list(np.argsort(centered @ principal_axis))

    order = _two_opt(order, distances)

    if coords[order[0], 1] > coords[order[-1], 1]:
        order.reverse()
    return [int(i) for i in order]


def _bin_indices(values, n_bins):
    """Partition index positions into `n_bins` groups along a 1-D coordinate."""
    n_bins = max(1, min(n_bins, len(values)))
    order = np.argsort(values)
    return [list(chunk) for chunk in np.array_split(order, n_bins) if len(chunk) >= 2]


def street_grid_axes(G):
    """
    Unit vectors for the local street grid: (along_avenues, along_cross_streets).

    Midtown avenues are not true north — they sit ~29° off. Binning corridors on
    raw x/y therefore cuts diagonally across the blocks and produces zig-zag
    "N/S" routes. Edge bearings from the OSM graph recover the real orientation.
    """
    angles = []
    weights = []
    for u, v, data in G.edges(data=True):
        dx = G.nodes[v]["x"] - G.nodes[u]["x"]
        dy = G.nodes[v]["y"] - G.nodes[u]["y"]
        length = float(data.get("length", np.hypot(dx, dy)))
        if length <= 0:
            continue
        # Undirected edge, folded into [0, π/2) so both avenue directions agree.
        ang = np.arctan2(dy, dx) % np.pi
        ang = ang % (np.pi / 2)
        angles.append(ang)
        weights.append(length)

    angles = np.asarray(angles)
    weights = np.asarray(weights)
    hist, bin_edges = np.histogram(angles, bins=36, range=(0, np.pi / 2), weights=weights)
    peak = 0.5 * (bin_edges[int(hist.argmax())] + bin_edges[int(hist.argmax()) + 1])

    # The two perpendicular street directions.
    dir_a = np.array([np.cos(peak), np.sin(peak)])
    dir_b = np.array([np.cos(peak + np.pi / 2), np.sin(peak + np.pi / 2)])

    # Call the more-northward of the two "avenues" (N/S), the other "streets" (E/W).
    if abs(dir_a[1]) >= abs(dir_b[1]):
        along_ns, along_ew = dir_a, dir_b
    else:
        along_ns, along_ew = dir_b, dir_a

    # Orient so "forward" is north / east.
    if along_ns[1] < 0:
        along_ns = -along_ns
    if along_ew[0] < 0:
        along_ew = -along_ew
    return along_ns, along_ew


def axis_preferring_path(G, origin, destination, along_axis, turn_penalty=4.0):
    """
    Shortest path that prefers edges parallel to `along_axis`.

    A plain Dijkstra path between two stops on the same avenue will still jog
    along cross-streets whenever that is a metre or two shorter. Weighting
    perpendicular edges more heavily keeps an N/S corridor on avenues (and an
    E/W corridor on cross-streets) unless a jog is unavoidable.
    """
    along = np.asarray(along_axis, dtype=float)
    along = along / np.linalg.norm(along)

    def weight(u, v, data):
        length = float(data.get("length", 0.0))
        dx = G.nodes[v]["x"] - G.nodes[u]["x"]
        dy = G.nodes[v]["y"] - G.nodes[u]["y"]
        norm = float(np.hypot(dx, dy))
        if norm == 0 or length == 0:
            return length if length > 0 else 1.0
        alignment = abs(np.dot([dx / norm, dy / norm], along))
        return length * (1.0 + turn_penalty * (1.0 - alignment))

    return nx.shortest_path(G, origin, destination, weight=weight)


def build_grid_routes(G, stop_nodes, dist_matrix, n_ns=2, n_ew=2,
                      corridor_half_width_m=220.0):
    """
    Turn a 2-D stop cloud into intersecting linear corridors.

    Midtown avenues are rotated relative to map x/y, so corridors are built in
    the local street-grid basis from `street_grid_axes`. Each N/S corridor is a
    narrow strip around an avenue spine (stops far from the spine are not on
    that route); each E/W corridor is the same for a cross-street spine. A stop
    may sit on one route of each axis, or only one if it only falls near one
    spine — passengers transfer where the corridors meet (or walk between the
    nearest gateway pair).

    Stop order on each corridor is strictly monotonic along that axis.
    """
    coords = stop_coordinates(G, stop_nodes)
    along_ns, along_ew = street_grid_axes(G)

    s = coords @ along_ns  # increases going "north" along avenues
    t = coords @ along_ew  # increases going "east" along cross-streets

    def _spines(values, n):
        """Evenly spaced corridor centerlines across the range of `values`."""
        n = max(1, min(n, len(values)))
        lo, hi = float(np.min(values)), float(np.max(values))
        if n == 1 or hi <= lo:
            return [0.5 * (lo + hi)]
        return list(np.linspace(lo, hi, n))

    def _assign(values, spines, half_width):
        """Nearest spine within half_width; -1 if the stop is too far from all."""
        assignment = []
        for value in values:
            distances = [abs(value - spine) for spine in spines]
            nearest = int(np.argmin(distances))
            if distances[nearest] <= half_width:
                assignment.append(nearest)
            else:
                assignment.append(-1)
        return assignment

    ns_spines = _spines(t, n_ns)
    ew_spines = _spines(s, n_ew)
    ns_of = _assign(t, ns_spines, corridor_half_width_m)
    ew_of = _assign(s, ew_spines, corridor_half_width_m)

    # Anything left on no corridor at all snaps to whichever spine is nearer,
    # so every stop is reachable. Stops near only one spine stay on that axis
    # alone — that is what keeps N/S routes from wandering east-west.
    for i in range(len(stop_nodes)):
        if ns_of[i] >= 0 or ew_of[i] >= 0:
            continue
        ns_dists = [abs(t[i] - spine) for spine in ns_spines]
        ew_dists = [abs(s[i] - spine) for spine in ew_spines]
        if min(ns_dists) <= min(ew_dists):
            ns_of[i] = int(np.argmin(ns_dists))
        else:
            ew_of[i] = int(np.argmin(ew_dists))

    routes = []
    for bin_i in range(len(ns_spines)):
        members = [i for i, owner in enumerate(ns_of) if owner == bin_i]
        if len(members) < 2:
            continue
        order = sorted(members, key=lambda i: s[i])
        routes.append({
            "id": f"NS{bin_i}",
            "axis": "NS",
            "stop_indices": [int(i) for i in order],
        })

    for bin_i in range(len(ew_spines)):
        members = [i for i, owner in enumerate(ew_of) if owner == bin_i]
        if len(members) < 2:
            continue
        order = sorted(members, key=lambda i: t[i])
        routes.append({
            "id": f"EW{bin_i}",
            "axis": "EW",
            "stop_indices": [int(i) for i in order],
        })

    return routes


# ---------------------------------------------------------------------------
# 5. Road classification -> traffic-capacity proxy
# ---------------------------------------------------------------------------

HIGHWAY_TRAFFIC_MULTIPLIERS = {
    "motorway": 1.0,
    "trunk": 1.2,
    "primary": 1.5,
    "secondary": 1.8,
    "tertiary": 2.0,
    "residential": 2.5,
}


def apply_traffic_multipliers(G, weights=None):
    """
    Tag each edge with a congestion multiplier based on its OSM road
    classification. OSM has no live traffic volume data, but road class
    is a documented, usable proxy (major/wide roads generally carry more
    through-traffic than small residential streets).
    """
    if weights is None:
        weights = HIGHWAY_TRAFFIC_MULTIPLIERS

    for u, v, k, data in G.edges(keys=True, data=True):
        hwy = data.get("highway", "residential")
        if isinstance(hwy, list):
            hwy = hwy[0]
        data["traffic_multiplier"] = weights.get(hwy, 2.0)

    return G


# ---------------------------------------------------------------------------
# 6. Per-city calibration
# ---------------------------------------------------------------------------

@dataclass
class CityCalibration:
    evasion_rate: float
    fare_price: float
    traffic_base: float  # legacy 1-5; used only if auto_volume is unset
    traffic_amplitude: float  # legacy diurnal swing on 1-5 scale
    num_stops: int  # soft target / docs only when using GTFS
    num_buses: int  # fleet budget floor; sim may add buses to hit target headway
    poi_radius_m: float  # neighborhood used for POI-driven arrival weights
    min_spacing_m: float  # corridor stop spacing (synthetic + GTFS corner placement)
    # Mean passenger arrivals per stop per minute (daytime). Scaled by POI weights.
    arrival_rate_per_min: float = 0.85
    # Paid NYC default: mixed with ~3% cash of all boardings (rest card among payers).
    fare_type: str = "mixed"
    card_share: float = 0.946  # P(card|pays); → ~3% cash overall at evasion=0.44
    # Clear-road rolling (25 mph) + Midtown lights; BPR scales T0 by (V/C).
    free_flow_speed_kmh: float = 40.23  # 25 mph
    signal_spacing_m: float = 80.0
    signal_mean_delay_sec: float = 14.0
    bpr_alpha: float = 0.15
    bpr_beta: float = 4.0
    # Bus exposure to auto congestion in BPR: T/T0 = 1 + α (κ V/C)^β.
    # 1.0 = mixed traffic (article baseline); <1 ≈ dedicated bus lane.
    bpr_kappa: float = 1.0
    auto_volume: float | None = None  # trips/min; None → derive from traffic_base
    auto_capacity: float = 50.0
    bus_flow_ref_per_min: float = 0.0  # 0 → set in simulate() from stop weights
    target_headway_min: float = 7.5
    min_buses_per_route: int = 2
    # Optional GTFS/MTA midday headways (min) by route id/name for fleet sizing.
    route_target_headway_min: dict | None = None
    od_gravity_meters: float = 700.0
    n_ns_routes: int = 2
    n_ew_routes: int = 2
    gtfs_feed: str | None = None  # e.g. "nyc_manhattan"; None = synthetic streets
    preferred_routes: tuple[str, ...] = ()
    include_all_gtfs_routes: bool = False  # True = every route/stop in the bbox
    target_stops: int | None = None  # collapse curb clusters to ~N corner hubs
    diurnal_profile: bool = False  # smooth AM/PM activity curve (see SimConfig)
    # Optional overrides for SimConfig diurnal Gaussian crests (None → defaults).
    am_peak_hour: float | None = None
    am_peak_amp: float | None = None
    pm_peak_hour: float | None = None
    pm_peak_amp: float | None = None
    night_dip_amp: float | None = None


# NOTE: NYC evasion_rate / fare_price from MTA-facing reporting. Arrival rates are
# Midtown *means across shared hubs*, not peak-stop rates. Routes are GTFS-clipped
# to the study bbox, so cycles are short. Fleet is sized from weekday midday
# scheduled headways (MTA/GTFS): n ≈ round(cycle / H) with min 1 bus per spine.
# Amsterdam: synthetic canal-belt spines (no GVB GTFS — tram-heavy core);
# priors from GVB fares/checks, Autoluwe screenline, Vervoerregio stop access.
# SF / small-town values remain thinner placeholders where noted.

# Weekday midday scheduled headways (minutes) for Midtown locals that appear in
# the nyc_times_square bbox. Sources: MTA timetables / GTFS terminal spacing.
NYC_MIDTOWN_ROUTE_HEADWAY_MIN = {
    "M42": 8.0,
    "M5": 10.0,
    "M7": 10.0,
    "M1": 10.0,
    "M2": 10.0,
    "M3": 10.0,
    "M4": 10.0,
    "M20": 25.0,
    "M55": 20.0,
    "M11": 12.0,
    "M50": 15.0,
    "M104": 8.0,
    "M34-SBS": 8.0,
    "M34A-SBS": 8.0,
    "M34": 8.0,
}

CALIBRATIONS = {
    "nyc_times_square": CityCalibration(
        evasion_rate=0.44,  # MTA bus Q1 2025
        fare_price=3.00,  # MTA local base effective Jan 2026
        fare_type="mixed",
        card_share=0.946,  # ~3% cash of all boardings at evasion=0.44
        traffic_base=3.2,  # legacy only
        traffic_amplitude=1.6,
        num_buses=13,  # ≈ Σ round(cycle/H) on thinned Midtown primaries
        poi_radius_m=120,
        min_spacing_m=200,  # Midtown target gap; GTFS paths place denser then thin
        # See conditions.py: Midtown shared-hub prior (~51/h mean).
        arrival_rate_per_min=0.85,
        free_flow_speed_kmh=40.23,
        signal_spacing_m=80.0,
        signal_mean_delay_sec=14.0,
        # Midtown BPR from N–S screenline (see nyc_midtown_screenline_* helpers).
        # C ≈ 5 aves × 3 lanes × 550 vph (DOT) = 8,250 vph ≈ 137.5 /min
        # V ≈ 5 × 1,200 vph = 6,000 vph ≈ 100 /min → V/C ≈ 0.73
        # Highway defaults (α=0.15, β=4) understate Midtown: even at low V there is
        # substantial interaction delay beyond T0 lights (turns, buses, curb friction),
        # while β=4 overstates the jump once already saturated. Urban prior: raise α,
        # lower β so V/C≈0.73 → T/T0≈1.45 (vs ≈1.04 on highway params).
        bpr_alpha=0.85,
        bpr_beta=2.0,
        bpr_kappa=1.0,  # mixed-traffic baseline; see nyc_midtown_bus_lane for κ<1
        auto_volume=100.0,
        auto_capacity=137.5,
        # ATR-calibrated weekday crests (see NYC_ATR_* constants above).
        am_peak_hour=NYC_ATR_AM_PEAK_HOUR,
        am_peak_amp=NYC_ATR_AM_PEAK_AMP,
        pm_peak_hour=NYC_ATR_PM_PEAK_HOUR,
        pm_peak_amp=NYC_ATR_PM_PEAK_AMP,
        night_dip_amp=NYC_ATR_NIGHT_DIP_AMP,
        target_headway_min=12.0,  # fallback when a route is not in the table
        min_buses_per_route=1,
        route_target_headway_min=dict(NYC_MIDTOWN_ROUTE_HEADWAY_MIN),
        od_gravity_meters=700.0,
        n_ns_routes=2,
        n_ew_routes=2,
        gtfs_feed="nyc_manhattan",
        preferred_routes=("M7", "M20", "M42", "M34", "M34A-SBS", "M4", "M5"),
        include_all_gtfs_routes=True,
        # Dense OSM corners first; CityConditions.min_stop_spacing_m thins to ~MTA gaps.
        num_stops=80,
        target_stops=80,
        diurnal_profile=True,
    ),
    "sf_financial": CityCalibration(
        evasion_rate=0.30,
        fare_price=2.75,
        traffic_base=1.8,
        traffic_amplitude=2.2,
        num_stops=12,
        num_buses=8,
        poi_radius_m=130,
        min_spacing_m=320,
        # Downtown Muni local mean ~15–20 boardings/hour/stop (not Stockton peaks).
        arrival_rate_per_min=0.28,
        target_headway_min=8.0,
        n_ns_routes=1,
        n_ew_routes=1,
        gtfs_feed="sf_muni",
    ),
    "amsterdam_centrum": CityCalibration(
        # GVB checks (2022–24): open tram ~2–4.5%, bus ~0.8%, conductor tram ~1%.
        # Centrum tram-substitute prior ≈ 3% (not NYC cash/tap evasion).
        evasion_rate=0.03,
        fare_price=3.40,  # GVB 1-hour ticket (2025/26)
        fare_type="card",  # OV-chipkaart / OVpay check-in (no cash board)
        card_share=1.0,
        # Autoluwe canal belt — see ams_centrum_screenline_* helpers.
        traffic_base=1.5,  # legacy only
        traffic_amplitude=0.9,
        num_stops=20,
        num_buses=10,  # floor; sim sizes to target headway
        poi_radius_m=100,
        # Vervoerregio: ≤400 m walk to local stop → along-route gaps ~250–350 m.
        min_spacing_m=250.0,
        # Centrum mean ~24 boardings/h/stop (0.40/min); POI weights lift
        # Dam/Centraal hubs. Below network-wide tram averages (many quiet
        # outer stops inflate those); above the old placeholder 0.20.
        arrival_rate_per_min=0.40,
        free_flow_speed_kmh=30.0,  # city 30 km/h zones; signals still in T0
        signal_spacing_m=90.0,  # irregular canal intersections
        signal_mean_delay_sec=14.0,
        # Mild urban BPR (same α/β family as Midtown); low V/C from Autoluwe.
        bpr_alpha=0.85,
        bpr_beta=2.0,
        bpr_kappa=1.0,
        auto_volume=ams_centrum_screenline_volume_per_min(),  # ≈16.7
        auto_capacity=ams_centrum_screenline_capacity_per_min(),  # ≈37.5
        target_headway_min=8.0,  # GVB daytime tram ~5–10 min
        min_buses_per_route=2,
        od_gravity_meters=550.0,  # short centrum hops
        n_ns_routes=2,
        n_ew_routes=2,
        diurnal_profile=True,
        am_peak_hour=AMS_DIURNAL_AM_PEAK_HOUR,
        am_peak_amp=AMS_DIURNAL_AM_PEAK_AMP,
        pm_peak_hour=AMS_DIURNAL_PM_PEAK_HOUR,
        pm_peak_amp=AMS_DIURNAL_PM_PEAK_AMP,
        night_dip_amp=AMS_DIURNAL_NIGHT_DIP_AMP,
    ),
    "small_town": CityCalibration(
        evasion_rate=0.10,
        fare_price=1.50,
        traffic_base=1.0,
        traffic_amplitude=0.4,
        num_stops=6,
        num_buses=2,
        poi_radius_m=150,
        min_spacing_m=400,
        arrival_rate_per_min=0.05,
        target_headway_min=15.0,
        min_buses_per_route=1,
    ),
}


# ---------------------------------------------------------------------------
# 7. Full pipeline for one city
# ---------------------------------------------------------------------------

def build_city(city_key, seed=None, conditions=None):
    """
    Build a city network for simulation.

    When the calibration names a GTFS feed, stops and corridors come from the
    agency schedule clipped to the city bbox. POIs only set relative arrival
    weights. Optional `conditions` (see `src.conditions`) thin/scale the
    template toward another city's operating regime.
    """
    from src.conditions import (
        CONDITIONS,
        apply_conditions_to_calibration,
        thin_routes,
    )
    from src.gtfs_network import load_gtfs_routes_in_bbox, place_stops_at_path_corners

    config = CITY_CONFIGS[city_key]
    calib = CALIBRATIONS[city_key]
    if conditions is None:
        if city_key.startswith("nyc"):
            conditions = CONDITIONS.get("nyc_midtown")
        elif city_key.startswith("amsterdam"):
            conditions = CONDITIONS.get("amsterdam_centrum")
    if isinstance(conditions, str):
        conditions = CONDITIONS[conditions]
    if conditions is not None:
        calib = apply_conditions_to_calibration(calib, conditions)

    print(f"[{city_key}] loading street graph (basemap + POIs)...")
    G_unprojected = load_city_graph_raw(config)
    G = ox.project_graph(G_unprojected)
    print(f"[{city_key}] graph loaded: {len(G.nodes())} nodes, {len(G.edges())} edges")
    G = apply_traffic_multipliers(G)

    print(f"[{city_key}] pulling points of interest...")
    pois = get_pois(G_unprojected)
    print(f"[{city_key}] found {len(pois)} POIs")

    stop_coords = None
    stop_ids = None
    stop_names = None

    if calib.gtfs_feed:
        bbox = config["bbox"]
        print(
            f"[{city_key}] loading GTFS feed '{calib.gtfs_feed}' "
            + ("(all routes/stops in bbox)..." if calib.include_all_gtfs_routes
               else f"({calib.n_ns_routes} N/S + {calib.n_ew_routes} E/W routes)...")
        )
        gtfs = load_gtfs_routes_in_bbox(
            calib.gtfs_feed,
            bbox=bbox,
            n_ns=calib.n_ns_routes,
            n_ew=calib.n_ew_routes,
            preferred_routes=list(calib.preferred_routes),
            include_all=calib.include_all_gtfs_routes,
        )
        # Project GTFS *paths* into the OSM graph CRS. Agency curb stops are
        # discarded — we only keep shapes as corridors.
        from pyproj import Transformer

        transformer = Transformer.from_crs("EPSG:4326", G.graph["crs"], always_xy=True)
        routes = []
        for route in gtfs["routes"]:
            if route.get("path_ll") is not None and len(route["path_ll"]):
                px, py = transformer.transform(route["path_ll"][:, 0], route["path_ll"][:, 1])
                path_xy = np.column_stack([px, py])
            else:
                continue
            routes.append({**route, "path_xy": path_xy, "stop_indices": [], "leg_meters": []})

        # Reclassify axis from projected path geometry (name heuristics can put
        # a clipped "crosstown" route on an avenue and draw E/W style N/S).
        from src.city_pipeline import street_grid_axes
        from src.gtfs_network import classify_axis_from_path

        along_ns, along_ew = street_grid_axes(G)
        for route in routes:
            route["axis"] = classify_axis_from_path(
                route["path_xy"], along_ns, along_ew, route.get("name")
            )

        print(
            f"[{city_key}] GTFS paths: {len(routes)} routes "
            f"({gtfs.get('n_routes_in_bbox', len(routes))} in bbox); "
            f"ignoring {gtfs.get('n_stops_in_bbox', 0)} agency curb stops"
        )

        target = calib.target_stops or calib.num_stops or 80
        # Place dense path corners (~block scale); Midtown ~220 m gaps come from
        # CityConditions.min_stop_spacing_m via thin_routes (per corridor).
        place_spacing = 70.0
        print(
            f"[{city_key}] placing ≤{target} stops at OSM corners "
            f"(pre-thin spacing {place_spacing:.0f}m) along those paths..."
        )
        placed = place_stops_at_path_corners(
            G,
            routes,
            target_n=target,
            path_buffer_m=40.0,
            min_spacing_m=place_spacing,
        )
        stop_coords = placed["stop_coords"]
        stop_ids = placed["stop_ids"]
        stop_names = placed["stop_names"]
        routes = placed["routes"]
        # Final paths are stop-to-stop snaps; reclassify so a clipped "crosstown"
        # route that rides an avenue isn't drawn/treated as E/W.
        along_ns, along_ew = street_grid_axes(G)
        for route in routes:
            path = route.get("path_xy")
            if path is not None and len(path) >= 2:
                route["axis"] = classify_axis_from_path(
                    path, along_ns, along_ew, route.get("name") or route.get("id")
                )
        n_before = len(routes)
        routes, stop_coords, stop_ids, stop_names, _kept = prune_stub_corridors(
            routes, stop_coords, stop_ids, stop_names, min_length_m=500.0
        )
        if len(routes) < n_before:
            print(
                f"[{city_key}] dropped {n_before - len(routes)} stub corridors "
                f"(<{500:g}m) and their orphan stops"
            )
        stop_nodes = list(range(len(stop_coords)))
        dist_matrix = _route_dist_matrix(stop_coords, routes)
        print(
            f"[{city_key}] {len(stop_coords)} corner stops "
            f"(from {placed['n_candidates']} path-adjacent junctions, "
            f"spacing≥{placed['min_spacing_m']:.0f}m)"
        )

        print(
            f"[{city_key}] corridors: "
            + ", ".join(
                f"{r['id']}({len(r['stop_indices'])})"
                for r in routes[:12]
            )
            + (" ..." if len(routes) > 12 else "")
        )
    else:
        print(
            f"[{city_key}] tracing synthetic street corridors "
            f"(spacing={calib.min_spacing_m}m)..."
        )
        stop_nodes, routes = build_street_corridors(
            G,
            n_ns=calib.n_ns_routes,
            n_ew=calib.n_ew_routes,
            min_spacing_m=calib.min_spacing_m,
        )
        dist_matrix = build_distance_matrix(G, stop_nodes)
        stop_coords = stop_coordinates(G, stop_nodes)
        for route in routes:
            idxs = route["stop_indices"]
            route["leg_meters"] = [
                float(dist_matrix[a, b]) for a, b in zip(idxs, idxs[1:])
            ]

    if conditions is not None and (
        conditions.stop_keep_fraction < 1.0 or conditions.min_stop_spacing_m
    ):
        print(f"[{city_key}] applying conditions '{conditions.label}' to thin network...")
        routes, survivors = thin_routes(routes, stop_coords, conditions)
        stop_coords = stop_coords[survivors]
        if stop_ids is not None:
            stop_ids = [stop_ids[i] for i in survivors]
            stop_names = [stop_names[i] for i in survivors]
        # After thinning, indices are remapped — no longer OSM node ids.
        stop_nodes = list(range(len(stop_coords)))
        dist_matrix = _route_dist_matrix(stop_coords, routes)
        thinned = True
    else:
        thinned = False

    print(f"[{city_key}] {len(routes)} corridors, {len(stop_nodes)} stops")

    # POI weights need a graph-local geometry. Project POI centroids into the
    # same local metre frame as stop_coords when using GTFS or after thinning
    # (remapped indices are not OSM node ids).
    print(f"[{city_key}] scoring stop arrival rates from POI density...")
    if calib.gtfs_feed or thinned:
        arrival_weights = _poi_weights_for_coords(
            stop_coords, pois, G.graph["crs"], radius_m=calib.poi_radius_m
        )
    else:
        arrival_weights = poi_arrival_weights(
            G, stop_nodes, pois, radius_m=calib.poi_radius_m
        )

    return {
        "city_key": city_key,
        "graph": G,
        "pois": pois,
        "stop_nodes": stop_nodes,
        "stop_ids": stop_ids,
        "stop_names": stop_names,
        "stop_coords": stop_coords,
        "routes": routes,
        "arrival_weights": arrival_weights,
        "dist_matrix": dist_matrix,
        "calibration": calib,
        "conditions": conditions,
    }


def _poi_weights_for_coords(stop_coords, pois, graph_crs, radius_m=120):
    """POI arrival weights when stops are already in the graph CRS."""
    pois_proj = pois.to_crs(graph_crs)
    poi_xy = np.column_stack([
        pois_proj.geometry.centroid.x.to_numpy(),
        pois_proj.geometry.centroid.y.to_numpy(),
    ])
    if len(poi_xy) == 0:
        return np.ones(len(stop_coords))

    raw = []
    for stop in stop_coords:
        d = np.linalg.norm(poi_xy - stop, axis=1)
        raw.append(float((d < radius_m).sum()) + 1.0)
    weights = np.asarray(raw, dtype=float)
    return weights / weights.mean()


def _route_dist_matrix(stop_coords, routes):
    n = len(stop_coords)
    dist = np.linalg.norm(stop_coords[:, None, :] - stop_coords[None, :, :], axis=-1)
    for route in routes:
        idxs = route["stop_indices"]
        legs = route["leg_meters"]
        prefix = [0.0]
        for leg in legs:
            prefix.append(prefix[-1] + leg)
        for i in range(len(idxs)):
            for j in range(len(idxs)):
                dist[idxs[i], idxs[j]] = abs(prefix[j] - prefix[i])
    return dist


def prune_stub_corridors(
    routes: list[dict],
    stop_coords: np.ndarray,
    stop_ids: list[str] | None = None,
    stop_names: list[str] | None = None,
    min_length_m: float = 500.0,
) -> tuple[list[dict], np.ndarray, list[str] | None, list[str] | None, list[int]]:
    """
    Drop clipped GTFS stubs and stops that only those stubs served.

    Fleet allocation already withholds buses from corridors shorter than
    `min_length_m`; keeping their stops around leaves unreachable dots on the
    map and walk-only demand with no route at the origin.
    """
    kept_routes = []
    for route in routes:
        idxs = list(route.get("stop_indices") or [])
        if len(idxs) < 2:
            continue
        length = float(sum(route.get("leg_meters") or []))
        if length <= 0.0:
            length = float(
                sum(
                    np.linalg.norm(stop_coords[b] - stop_coords[a])
                    for a, b in zip(idxs, idxs[1:])
                )
            )
        if length < min_length_m:
            continue
        kept_routes.append(route)

    used = sorted({i for r in kept_routes for i in r["stop_indices"]})
    if not used:
        return routes, stop_coords, stop_ids, stop_names, list(range(len(stop_coords)))

    old_to_new = {old: new for new, old in enumerate(used)}
    remapped = []
    for route in kept_routes:
        idxs = [old_to_new[i] for i in route["stop_indices"] if i in old_to_new]
        if len(idxs) < 2:
            continue
        legs = [
            float(np.linalg.norm(stop_coords[used[b]] - stop_coords[used[a]]))
            for a, b in zip(idxs, idxs[1:])
        ]
        path_xy = route.get("path_xy")
        remapped.append({
            **route,
            "stop_indices": idxs,
            "leg_meters": legs,
            "path_xy": path_xy,
        })

    new_coords = stop_coords[used]
    new_ids = [stop_ids[i] for i in used] if stop_ids is not None else None
    new_names = [stop_names[i] for i in used] if stop_names is not None else None
    return remapped, new_coords, new_ids, new_names, used


# ---------------------------------------------------------------------------
# 8. Visualization helper
# ---------------------------------------------------------------------------

def plot_city_stops(city_data, ax=None, show_pois=True):
    """Sanity-check plot: street network + POIs (faint) + chosen stops (red)."""
    G = city_data["graph"]
    stop_nodes = city_data["stop_nodes"]

    fig, ax = ox.plot_graph(
        G,
        ax=ax,
        show=False,
        close=False,
        node_size=0,
        edge_color="#999999",
        edge_linewidth=0.5,
        bgcolor="white",
    )

    if show_pois:
        pois_projected = city_data["pois"].to_crs(G.graph["crs"])
        poi_pts = pois_projected.geometry.centroid
        ax.scatter(
            poi_pts.x,
            poi_pts.y,
            c="lightblue",
            s=4,
            alpha=0.5,
            zorder=3,
            label="POIs",
        )

    xs = [G.nodes[n]["x"] for n in stop_nodes]
    ys = [G.nodes[n]["y"] for n in stop_nodes]
    ax.scatter(xs, ys, c="red", s=50, zorder=5, label="Bus stops")

    ax.set_title(city_data["city_key"])
    ax.legend()
    return ax
