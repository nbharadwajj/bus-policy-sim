# bus-policy-sim

A simulation for studying public policy questions around bus transit.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

Build the NYC Midtown street graph, place POI-weighted bus stops, run the
simulation on them, and save the figures:

```bash
source .venv/bin/activate
python -m src.main
```

Outputs land in `outputs/nyc_route_map.png` and `outputs/nyc_sim_metrics.png`.

## How it fits together

1. `build_city()` loads an OSM basemap + POIs, then — for NYC/SF — pulls real
   agency GTFS routes/stops inside the study bbox.
2. Nearby POI density only sets each stop's *relative* arrival weight. The mean
   arrival rate comes from `CityCalibration.arrival_rate_per_min` (NYC Midtown
   default 0.85/min ≈ 51 boardings/hour/stop) and is overridable via
   `CityConditions` for fare-policy experiments.
3. `CityConditions` can thin the same GTFS skeleton (drop stops, enforce longer
   spacing, scale demand/fleet/evasion) to mimic another city's operating
   regime — e.g. `sf_like_from_nyc` — so policy can be compared under different
   conditions without changing geography.
4. `simulate()` runs the SimPy model: POI-weighted Poisson arrivals, transfers,
   door times, capacity, traffic-scaled travel.

```bash
python -m src.main   # writes nyc_midtown_* and sf_like_from_nyc_* plots
```

## Layout

- `src/city_pipeline.py` — OSM basemap/POIs, calibration, `build_city`
- `src/gtfs_network.py` — agency GTFS → Midtown-scale corridors
- `src/conditions.py` — thin/scale templates (e.g. SF-like from NYC)
- `src/simulation.py` — SimPy bus simulation
- `src/viz.py` — route map and metrics plots
- `src/main.py` — NYC baseline + SF-like comparison
- `cache/gtfs/` — downloaded GTFS zips
- `outputs/` — figures (git-ignored)

## Notice

This code was written with [Cursor](https://cursor.com).
