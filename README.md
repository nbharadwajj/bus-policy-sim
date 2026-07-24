# bus-policy-sim

A simulation for studying public policy questions around bus transit.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
python -m src.main
```

## Layout

- `src/` — simulation code
- `data/` — input data (schedules, routes, ridership)
- `outputs/` — generated results and figures (git-ignored)
