# example_agent.py
"""
WS2526 Cave-Finding — simple baseline agent (mid-step timing fixed).

What this agent does
--------------------
1) If a 'map' is provided in the request, we:
   - Build a fixed, start-agnostic action pattern (NESW repeated) and cap it by 'max-time'.
   - Compute success-chance as the fraction of 'M' start cells from which executing the
     SAME action sequence reaches a 'W' within the step budget.
   - Compute expected-time as the average **mid-step** hit time:
         time = 0.0 if hit at step 0 (start already on W)
               = step_index - 0.5 for hits during movement
     If no start hits W, we return the mid-step fallback: max(budget - 0.5, 0.0).

2) If no 'map' is provided, we return no actions and 0/0 for the metrics
   (unless hints are present).

Run locally against your config with:
    python3 example_agent.py path/to/your/config.json
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional

# ---- Legal action strings ----
GO = {
    "N": "GO north",
    "S": "GO south",
    "E": "GO east",
    "W": "GO west",
}

# Grid move deltas
DELTAS = {
    GO["N"]: (-1, 0),
    GO["S"]: ( 1, 0),
    GO["E"]: ( 0, 1),
    GO["W"]: ( 0,-1),
}

BLOCKED = {"P", "#"}   # impassable cells
PASSABLE = {"M", "W"}  # traversable; 'W' is the goal


# ------------------------ utils ------------------------

def _num_from(request: Dict[str, Any], keys: List[str], default: Optional[float]) -> Optional[float]:
    for k in keys:
        if k in request:
            try:
                return float(request[k])
            except (TypeError, ValueError):
                pass
    return default


def _parse_grid(map_text: str) -> List[List[str]]:
    rows = [line.strip() for line in map_text.splitlines() if line.strip() != ""]
    return [list(row) for row in rows]


def _in_bounds(r: int, c: int, R: int, C: int) -> bool:
    return 0 <= r < R and 0 <= c < C


def _simulate_path_from(grid: List[List[str]], start_rc: Tuple[int, int], actions: List[str]) -> Tuple[bool, Optional[int]]:
    """
    Follow `actions` from `start_rc` on `grid`.
    Returns (reached_W, first_hit_step_index) where the index is 1-based (the step number when W is first entered).
    If already on W at start, returns (True, 0).
    If never reach W (or illegal move), returns (False, None).
    """
    r, c = start_rc
    R, C = len(grid), len(grid[0])

    if grid[r][c] == "W":
        return True, 0

    for step_idx, act in enumerate(actions, start=1):
        dr, dc = DELTAS[act]
        nr, nc = r + dr, c + dc
        if not _in_bounds(nr, nc, R, C):
            return False, None
        if grid[nr][nc] in BLOCKED:
            return False, None

        r, c = nr, nc
        if grid[r][c] == "W":
            return True, step_idx

    return False, None


def _build_universal_plan(budget: int) -> List[str]:
    """
    Start-agnostic plan: NESW cycling, cut to `budget`.
    """
    pattern = [GO["N"], GO["E"], GO["S"], GO["W"]]
    if budget <= 0:
        return []
    out = []
    k = 0
    while len(out) < budget:
        out.append(pattern[k % len(pattern)])
        k += 1
    return out


def _midstep_time(step_idx: int) -> float:
    """
    Convert a first-hit step index to mid-step time:
      - step 0 (start already on W) => 0.0
      - step k>=1 => k - 0.5
    """
    if step_idx <= 0:
        return 0.0
    return float(step_idx) - 0.5


def _analyze_map_request(req: Dict[str, Any]) -> Dict[str, Any]:
    map_text = req.get("map", "")
    grid = _parse_grid(map_text)
    if not grid:
        return {"actions": [], "success-chance": 0.0, "expected-time": 0.0}

    R, C = len(grid), len(grid[0])

    # Time budget
    max_time = None
    obs = req.get("observations", {})
    if isinstance(obs, dict) and "max-time" in obs:
        max_time = _num_from(obs, ["max-time", "max_time"], None)
    if max_time is None:
        max_time = _num_from(req, ["max-time", "max_time", "time_budget"], 2.0)

    budget = max(0, int(max_time))

    actions = _build_universal_plan(budget)

    # Start distribution: all 'M' cells
    M_starts = [(r, c) for r in range(R) for c in range(C) if grid[r][c] == "M"]

    if not M_starts or budget == 0 or not actions:
        success_chance = 0.0
        expected_time = 0.0 if budget == 0 else max(budget - 0.5, 0.0)
    else:
        successes = 0
        hit_times: List[float] = []

        for s in M_starts:
            reached, first_step = _simulate_path_from(grid, s, actions)
            if reached:
                successes += 1
                hit_times.append(_midstep_time(first_step if first_step is not None else budget))

        success_chance = successes / len(M_starts) if M_starts else 0.0
        if hit_times:
            expected_time = float(sum(hit_times) / len(hit_times))
        else:
            # no hits from any start → fallback to mid-step budget
            expected_time = max(budget - 0.5, 0.0)

    # Honor hints if present
    sc_hint = _num_from(req, ["success-chance", "success_chance", "p_success", "expected_success"], None)
    et_hint = _num_from(req, ["expected-time", "expected_time", "eta", "time_hint"], None)
    if sc_hint is not None:
        success_chance = float(sc_hint)
    if et_hint is not None:
        expected_time = float(et_hint)

    return {
        "actions": actions,
        "success-chance": float(success_chance),
        "expected-time": float(expected_time),
    }


def _fallback_observation_only(req: Dict[str, Any]) -> Dict[str, Any]:
    actions: List[str] = []
    success_chance = 0.0
    expected_time = 0.0

    sc_hint = _num_from(req, ["success-chance", "success_chance", "p_success", "expected_success"], None)
    et_hint = _num_from(req, ["expected-time", "expected_time", "eta", "time_hint"], None)
    if sc_hint is not None:
        success_chance = float(sc_hint)
    if et_hint is not None:
        expected_time = float(et_hint)

    return {
        "actions": actions,
        "success-chance": float(success_chance),
        "expected-time": float(expected_time),
    }


# ------------------------ agent entrypoint ------------------------

def agent_function(request_dict: Dict[str, Any], _info: Any) -> Dict[str, Any]:
    """
    Must return:
      - "actions": List[str]        # {"GO north","GO south","GO east","GO west"}
      - "success-chance": float
      - "expected-time": float
    """
    if "map" in request_dict and isinstance(request_dict["map"], str):
        return _analyze_map_request(request_dict)
    else:
        return _fallback_observation_only(request_dict)


# ------------------------ standalone run harness ------------------------

if __name__ == "__main__":
    try:
        from client import run
    except ImportError:
        raise ImportError("You need to have the client.py file in the same directory as this file")

    import logging, sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python3 example_agent.py path/to/your/config.json")

    config_file = sys.argv[1]

    run(
        config_file,
        agent_function,
        processes=1,
        run_limit=1000,
        parallel_runs=True,  # easier debugging (no batching)
    )
