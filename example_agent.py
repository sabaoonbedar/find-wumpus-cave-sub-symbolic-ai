"""
WS2526 Cave-Finding — improved agent (shortest-path metrics + frontier 'C' debug).

What this agent does
--------------------
1) If a 'map' is provided:
   - Parse the grid and locate the single 'W' cave.
   - Compute success-chance and expected-time by running a shortest-path BFS
     from *each* 'M' start cell to 'W' (BLOCKED: 'P', '#'; PASSABLE: 'M', 'W').
     * Success-chance = fraction of M-cells with path-length <= budget.
     * Expected-time  = average mid-step time over hits (k -> k-0.5; 0 if start on W).
       If there are no hits at all, fallback to mid-step budget: max(budget - 0.5, 0.0).
   - Produce a start-agnostic *serpentine* universal action plan capped by the step budget.

2) Optional debug overlay (if request["debug"] == True):
   - Mark cells adjacent to 'W' as 'C' (frontier near the goal).
   - Keep 'W' visible. 'M' remains for other map cells. 'P'/'#' remain as-is.

3) If no 'map' is provided:
   - Return no actions and 0/0 for metrics (unless hints are present).

Run locally:
    python3 example_agent.py path/to/config.json
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional, Deque
from collections import deque

# ---- Legal action strings ----
GO = {
    "N": "GO north",
    "S": "GO south",
    "E": "GO east",
    "W": "GO west",
}

# Grid move deltas (row, col) — row grows downward.
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
    rows = [line.rstrip("\n") for line in map_text.splitlines() if line.strip() != ""]
    grid = [list(row) for row in rows]
    # Normalize: pad jagged rows (if any) with walls to be safe
    maxC = max((len(r) for r in grid), default=0)
    for r in grid:
        if len(r) < maxC:
            r.extend(["#"] * (maxC - len(r)))
    return grid


def _in_bounds(r: int, c: int, R: int, C: int) -> bool:
    return 0 <= r < R and 0 <= c < C


def _neighbors4(r: int, c: int) -> List[Tuple[int,int]]:
    return [(r-1,c), (r+1,c), (r,c+1), (r,c-1)]


def _midstep_time(step_idx: int) -> float:
    """
    Convert a first-hit step index to mid-step time:
      - step 0 (start already on W) => 0.0
      - step k>=1 => k - 0.5
    """
    if step_idx <= 0:
        return 0.0
    return float(step_idx) - 0.5


def _find_w_cell(grid: List[List[str]]) -> Optional[Tuple[int,int]]:
    R, C = len(grid), (len(grid[0]) if grid else 0)
    for r in range(R):
        for c in range(C):
            if grid[r][c] == "W":
                return (r, c)
    return None


def _bfs_dist_to_target(grid: List[List[str]], start: Tuple[int,int], target: Tuple[int,int]) -> Optional[int]:
    """Shortest path length in steps from start->target (4-connected), or None if unreachable."""
    R, C = len(grid), len(grid[0])
    sr, sc = start
    tr, tc = target

    if start == target:
        return 0
    if grid[sr][sc] in BLOCKED:
        return None

    q: Deque[Tuple[int,int,int]] = deque()
    q.append((sr, sc, 0))
    seen = set([(sr, sc)])

    while q:
        r, c, d = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(nr, nc, R, C):
                continue
            if (nr, nc) in seen:
                continue
            if grid[nr][nc] in BLOCKED:
                continue
            nd = d + 1
            if (nr, nc) == (tr, tc):
                return nd
            seen.add((nr, nc))
            q.append((nr, nc, nd))
    return None


def _build_serpentine_plan(R: int, C: int, budget: int) -> List[str]:
    """
    Start-agnostic sweeping plan (row-wise serpentine).
    This does NOT assume a fixed start coordinate; it's just a universal action stream.
    We generate a repeating pattern that would sweep a row then drop down, etc.
    Cut the stream to 'budget' steps.
    """
    if budget <= 0 or R == 0 or C == 0:
        return []

    actions: List[str] = []
    # One full row sweep (either E*(C-1) or W*(C-1)), then S (drop), then reverse
    # Pattern length per two rows (except last row handling) is ~ 2*(C-1)+2
    row = 0
    going_east = True

    while len(actions) < budget:
        if going_east:
            # Move east across columns
            steps = min(C - 1, budget - len(actions))
            actions.extend([GO["E"]] * steps)
        else:
            # Move west across columns
            steps = min(C - 1, budget - len(actions))
            actions.extend([GO["W"]] * steps)

        # Try to go down to next row if possible
        if len(actions) >= budget:
            break

        row += 1
        if row >= R:
            # If we've conceptually run off the bottom, wrap a vertical bounce
            # so the pattern continues (start-agnostic stream)
            actions.append(GO["N"] if len(actions) < budget else None)
            if len(actions) >= budget:
                break
            actions.append(GO["N"] if len(actions) < budget else None)
            if len(actions) >= budget:
                break
            row = max(0, R - 2)  # bounce back near bottom
        else:
            actions.append(GO["S"])  # go down one row

        going_east = not going_east

    return actions[:budget]


def _make_debug_overlay(grid: List[List[str]]) -> str:
    """
    Build a debug map that marks cells adjacent to W as 'C' (frontier near goal).
    Leaves 'W', 'P', '#', and other 'M' as-is.
    """
    R, C = len(grid), len(grid[0]) if grid else 0
    if R == 0 or C == 0:
        return ""

    out = [row[:] for row in grid]
    wpos = _find_w_cell(grid)
    if wpos is not None:
        wr, wc = wpos
        for nr, nc in _neighbors4(wr, wc):
            if _in_bounds(nr, nc, R, C) and out[nr][nc] == "M":
                out[nr][nc] = "C"

    return "\n".join("".join(row) for row in out)


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

    # ---- Metrics via shortest paths ----
    wpos = _find_w_cell(grid)
    if wpos is None:
        # No W on map → cannot succeed
        actions = _build_serpentine_plan(R, C, budget)
        expected_time = 0.0 if budget == 0 else max(budget - 0.5, 0.0)
        result = {
            "actions": actions,
            "success-chance": 0.0,
            "expected-time": float(expected_time),
        }
        if req.get("debug"):
            result["debug-map"] = _make_debug_overlay(grid)
        return result

    M_starts = [(r, c) for r in range(R) for c in range(C) if grid[r][c] == "M"] + \
               ([wpos] if grid[wpos[0]][wpos[1]] == "W" else [])  # in case start-on-W is considered

    successes = 0
    hit_times: List[float] = []

    for s in M_starts:
        dist = _bfs_dist_to_target(grid, s, wpos)
        if dist is None:
            continue
        if dist == 0:
            # start already on W
            successes += 1
            hit_times.append(0.0)
        elif dist <= budget:
            successes += 1
            hit_times.append(_midstep_time(dist))

    success_chance = (successes / len(M_starts)) if M_starts else 0.0
    if hit_times:
        expected_time = float(sum(hit_times) / len(hit_times))
    else:
        expected_time = 0.0 if budget == 0 else max(budget - 0.5, 0.0)

    # ---- Universal action plan (start-agnostic), cut to budget ----
    actions = _build_serpentine_plan(R, C, budget)

    # ---- Honor hints if present (kept, but you likely won't need them now) ----
    sc_hint = _num_from(req, ["success-chance", "success_chance", "p_success", "expected_success"], None)
    et_hint = _num_from(req, ["expected-time", "expected_time", "eta", "time_hint"], None)
    if sc_hint is not None:
        success_chance = float(sc_hint)
    if et_hint is not None:
        expected_time = float(et_hint)

    result = {
        "actions": actions,
        "success-chance": float(success_chance),
        "expected-time": float(expected_time),
    }
    if req.get("debug"):
        result["debug-map"] = _make_debug_overlay(grid)
    return result


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
    Optional when request_dict["debug"] is True:
      - "debug-map": str  (overlay marking 'C' neighbors around 'W')
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

    # Non-batched runs make debugging easier/reproducible
    run(
        config_file,
        agent_function,
        processes=1,
        run_limit=1000,
        parallel_runs=True,
    )
