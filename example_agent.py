from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional, Deque
from collections import deque
import math

GO = {"N": "GO north", "S": "GO south", "E": "GO east", "W": "GO west"}
DELTAS = {GO["N"]: (-1, 0), GO["S"]: (1, 0), GO["E"]: (0, 1), GO["W"]: (0, -1)}

# In this assignment, '#' is a wall and 'P' is a pit; both are impassable.
BLOCKED = {"P", "#"}


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
    maxC = max((len(r) for r in grid), default=0)
    for r in grid:
        if len(r) < maxC:
            r.extend(["#"] * (maxC - len(r)))
    return grid

def _in_bounds(r: int, c: int, R: int, C: int) -> bool:
    return 0 <= r < R and 0 <= c < C

def _neighbors4(r: int, c: int) -> List[Tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c + 1), (r, c - 1)]

def _find_all_w_cells(grid: List[List[str]]) -> List[Tuple[int, int]]:
    R = len(grid)
    C = len(grid[0]) if R else 0
    return [(r, c) for r in range(R) for c in range(C) if grid[r][c] == "W"]

def _multi_target_dist(grid: List[List[str]]) -> List[List[Optional[int]]]:
    R = len(grid)
    C = len(grid[0]) if R else 0
    dist: List[List[Optional[int]]] = [[None] * C for _ in range(R)]
    q: Deque[Tuple[int, int]] = deque()
    for r, c in _find_all_w_cells(grid):
        dist[r][c] = 0
        q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(nr, nc, R, C):
                continue
            if grid[nr][nc] in BLOCKED:
                continue
            if dist[nr][nc] is None:
                dist[nr][nc] = dist[r][c] + 1  # type: ignore
                q.append((nr, nc))
    return dist

# ----- Simple plan (row-wise "serpentine") -----
def _build_serpentine_plan(R: int, C: int, budget_steps: int) -> List[str]:
    """
    Produces a fixed action sequence (independent of the unknown start cell):
    zig-zag horizontally; at row end, go one step south (if possible), then reverse direction.
    Stops when 'budget_steps' actions have been generated.
    """
    if budget_steps <= 0 or R == 0 or C == 0:
        return []

    actions: List[str] = []
    going_east = True

    row_moves = [GO["E"]] * (C - 1)
    row_moves_rev = [GO["W"]] * (C - 1)

    r = 0
    while len(actions) < budget_steps and r < R:
        sweep = row_moves if going_east else row_moves_rev
        for a in sweep:
            if len(actions) >= budget_steps:
                break
            actions.append(a)
        if len(actions) >= budget_steps:
            break
        if r < R - 1:
            actions.append(GO["S"])
            r += 1
            going_east = not going_east
        else:
            break

    return actions[:budget_steps]

# ----- Observation prior and plan simulation -----
def _prior_from_observations(grid: List[List[str]], obs: Dict[str, Any]) -> Dict[Tuple[int, int], float]:
    """
    Build a prior over start cells.

    Priority:
    1) If observations specify "current-cell": "<symbol>", use exactly those cells.
    2) Else, if there are any 'M' cells, use a uniform over all 'M'.
    3) Else, use a uniform over passable NON-'W' cells (avoid free success at t=0).
    """
    R, C = len(grid), len(grid[0]) if grid else 0

    def cells_where(pred) -> List[Tuple[int, int]]:
        out = []
        for r in range(R):
            for c in range(C):
                if pred(r, c):
                    out.append((r, c))
        return out

    target_symbol = None
    if isinstance(obs, dict) and "current-cell" in obs:
        target_symbol = str(obs["current-cell"]).strip()

    if target_symbol is not None:
        candidates = cells_where(lambda r, c: grid[r][c] == target_symbol)
    else:
        m_cells = cells_where(lambda r, c: grid[r][c] == "M")
        if m_cells:
            candidates = m_cells
        else:
            candidates = cells_where(lambda r, c: grid[r][c] not in BLOCKED and grid[r][c] != "W")

    if not candidates:
        return {}

    p = 1.0 / len(candidates)
    prior = {rc: p for rc in candidates}

    s = sum(prior.values())
    if s > 0 and abs(s - 1.0) > 1e-12:
        for k in prior:
            prior[k] /= s

    return prior


def _apply_move(grid: List[List[str]], r: int, c: int, action: str) -> Tuple[int, int]:
    dr, dc = DELTAS.get(action, (0, 0))
    nr, nc = r + dr, c + dc
    R, C = len(grid), len(grid[0]) if grid else 0
    if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] not in BLOCKED:
        return nr, nc
    return r, c  # bump -> stay

def _simulate_plan_success(
    grid: List[List[str]],
    prior: Dict[Tuple[int, int], float],
    actions: List[str],
    max_time: float,
) -> Tuple[float, float]:
    """
    Returns (success_chance, expected_time), where expected_time is conditional on success
    and uses mid-step timing: if you first land on W after the t-th move, time = t - 0.5.
    If already on W at start, time = 0.0.
    """
    steps = max(0, math.floor(max_time))
    if not prior or steps == 0:
        succ_mass = 0.0
        t_weighted = 0.0
        for (sr, sc), p in prior.items():
            if grid[sr][sc] == "W":
                succ_mass += p
                # time = 0.0
        if succ_mass == 0.0:
            return 0.0, 0.0
        return succ_mass, 0.0

    succ_mass = 0.0
    t_weighted = 0.0

    for (sr, sc), p in prior.items():
        r, c = sr, sc
        if grid[r][c] == "W":
            succ_mass += p
            # time = 0.0
            continue

        t_hit: Optional[float] = None
        for t in range(1, steps + 1):
            if t - 1 < len(actions):
                r, c = _apply_move(grid, r, c, actions[t - 1])
            if grid[r][c] == "W":
                t_hit = t - 0.5  # mid-step timing
                break

        if t_hit is not None:
            succ_mass += p
            t_weighted += p * t_hit

    if succ_mass == 0.0:
        return 0.0, 0.0
    return succ_mass, t_weighted / succ_mass

# ----- Debug overlay (optional) -----
def _make_debug_overlay(grid: List[List[str]]) -> str:
    R, C = len(grid), len(grid[0]) if grid else 0
    if R == 0 or C == 0:
        return ""
    out = [row[:] for row in grid]
    for wr, wc in _find_all_w_cells(grid):
        for nr, nc in _neighbors4(wr, wc):
            if _in_bounds(nr, nc, R, C) and out[nr][nc] == "M":
                out[nr][nc] = "C"
    return "\n".join("".join(row) for row in out)

def _analyze_map_request(req: Dict[str, Any]) -> Dict[str, Any]:
    grid = _parse_grid(req.get("map", ""))
    if not grid:
        return {"actions": [], "success-chance": 0.0, "expected-time": 0.0}

    R, C = len(grid), len(grid[0])

    # time budget
    obs = req.get("observations", {})
    max_time = None
    if isinstance(obs, dict) and "max-time" in obs:
        max_time = _num_from(obs, ["max-time", "max_time"], None)
    if max_time is None:
        max_time = _num_from(req, ["max-time", "max_time", "time_budget"], 2.0)
    if max_time is None:
        max_time = 2.0
    budget_steps = max(0, math.floor(max_time))

    actions = _build_serpentine_plan(R, C, budget_steps)

    prior = _prior_from_observations(grid, obs if isinstance(obs, dict) else {})
    success_chance, expected_time = _simulate_plan_success(grid, prior, actions, max_time)

    result: Dict[str, Any] = {
        "actions": actions,
        "success-chance": float(success_chance),
        "expected-time": float(expected_time),
    }
    if req.get("debug"):
        result["debug-map"] = _make_debug_overlay(grid)
    return result

def _fallback_observation_only(req: Dict[str, Any]) -> Dict[str, Any]:
    # No map -> nothing to do;then don't guess.
    return {"actions": [], "success-chance": 0.0, "expected-time": 0.0}

def agent_function(request_dict: Dict[str, Any], _info: Any) -> Dict[str, Any]:
    if "map" in request_dict and isinstance(request_dict["map"], str):
        return _analyze_map_request(request_dict)
    else:
        return _fallback_observation_only(request_dict)

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
    run(config_file, agent_function, processes=1, run_limit=1000, parallel_runs=True)
