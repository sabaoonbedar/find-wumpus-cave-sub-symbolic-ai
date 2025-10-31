# Cave-Finding Agent (Find the Wumpus Cave assignment 2.0 (warm up))

## Overview

This project contains a Python script called `example_agent.py`.  
It defines a small but smart cave-finding agent that roams a grid map to locate a cave marked with `'W'`.

The agent reports two things:

- **Success Chance** – how likely it is to reach a cave within the allowed number of moves (the **time budget**), given what we know about where it might start.
- **Expected Time (conditional)** – how long it takes on average **when it does succeed**.

To do this, the agent evaluates a simple, universal **serpentine (snake-like)** plan by simulating it from all feasible starting cells (e.g., uniformly over `'M'` cells when `"current-cell": "M"`). It also computes a BFS distance field as a helpful diagnostic, but the reported metrics are based on **plan simulation**, not just shortest paths.

---

## How the Agent Works

### 1. Map Parsing

When a map is provided, the agent:

- Turns the ASCII map into a 2D grid.
- Detects any cave cells `'W'`.
- Finds all `'M'` cells, i.e., possible start positions.
- Treats pits `'P'` and walls `'#'` as **impassable**.

This sets up the world so planning and evaluation are straightforward.

---

### 2. Plan-Conditioned Evaluation (the important bit)

Rather than only looking at shortest paths, the agent **simulates the exact action sequence it returns** from every feasible starting cell, weighted by the observations:

- If `"current-cell": "M"` is given, it assumes a uniform prior over all `'M'` cells.  
  (No observation? Then it spreads probability over all passable cells.)
- It executes up to `floor(max_time)` actions from the plan. Hitting a wall or pit means the agent **stays put** for that move.
- If the agent first lands on a `'W'` after the *t*-th move, it records the time as **`t - 0.5`** (a “mid-step” timing convention).  
  If it starts already on `'W'`, that counts as success at time **`0.0`**.

From these rollouts it computes:

- **Success Chance:** Total prior probability that reaches any `'W'` within the budget.
- **Expected Time:** The **average time to first success, conditioned on success**. Failures don’t contribute time.

> The BFS distance map is still built for insight and debugging, but it’s not used to directly set the reported numbers. That keeps the grading aligned with what the agent actually plans to do.

---

### 3. Serpentine Exploration Plan

The agent produces a compact, **start-agnostic** action plan that covers the map in a snake-like pattern:

- Sweep across a row (east or west).
- Step down a row (if possible).
- Reverse direction and repeat until the time budget runs out.

It’s simple, predictable, and surprisingly effective for warm-up exploration.

---

### 4. Debug Map (Optional)

Enable `"debug": true` in the request to get a quick visual aid:

- Cells adjacent to a cave `'W'` are marked as `'C'`.
- Terrain (`'#'`, `'P'`), starts (`'M'`), and caves (`'W'`) stay visible.

This helps you eyeball which areas are close to the target.

---

### 5. No Map? No Problem (Fallback)

If no map is provided, the agent returns:

- `actions: []`
- `success-chance: 0.0`
- `expected-time: 0.0`

Fail-safe behavior beats mysterious crashes.

---

## Methods and Techniques Used

| Step | Method | Purpose |
|------|--------|---------|
| Evaluation | **Plan simulation with observation-based prior** | Measures success chance and conditional expected time for the **actual returned plan**. |
| Planning | **Serpentine sweep** | Systematic, start-agnostic coverage under a tight time budget. |
| Diagnostics | **BFS distance field to `'W'`** | Visual intuition and debugging support (not used to set the final metrics). |
| Timing | **Mid-step timing** (`t - 0.5`) | Consistent time stamps when a cave is reached during a move. |
| Debugging | **Cave frontier marking** (`'C'`) | Highlights cells adjacent to caves for quick inspection. |

---

## Key Metrics

- **Success Chance:** Probability (under the observation prior) that the plan reaches a cave within the time budget.  
- **Expected Time (conditional):** Average time to first cave **given that it succeeds**, using mid-step timing.  
- **Serpentine Plan:** A general movement plan that works from any start cell.

---

## How to Run the Agent

> Note: run it from the **project root** folder.

```bash
python3 example_agent.py agent-configs/env-1.json
