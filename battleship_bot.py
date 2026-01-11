#!/usr/bin/env python3
"""
Code Clash Battleship Bot Challenge - CREATE UofT - Winter 2026

Strategy overview (matches the approach you described):
- Pick abilities: HS (Hailstorm) + SP (Sonar Pulse)
- If we have a hit/blocked cell, shoot adjacent unknown cells (candidate targeting)
- Otherwise, compute a probability heatmap from remaining legal ship placements and shoot the best cell
- SP policy:
  1) If no ship evidence after 2 shots, use SP (early-hunt accelerator)
  2) If SP is still unused by shot #10, force SP (midgame value guarantee)
  3) SP targets the best 3*3 region by maximizing expected ship-occupancy score
- HS policy:
  - Use HS on the very first combat turn (if available) to quickly create early evidence

All debug printing (if enabled) goes to stderr so stdout stays valid JSON.
"""

from __future__ import annotations

import os
import random
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from battleship_api import BattleshipBotAPI, run_bot, ABILITY_CODES

# ---- Tunables ----
DEBUG_HEATMAP = os.getenv("DEBUG_HEATMAP", "0") == "1"

EVIDENCE_SEARCH_BUDGET_S = 2.0
EVIDENCE_MAX_FLEETS = 250_000

TOP_TIE_REL = 0.01  # randomize among cells within 1% of best score

# Target-mode bonuses
ADJ_BASE_BONUS = 1.25
LINE_EXT_BONUS = 1.45


class MyBattleshipBot(BattleshipBotAPI):
    # Officially documented symbols are N/H/M/B.
    # We also accept common sonar markers if the engine puts them on the grid.
    _SHOT_MARKERS = {"H", "M", "B"}
    _SHIPINFO_MARKERS = {"H", "B", "S"}
    _WATER_MARKERS = {"M", "E"}

    _ENEMY_SHIPS = ("ship_1x4", "ship_1x3", "ship_2x3", "ship_1x2")

    # Cache placements per process
    _PLACEMENTS: Optional[Dict[str, List[Tuple[int, Tuple[int, ...]]]]] = None
    _PLACEMENTS_BY_CELL: Optional[Dict[str, Dict[int, Tuple[int, ...]]]] = None

    # -------------------------- basic utils --------------------------
    @staticmethod
    def _idx(r: int, c: int) -> int:
        return r * 8 + c

    @staticmethod
    def _bit(r: int, c: int) -> int:
        return 1 << (r * 8 + c)

    @staticmethod
    def _iter_bits(mask: int):
        while mask:
            lsb = mask & -mask
            yield lsb.bit_length() - 1
            mask ^= lsb

    # -------------------------- required hooks --------------------------
    def ability_selection(self) -> List[str]:
        # Strategy chooses Sonar Pulse (SP) + Hailstorm (HS)
        return ["SP", "HS"]

    def place_ship_strategy(self, ship_name: str, game_state: Dict[str, Any]) -> Dict[str, Any]:
        placed_coords = self._get_placed_coordinates(game_state)

        # Randomize with anti-adjacency bias
        for _ in range(600):
            sr = random.randint(0, 7)
            sc = random.randint(0, 7)
            orient = random.choice(["H", "V"])
            cells = self._get_ship_cells(ship_name, sr, sc, orient)
            if not cells or not self._is_valid_placement(cells, placed_coords):
                continue

            adjacent = False
            for (r, c) in cells:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if (r + dr, c + dc) in placed_coords:
                        adjacent = True
                        break
                if adjacent:
                    break
            if adjacent:
                continue

            return {"placement": {"name": ship_name, "cell": [sr, sc], "direction": orient}}

        placement = self._get_random_placement(ship_name, placed_coords)
        if placement:
            return placement
        return {"placement": {"name": ship_name, "cell": [0, 0], "direction": "H"}}

    # -------------------------- ability availability + sonar parsing --------------------------
    def _available_abilities_robust(self, game_state: Dict[str, Any]) -> List[str]:
        """Treat an ability as available iff its info contains key "None"."""
        out: List[str] = []
        for obj in game_state.get("player_abilities", []) or []:
            if not isinstance(obj, dict):
                continue
            ab = obj.get("ability")
            if ab not in ABILITY_CODES:
                continue
            info = obj.get("info", {})
            if isinstance(info, dict) and "None" in info:
                out.append(ab)
        return out

    @staticmethod
    def _flatten_sp_list(sp_val: Any) -> Iterable[dict]:
        if sp_val is None:
            return []
        if isinstance(sp_val, list):
            if sp_val and all(isinstance(x, list) for x in sp_val):
                for row in sp_val:
                    if isinstance(row, list):
                        for item in row:
                            if isinstance(item, dict):
                                yield item
                return
            for item in sp_val:
                if isinstance(item, dict):
                    yield item

    def _sonar_evidence_from_state(self, game_state: Dict[str, Any]) -> Tuple[int, int, List[List[int]]]:
        """Parse SP info payload into ship_mask, water_mask, and unshot ship cells."""
        ship_mask = 0
        water_mask = 0
        ship_cells: List[List[int]] = []

        opponent_grid = self._get_opponent_grid(game_state)

        for obj in game_state.get("player_abilities", []) or []:
            if not isinstance(obj, dict) or obj.get("ability") != "SP":
                continue
            info = obj.get("info", {})
            if not isinstance(info, dict) or "SP" not in info:
                continue
            sp_val = info.get("SP")
            for sp_obj in self._flatten_sp_list(sp_val):
                cell = sp_obj.get("cell")
                res = sp_obj.get("result")
                if not (isinstance(cell, list) and len(cell) == 2):
                    continue
                r, c = cell
                if not (isinstance(r, int) and isinstance(c, int) and 0 <= r < 8 and 0 <= c < 8):
                    continue
                if res == "Ship":
                    ship_mask |= self._bit(r, c)
                    if opponent_grid[r][c] == "N":
                        ship_cells.append([r, c])
                elif res == "Water":
                    water_mask |= self._bit(r, c)

        return ship_mask, water_mask, ship_cells

    def _masks_from_grid(self, grid: List[List[str]]) -> Tuple[int, int, int]:
        shot_mask = 0
        water_mask = 0
        shipinfo_mask = 0
        for r in range(8):
            row = grid[r]
            for c in range(8):
                v = row[c]
                b = self._bit(r, c)
                if v in self._SHOT_MARKERS:
                    shot_mask |= b
                if v in self._WATER_MARKERS:
                    water_mask |= b
                if v in self._SHIPINFO_MARKERS:
                    shipinfo_mask |= b
        return shot_mask, water_mask, shipinfo_mask

    # -------------------------- placement cache --------------------------
    def _ensure_placement_cache(self) -> None:
        if self._PLACEMENTS is not None and self._PLACEMENTS_BY_CELL is not None:
            return

        placements: Dict[str, List[Tuple[int, Tuple[int, ...]]]] = {}
        by_cell: Dict[str, Dict[int, List[int]]] = {}

        for ship in self._ENEMY_SHIPS:
            ship_pl: List[Tuple[int, Tuple[int, ...]]] = []
            ship_by_cell: Dict[int, List[int]] = {}

            for orient in ("H", "V"):
                for r in range(8):
                    for c in range(8):
                        cells = self._get_ship_cells(ship, r, c, orient)
                        if not cells:
                            continue
                        mask = 0
                        idxs: List[int] = []
                        for rr, cc in cells:
                            i = self._idx(rr, cc)
                            mask |= 1 << i
                            idxs.append(i)
                        idxs_t = tuple(idxs)
                        pi = len(ship_pl)
                        ship_pl.append((mask, idxs_t))
                        for i in idxs_t:
                            ship_by_cell.setdefault(i, []).append(pi)

            placements[ship] = ship_pl
            by_cell[ship] = ship_by_cell

        frozen_by_cell: Dict[str, Dict[int, Tuple[int, ...]]] = {}
        for ship, d in by_cell.items():
            frozen_by_cell[ship] = {i: tuple(lst) for i, lst in d.items()}

        self._PLACEMENTS = placements
        self._PLACEMENTS_BY_CELL = frozen_by_cell

    # -------------------------- heatmap core --------------------------
    def _heatmap(self, water_mask: int, shipinfo_mask: int) -> Tuple[List[int], Optional[int]]:
        """Return (scores[64], fleet_count or None)."""
        self._ensure_placement_cache()
        assert self._PLACEMENTS is not None
        assert self._PLACEMENTS_BY_CELL is not None

        placements = self._PLACEMENTS
        placements_by_cell = self._PLACEMENTS_BY_CELL

        scores = [0] * 64

        # Hunt mode: no ship evidence yet -> independent ship placement overlap counting
        if shipinfo_mask == 0:
            for ship in self._ENEMY_SHIPS:
                for pmask, idxs in placements[ship]:
                    if pmask & water_mask:
                        continue
                    for i in idxs:
                        scores[i] += 1
            return scores, None

        # Evidence mode: build fleets consistent with all ship-evidence cells and water
        legal_flags: Dict[str, List[bool]] = {}
        for ship in self._ENEMY_SHIPS:
            flags = [True] * len(placements[ship])
            for pi, (pmask, _) in enumerate(placements[ship]):
                if pmask & water_mask:
                    flags[pi] = False
            legal_flags[ship] = flags

        required_cells = [i for i in self._iter_bits(shipinfo_mask)]

        start_t = time.time()
        fleet_count = 0

        cover_cache: Dict[Tuple[str, int], Tuple[int, ...]] = {}

        def legal_cover(ship: str, cell_i: int) -> Tuple[int, ...]:
            key = (ship, cell_i)
            if key in cover_cache:
                return cover_cache[key]
            cover = placements_by_cell[ship].get(cell_i, ())
            out = tuple(pi for pi in cover if legal_flags[ship][pi])
            cover_cache[key] = out
            return out

        def pick_required_cell(required_mask: int, remaining: Tuple[str, ...]) -> int:
            best_i = None
            best_cnt = 10**9
            for i in self._iter_bits(required_mask):
                cnt = 0
                for ship in remaining:
                    cnt += len(legal_cover(ship, i))
                if cnt < best_cnt:
                    best_cnt = cnt
                    best_i = i
            return best_i if best_i is not None else -1

        def dfs(remaining_ships: Tuple[str, ...], used_mask: int, required_mask: int):
            nonlocal fleet_count, scores

            if fleet_count >= EVIDENCE_MAX_FLEETS:
                return
            if time.time() - start_t > EVIDENCE_SEARCH_BUDGET_S:
                return

            if not remaining_ships:
                if required_mask == 0:
                    fleet_count += 1
                    for i in self._iter_bits(used_mask):
                        scores[i] += 1
                return

            ship = remaining_ships[0]
            rest = remaining_ships[1:]

            if required_mask != 0:
                cell_i = pick_required_cell(required_mask, remaining_ships)
                candidates = legal_cover(ship, cell_i)
                if not candidates:
                    # Maybe another ship covers this required cell; skip choosing by this ship.
                    candidates = tuple(pi for pi, ok in enumerate(legal_flags[ship]) if ok)
            else:
                candidates = tuple(pi for pi, ok in enumerate(legal_flags[ship]) if ok)

            # Randomize to avoid worst-case ordering
            if len(candidates) > 32:
                cand_list = list(candidates)
                random.shuffle(cand_list)
                candidates = tuple(cand_list)

            for pi in candidates:
                pmask, idxs = placements[ship][pi]
                if pmask & used_mask:
                    continue

                new_used = used_mask | pmask
                new_required = required_mask & ~pmask
                dfs(rest, new_used, new_required)

        dfs(tuple(self._ENEMY_SHIPS), 0, shipinfo_mask)

        return scores, fleet_count if fleet_count > 0 else None

    # -------------------------- targeting: hit clusters -> adjacent candidates --------------------------
    def _hit_components(self, grid: List[List[str]]) -> List[List[Tuple[int, int]]]:
        """Return connected components of ship-evidence cells (H or B), using 4-neighborhood."""
        seen = [[False] * 8 for _ in range(8)]
        comps: List[List[Tuple[int, int]]] = []

        for r in range(8):
            for c in range(8):
                if grid[r][c] not in ("H", "B") or seen[r][c]:
                    continue
                q = [(r, c)]
                seen[r][c] = True
                comp: List[Tuple[int, int]] = []
                while q:
                    rr, cc = q.pop()
                    comp.append((rr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < 8 and 0 <= nc < 8 and not seen[nr][nc] and grid[nr][nc] in ("H", "B"):
                            seen[nr][nc] = True
                            q.append((nr, nc))
                comps.append(comp)

        return comps

    def _adjacent_unknowns_with_bonus(self, grid: List[List[str]]) -> Dict[Tuple[int, int], float]:
        """Return dict of candidate (r,c)->bonus for N cells adjacent to any ship-evidence (H/B).

        Always includes all N neighbors of evidence components.
        Adds a bonus to line-extension cells when a component is a straight line.
        """
        comps = self._hit_components(grid)
        cand: Dict[Tuple[int, int], float] = {}

        # All adjacent unknowns
        for comp in comps:
            for (r, c) in comp:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 8 and 0 <= nc < 8 and grid[nr][nc] == "N":
                        cand[(nr, nc)] = max(cand.get((nr, nc), 1.0), ADJ_BASE_BONUS)

        # Line extension heuristic
        for comp in comps:
            if len(comp) < 2:
                continue
            rows = {r for r, _ in comp}
            cols = {c for _, c in comp}
            if len(rows) == 1:
                r = next(iter(rows))
                minc = min(c for _, c in comp)
                maxc = max(c for _, c in comp)
                for nc in (minc - 1, maxc + 1):
                    if 0 <= nc < 8 and grid[r][nc] == "N":
                        cand[(r, nc)] = max(cand.get((r, nc), 1.0), LINE_EXT_BONUS)
            elif len(cols) == 1:
                c = next(iter(cols))
                minr = min(r for r, _ in comp)
                maxr = max(r for r, _ in comp)
                for nr in (minr - 1, maxr + 1):
                    if 0 <= nr < 8 and grid[nr][c] == "N":
                        cand[(nr, c)] = max(cand.get((nr, c), 1.0), LINE_EXT_BONUS)

        return cand

    # -------------------------- SP targeting --------------------------
    def _best_sp_center(self, cell_value: List[float], shot_mask: int) -> List[int]:
        best_score = -1.0
        best_center = [4, 4]
        for r in range(8):
            for c in range(8):
                s = 0.0
                for rr in (r - 1, r, r + 1):
                    if rr < 0 or rr >= 8:
                        continue
                    base = rr * 8
                    for cc in (c - 1, c, c + 1):
                        if cc < 0 or cc >= 8:
                            continue
                        i = base + cc
                        if shot_mask & (1 << i):
                            continue
                        s += cell_value[i]
                if s > best_score:
                    best_score = s
                    best_center = [r, c]
        return best_center

    # -------------------------- debug --------------------------
    def _debug_print_heatmap(
        self,
        grid: List[List[str]],
        cell_prob: List[float],
        total_shots: int,
        fleet_count: Optional[int],
        best1: List[int],
        best2: Optional[List[int]],
        p1: float,
        p2: float,
        target_mode: bool,
    ) -> None:
        if not DEBUG_HEATMAP:
            return
        print(f"[DBG] shots={total_shots} fleet_count={fleet_count} target_mode={target_mode}", file=sys.stderr)
        print(f"[DBG] best1={best1} p1={p1:.3f} best2={best2} p2={p2:.3f}", file=sys.stderr)

    # -------------------------- main combat strategy --------------------------
    def combat_strategy(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        available_abilities = set(self._available_abilities_robust(game_state))
        grid = self._get_opponent_grid(game_state)

        shot_mask, water_mask, shipinfo_mask = self._masks_from_grid(grid)
        total_shots = shot_mask.bit_count()

        # Add sonar evidence from state
        sonar_ship_mask, sonar_water_mask, sonar_ship_cells = self._sonar_evidence_from_state(game_state)
        water_mask |= sonar_water_mask
        shipinfo_mask |= sonar_ship_mask

        # Immediate shots from sonar ship cells (after a previous SP)
        if sonar_ship_cells:
            return {"combat": {"cell": sonar_ship_cells[0], "ability": {"None": {}}}}

        # --- HS opening: use Hailstorm on the first combat turn (if available) ---
        # HS fires 4 random shots; we use it immediately to generate early evidence for candidate targeting.
        if "HS" in available_abilities and total_shots == 0:
            return {"combat": {"cell": [0, 0], "ability": {"HS": {}}}}

        # Compute heatmap
        scores_int, fleet_count = self._heatmap(water_mask=water_mask, shipinfo_mask=shipinfo_mask)

        # Convert to comparable probabilities
        if fleet_count is not None and fleet_count > 0:
            cell_prob = [s / float(fleet_count) for s in scores_int]
        else:
            mx = max(scores_int) if scores_int else 1
            if mx <= 0:
                mx = 1
            cell_prob = [s / float(mx) for s in scores_int]

        # -------- TARGET MODE: if we have any hit(s), shoot adjacent unknowns first --------
        cand_bonus = self._adjacent_unknowns_with_bonus(grid)
        target_mode = len(cand_bonus) > 0

        # Decide candidate list
        if target_mode:
            candidates = [[r, c] for (r, c) in cand_bonus.keys()]
        else:
            # Global fireable cells
            candidates = [[r, c] for r in range(8) for c in range(8) if grid[r][c] == "N"]

        if not candidates:
            # Should not happen, but safe fallback
            return {"combat": {"cell": [random.randint(0, 7), random.randint(0, 7)], "ability": {"None": {}}}}

        # Score candidates
        def cand_score(rc: List[int]) -> float:
            r, c = rc
            base = cell_prob[self._idx(r, c)]
            if target_mode:
                base *= cand_bonus.get((r, c), 1.0)
            return base

        candidates.sort(key=cand_score, reverse=True)
        best1_score = cand_score(candidates[0])

        # Randomize among near ties for best1
        top = []
        if best1_score > 0:
            cutoff = best1_score * (1.0 - TOP_TIE_REL)
            for rc in candidates:
                if cand_score(rc) >= cutoff:
                    top.append(rc)
                else:
                    break
        else:
            top = [candidates[0]]

        best1 = random.choice(top)

        # best2 = next best distinct (used only for debug printing)
        best2 = None
        for rc in candidates:
            if rc != best1:
                best2 = rc
                break

        p1 = cell_prob[self._idx(best1[0], best1[1])]
        p2 = cell_prob[self._idx(best2[0], best2[1])] if best2 else 0.0

        self._debug_print_heatmap(grid, cell_prob, total_shots, fleet_count, best1, best2, p1, p2, target_mode)

        # --- SP policy ---
        if "SP" in available_abilities:
            # (1) no ship evidence after 2 shots -> SP
            # (2) SP unused by shot #10 -> force
            if (shipinfo_mask == 0 and total_shots >= 2) or (total_shots >= 9):
                center = self._best_sp_center(cell_prob, shot_mask)
                return {"combat": {"cell": [0, 0], "ability": {"SP": center}}}

        return {"combat": {"cell": best1, "ability": {"None": {}}}}


if __name__ == "__main__":
    run_bot(MyBattleshipBot)
