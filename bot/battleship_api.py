#!/usr/bin/env python3
"""
Code Clash Battleship Bot Challenge - CREATE UofT - Winter 2026

Battleship Bot API - DO NOT MODIFY

Contains all the infrastructure code that must remain unchanged.
"""

import sys
import json
import random
from typing import Dict, List, Any, Optional, Set, Tuple

# ============================================================================
# GAME CONSTANTS - DO NOT MODIFY
# ============================================================================

BOARD_SIZE = 8
SHIP_TYPES = ["ship_1x4", "ship_1x3", "ship_2x3", "ship_1x2"]
SHIP_SIZES = {
    "ship_1x4": (1, 4),
    "ship_1x3": (1, 3), 
    "ship_2x3": (2, 3),
    "ship_1x2": (1, 2)
}
ABILITY_CODES = ["SP", "RF", "SD", "HS"]

# ============================================================================
# BATTLESHIP BOT API CLASS
# ============================================================================

class BattleshipBotAPI:
    """
    Base class for all battleship bots.
    Participants inherit from this and override strategy methods.
    """
    
    def __init__(self):
        """Initialize bot - override if needed"""
        pass
    
    # ------------------------------------------------------------------------
    # INFRASTRUCTURE METHODS - DO NOT OVERRIDE
    # ------------------------------------------------------------------------
    
    def _get_ship_cells(self, ship_name: str, start_row: int, start_col: int, orientation: str) -> List[Tuple[int, int]]:
        """Calculate cells occupied by a ship."""
        if ship_name not in SHIP_SIZES:
            return []
            
        rows, cols = SHIP_SIZES[ship_name]
        cells = []
        
        if orientation == 'H':  # Horizontal
            for c in range(cols):
                for r in range(rows):
                    new_row = start_row + r
                    new_col = start_col + c
                    if new_row >= BOARD_SIZE or new_col >= BOARD_SIZE:
                        return []  # Out of bounds
                    cells.append((new_row, new_col))
        else:  # Vertical  
            for r in range(cols):
                for c in range(rows):
                    new_row = start_row + r
                    new_col = start_col + c
                    if new_row >= BOARD_SIZE or new_col >= BOARD_SIZE:
                        return []  # Out of bounds
                    cells.append((new_row, new_col))
        return cells
    
    def _is_valid_placement(self, cells: List[Tuple[int, int]], placed_coords: Set[Tuple[int, int]]) -> bool:
        """Check if ship placement doesn't overlap."""
        for cell in cells:
            if cell in placed_coords:
                return False
        return True
    
    def _get_placed_coordinates(self, game_state: Dict[str, Any]) -> Set[Tuple[int, int]]:
        """Get coordinates of already placed ships."""
        placed_coords = set()
        for ship in game_state.get("player_ships", []):
            if isinstance(ship, dict):
                for coord in ship.get("coordinates", []):
                    placed_coords.add(tuple(coord))
        return placed_coords
    
    def _get_available_cells(self, opponent_grid: List[List[str]]) -> List[List[int]]:
        """Get cells that haven't been shot at yet."""
        available_cells = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if opponent_grid[row][col] == 'N':
                    available_cells.append([row, col])
        return available_cells
    
    def _get_random_placement(self, ship_name: str, placed_coords: Set[Tuple[int, int]]) -> Optional[Dict[str, Any]]:
        """Generate random valid ship placement."""
        for _ in range(100):
            start_row = random.randint(0, BOARD_SIZE - 1)
            start_col = random.randint(0, BOARD_SIZE - 1)
            orientation = random.choice(['H', 'V'])
            
            cells = self._get_ship_cells(ship_name, start_row, start_col, orientation)
            if cells and self._is_valid_placement(cells, placed_coords):
                return {
                    "placement": {
                        "name": ship_name,
                        "cell": [start_row, start_col],
                        "direction": orientation
                    }
                }
        return None
    
    def _get_next_ship_to_place(self, game_state: Dict[str, Any]) -> Optional[str]:
        """Determine which ship needs to be placed next."""
        placed_ship_names = set()
        for ship in game_state.get("player_ships", []):
            if isinstance(ship, dict):
                placed_ship_names.add(ship.get("name", ""))
        
        ships_to_place = [ship for ship in SHIP_TYPES if ship not in placed_ship_names]
        return ships_to_place[0] if ships_to_place else None
    
    # ------------------------------------------------------------------------
    # DATA ACCESS HELPERS
    # ------------------------------------------------------------------------
    
    def _get_available_abilities(self, game_state: Dict[str, Any]) -> List[str]:
        """Get abilities you still have available."""
        abilities = []
        for ability_obj in game_state.get("player_abilities", []):
            if isinstance(ability_obj, dict):
                abilities.append(ability_obj.get("ability", ""))
        return [a for a in abilities if a in ABILITY_CODES]
    
    def _get_opponent_abilities(self, game_state: Dict[str, Any]) -> List[str]:
        """Get opponent's remaining abilities."""
        abilities = []
        for ability_obj in game_state.get("opponent_abilities", []):
            if isinstance(ability_obj, dict):
                abilities.append(ability_obj.get("ability", ""))
        return [a for a in abilities if a in ABILITY_CODES]
    
    def _get_opponent_grid(self, game_state: Dict[str, Any]) -> List[List[str]]:
        """Get grid showing your shots on opponent's board."""
        return game_state.get("opponent_grid", [['N'] * 8 for _ in range(8)])
    
    def _get_own_grid(self, game_state: Dict[str, Any]) -> List[List[str]]:
        """Get grid showing opponent's shots on your board."""
        return game_state.get("player_grid", [['N'] * 8 for _ in range(8)])
    
    def _get_own_ships(self, game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get your ships with coordinates and hit status."""
        return game_state.get("player_ships", [])
    
    # ------------------------------------------------------------------------
    # STRATEGY METHODS - OVERRIDE THESE
    # ------------------------------------------------------------------------
    
    def ability_selection(self) -> List[str]:
        """CHOOSE 2 abilities for the game. Override this."""
        return random.sample(ABILITY_CODES, 2)
    
    def place_ship_strategy(self, ship_name: str, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """PLACE a ship on your board. Override this."""
        placed_coords = self._get_placed_coordinates(game_state)
        placement = self._get_random_placement(ship_name, placed_coords)
        if placement:
            return placement
        return {
            "placement": {
                "name": ship_name,
                "cell": [0, 0],
                "direction": 'H'
            }
        }
    
    def combat_strategy(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """CHOOSE a combat move. Override this."""
        available_cells = self._get_available_cells(game_state["opponent_grid"])
        if available_cells:
            target = random.choice(available_cells)
        else:
            target = [random.randint(0, 7), random.randint(0, 7)]
        
        return {
            "combat": {
                "cell": target,
                "ability": {"None": {}}
            }
        }

# ============================================================================
# BOT EXECUTION LOGIC - DO NOT MODIFY
# ============================================================================

def run_bot(bot_class):
    """
    Main execution logic for bots.
    Participants DO NOT modify this function.
    """
    if len(sys.argv) != 2:
        print("ERROR: Usage: python3 bot.py <state.json>", file=sys.stderr)
        sys.exit(1)
    
    try:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            game_state = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load game state: {e}", file=sys.stderr)
        sys.exit(1)
    
    bot = bot_class()
    
    try:
        # Determine phase
        if "player_ships" not in game_state:
            # Ability selection phase
            move = {"abilitySelect": bot.ability_selection()}
        elif len(game_state["player_ships"]) < len(SHIP_TYPES):
            # Placement phase
            next_ship = bot._get_next_ship_to_place(game_state)
            if next_ship:
                move = bot.place_ship_strategy(next_ship, game_state)
            else:
                move = bot.place_ship_strategy(SHIP_TYPES[0], game_state)
        else:
            # Combat phase
            move = bot.combat_strategy(game_state)
        
        print(json.dumps(move))
        
    except Exception as e:
        print(f"ERROR: Bot strategy failed: {e}", file=sys.stderr)
        sys.exit(1)