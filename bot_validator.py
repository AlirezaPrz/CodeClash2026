#!/usr/bin/env python3
"""
Code Clash Battleship Bot Challenge - CREATE UofT - Winter 2026

Bot Validator - Test your bot against the JSON schema before submission
"""

import json
import random
import subprocess
import tempfile
import os
import sys

# ANSI color codes for terminal output
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

SHIP_TYPES = ["ship_1x4", "ship_1x3", "ship_2x3", "ship_1x2"]

def create_test_state(phase="combat"):
    """Create a test game state for validation."""
    if phase == "ability_selection":
        return {
            "player_grid": [['N'] * 8 for _ in range(8)],
            "opponent_grid": [['N'] * 8 for _ in range(8)],
            "player_abilities": [],
            "opponent_abilities": []
        }
    
    elif phase == "placement":
        # During placement, only show already-placed ships
        # Let's say 1 ship is already placed
        return {
            "player_ships": [
                {
                    "name": "ship_1x4",
                    "coordinates": [[0, 0], [0, 1], [0, 2], [0, 3]],
                    "hits": []
                }
            ],
            "player_grid": [['N'] * 8 for _ in range(8)],
            "opponent_grid": [['N'] * 8 for _ in range(8)],
            "player_abilities": [],
            "opponent_abilities": []
        }
    
    else:  # Combat - all ships placed
        player_grid = [['N'] * 8 for _ in range(8)]
        opponent_grid = [['N'] * 8 for _ in range(8)]
        
        # Add some random hits/misses for realism
        for i in range(5):
            r, c = random.randint(0, 7), random.randint(0, 7)
            player_grid[r][c] = 'H'
            r, c = random.randint(0, 7), random.randint(0, 7)
            opponent_grid[r][c] = 'H'
        
        for i in range(7):
            r, c = random.randint(0, 7), random.randint(0, 7)
            player_grid[r][c] = 'M'
            r, c = random.randint(0, 7), random.randint(0, 7)
            opponent_grid[r][c] = 'M'
        
        for i in range(2):
            r, c = random.randint(0, 7), random.randint(0, 7)
            player_grid[r][c] = 'B'
        
        return {
            "player_ships": [
                {
                    "name": "ship_1x4",
                    "coordinates": [[0, 0], [0, 1], [0, 2], [0, 3]],
                    "hits": [[0, 1]]
                },
                {
                    "name": "ship_1x3",
                    "coordinates": [[2, 2], [3, 2], [4, 2]],
                    "hits": []
                },
                {
                    "name": "ship_2x3",
                    "coordinates": [[5, 0], [5, 1], [5, 2], [6, 0], [6, 1], [6, 2]],
                    "hits": []
                },
                {
                    "name": "ship_1x2",
                    "coordinates": [[7, 5], [7, 6]],
                    "hits": []
                }
            ],
            "player_grid": player_grid,
            "opponent_grid": opponent_grid,
            "player_abilities": [
                {"ability": "SP", "info": {"None": {}}},
                {"ability": "RF", "info": {"None": {}}}
            ],
            "opponent_abilities": [
                {"ability": "SD", "info": {"None": {}}},
                {"ability": "HS", "info": {"None": {}}}
            ]
        }

def validate_bot_output(output, phase):
    """Validate bot output for specific phase."""
    try:
        data = json.loads(output)
        
        if phase == "ability_selection":
            if "abilitySelect" not in data:
                return False, "Missing 'abilitySelect' key"
            abilities = data["abilitySelect"]
            if not isinstance(abilities, list) or len(abilities) != 2:
                return False, "Must provide exactly 2 abilities"
            valid = {"SP", "RF", "SD", "HS"}
            for ab in abilities:
                if ab not in valid:
                    return False, f"Invalid ability: {ab}"
            return True, "Valid ability selection"
        
        elif phase == "placement":
            if "placement" not in data:
                return False, "Missing 'placement' key"
            placement = data["placement"]
            if "cell" not in placement or "direction" not in placement:
                return False, "Placement missing cell or direction"
            cell = placement["cell"]
            if not isinstance(cell, list) or len(cell) != 2:
                return False, "Cell must be [row, col] array"
            if cell[0] < 0 or cell[0] > 7 or cell[1] < 0 or cell[1] > 7:
                return False, f"Cell out of bounds: {cell}"
            if placement["direction"] not in ["H", "V"]:
                return False, "Direction must be 'H' or 'V'"
            return True, "Valid placement"
        
        else:  # combat
            if "combat" not in data:
                return False, "Missing 'combat' key"
            combat = data["combat"]
            if "cell" not in combat or "ability" not in combat:
                return False, "Combat missing cell or ability"
            cell = combat["cell"]
            if not isinstance(cell, list) or len(cell) != 2:
                return False, "Cell must be [row, col] array"
            if cell[0] < 0 or cell[0] > 7 or cell[1] < 0 or cell[1] > 7:
                return False, f"Cell out of bounds: {cell}"
            return True, "Valid combat move"
    
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"

def test_bot(bot_path, phase="combat"):
    """Test if bot produces valid output for given phase."""
    test_state = create_test_state(phase)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_state, f)
        temp_file = f.name
    
    try:
        # Run the bot
        result = subprocess.run(
            ['python3', bot_path, temp_file],
            capture_output=True,
            text=True,
            timeout=3
        )
        
        if result.returncode != 0:
            print(f"{RED}❌ Bot crashed with return code {result.returncode}{RESET}")
            if result.stderr:
                print(f"{YELLOW}Stderr:{RESET} {result.stderr[:200]}...")
            return False
        
        valid, message = validate_bot_output(result.stdout.strip(), phase)
        
        if valid:
            print(f"{GREEN}✅ {phase.upper()}: {message}{RESET}")
            print(f"{BLUE}   Output: {result.stdout.strip()}{RESET}")
            return True
        else:
            print(f"{RED}❌ {phase.upper()}: {message}{RESET}")
            print(f"{YELLOW}   Output: {result.stdout.strip()}{RESET}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"{RED}❌ {phase.upper()}: Bot timed out (3 seconds){RESET}")
        return False
    except Exception as e:
        print(f"{RED}❌ {phase.upper()}: Unexpected error: {e}{RESET}")
        return False
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"{RED}Usage: python3 bot_validator.py <path_to_bot.py>{RESET}")
        sys.exit(1)
    
    bot_path = sys.argv[1]
    
    if not os.path.exists(bot_path):
        print(f"{RED}Error: Bot file '{bot_path}' not found{RESET}")
        sys.exit(1)
    
    print(f"{BLUE}═" * 50)
    print(f"VALIDATING: {bot_path}")
    print("═" * 50 + f"{RESET}")
    
    phases = ["ability_selection", "placement", "combat"]
    all_passed = True
    results = []
    
    for phase in phases:
        passed = test_bot(bot_path, phase)
        results.append((phase, passed))
        all_passed = all_passed and passed
    
    print(f"\n{BLUE}═" * 50)
    print("VALIDATION SUMMARY")
    print("═" * 50 + f"{RESET}")
    
    for phase, passed in results:
        status = f"{GREEN}✅ PASS{RESET}" if passed else f"{RED}❌ FAIL{RESET}"
        print(f"  {phase:20} {status}")
    
    if all_passed:
        print(f"\n{GREEN}✨ All tests passed! Your bot is ready for submission. ✨{RESET}")
        print(f"\n{YELLOW}⚠️  IMPORTANT: This validator tests basic functionality.")
        print(f"   For comprehensive testing, run your bot against sample bots.")
        print(f"   See dev_guide.md for more testing instructions.{RESET}")
    else:
        print(f"\n{RED}⚠️  Some tests failed. Please fix your bot before submission.{RESET}")
        sys.exit(1)