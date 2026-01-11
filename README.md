# CREATE Code Clash: Battleship Bot Challenge - Winter 2026

## Table of Contents
1. [Competition Overview](#competition-overview)
2. [Game Rules](#game-rules)
3. [Bot Development Guide](#bot-development-guide)
4. [JSON Interface Specification](#json-interface-specification)
5. [Submission Guidelines](#submission-guidelines)
6. [Technical Requirements](#technical-requirements)

## Competition Overview

Welcome to Code Clash: Battleship Bot Challenge! This competition challenges you to create an AI bot that plays a enhanced version of Battleship against other participants' bots. Your bot will compete in automated matches, with the top performers advancing to the live finals during our closing ceremony.

### Key Features
- **Enhanced Battleship**: 8×8 grid with special abilities
- **Special Abilities**: Choose 2 of 4 unique powers
- **Live Finals**: Top 8 bots compete in live-streamed matches

## Game Rules

### Board and Ships
- **Board Size**: 8×8 grid (0-based coordinates: 0-7)
- **Ships**: 4 ships with different sizes:
  - 1×4 (4 cells)
  - 1×3 (3 cells) 
  - 2×3 (6 cells)
  - 1×2 (2 cells)

### Game Phases
1. **Ability Selection**: Choose 2 of 4 abilities for the entire game
2. **Ship Placement**: Place all 4 ships on your board
3. **Combat**: Take turns shooting or using abilities until one player wins

### Special Abilities
Each ability can be used **once per game** (instead of shooting):

| Code | Name | Effect |
|------|------|--------|
| SP | Sonar Pulse | Reveal 3×3 area |
| RF | Rapid Fire | Fire 2 simultaneous shots |
| SD | Shield | Protect a ship for 2 turns |
| HS | Hailstorm | Fire 4 random shots across grid |

### Win Condition
The first player to sink all opponent's ships wins the match.

## Bot Development Guide

### Bot Interface
Your bot must be a **single executable file** that:
- Accepts one command-line argument: path to `state.json`
- Reads game state from the provided JSON file
- Outputs a valid JSON move
- Exits with code `0` on success, non-zero on error
- Completes within **3 seconds** per move

### Execution Pattern
```bash
./your_bot /path/to/state.json
```

## Starter Code

We provide starter code in three languages:
- **Python**: `battleship_bot.py` + `battleship_api.py`
- **C++**: `battleship_bot.cpp` + `battleship_api.h`
- **C**: `battleship_bot.c` + `battleship_api.h`

## JSON Interface Specification

### Input: Game State (`state.json`)

The game state provided to your bot varies by phase:

**Ability Selection Phase**
```json
{}
```

**Placement Phase**
```json
{
    "player_ships": [
        {
        "name": "ship_1x4",
        "coordinates": [[0,0], [0,1], [0,2], [0,3]],
        "hits": []
        }
    ],
    "player_grid": [
        ["N","N","N","N","N","N","N","N"],
        ["N","N","N","N","N","N","N","N"],
        ["N","N","N","N","N","N","N","N"],
        ["N","N","N","N","N","N","N","N"],
        ["N","N","N","N","N","N","N","N"],
        ["N","N","N","N","N","N","N","N"],
        ["N","N","N","N","N","N","N","N"],
        ["N","N","N","N","N","N","N","N"]
    ],
    "opponent_grid": [
        ["N","N","N","N","N","N","N","N"],
        ["N","N","N","N","N","N","N","N"],
        ["N","N","N","N","N","N","N","N"],
        ["N","N","N","N","N","N","N","N"],
        ["N","N","N","N","N","N","N","N"],
        ["N","N","N","N","N","N","N","N"],
        ["N","N","N","N","N","N","N","N"],
        ["N","N","N","N","N","N","N","N"]
    ],
    "player_abilities": [],
    "opponent_abilities": [],
    "current_ship": "ship_1x3" # Ship to be placed in current call
}
```

**Combat Phase**
```json
{
    "player_ships": [
        {
        "name": "ship_1x4",
        "coordinates": [[0,0], [0,1], [0,2], [0,3]],
        "hits": []
        },
        # 3 other ships here
    ],
    "player_grid": [
        ["N","N","N","N","N","N","N","N"],
        ["N","N","N","N","N","N","N","N"],
        ["N","N","N","N","N","N","N","N"],
        ["N","N","N","N","N","N","N","N"],
        ["N","N","N","N","N","N","N","N"],
        ["N","N","N","N","N","N","N","N"],
        ["N","N","N","N","N","N","N","N"],
        ["N","N","N","N","N","N","N","N"]
    ],
    "opponent_grid": [
        ["N","N","N","N","N","N","N","N"],
        ["N","N","N","N","N","N","N","N"],
        ["N","N","N","N","N","N","N","N"],
        ["N","N","N","N","N","N","N","N"],
        ["N","N","N","N","N","N","N","N"],
        ["N","N","N","N","N","N","N","N"],
        ["N","N","N","N","N","N","N","N"],
        ["N","N","N","N","N","N","N","N"]
    ],
    "player_abilities": [
        {"ability": "SP", "info": {"None": {}}},
        {"ability": "RF", "info": {"None": {}}}
    ],
    "opponent_abilities": [
        {"ability": "SD", "info": {"None": {}}},
        {"ability": "HS", "info": {"None": {}}}
    ]
}
```

**Grid Cell Status**
- **N**: No shot attempted
- **H**: Hit (successful shot)
- **M**: Miss (unsuccessful shot)
- **B**: Blocked (shot blocked by shield)

### Output: Bot Moves

Your bot must output exactly one of these JSON response format:

**Ability Selection**
```json
{"abilitySelect": ["SP", "RF"]}
```

**Ship Placement**
```json
{
  "placement": {
    "name": "ship_1x4",
    "cell": [0, 0],
    "direction": "H"
  }
}
```

**Combat Move (Shoot)**
```json
{
  "combat": {
    "cell": [3, 4],
    "ability": {"None": {}}
  }
}
```

**Combat Move (Use Ability)**
```json
{
  "combat": {
    "cell": [0, 0],
    "ability": {"SP": [3, 3]}
  }
}
```

## Submission Guidelines

### File Structure

Submit a .py file. (battleship_bot, NOT battleship_api). do NOT change battleship_api btw.

### Executable Requirements
- **Single file:** Must run as standalone executable
- **Ubuntu 24.04:** Tested on our competition environment
- **No dependencies:** Except standard system libraries
- **3-second timeout:** Must complete moves within time limit. Failure to comply will result in random behaviour for that turn.

### Build Instructions
TODO: complete this section

### Validation Checklist
- Bot reads state.json from command line argument
- Bot outputs valid JSON
- Executable runs on Ubuntu 24.04
- All moves complete within 3 seconds
- No external network/database dependencies
- ZIP contains executable + source code + README
