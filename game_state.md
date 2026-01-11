# BOT INPUT GAME STATE Structure Documentation
This markdown file describes the JSON Schema used in the game-state JSON file which teams' bots will process, representing the match history of the Code Clash 2025 Battleship game. 

## Summary
Each turn, a player will be provided JSON file describing every turn  throughout the current game. I.e., the result of every move (whether a hit, miss, or block) from the first turn to the most previous, will be recorded, along with the complete location of the player's ships (own ships, *not* the enemy's ships). Additionally, information of active abilities (e.g. Sonar Pulse, Rapid Fire, etc.) of both the player and opponent will be included. The rest of this document specified the structure of this information.

## Top-Level Structure
The game state will be specified by a JSON object with the following properties:
- **player_ships**: Array of Ship object
- **player_grid**: 8x8 Array of enums equal to 'H', 'M', or 'B', 'N', representing Hit, Miss, Block, or None respectively, coming from the opponent, unto the player's board
- **opponent_grid**: 8x8 Array of enums equal to 'H', 'M', or 'B', 'N', representing Hit, Miss, Block, or None respectively, coming from the player, unto the opponent's board
- **player_abilities**: List of 2 Ability objects representing the player's equipped abilities
- **opponent_abilities**: List of 2 Ability objects representing the opponent's equipped abilties

## Objects
The objects described will be specified by a JSON object as follows:

## Ship Object
Represents a player's own ship
- **coordinates**: Array of 2D coordinates occupied by Ship
- **hits**: Array of 2D coordinates previously hit on Ship

## Ability Object
- **ability**: enum equal to 'SP', 'RF', 'SD', or 'HS', for Sonar Pulse, Rapid Fire, Shield, or Hailstorm, respectively, specifying the ability
- **info**: An object containing ONE of the following possible set of properties
    - **None**: Empty object {}
    - **SD**: Empty object {} (location of opponent ship being shielded is known only to the opponent)
    - **SP**: 3x3 List of SonarPulse objects activated by player
    - **RF**: Array of 2 coordinates (i.e. Array of 4 ints, [y1, x1, y2, x2]) for both Rapid Fire shots
    - **HS**: Array of 4 Bomb objects, representing projectiles randomly fired by Hailstorm

    ## Sonar Pulse Object
    - **cell**: 2D coordinate representing cell covered by Sonar Pulse
    - **result**: enum equal to 'Ship' or 'Water'

    ## Bomb Object
    Represents a cell-bomb by the player
    - **cell**: 2D coordinate representing cell targetted by bomb
    - **result**: enum equal to 'H', 'M', or 'B', 'N', representing Hit, Miss, Block, or None respectively