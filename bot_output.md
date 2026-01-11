# BOT OUTPUT Structure Documentation
This markdown file described the JSON Schema which players are expected to follow when describing their bots' output JSON.

## Top-Level Structure
The bot output should describe the player's move with a JSON object with ONE of the following properties:

- **abilitySelect**: enum equal to 'SP', 'RF', 'SD', or 'HS' for Sonar Pulse, Rapid Fire, Shield, or Hailstorm, respectively
- **placement**: A Placement object
- **combat**: A Combat object

## Objects
The objects described will be specified by a JSON object as follows:

## Placement Object
Represents a player's move during the *placement* phase
- **ship**: enum equal to '1x4', '1x3', '2x3', or '1x2'
- **cell**: 2D coordinate of where the front of ship will be
- **direction**: enum equal to 'V' or 'H' for Vertical or Horizontal, respectively, such that the rest of the cells follow rightwards or downwards (e.g, a ship at (7,7) with direction 'H' or 'V' will not be within the board, but a ship at (0,0) will always be within the board)

## Combat Object
Represents a player's move during the *combat* phase
- **cell**: 2D coordinate of the cell to bomb
- **ability**: an object with ONE of the following properties:
    - **None**: Empty object {}
    - **HS**: Empty object {} (recall that Hailstorm *randomly* selects cells to bomb)
    - **SP**: 2D coordinate where the center of Sonar Pulse will be
    - **RF**: Array of 2 coordinates (i.e. Array of 4 ints, [y1, x1, y2, x2]) for both Rapid Fire shots
    - **SD**: 2D coordinate of player Ship to protect