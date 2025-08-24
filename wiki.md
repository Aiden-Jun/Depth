# Operations: Chess Engine Mechanics

This document explains the internal workings of the chess engine, including how moves are calculated, evaluated, and filtered for legality.

## Board Representation

- The engine uses a **120-square board** (10x12) to simplify move generation and boundary checking.
  - Squares outside the 8x8 playable board are treated as **off-board** and stored in `ob`.
  - Indices 22–29, 32–39, 52–59, 62–69 represent the playable board squares.
- Each square contains a piece ID from the `pieces` dictionary or `0` for empty (`e`).

## Piece Definitions

- Pieces are assigned numeric IDs:
  - White: pawn (1), knight (2), bishop (3), rook (4), queen (5), king (6)
  - Black: pawn (7), knight (8), bishop (9), rook (10), queen (11), king (12)
- `pvm` defines piece values for evaluation.
- `pst` defines **piece-square tables** to adjust positional evaluation.

## Move Representation

- `Move` objects contain:
  - `sp`: starting square
  - `dp`: destination square
  - `promo`: promotion piece (if any)
  - `is_capture`: whether the move captures a piece
- UCI strings are generated using `index_to_pos()` converting 120-index to chess notation.

## Move Generation Logic

The `Logics` class contains functions for generating **pseudo-legal moves** for each piece type.

### Pawns (`pawn_moves`)
- Move offsets differ by color:
  - White: single (-10), double (-20), captures (-11 left, -9 right)
  - Black: single (+10), double (+20), captures (+9 left, +11 right)
- Pawn moves:
  1. **Single step** if the square is empty.
  2. **Double step** from the starting rank if both squares are empty.
  3. **Captures** diagonally if an enemy piece exists.
  4. **Promotion** if reaching the back rank: generates 4 moves (queen, rook, bishop, knight).
  5. **En passant**: adds a capture move if en passant target exists.
- Moves going off-board are filtered using `ob`.

### Knights (`knight_moves`)
- Knight offsets: `[-17, -15, -10, -6, 6, 10, 15, 17]` (120-index).
- A move is valid if:
  - The target is **not off-board**.
  - The target is empty or contains an enemy piece.

### Bishops (`bishop_moves`)
- Move offsets: `[-11, -9, 9, 11]` (diagonal directions).
- Moves continue in each direction until:
  - A piece blocks the path.
  - An enemy piece is captured (stop after capture).
- Off-board squares are skipped using `ob`.

### Rooks (`rook_moves`)
- Move offsets: `[-10, -1, 1, 10]` (straight directions).
- Moves continue in each direction until blocked or capturing an enemy.

### Queens (`queen_moves`)
- Combines rook and bishop moves (all 8 directions).

### King (`king_moves`)
- Move offsets: `[-11, -10, -9, -1, 1, 9, 10, 11]`.
- Checks for:
  - **Castling rights**:
    - King-side: squares between king and rook must be empty and not attacked.
    - Queen-side: same logic.
  - Captures and single-square moves like normal.

## Attack Detection

- `index_attacked(board, color, square)` determines if a square is attacked:
  1. Check for pawn attacks based on enemy pawn direction.
  2. Check for knight attacks using knight offsets.
  3. Check for sliding attacks (bishop, rook, queen) along rays until blocked.
  4. Check for king proximity attacks.

## Move Filtering

- After generating pseudo-legal moves, the engine:
  - Filters **illegal moves** (leaving king in check) using `is_legal()`.
  - Removes moves that go off-board or into friendly pieces.
  - Handles **special moves**: castling, en passant, promotion.

## Evaluation

- `evaluate(board)` calculates a score for a given position:
  1. Piece value sum (material count).
  2. Positional bonuses/penalties from `pst`.
  3. Additional penalties for check, mobility, or king safety.

## Summary

- The engine **represents the board in 120-index** for simplicity.
- Each piece type has **specific movement offsets**.
- Moves are first generated **pseudo-legally**, then filtered for legality.
- Attacks are detected using a combination of offsets and ray checks.
- Special moves and promotions are handled explicitly.
- Evaluation relies on **material + position** heuristics.

## Search

The search module is responsible for determining the best move from a given board state. It uses a **minimax algorithm** with **alpha-beta pruning**. Here's a breakdown of how it works.

### Minimax Algorithm

The minimax algorithm is a recursive decision-making process used to find the optimal move in two-player, turn-based games like chess.

1. **Base Case:**  
   If the search depth is 0 or the game is over (checkmate/stalemate), the algorithm evaluates the board using a static evaluation function (`eval`) and returns a score.

2. **Recursive Case:**  
   - Generate all legal moves for the current player.  
   - Recursively simulate each move by calling `minimax` with `depth - 1` and switching the player (`maximizing_player` toggles).  
   - Evaluate each move's resulting board using the same algorithm until reaching the base case.

3. **Maximizing vs Minimizing:**  
   - If the current player is the **maximizing player** (e.g., White), choose the move with the **highest evaluation score**.  
   - If the current player is the **minimizing player** (e.g., Black), choose the move with the **lowest evaluation score**.  

### Alpha-Beta Pruning

Alpha-beta pruning optimizes minimax by **eliminating branches that cannot possibly affect the final decision**.

- `alpha` = best score found so far for the **maximizing player**.  
- `beta` = best score found so far for the **minimizing player**.  

#### Rules:

1. **Maximizing Player:**  
   - If a move's evaluation ≥ `beta`, stop exploring further moves (beta cutoff).  
   - Update `alpha` if the move's evaluation > current `alpha`.

2. **Minimizing Player:**  
   - If a move's evaluation ≤ `alpha`, stop exploring further moves (alpha cutoff).  
   - Update `beta` if the move's evaluation < current `beta`.

This reduces the number of nodes evaluated without affecting the final decision.

### Move Evaluation

For each leaf node (or board at depth 0):

1. **Material Score:**  
   Sum of piece values (pawns, knights, bishops, rooks, queens). White pieces are positive, black pieces are negative.

2. **Positional Score (Piece-Square Tables):**  
   Each piece has a predefined value for each square (`pst`). Positional value is multiplied by a strength factor (`pst_strength`) and added to the material score.  
   - White pieces: use the table as-is.  
   - Black pieces: the table is mirrored vertically to reflect perspective.

3. **Final Evaluation:**  
   `eval(board) = material_score + positional_score`  

This score represents how favorable the board is for the current player.

### Best Move Selection

Once all branches are evaluated:

- The **highest evaluation** for the maximizing player or **lowest evaluation** for the minimizing player is selected as the **best move**.  
- `search_best_move` returns this move along with its evaluation.

### Summary

1. Generate all legal moves.  
2. For each move, recursively simulate using minimax.  
3. Use alpha-beta pruning to cut off unpromising branches.  
4. Evaluate leaf boards with material + positional values.  
5. Return the move with the optimal evaluation.
