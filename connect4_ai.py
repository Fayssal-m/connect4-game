"""Puissance 4 - Simplified AI Module using Minimax with Alpha-Beta Pruning.

This version uses the standard 2D board grid (no bitboards, no complex bitwise shifts),
making it simple to understand, debug, and present.

Key components:
1. Heuristic B: Evaluates all 69 possible 4-cell windows (horizontal, vertical, diagonal).
2. Center bonus: Gives higher weight to the center column (column 3).
3. Threats & Parity (Allis rule): Detects potential winning spots and scores them by row parity.
4. Minimax with Alpha-Beta pruning: Explores the game tree efficiently.
5. In-place Drop & Undo: Modifies and reverts the grid without heavy copying.
"""

from connect4_logic import ROWS, COLS, EMPTY, P1, P2

# Center-out order: column 3 is tested first, then columns 2 and 4, etc.
# This causes Alpha-Beta to prune bad branches much earlier.
COLUMN_ORDER = [3, 2, 4, 1, 5, 0, 6]

WIN_SCORE = 1_000_000


def build_all_windows():
    r"""Precompute all 69 groups of 4 connected cells on a 6x7 board.
    
    - 24 horizontal windows
    - 21 vertical windows
    - 12 diagonal / windows
    - 12 diagonal \ windows
    """
    windows = []
    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            windows.append([(r, c + i) for i in range(4)])
    # Vertical
    for r in range(ROWS - 3):
        for c in range(COLS):
            windows.append([(r + i, c) for i in range(4)])
    # Diagonal / (bottom-left to top-right)
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            windows.append([(r - i, c + i) for i in range(4)])
    # Diagonal \ (top-left to bottom-right)
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            windows.append([(r + i, c + i) for i in range(4)])
    return windows


WINDOWS = build_all_windows()


def is_fast_win(grid, r, c, player):
    """Check if the piece dropped at (r, c) creates 4 aligned pieces for player."""
    for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        count = 1
        # Check forward along the direction
        i = 1
        while 0 <= r + i * dr < ROWS and 0 <= c + i * dc < COLS and grid[r + i * dr][c + i * dc] == player:
            count += 1
            i += 1
        # Check backward along the opposite direction
        i = 1
        while 0 <= r - i * dr < ROWS and 0 <= c - i * dc < COLS and grid[r - i * dr][c - i * dc] == player:
            count += 1
            i += 1
        if count >= 4:
            return True
    return False


def evaluate_board(board, ai_player):
    """Heuristic function: gives a numerical score to a board state."""
    hu_player = P2 if ai_player == P1 else P1
    grid = board.grid
    score = 0
    ai_threats = []
    hu_threats = []

    # 1. Evaluate all 69 windows (Heuristic B)
    for w in WINDOWS:
        vals = [grid[r][c] for r, c in w]
        ai_count = vals.count(ai_player)
        hu_count = vals.count(hu_player)
        empty_count = vals.count(EMPTY)

        # If both players have pieces in this window, nobody can win with it
        if ai_count > 0 and hu_count > 0:
            continue

        if ai_count == 3 and empty_count == 1:
            score += 20
            # Check if this empty spot is an unplayable threat (not yet supported)
            for r, c in w:
                if grid[r][c] == EMPTY and r < ROWS - 1 and grid[r + 1][c] == EMPTY:
                    ai_threats.append((r, c))
        elif ai_count == 2 and empty_count == 2:
            score += 5

        if hu_count == 3 and empty_count == 1:
            score -= 20
            for r, c in w:
                if grid[r][c] == EMPTY and r < ROWS - 1 and grid[r + 1][c] == EMPTY:
                    hu_threats.append((r, c))
        elif hu_count == 2 and empty_count == 2:
            score -= 5

    # 2. Center Column Bonus (Column 3)
    # The center column offers the highest tactical value
    for r in range(ROWS):
        if grid[r][3] == ai_player:
            score += 4
        elif grid[r][3] == hu_player:
            score -= 4

    # 3. Threat Parity (Victor Allis Rule)
    # Player 1 naturally controls odd rows (1, 3, 5 from bottom)
    # Player 2 controls even rows (2, 4, 6 from bottom)
    for r, c in set(ai_threats):
        row_height = ROWS - r  # 1 at the bottom, 6 at the top
        favorable = (row_height % 2 == 1) if ai_player == P1 else (row_height % 2 == 0)
        score += 40 if favorable else 10

    for r, c in set(hu_threats):
        row_height = ROWS - r
        favorable = (row_height % 2 == 1) if hu_player == P1 else (row_height % 2 == 0)
        score -= 40 if favorable else 10

    return score


class AlphaBetaAI:
    """Minimax AI with Alpha-Beta pruning."""

    def __init__(self, depth=5):
        self.depth = depth
        self.nodes = 0

    def best_move(self, board, ai_player):
        """Find the best column for ai_player."""
        self.nodes = 0
        valid_cols = board.valid_columns()

        if not valid_cols:
            return None

        # Optimization: play center on empty board
        if len(valid_cols) == COLS and all(board.grid[ROWS - 1][c] == EMPTY for c in range(COLS)):
            return 3

        hu_player = P2 if ai_player == P1 else P1

        # Check immediate win or immediate block at root (depth 1 check)
        for c in valid_cols:
            r = board.drop(c, ai_player)
            if is_fast_win(board.grid, r, c, ai_player):
                board.undo(c)
                return c
            board.undo(c)

        for c in valid_cols:
            r = board.drop(c, hu_player)
            if is_fast_win(board.grid, r, c, hu_player):
                board.undo(c)
                return c
            board.undo(c)

        # Minimax recursive search
        best_col, _ = self._max_value(
            board=board,
            depth=self.depth,
            alpha=-float("inf"),
            beta=float("inf"),
            ai_player=ai_player,
            hu_player=hu_player
        )
        return best_col

    def _max_value(self, board, depth, alpha, beta, ai_player, hu_player):
        self.nodes += 1
        valid_cols = board.valid_columns()

        if not valid_cols:
            return None, 0  # Draw

        if depth == 0:
            return None, evaluate_board(board, ai_player)

        ordered_cols = [c for c in COLUMN_ORDER if c in valid_cols]
        best_val = -float("inf")
        best_col = ordered_cols[0]

        for col in ordered_cols:
            r = board.drop(col, ai_player)
            if is_fast_win(board.grid, r, col, ai_player):
                board.undo(col)
                return col, WIN_SCORE + depth

            _, val = self._min_value(board, depth - 1, alpha, beta, ai_player, hu_player)
            board.undo(col)

            if val > best_val:
                best_val = val
                best_col = col

            alpha = max(alpha, best_val)
            if alpha >= beta:
                break  # Beta cut-off

        return best_col, best_val

    def _min_value(self, board, depth, alpha, beta, ai_player, hu_player):
        self.nodes += 1
        valid_cols = board.valid_columns()

        if not valid_cols:
            return None, 0

        if depth == 0:
            return None, evaluate_board(board, ai_player)

        ordered_cols = [c for c in COLUMN_ORDER if c in valid_cols]
        best_val = float("inf")
        best_col = ordered_cols[0]

        for col in ordered_cols:
            r = board.drop(col, hu_player)
            if is_fast_win(board.grid, r, col, hu_player):
                board.undo(col)
                return col, -(WIN_SCORE + depth)

            _, val = self._max_value(board, depth - 1, alpha, beta, ai_player, hu_player)
            board.undo(col)

            if val < best_val:
                best_val = val
                best_col = col

            beta = min(beta, best_val)
            if beta <= alpha:
                break  # Alpha cut-off

        return best_col, best_val
