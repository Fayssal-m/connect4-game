"""Puissance 4 - Computer player: MinMax with Alpha-Beta pruning.

The search works on BITBOARDS (one integer per player) instead of the
6x7 grid, because a depth of 8 means hundreds of thousands of positions
and plain Python lists would be too slow.

Bit layout: column c, row r (r = 0 is the BOTTOM) -> bit number c*7 + r.
Each column has 7 bits: 6 for cells + 1 empty "sentinel" bit on top, which
stops alignments from wrapping from one column to the next.
"""
from connect4_logic import ROWS, COLS, EMPTY, P1

H = ROWS + 1                          # bits per column (6 cells + sentinel)
WIN_SCORE = 1_000_000                 # value of a won position
INF = float("inf")
COLUMN_ORDER = [3, 2, 4, 1, 5, 0, 6]  # try the centre first (better pruning)

popcount = int.bit_count if hasattr(int, "bit_count") else (lambda x: bin(x).count("1"))

# ---------------------------------------------------------------- heuristic
W2 = 5                  # window with 2 of my pieces + 2 empty
W3 = 20                 # window with 3 of my pieces + 1 empty
THREAT_GOOD = 40        # threat cell on the row that parity favours
THREAT_OTHER = 10       # threat cell on the other rows
CENTER_WEIGHT = [1, 2, 3, 4, 3, 2, 1]   # bonus per piece, by column


def _bit(col, row):
    return 1 << (col * H + row)


def _build_windows():
    """Every group of 4 aligned cells on the board, as a bit mask (69 total)."""
    windows = []
    for col in range(COLS):
        for row in range(ROWS):
            for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
                cells = [(col + i * dc, row + i * dr) for i in range(4)]
                if all(0 <= c < COLS and 0 <= r < ROWS for c, r in cells):
                    windows.append(sum(_bit(c, r) for c, r in cells))
    return windows


WINDOWS = _build_windows()
COLUMN_MASK = [((1 << ROWS) - 1) << (c * H) for c in range(COLS)]


def is_win(b):
    """True if bitboard `b` contains 4 aligned pieces."""
    for s in (1, H, H - 1, H + 1):        # vertical, horizontal, 2 diagonals
        m = b & (b >> s)
        if m & (m >> (2 * s)):
            return True
    return False


def _threat_value(threats, owner_is_first):
    """Score the empty cells that would complete a 4 for one player.

    Parity rule (zugzwang): when the board fills up, the FIRST player
    tends to get the odd rows (1,3,5 counted from the bottom) and the
    SECOND player the even rows (2,4,6). A threat on "your" parity is worth more.
    """
    total = 0
    while threats:
        low = threats & -threats            # isolate one threat cell
        threats ^= low
        row = (low.bit_length() - 1) % H + 1   # 1..6, from the bottom
        good = (row % 2 == 1) if owner_is_first else (row % 2 == 0)
        total += THREAT_GOOD if good else THREAT_OTHER
    return total


def evaluate(ai, hu, ai_is_first):
    """Heuristic value of a position, from the computer's point of view."""
    occupied = ai | hu
    score = 0
    ai_threats = hu_threats = 0

    # 1) window scoring (+ collect the threat cells)
    for w in WINDOWS:
        a = popcount(ai & w)
        h = popcount(hu & w)
        if a and h:
            continue                         # dead window: nobody can finish it
        if h == 0:
            if a == 2:
                score += W2
            elif a == 3:
                score += W3
                ai_threats |= w & ~occupied
        elif a == 0:
            if h == 2:
                score -= W2
            elif h == 3:
                score -= W3
                hu_threats |= w & ~occupied

    # 2) centre bonus
    for c in range(COLS):
        mask = COLUMN_MASK[c]
        score += CENTER_WEIGHT[c] * (popcount(ai & mask) - popcount(hu & mask))

    # 3) threats + parity
    score += _threat_value(ai_threats, ai_is_first)
    score -= _threat_value(hu_threats, not ai_is_first)
    return score


# ------------------------------------------------------------------- search
class AlphaBetaAI:
    def __init__(self, depth=8):
        self.depth = depth
        self.nodes = 0                      # positions explored (for statistics)
        self.ai_is_first = False

    def best_move(self, board, ai_player):
        """Return the best column for `ai_player` on `board`."""
        ai, hu, heights, played = self._to_bitboards(board, ai_player)
        self.ai_is_first = (ai_player == P1)
        self.nodes = 0

        if played == 0:
            return COLS // 2                # opening: the centre column is best

        best_col, best_val = None, -INF
        alpha, beta = -INF, INF
        for c in COLUMN_ORDER:
            if heights[c] >= ROWS:
                continue
            new_ai = ai | (1 << (c * H + heights[c]))
            if is_win(new_ai):
                return c                    # winning move: play it
            heights[c] += 1
            val = self._min_value(new_ai, hu, heights, self.depth - 1,
                                  alpha, beta, played + 1)
            heights[c] -= 1
            if val > best_val:
                best_val, best_col = val, c
            alpha = max(alpha, val)
        return best_col

    def _max_value(self, ai, hu, heights, depth, alpha, beta, played):
        """Computer's turn: choose the move with the HIGHEST value."""
        self.nodes += 1
        if played == ROWS * COLS:
            return 0                        # board full: draw
        if depth == 0:
            return evaluate(ai, hu, self.ai_is_first)
        value = -INF
        for c in COLUMN_ORDER:
            if heights[c] >= ROWS:
                continue
            new_ai = ai | (1 << (c * H + heights[c]))
            if is_win(new_ai):
                return WIN_SCORE + depth    # + depth: prefer a FASTER win
            heights[c] += 1
            v = self._min_value(new_ai, hu, heights, depth - 1,
                                alpha, beta, played + 1)
            heights[c] -= 1
            if v > value:
                value = v
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break                       # beta cut-off: MIN will avoid this branch
        return value

    def _min_value(self, ai, hu, heights, depth, alpha, beta, played):
        """Opponent's turn: assume they choose the move with the LOWEST value."""
        self.nodes += 1
        if played == ROWS * COLS:
            return 0
        if depth == 0:
            return evaluate(ai, hu, self.ai_is_first)
        value = INF
        for c in COLUMN_ORDER:
            if heights[c] >= ROWS:
                continue
            new_hu = hu | (1 << (c * H + heights[c]))
            if is_win(new_hu):
                return -(WIN_SCORE + depth)  # opponent wins (sooner = worse)
            heights[c] += 1
            v = self._max_value(ai, new_hu, heights, depth - 1,
                                alpha, beta, played + 1)
            heights[c] -= 1
            if v < value:
                value = v
            if value < beta:
                beta = value
            if alpha >= beta:
                break                       # alpha cut-off: MAX will avoid this branch
        return value

    @staticmethod
    def _to_bitboards(board, ai_player):
        """Convert the grid (list of lists) into two bitboards + column heights."""
        ai = hu = played = 0
        heights = [0] * COLS
        for c in range(COLS):
            h = 0
            for r in range(ROWS - 1, -1, -1):        # grid bottom -> top
                v = board.grid[r][c]
                if v == EMPTY:
                    break
                bit = 1 << (c * H + h)
                if v == ai_player:
                    ai |= bit
                else:
                    hu |= bit
                h += 1
            heights[c] = h
            played += h
        return ai, hu, heights, played
