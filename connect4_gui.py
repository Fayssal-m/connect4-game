"""Puissance 4 - Human vs Human and Human vs Computer (Alpha-Beta, depth 8).

Run:  python connect4_gui.py
"""
import queue
import threading
import time
import tkinter as tk

from connect4_logic import Board, ROWS, COLS, EMPTY, P1, P2
from connect4_ai import AlphaBetaAI

CELL = 90
PAD = 8
DEPTH = 8
COLORS = {EMPTY: "white", P1: "red", P2: "gold"}
NAMES = {P1: "Rouge", P2: "Jaune"}


class App:
    def __init__(self, root):
        self.root = root
        root.title("Puissance 4")
        self.results = queue.Queue()     # the AI thread puts its answer here
        self.game_id = 0                 # lets us ignore answers of an old game

        # --- options (applied when "Nouvelle partie" is pressed)
        self.mode = tk.StringVar(value="hc")
        self.first = tk.StringVar(value="human")
        menu = tk.Frame(root)
        menu.pack(pady=4)
        tk.Radiobutton(menu, text="Humain vs Humain", variable=self.mode,
                       value="hh").grid(row=0, column=0, padx=8, sticky="w")
        tk.Radiobutton(menu, text="Humain vs Ordinateur", variable=self.mode,
                       value="hc").grid(row=0, column=1, padx=8, sticky="w")
        tk.Radiobutton(menu, text="Je joue Rouge (je commence)", variable=self.first,
                       value="human").grid(row=1, column=0, padx=8, sticky="w")
        tk.Radiobutton(menu, text="Je joue Jaune (l'ordinateur commence)",
                       variable=self.first, value="ai").grid(row=1, column=1, padx=8, sticky="w")

        self.status = tk.Label(root, font=("Arial", 16))
        self.status.pack(pady=4)
        self.canvas = tk.Canvas(root, width=COLS * CELL, height=ROWS * CELL,
                                bg="royalblue", highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)
        self.info = tk.Label(root, font=("Arial", 11), fg="gray30")
        self.info.pack()
        tk.Button(root, text="Nouvelle partie", command=self.reset).pack(pady=6)
        self.reset()

    # ------------------------------------------------------------- game flow
    def reset(self):
        self.game_id += 1
        self.board = Board()
        self.current = P1                      # Red always starts
        self.game_over = False
        self.thinking = False
        self.win_cells = []
        self.active_mode = self.mode.get()
        if self.active_mode == "hc":
            self.human = P1 if self.first.get() == "human" else P2
            self.ai_player = P2 if self.human == P1 else P1
        self.info.config(text="")
        self.draw()
        self.update_status()
        self.maybe_start_ai()

    def on_click(self, event):
        if self.game_over or self.thinking:
            return
        if self.active_mode == "hc" and self.current != self.human:
            return
        col = event.x // CELL
        if col in self.board.valid_columns():
            self.play(col)

    def play(self, col):
        """Play `col` for the current player (human or computer)."""
        self.board.drop(col, self.current)
        cells = self.board.winning_cells(self.current)
        if cells:
            self.win_cells = cells
            self.game_over = True
        elif self.board.is_full():
            self.game_over = True
        else:
            self.current = P2 if self.current == P1 else P1
        self.draw()
        self.update_status(winner=self.current if cells else None)
        self.maybe_start_ai()

    # ------------------------------------------------------------ computer
    def maybe_start_ai(self):
        if (self.game_over or self.active_mode != "hc"
                or self.current != self.ai_player):
            return
        self.thinking = True
        self.update_status()
        gid, board, player = self.game_id, self.board.copy(), self.ai_player

        def work():                            # runs in a background thread
            t0 = time.time()
            ai = AlphaBetaAI(DEPTH)
            col = ai.best_move(board, player)
            self.results.put((gid, col, ai.nodes, time.time() - t0))

        threading.Thread(target=work, daemon=True).start()
        self.root.after(100, self.poll_ai)

    def poll_ai(self):
        """Check (without blocking the window) whether the AI has answered."""
        if not self.thinking:
            return
        try:
            gid, col, nodes, dt = self.results.get_nowait()
        except queue.Empty:
            self.root.after(100, self.poll_ai)
            return
        if gid != self.game_id:                # answer of a game that was restarted
            self.root.after(100, self.poll_ai)
            return
        self.thinking = False
        self.info.config(text=f"Ordinateur : {nodes:,} noeuds explorés en {dt:.2f} s")
        self.play(col)

    # ------------------------------------------------------------- display
    def update_status(self, winner=None):
        if winner:
            text = f"Le joueur {NAMES[winner]} gagne !"
        elif self.game_over:
            text = "Partie nulle !"
        elif self.thinking:
            text = f"L'ordinateur ({NAMES[self.current]}) réfléchit..."
        else:
            text = f"Au tour du joueur {NAMES[self.current]}"
        self.status.config(text=text)

    def draw(self):
        self.canvas.delete("all")
        for r in range(ROWS):
            for c in range(COLS):
                x0, y0 = c * CELL + PAD, r * CELL + PAD
                x1, y1 = (c + 1) * CELL - PAD, (r + 1) * CELL - PAD
                win = (r, c) in self.win_cells
                self.canvas.create_oval(
                    x0, y0, x1, y1, fill=COLORS[self.board.grid[r][c]],
                    outline="black" if win else "navy", width=5 if win else 2)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
