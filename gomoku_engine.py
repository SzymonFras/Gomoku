from typing import List, Tuple, Optional
import uuid
class GomokuEngine:
    def __init__(self, board_size: int = 15):
        self.board_size = board_size
        self.game_id = str(uuid.uuid4())
        self.board: List[List[str]] = [['.' for _ in range(board_size)] for _ in range(board_size)]
        self.current_player = 'X'
        self.move_number = 0
        self.winner: Optional[str] = None
        self.is_game_over = False
        self.history: List[Tuple[int, str, int, int]] = []
        self.opening_state: str = ""

    def reset_game(self) -> None:
        self.board = [['.' for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.game_id = str(uuid.uuid4())
        self.current_player = 'X'
        self.move_number = 0
        self.winner = None
        self.is_game_over = False
        self.history = []
        self.opening_state = ""

    def get_legal_moves(self) -> List[Tuple[int, int]]:
        if self.move_number == 0:
            center = self.board_size // 2
            return [(center, center)]
        legal_moves = []
        radius = 2
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r][c] == '.':
                    has_neighbor = False
                    r_min = max(0, r - radius)
                    r_max = min(self.board_size - 1, r + radius)
                    c_min = max(0, c - radius)
                    c_max = min(self.board_size - 1, c + radius)

                    for nr in range(r_min, r_max + 1):
                        for nc in range(c_min, c_max + 1):
                            if self.board[nr][nc] != '.':
                                has_neighbor = True
                                break
                        if has_neighbor:
                            break

                    if has_neighbor:
                        legal_moves.append((r, c))
        if not legal_moves:
            for r in range(self.board_size):
                for c in range(self.board_size):
                    if self.board[r][c] == '.':
                        legal_moves.append((r, c))
        return legal_moves

    def make_move(self, row: int, col: int) -> bool:
        if self.is_game_over:
            return False
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            print("Move outside the board")
            return False

        if self.board[row][col] != '.':
            print("Field is taken")
            return False
        self.board[row][col] = self.current_player
        self.move_number += 1
        self.history.append((self.move_number, self.current_player, row, col))
        if self._check_winner(row, col,5):
            self.winner = self.current_player
            self.is_game_over = True
        elif self.move_number == self.board_size * self.board_size:
            self.winner = 'Draw'
            self.is_game_over = True
        else:
            self.current_player = 'O' if self.current_player == 'X' else 'X'
        return True

    def _check_winner(self, last_row: int, last_col: int, length: int = 5) -> bool:
        player = self.board[last_row][last_col]
        directions = [
            (0, 1),
            (1, 0),
            (1, 1),
            (1, -1)
        ]

        for dr, dc in directions:
            count = 1
            r, c = last_row + dr, last_col + dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r][c] == player:
                count += 1
                r += dr
                c += dc
            r, c = last_row - dr, last_col - dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r][c] == player:
                count += 1
                r -= dr
                c -= dc

            if count >= length:
                return True
        return False
    def play_game(self, player_x_fn, player_o_fn,opening_moves: List[Tuple[int, int]] = None) -> None:
        self.reset_game()
        if not opening_moves:
            opening_moves = []
        if len(opening_moves) == 0:
            self.opening_state = ""
        else:
            self.opening_state = ";".join(f"{r},{c}" for r, c in opening_moves)
        for row, col in opening_moves:
            self.make_move(row, col)
        while not self.is_game_over:
            if self.current_player == 'X':
                row, col = player_x_fn(self)
            else:
                row, col = player_o_fn(self)

            self.make_move(row, col)


    def get_data(self) -> List[dict]:
        rows = []
        for move_num, player, r, c in self.history:
            rows.append({
                "game_id": self.game_id,
                "move_number": move_num,
                "player": player,
                "move_x": r,
                "move_y": c,
                "winner": self.winner,
                "opening_state": self.opening_state
            })
        return rows

    def render(self) -> None:
        print("   " + " ".join(f"{i:2}" for i in range(self.board_size)))
        for idx, row in enumerate(self.board):
            print(f"{idx:2} " + "  ".join(row))
        print(f"Status: Game Over={self.is_game_over} | Winner={self.winner}\n")
