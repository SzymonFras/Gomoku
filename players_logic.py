import numpy as np
from typing import List, Tuple
import random
import torch.nn.functional as F
import torch

def select_weighted_move(engine):
    legal_moves = engine.get_legal_moves()
    if engine.move_number == 0:
        center = engine.board_size // 2
        return (center, center)
    player = engine.current_player
    opponent = 'O' if player == 'X' else 'X'
    for row, col in legal_moves:
        engine.board[row][col] = player
        is_winning = engine._check_winner(row, col ,5)
        engine.board[row][col] = '.'
        if is_winning:
            return row, col
    for row, col in legal_moves:
        engine.board[row][col] = opponent
        if engine._check_winner(row, col, 5):
            engine.board[row][col] = '.'
            return row, col
        engine.board[row][col] = '.'
    for row, col in legal_moves:
        engine.board[row][col] = player
        if engine._check_winner(row, col, 4):
            engine.board[row][col] = '.'
            return row, col
        engine.board[row][col] = '.'
    for row, col in legal_moves:
        engine.board[row][col] = opponent
        if engine._check_winner(row, col, 4):
            engine.board[row][col] = '.'
            return row, col
        engine.board[row][col] = '.'
    move_scores = []
    for row, col in legal_moves:
        score = 1
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r, c = row + dr, col + dc
                if 0 <= r < engine.board_size and 0 <= c < engine.board_size:
                    neighbor = engine.board[r][c]
                    if neighbor == player:
                        score += 1000
                    elif neighbor == opponent:
                        score += 700

        move_scores.append(score)
    chosen_move = random.choices(legal_moves, weights=move_scores, k=1)[0]
    return chosen_move

def make_cnn_prediction(model: torch.nn.Module,
                        board_matrix: List[List[str]],
                        legal_moves: List[Tuple[int, int]],
                        current_player: str,
                        device: str = "cuda",
                        temperature: float = 0.0,
                        top_k: int = 1,
                        return_state: bool = False
                        ) -> Tuple[int, int]:
    board_size = len(board_matrix)  # 15
    opponent_player = 'O' if current_player == 'X' else 'X'

    channel_my = np.zeros((board_size, board_size), dtype=np.float32)
    channel_opp = np.zeros((board_size, board_size), dtype=np.float32)
    channel_empty = np.zeros((board_size, board_size), dtype=np.float32)

    for r in range(board_size):
        for c in range(board_size):
            cell = board_matrix[r][c]
            if cell == current_player:
                channel_my[r, c] = 1.0
            elif cell == opponent_player:
                channel_opp[r, c] = 1.0
            else:
                channel_empty[r, c] = 1.0

    state_3c = np.stack([channel_my, channel_opp, channel_empty], axis=0)
    tensor_input = torch.tensor(state_3c, dtype=torch.float32).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda" if "cuda" in str(device) else "cpu"):
            logits = model(tensor_input).squeeze(0)
        #probabilities = logits.squeeze(0).cpu().numpy()
    if not legal_moves:
        return None
    mask = torch.full((board_size * board_size,), float('-inf'), device=device)

    for r, c in legal_moves:
        mask[r * board_size + c] = 0.0

    masked_logits = logits + mask

    if temperature == 0.0 or top_k <= 1:
        move_idx = torch.argmax(masked_logits).item()
    else:
        actual_top_k = min(top_k, len(legal_moves))
        top_values, top_indices = torch.topk(masked_logits, actual_top_k)
        filtered_logits = torch.full_like(masked_logits, float('-inf'))
        filtered_logits.scatter_(0, top_indices, top_values)
        probs = F.softmax(filtered_logits / temperature, dim=0)
        move_idx = torch.multinomial(probs, 1).item()

    best_move = (move_idx // board_size, move_idx % board_size)

    if return_state:
        return best_move, state_3c

    return best_move
