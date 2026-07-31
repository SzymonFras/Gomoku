import lightgbm as lgb
import numpy as np
from typing import List, Tuple
import random
from gomuku_engine import GomokuEngine
from pathlib import Path

def read_model(model_txt_path: str):
    CURRENT_DIR = Path(__file__).resolve().parent
    MODEL_PATH = CURRENT_DIR / 'models' / model_txt_path
    bst = lgb.Booster(model_file=MODEL_PATH )
    return bst
bst1 = read_model('gomoku_lgb_champion_v1.txt')
bst2 = read_model('gomoku_lgb_champion_v2.txt')
bst4 = read_model('gomoku_lgb_champion.txt')
bst3 = read_model('gomoku_lgb_champion_v3.txt')
bst_v2_1 = read_model('v2_gomoku_lgb_champion_v2.txt')
def heuristic_player(engine: GomokuEngine):
    return engine.select_weighted_move()
def v2_model_v1_player(engine: GomokuEngine) -> Tuple[int, int]:
    return make_model_prediction(
        bst=bst_v2_1,
        board_matrix=engine.board,
        legal_moves=engine.get_legal_moves(),
        current_player=engine.current_player
    )
def model_v3_player(engine: GomokuEngine) -> Tuple[int, int]:
    return make_model_prediction(
        bst=bst3,
        board_matrix=engine.board,
        legal_moves=engine.get_legal_moves(),
        current_player=engine.current_player
    )
def model_v1_player(engine: GomokuEngine) -> Tuple[int, int]:
    return make_model_prediction(
        bst=bst1,
        board_matrix=engine.board,
        legal_moves=engine.get_legal_moves(),
        current_player=engine.current_player
    )
def model_v4_player(engine: GomokuEngine) -> Tuple[int, int]:
    return make_model_prediction(
        bst=bst4,
        board_matrix=engine.board,
        legal_moves=engine.get_legal_moves(),
        current_player=engine.current_player
    )
def model_v2_player(engine: GomokuEngine) -> Tuple[int, int]:
    return make_model_prediction(
        bst=bst2,
        board_matrix=engine.board,
        legal_moves=engine.get_legal_moves(),
        current_player=engine.current_player
    )

def make_model_prediction(bst, board_matrix: List[List[str]], legal_moves: List[Tuple[int, int]],
                          current_player: str) -> Tuple[int, int]:
    flat_board = []
    for r in range(len(board_matrix)):
        for c in range(len(board_matrix[r])):
            cell = board_matrix[r][c]
            if cell == 'X':
                flat_board.append(1)
            elif cell == 'O':
                flat_board.append(-1)
            else:
                flat_board.append(0)

    state_1d = np.array(flat_board).reshape(1, -1)
    probabilities = bst.predict(state_1d)[0]

    best_move = None
    best_prob = -1
    for row, col in legal_moves:
        move_idx = row * len(board_matrix) + col
        if probabilities[move_idx] > best_prob:
            best_prob = probabilities[move_idx]
            best_move = (row, col)
    if best_move is None:
        best_move = random.choice(legal_moves)
    return best_move