from gomoku_engine import GomokuEngine
from players_logic import make_cnn_prediction, select_weighted_move
from utils import load_cnn_model

def human_player(engine):
    engine.render()
    while True:
        try:
            move_str = input(f"Human move as {engine.current_player} (row, col for example. '7 8'): ")
            r_str, c_str = move_str.strip().split()
            row, col = int(r_str), int(c_str)
            if (row, col) in engine.get_legal_moves():
                return row, col
            else:
                print("not available")
        except ValueError:
            print("incorrect format")
def heuristic_player(engine: GomokuEngine):
    return select_weighted_move(engine)

def cnn_player(engine,path,structure):
    return make_cnn_prediction(
        model=load_cnn_model(path, structure),
        board_matrix=engine.board,
        legal_moves=engine.get_legal_moves(),
        current_player=engine.current_player
    )
