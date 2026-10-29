import yaml
import time
import copy
import random
import torch
from gomoku_engine import GomokuEngine
from players_logic import make_cnn_prediction
def load_config(config_path="config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as file:
        config_dict = yaml.safe_load(file)
    return config_dict

def load_cnn_model(path: str, structure):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = copy.deepcopy(structure).to(device)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def watch_generated_games(model, num_games=2):
    print(f"Loading model {model} for analysis...")
    center_coords = [6, 7, 8]
    central_moves = [(r, c) for r in center_coords for c in center_coords]

    for i in range(num_games):
        print(f"\n======================================")
        print(f"          Show Match {i + 1}            ")
        print(f"======================================")

        engine = GomokuEngine()
        opening_moves = random.sample(central_moves, k=4)
        for r, c in opening_moves:
            engine.make_move(r, c)

        print("State after opening moves:")
        engine.render()
        time.sleep(2)

        while not engine.is_game_over:
            current_p = engine.current_player
            moves_played = engine.move_number
            temp = 1.0 if moves_played < 12 else 0.1
            (row, col) = make_cnn_prediction(
                model=model,
                board_matrix=engine.board,
                legal_moves=engine.get_legal_moves(),
                current_player=current_p,
                temperature=temp
            )
            engine.make_move(row, col)

            print(f"\nMove no {moves_played + 1} | Player {current_p} game: ({row}, {col}) | Temp: {temp}")
            engine.render()
            time.sleep(0.8)

        print(f"\n>>> Game finished! Winner: {engine.winner} <<<")
        time.sleep(3)
