from architecture import AlphaZeroGomokuNet
import torch
import numpy as np
import random
from gomoku_engine import GomokuEngine
from players import heuristic_player
import os


def generate_warmup_data(num_games=1000, output_path="Data_rfl/warmup_buffer.pt", opening_moves_k=2):
    print(f"Generating {num_games} warmup games")
    os.makedirs('Data_rfl', exist_ok=True)
    all_data = []

    center_coords = [6, 7, 8]
    central_moves = [(r, c) for r in center_coords for c in center_coords]

    for i in range(num_games):
        engine = GomokuEngine()
        game_history = []
        opening_moves = random.sample(central_moves, k=opening_moves_k)
        for r, c in opening_moves:
            engine.make_move(r, c)

        while not engine.is_game_over:
            current_p = engine.current_player
            row, col = heuristic_player(engine)

            pi = np.zeros(engine.board_size * engine.board_size, dtype=np.float32)
            action_idx = row * engine.board_size + col
            pi[action_idx] = 1.0

            board_np = np.array(engine.board)
            channel_my = (board_np == current_p).astype(np.float32)
            channel_opp = (board_np == ('O' if current_p == 'X' else 'X')).astype(np.float32)
            channel_empty = (board_np == '.').astype(np.float32)
            state_3c = np.stack([channel_my, channel_opp, channel_empty], axis=0)

            game_history.append({'player': current_p, 'state': state_3c, 'pi': pi})
            engine.make_move(row, col)

        winner = engine.winner
        for step in game_history:
            if winner == 'Draw':
                reward = 0.0
            elif step['player'] == winner:
                reward = 1.0
            else:
                reward = -1.0
            all_data.append((step['state'], step['pi'], reward))
        if (i + 1)%100==0:
            print(f"Game {i + 1}/{num_games} finished. Result: {winner}. Moves: {engine.move_number}")

    torch.save(all_data, output_path)
    print(f"Saving {len(all_data)} moves to {output_path}")

def init_fresh_alphazero_model(models_dir="models", filename="model_20_rl.pt"):
    os.makedirs(models_dir, exist_ok=True)
    model = AlphaZeroGomokuNet()
    save_path = os.path.join(models_dir, filename)
    torch.save(model.state_dict(), save_path)

    print(f"Sucessfully saved model to {save_path}")


if __name__ == "__main__":
    #init_fresh_alphazero_model()
    generate_warmup_data(num_games=5000, output_path="Data_rfl/warmup_buffer.pt")