from datetime import datetime
import os
import random
import numpy as np
import torch
from gomoku_engine import GomokuEngine
from mcts import MCTS


def generate_bootstrapping_data(model, num_games=10, num_simulations=25, c_puct=1.5,
                                temp_moves=12, temp=0.5, opening_moves_k=2):
    all_training_data = []
    print(f"Starting to generate games. {num_games} games to play...")
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs('Data_rfl', exist_ok=True)
    filename = f"Data_rfl/replay_buffer_{current_time}.pt"
    mcts = MCTS(model, num_simulations=num_simulations, c_puct=c_puct)
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
            moves_played = engine.move_number

            current_temp = temp if moves_played < temp_moves else 0.0
            pi = mcts.get_action_probs(engine, temperature=current_temp)

            board_np = np.array(engine.board)
            channel_my = (board_np == current_p).astype(np.float32)
            channel_opp = (board_np == ('O' if current_p == 'X' else 'X')).astype(np.float32)
            channel_empty = (board_np == '.').astype(np.float32)
            state_3c = np.stack([channel_my, channel_opp, channel_empty], axis=0)

            game_history.append({
                'player': current_p,
                'state': state_3c,
                'pi': pi
            })

            if current_temp == 0:
                action = np.argmax(pi)
            else:
                action = np.random.choice(len(pi), p=pi)

            row, col = action // engine.board_size, action % engine.board_size
            engine.make_move(row, col)

        winner = engine.winner
        for step in game_history:
            if winner == 'Draw':
                reward = 0.0
            elif step['player'] == winner:
                reward = 1.0
            else:
                reward = -1.0

            state = step['state']  # shape: (3, 15, 15)
            pi_2d = step['pi'].reshape(engine.board_size, engine.board_size)

            for k in range(4):
                rot_state = np.rot90(state, k=k, axes=(1, 2))
                rot_pi = np.rot90(pi_2d, k=k)
                all_training_data.append((rot_state.copy(), rot_pi.flatten().copy(), reward))

            # Reflection
                flip_state = np.flip(rot_state, axis=2)
                flip_pi = np.fliplr(rot_pi)
                all_training_data.append((flip_state.copy(), flip_pi.flatten().copy(), reward))

        if (i + 1) % 10 == 0:
            print(f"Game {i + 1}/{num_games} finished. Winner: {winner}. Number of moves: {engine.move_number}")

    torch.save(all_training_data, filename)
    print(f"Simulation finished. Saved {len(all_training_data)} games to file {filename}")

