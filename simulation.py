from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import os
import random
import numpy as np
import torch
from gomoku_engine import GomokuEngine
from mcts import MCTS
from players_logic import select_weighted_move


def play_single_game(model, num_simulations, c_puct, temp_moves, temp, opening_moves_k, gamma, player_x_type, player_o_type):
    engine = GomokuEngine()
    mcts = MCTS(model, num_simulations=num_simulations, c_puct=c_puct)
    if opening_moves_k>0:
        center_coords = [6, 7, 8]
        central_moves = [(r, c) for r in center_coords for c in center_coords]
        opening_moves = random.sample(central_moves, k=int(opening_moves_k))

        for r, c in opening_moves:
            engine.make_move(r, c)

    game_history = []

    while not engine.is_game_over:
        current_p = engine.current_player
        moves_played = engine.move_number
        current_type = player_x_type if current_p == 'X' else player_o_type

        if current_type == 'M':
            pi = mcts.get_action_probs(engine, temperature=1.0, add_noise=True)
            current_temp = temp if moves_played < temp_moves else 0.0

            if current_temp == 0.0:
                action = int(np.argmax(pi))
            else:
                scaled_pi = pi ** (1.0 / current_temp)
                scaled_pi /= np.sum(scaled_pi)
                action = np.random.choice(len(pi), p=scaled_pi)

            row, col = action // engine.board_size, action % engine.board_size

            board_np = np.array(engine.board)
            channel_my = (board_np == current_p).astype(np.float32)
            channel_opp = (board_np == ('O' if current_p == 'X' else 'X')).astype(np.float32)
            channel_empty = (board_np == '.').astype(np.float32)
            state_3c = np.stack([channel_my, channel_opp, channel_empty], axis=0)

            game_history.append({
                'player': current_p,
                'state': state_3c,
                'pi': pi,
                'move_no': moves_played
            })
        else:
            row, col = select_weighted_move(engine)
            action = row * engine.board_size + col
        engine.make_move(row, col)
        mcts.update_with_move(action)

    winner = engine.winner
    total_moves = engine.move_number
    gamma = gamma
    game_data = []

    for step in game_history:
        if winner == 'Draw':
            base_reward = 0.0
        elif step['player'] == winner:
            base_reward = 1.0
        else:
            base_reward = -1.0

        steps_from_end = max(0, (total_moves - 1) - step['move_no'])
        reward = float(base_reward * (gamma ** steps_from_end))

        state = step['state']
        pi_2d = step['pi'].reshape(engine.board_size, engine.board_size)

        for k in range(4):
            rot_state = np.rot90(state, k=k, axes=(1, 2))
            rot_pi = np.rot90(pi_2d, k=k)
            game_data.append((rot_state.copy(), rot_pi.flatten().copy(), reward))

            flip_state = np.flip(rot_state, axis=2)
            flip_pi = np.fliplr(rot_pi)
            game_data.append((flip_state.copy(), flip_pi.flatten().copy(), reward))

    return game_data, winner, engine.move_number


def generate_bootstrapping_data(model, num_games=10, num_simulations=25, c_puct=1.5,
                                temp_moves=12, temp=0.5, opening_moves_k=2,
                                heuristic_ratio=0.2, max_threads=10, gamma=0.97):
    print(f"Starting to generate {num_games} games in parallel (Threads: {max_threads})...")
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('Data_rfl', exist_ok=True)
    filename = f"Data_rfl/replay_buffer_{current_time}.pt"

    model.eval()

    num_mixed = int(num_games * heuristic_ratio)
    num_heur_x = num_mixed // 2
    num_heur_o = num_mixed - num_heur_x
    num_mcts = num_games - num_mixed

    matchups = [('M', 'M')] * num_mcts + [('H', 'M')] * num_heur_x + [('M', 'H')] * num_heur_o
    random.shuffle(matchups)

    all_training_data = []

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        future_to_matchup = {}
        for px, po in matchups:
            future = executor.submit(
                play_single_game,
                model, num_simulations, c_puct, temp_moves, temp, opening_moves_k, gamma, px, po
            )
            future_to_matchup[future] = (px, po)

        completed_count = 0
        for future in as_completed(future_to_matchup):
            px, po = future_to_matchup[future]
            game_data, winner, moves = future.result()

            all_training_data.extend(game_data)
            completed_count += 1

            if completed_count % 5 == 0 or completed_count == num_games:
                print(f"Game {completed_count}/{num_games} finished [{px} vs {po}]. Winner: {winner}. Moves: {moves}")

    torch.save(all_training_data, filename)
    print(f"Simulation finished. Saved {len(all_training_data)} samples to file {filename}")