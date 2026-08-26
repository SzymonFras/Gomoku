from gomuku_engine import GomokuEngine
from datetime import datetime
import pandas as pd
from players import heuristic_player
from players_logic import make_cnn_prediction
from architecture import model_4_structure
from utils import load_cnn_model
import random
import torch
import os

def generate_bootstrapping_data(model, num_games=1000, temp_moves=12, self_play_ratio=0.75, discount_factor=0.9, top_k=3, temp=1.0, end_temp=0.1, end_top_k=1):
    all_training_data = []
    num_self_play = int(num_games * self_play_ratio)
    print(f"Starting to generate games. {num_games} games to play...")
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Data_rfl\\replay_buffer_{current_time}.pt"
    center_coords = [6, 7, 8]
    central_moves = [(r, c) for r in center_coords for c in center_coords]

    for i in range(num_games):
        engine = GomokuEngine()
        game_history = []
        opening_moves = random.sample(central_moves, k=4)
        for r, c in opening_moves:
            engine.make_move(r, c)
        is_self_play = (i < num_self_play)
        model_color = 'O' if not is_self_play else None
        while not engine.is_game_over:
            current_p = engine.current_player
            moves_played = sum(row.count('X') + row.count('O') for row in engine.board)
            current_temp = temp if moves_played < temp_moves else end_temp
            current_top_k = top_k if moves_played < temp_moves else end_top_k
            if is_self_play or current_p == model_color:
                (row, col), state_3c = make_cnn_prediction(
                    model=model,
                    board_matrix=engine.board,
                    legal_moves=engine.get_legal_moves(),
                    current_player=current_p,
                    temperature=current_temp,
                    top_k=current_top_k,
                    return_state=True
                )

                game_history.append({
                    'player': current_p,
                    'state': state_3c,
                    'move': row * 15 + col
                })
            else:
                row, col = heuristic_player(engine)
                game_history.append({
                    'player': current_p,
                    'is_model': False
                })

            engine.make_move(row, col)


        winner = engine.winner
        total_moves = len(game_history)
        for idx, step in enumerate(game_history):
            if not step.get('is_model', True):
                continue
            if winner is None:
                base_reward = 0.0
            elif step['player'] == winner:
                base_reward = 1.0
            else:
                base_reward = -1.0
            moves_to_end = total_moves - idx - 1
            discount_factor = discount_factor ** moves_to_end
            final_reward = base_reward * discount_factor

            all_training_data.append((step['state'], step['move'], final_reward))
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{num_games} Games simulated. {len(all_training_data)} moves gathered.")

    torch.save(all_training_data, filename)
    print(f"Done. Saving {len(all_training_data)} moves to file {filename}")

def game_simul(players_paths, players2_paths,heur, games = 50, n_save=1):
    def create_bot(path):
        model = load_cnn_model(path, model_4_structure)
        def bot_fn(engine):
            return make_cnn_prediction(model, engine.board, engine.get_legal_moves(), engine.current_player,
                                       temperature=0.0)
        bot_fn.__name__ = os.path.basename(path)
        return bot_fn
    players = [create_bot(path) for path in players_paths]
    players2 = [create_bot(path) for path in players2_paths]
    if heur:
        players.append(heuristic_player)
        players2.append(heuristic_player)
    for x in range(n_save):
        data = []
        match_summary_data = []
        for p1 in players:
            for p2 in players2:
                if p1 == p2:
                    continue
                print(f"Match between {p1.__name__} and {p2.__name__}")
                p1_wins_as_X = 0
                p2_wins_as_O = 0
                p2_wins_as_X = 0
                p1_wins_as_O = 0
                draws = 0
                center_coords = [6, 7, 8]
                central_moves = [(r, c) for r in center_coords for c in center_coords]
                match_openings = [random.sample(central_moves, k=4) for _ in range(games)]
                for i in range(games):
                    game = GomokuEngine()
                    game.play_game(p1,p2,match_openings[i])
                    data.append(game.get_data())
                    #game.render()
                    if game.winner == 'X':
                        p1_wins_as_X += 1
                    elif game.winner == 'O':
                        p2_wins_as_O += 1
                    else:
                        draws += 1
                for i in range(games):
                    game = GomokuEngine()
                    game.play_game(p2,p1,match_openings[i])
                    data.append(game.get_data())
                    #game.render()
                    if game.winner == 'X':
                        p2_wins_as_X += 1
                    elif game.winner == 'O':
                        p1_wins_as_O += 1
                    else:
                        draws += 1
                print(f"\n==========================================")
                print(f"Duel finished!")
                print(f"Match between {p1.__name__} vs {p2.__name__}")
                print(f"------------------------------------------")
                print(f"Player {p1.__name__}:")
                print(f"  As X : {p1_wins_as_X} / {games}")
                print(f"  As O : {p1_wins_as_O} / {games}")
                print(f"Player {p2.__name__}:")
                print(f"  As X : {p2_wins_as_X} / {games}")
                print(f"  As O : {p2_wins_as_O} / {games}")
                print(f"Total Draws: {draws} / {games * 2}")
                print(f"==========================================")
                match_summary_data.append({
                    "p1": p1.__name__, "p2": p2.__name__,
                    "p1_wins_X": p1_wins_as_X, "p1_wins_O": p1_wins_as_O,
                    "p2_wins_X": p2_wins_as_X, "p2_wins_O": p2_wins_as_O,
                    "draws": draws, "total_games": games * 2
                })
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        pd.DataFrame(data=data).to_csv(f'game_simul_data/bq_data{current_time}.csv')
        pd.DataFrame(data=match_summary_data).to_csv(f'game_simul_data/match_results_{current_time}.csv', index=False)

