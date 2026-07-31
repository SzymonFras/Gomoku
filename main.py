from gomuku_engine import GomokuEngine
from datetime import datetime
import pandas as pd
from LGB_player import model_v3_player, heuristic_player, model_v1_player, model_v2_player, model_v4_player, v2_model_v1_player
import random
def main():
    #players = [model_v3_player, model_v1_player]
    for x in range(10):
        bq_data = []
        players = [heuristic_player, model_v1_player, model_v2_player, model_v3_player, model_v4_player]
        player = [v2_model_v1_player]
        for p1 in players:
            for p2 in player:
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
                match_openings = [random.sample(central_moves, k=4) for _ in range(50)]
                for i in range(50):
                    game = GomokuEngine()
                    game.play_game(p1,p2,match_openings[i])
                    bq_data.append(game.get_bigquery_data())
                    #game.render()
                    if game.winner == 'X':
                        p1_wins_as_X += 1
                    elif game.winner == 'O':
                        p2_wins_as_O += 1
                    else:
                        draws += 1
                for i in range(50):
                    game = GomokuEngine()
                    game.play_game(p2,p1,match_openings[i])
                    bq_data.append(game.get_bigquery_data())
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
                print(f"  As X : {p1_wins_as_X} wins / 50")
                print(f"  As O : {p1_wins_as_O} wins / 50")
                print(f"Player {p2.__name__}:")
                print(f"  As X : {p2_wins_as_X} wins / 50")
                print(f"  As O : {p2_wins_as_O} wins / 50")
                print(f"Draws: {draws}")
                print(f"==========================================")
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        pd.DataFrame(data=bq_data).to_csv(f'bq_data{current_time}.csv')
        print(f"rows insterted: {len(bq_data)}")
if __name__ == "__main__":
    main()