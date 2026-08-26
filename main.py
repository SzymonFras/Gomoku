from gui import play_with_gui, bot
from simulation import game_simul, generate_bootstrapping_data
from utils import watch_generated_games, load_cnn_model
from rl_train import main_rl_loop
from config import generate_bootstrapping_data_params, main_rl_loop_params, load_cnn_model_params, game_simul_params

def main():
    #generate_bootstrapping_data(model = load_cnn_model(**load_cnn_model_params()), **generate_bootstrapping_data_params())
    #main_rl_loop(**main_rl_loop_params())
    #watch_generated_games(load_cnn_model(**load_cnn_model_params()))
    #game_simul(**game_simul_params())
    #play_with_gui(human_starts=True, bot_fn=bot)

if __name__ == "__main__":
    main()