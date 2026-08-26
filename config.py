from utils import load_config
from architecture import *
AVAILABLE_STRUCTURES = {
        'model_1_structure': model_1_structure,
        'model_2_structure': model_2_structure,
        'model_3_structure': model_3_structure,
        'model_4_structure': model_4_structure,
    }
def main_rl_loop_params():
    config = load_config()
    return config['main_rl_loop']
def evaluate_candidate_params():
    config = load_config()
    return config['evaluate_candidate']
def generate_bootstrapping_data_params():
    config = load_config()
    return config['generator']
def run_rl_training_params():
    config = load_config()
    return config['CNN']
def game_simul_params():
    config = load_config()
    return config['game_simul']
def load_cnn_model_params(AVAILABLE_STRUCTURES=AVAILABLE_STRUCTURES):
    config = load_config()
    path = config['load_cnn_model']['path']
    structure_name = config['load_cnn_model']['structure']
    real_structure = AVAILABLE_STRUCTURES[structure_name]
    return {'structure': real_structure, 'path': path}

