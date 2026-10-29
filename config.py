from utils import load_config
from architecture import  AlphaZeroGomokuNet
AVAILABLE_STRUCTURES = {
        'alphazero_structure': AlphaZeroGomokuNet()
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

def gui_params(AVAILABLE_STRUCTURES=AVAILABLE_STRUCTURES):
    config = load_config()['gui']
    structure_name = config['structure']
    return {
        'path': config['path'],
        'structure': AVAILABLE_STRUCTURES[structure_name],
        'num_simulations': config['num_simulations'],
        'c_puct': config['c_puct'],
        'human_starts': config['human_starts']
    }

