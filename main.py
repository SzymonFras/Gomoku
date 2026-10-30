from gui import play_with_gui
from rl_train import main_rl_loop
from config import main_rl_loop_params, gui_params
import torch
torch.backends.cudnn.benchmark = True

def main():
    main_rl_loop(**main_rl_loop_params())
    #play_with_gui(**gui_params())


if __name__ == "__main__":
    main()