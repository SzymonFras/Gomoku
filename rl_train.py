import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from architecture import AlphaZeroGomokuNet, AlphaZeroLoss
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import load_cnn_model
import shutil
import glob
from gomoku_engine import GomokuEngine
from simulation import generate_bootstrapping_data
import random
import re
from config import evaluate_candidate_params, generate_bootstrapping_data_params, run_rl_training_params
from mcts import MCTS
import numpy as np
import time
from stats_logger import log_iteration_stats
from players_logic import select_weighted_move

class RLDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path, weights_only=False)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, pi, reward = self.data[idx]
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(pi, dtype=torch.float32),
            torch.tensor(reward, dtype=torch.float32)
        )
def train_rl_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0

    for batch_X, batch_y, batch_w in loader:
        batch_X, batch_y, batch_w = batch_X.to(device), batch_y.to(device), batch_w.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            policy_logits, value_preds = model(batch_X)
            loss = criterion(policy_logits, value_preds, batch_y, batch_w)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * batch_X.size(0)
    return total_loss / len(loader.dataset)


def run_rl_training(model_path, data_path, save_path, epochs=5, batch_size=256, lr=0.00001, weight_decay=0.0001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using: {device}")
    model = load_cnn_model(model_path, AlphaZeroGomokuNet())
    model.to(device)
    dataset = RLDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = AlphaZeroLoss()

    scaler = torch.amp.GradScaler('cuda')
    epoch_losses = []
    for epoch in range(epochs):
        loss = train_rl_epoch(model, loader, optimizer, criterion, device, scaler)
        epoch_losses.append(loss)
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Training finished. New model saved as: {save_path}")
    return epoch_losses


def get_sliding_window_data(window_size=2,target_moves=20000, output_path="Data_rfl/aggregated_buffer.pt",only_winners=False,
                            warmup_path="Data_rfl/warmup_buffer.pt", warmup_ratio=0.0):
    files = sorted(glob.glob('Data_rfl/replay_buffer_*.pt'))

    if not files:
        return output_path, 0,0

    all_data = []
    files_used = 0

    for f in reversed(files):
        file_data = torch.load(f, weights_only=False)
        filtered_data = [item for item in file_data if item[2] > 0] if only_winners else file_data
        all_data.extend(filtered_data)
        files_used += 1
        if files_used >= window_size and len(all_data) >= target_moves:
            break
    mcts_moves = len(all_data)
    if 0.0 < warmup_ratio <= 1.0:
        warmup_data = torch.load(warmup_path, weights_only=False)
        if warmup_ratio == 1.0:
            all_data.extend(warmup_data)
        else:
            n_warmup = int(mcts_moves * warmup_ratio / (1.0 - warmup_ratio))
            all_data.extend(random.sample(warmup_data, min(n_warmup, len(warmup_data))))
    torch.save(all_data, output_path)
    return output_path, len(all_data), files_used


def play_single_eval_game(candidate_model, opponent_model, opponent_type,
                          candidate_plays_x, num_simulations, c_puct, opening_moves_k):

    engine = GomokuEngine()

    center_coords = [6, 7, 8]
    central_moves = [(r, c) for r in center_coords for c in center_coords]
    for r, c in random.sample(central_moves, k=int(opening_moves_k)):
        engine.make_move(r, c)

    mcts_candidate = MCTS(candidate_model, num_simulations=num_simulations, c_puct=c_puct)
    mcts_opponent = None
    if opponent_type == 'mcts':
        mcts_opponent = MCTS(opponent_model, num_simulations=num_simulations, c_puct=c_puct)

    candidate_symbol = 'X' if candidate_plays_x else 'O'

    while not engine.is_game_over:
        current_p = engine.current_player
        action = None

        if current_p == candidate_symbol:
            pi = mcts_candidate.get_action_probs(engine, temperature=0.0)
            action = int(np.argmax(pi))
            row, col = action // engine.board_size, action % engine.board_size
        else:
            if opponent_type == 'mcts':
                pi = mcts_opponent.get_action_probs(engine, temperature=0.0)
                action = int(np.argmax(pi))
                row, col = action // engine.board_size, action % engine.board_size
            elif opponent_type == 'heuristic':
                row, col = select_weighted_move(engine)
                action = row * engine.board_size + col
            elif opponent_type == 'raw_net':
                device = next(opponent_model.parameters()).device
                board_np = np.array(engine.board)
                ch_my = (board_np == current_p).astype(np.float32)
                ch_opp = (board_np == ('O' if current_p == 'X' else 'X')).astype(np.float32)
                ch_empty = (board_np == '.').astype(np.float32)
                state_t = torch.tensor(np.stack([ch_my, ch_opp, ch_empty]), dtype=torch.float32,
                                       device=device).unsqueeze(0)

                with torch.inference_mode():
                    logits, _ = opponent_model(state_t)
                    probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

                legal_mask = (board_np.flatten() == '.')
                probs[~legal_mask] = -1.0
                action = int(np.argmax(probs))
                row, col = action // engine.board_size, action % engine.board_size

        engine.make_move(row, col)
        mcts_candidate.update_with_move(action)
        if opponent_type == 'mcts':
            mcts_opponent.update_with_move(action)
    if engine.winner == 'Draw':
        return 0.5
    elif engine.winner == candidate_symbol:
        return 1.0
    else:
        return 0.0


def run_parallel_match(candidate_model, opponent_model, opponent_type, num_games,
                       num_simulations, c_puct, opening_moves_k, max_threads=8):

    if num_games <= 0:
        return 100.0

    score = 0.0
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = []
        for i in range(num_games):
            candidate_plays_x = (i % 2 == 0)
            futures.append(
                executor.submit(
                    play_single_eval_game,
                    candidate_model, opponent_model, opponent_type,
                    candidate_plays_x, num_simulations, c_puct, opening_moves_k
                )
            )

        for future in as_completed(futures):
            score += future.result()

    return (score / num_games) * 100.0


def evaluate_candidate(candidate_model_path, current_master_model_path, baseline_model_path,
                       num_games_old=40, num_games_heur=40, num_games_base=20,
                       threshold_old=52.0, threshold_base=0.0, threshold_heur=0.0,
                       num_simulations=300, c_puct=1.0, opening_moves_k=2,
                       max_threads=8):
    print("\n--- Evaluating candidate (Parallel) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.inference_mode():
        candidate_model = AlphaZeroGomokuNet().to(device)
        candidate_model.load_state_dict(torch.load(candidate_model_path, map_location=device, weights_only=True))
        candidate_model.eval()

        current_master_model = AlphaZeroGomokuNet().to(device)
        current_master_model.load_state_dict(torch.load(current_master_model_path, map_location=device, weights_only=True))
        current_master_model.eval()

        win_ratio_old = run_parallel_match(
            candidate_model=candidate_model,
            opponent_model=current_master_model,
            opponent_type='mcts',
            num_games=num_games_old,
            num_simulations=num_simulations,
            c_puct=c_puct,
            opening_moves_k=opening_moves_k,
            max_threads=max_threads
        )
        print(f"Result [Candidate vs Current Master]: {win_ratio_old:.1f}% win ratio")

        if win_ratio_old < threshold_old:
            print(">>> Candidate Failed to pass evaluation (Lost to Master)")
            return {"passed": False, "winrate_master": win_ratio_old, "winrate_base": 0.0, "winrate_heur": 0.0}

        base_model = AlphaZeroGomokuNet().to(device)
        base_model.load_state_dict(torch.load(baseline_model_path, map_location=device, weights_only=True))
        base_model.eval()

        win_ratio_base = run_parallel_match(
            candidate_model=candidate_model,
            opponent_model=base_model,
            opponent_type='mcts',
            num_games=num_games_base,
            num_simulations=num_simulations,
            c_puct=c_puct,
            opening_moves_k=opening_moves_k,
            max_threads=max_threads
        )
        print(f"Result [Candidate vs Base CNN model]: {win_ratio_base:.1f}% win ratio")

        if win_ratio_base < threshold_base:
            print(">>> Candidate Failed to pass evaluation (Lost to Base model)")
            return {"passed": False, "winrate_master": win_ratio_old, "winrate_base": win_ratio_base, "winrate_heur": 0.0}

        win_ratio_heur = run_parallel_match(
            candidate_model=candidate_model,
            opponent_model=None,
            opponent_type='heuristic',
            num_games=num_games_heur,
            num_simulations=num_simulations,
            c_puct=c_puct,
            opening_moves_k=opening_moves_k,
            max_threads=max_threads
        )
        print(f"Result [Candidate vs Heuristic]:   {win_ratio_heur:.1f}% win ratio")

    passed = (win_ratio_old >= threshold_old and win_ratio_base >= threshold_base and win_ratio_heur >= threshold_heur)

    if passed:
        print(">>> Candidate PASSED evaluation and becomes the new Master!")
    else:
        print(">>> Candidate Failed to pass evaluation")

    return {"passed": passed, "winrate_master": win_ratio_old, "winrate_base": win_ratio_base, "winrate_heur": win_ratio_heur}


def main_rl_loop(iterations=5, num_games=3000, window_size=3, target_moves = 100000,only_winners=False,skip_first_gen=False,
                 warmup_data_path="Data_rfl/warmup_buffer.pt", warmup_ratios=None, consecutive_failures=0):
    current_version, current_master = get_latest_master()
    candidate_model = 'models/model_candidate.pt'
    if warmup_ratios is None:
        warmup_ratios = [0.0]
    print(f"\n[*] Starting reincforcement learning loop")
    print(f"[*] Current Master: {current_master} (Version: {current_version})")
    for i in range(iterations):
        iter_start_time = time.time()
        master_before = current_master
        print(f"\n=============================================")
        print(f"           Iteration of RL {i + 1}/{iterations}            ")
        print(f"=============================================")
        dynamic_window = window_size + consecutive_failures

        if consecutive_failures > 0:
            print(f">>> [!] Consecutive failures: {consecutive_failures}. Expanding buffer!")
        model_to_play = load_cnn_model(current_master, AlphaZeroGomokuNet())
        ratio = warmup_ratios[i] if i < len(warmup_ratios) else 0.0
        if i == 0 and skip_first_gen:
            print(f">>> Using wamup data: {warmup_data_path}")
            training_data_path = warmup_data_path
            current_epochs = 20
            current_lr = 0.0001
            num_winning_moves = len(torch.load(warmup_data_path, weights_only=False))
            files_used = 1
        else:
            current_epochs = run_rl_training_params().get('epochs', 5)
            current_lr = run_rl_training_params().get('lr', 0.00002)
            generate_bootstrapping_data(model_to_play,  **(generate_bootstrapping_data_params()| {'num_games': num_games}))
            while True:
                training_data_path, num_winning_moves, files_used = get_sliding_window_data(window_size=dynamic_window,
                                                                                            only_winners=only_winners,
                                                                                            warmup_ratio=ratio,
                                                                                            target_moves=target_moves,
                                                                                            warmup_path=warmup_data_path)
                if num_winning_moves >= target_moves and files_used >= dynamic_window:
                    print(f">>> {num_winning_moves} moves found in {dynamic_window} files. Starting the training")
                    break
                else:
                    print(f">>> Not enough moves in the window {num_winning_moves}/{target_moves}")
                    print(f">>> Generating {num_games} new games to gather more moves")
                    generate_bootstrapping_data(model_to_play, **(generate_bootstrapping_data_params() | {'num_games': num_games}))
        epoch_losses = run_rl_training(
            model_path=current_master,
            data_path=training_data_path,
            save_path=candidate_model,
            **(run_rl_training_params() | {'epochs': current_epochs, 'lr': current_lr})
        )
        eval_results = evaluate_candidate(candidate_model, current_master, **evaluate_candidate_params())

        if eval_results["passed"]:
            current_version += 1
            consecutive_failures = 0
            new_master_path = f'models/model_{current_version}_rl.pt'

            print(f">>> Succes! Candidate passed evaluation")
            print(f">>> Saving new master as: {new_master_path}")
            shutil.copy(candidate_model, new_master_path)
            current_master = new_master_path
        else:
            consecutive_failures += 1
            print(">>> Candidate Failed to pass evaluation")
        log_iteration_stats(
            iteration=i + 1,
            master_before=master_before,
            master_after=current_master,
            epoch_losses=epoch_losses,
            eval_results=eval_results,
            buffer_samples=num_winning_moves,
            files_used=files_used,
            warmup_ratio=ratio,
            epochs=current_epochs,
            lr=current_lr,
            iter_start_time=iter_start_time
        )

    print("\n[Iteration finished] - check file", current_master)


def get_latest_master(models_dir="models"):
    files = glob.glob(f"{models_dir}/model_*.pt")
    highest_version = 0
    latest_model_path = f"{models_dir}/model_20_rl.pt"

    for f in files:
        if "candidate" in f:
            continue
        match = re.search(r'model_(\d+)', f)
        if match:
            version = int(match.group(1))
            if version > highest_version:
                highest_version = version
                latest_model_path = f

    return highest_version, latest_model_path