import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from architecture import model_4_structure
from players import load_cnn_model, make_cnn_prediction, heuristic_player
import shutil
import glob
from gomuku_engine import GomokuEngine
from simulation import generate_bootstrapping_data
import random
import re
from config import evaluate_candidate_params, generate_bootstrapping_data_params, run_rl_training_params

class RLDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path, weights_only=False)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, action, reward = self.data[idx]
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(reward, dtype=torch.float32)
        )
def train_rl_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0

    for batch_X, batch_y, batch_w in loader:
        batch_X, batch_y, batch_w = batch_X.to(device), batch_y.to(device), batch_w.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(batch_X)
            loss = (criterion(outputs, batch_y) * batch_w).mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * batch_X.size(0)
    return total_loss / len(loader.dataset)


def run_rl_training(model_path, data_path, save_path, epochs=5, batch_size=256, lr=0.00001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using: {device}")
    model = load_cnn_model(model_path, model_4_structure)
    model.to(device)
    dataset = RLDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction='none')
    scaler = torch.amp.GradScaler('cuda')
    print(f"Starting training on {len(dataset)} moves...")

    for epoch in range(epochs):
        loss = train_rl_epoch(model, loader, optimizer, criterion, device, scaler)
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Training finished. New model saved as: {save_path}")


def get_sliding_window_data(window_size=2, output_path="Data_rfl/aggregated_buffer.pt"):
    files = sorted(glob.glob('Data_rfl/replay_buffer_*.pt'))
    recent_files = files[-window_size:]

    if not recent_files:
        return output_path, 0

    print(f"Agregating  {len(recent_files)} latest files...")
    all_winning_data = []

    for f in recent_files:
        file_data = torch.load(f, weights_only=False)
        winning_moves = [item for item in file_data if item[2] > 0]
        all_winning_data.extend(winning_moves)

    torch.save(all_winning_data, output_path)
    return output_path, len(all_winning_data)


def evaluate_candidate(candidate_path, current_model_path, num_games_old=1000, num_games_heur=1000, num_games_base=1000, threshold_old=52,threshold_base=55,threshold_heur=60):
    print(f"\n--- Evaluating candidate ---")
    candidate = load_cnn_model(candidate_path, model_4_structure)
    current_best = load_cnn_model(current_model_path, model_4_structure)
    baseline_model_path = 'models/model_4.pt'
    baseline_model = load_cnn_model(baseline_model_path, model_4_structure)

    def cand_move(eng):
        return make_cnn_prediction(candidate, eng.board, eng.get_legal_moves(), eng.current_player, temperature=0.0)

    def best_move(eng):
        return make_cnn_prediction(current_best, eng.board, eng.get_legal_moves(), eng.current_player, temperature=0.0)

    def baseline_move(eng):
        return make_cnn_prediction(baseline_model, eng.board, eng.get_legal_moves(), eng.current_player,
                                   temperature=0.0)

    center_coords = [6, 7, 8]
    central_moves = [(r, c) for r in center_coords for c in center_coords]

    def run_duel(p1, p2, games):
        p1_wins = 0
        draws = 0
        match_openings = [random.sample(central_moves, k=4) for _ in range(games)]

        for i in range(games // 2):
            game = GomokuEngine()
            game.play_game(p1, p2, match_openings[i])
            if game.winner == 'X':
                p1_wins += 1
            elif game.winner is None:
                draws += 1

        for i in range(games // 2, games):
            game = GomokuEngine()
            game.play_game(p2, p1, match_openings[i])
            if game.winner == 'O':
                p1_wins += 1
            elif game.winner is None:
                draws += 1

        return ((p1_wins + (draws * 0.5)) / games) * 100.0

    # Candidate vs current master
    winrate_old = run_duel(cand_move, best_move, num_games_old)
    print(f"Result [Candidate vs Current Master]: {winrate_old:.1f}% win ratio")
    # Candidate vs base model
    winrate_baseline = run_duel(cand_move, baseline_move, num_games_base)
    print(f"Result [Candidate vs Base CNN model]: {winrate_baseline:.1f}% win ratio")
    # Candidate vs heuristic
    winrate_heur = run_duel(cand_move, heuristic_player, num_games_heur)
    print(f"Result [Candidate vs Heuristic]:   {winrate_heur:.1f}% win ratio")

    passed = (winrate_old >= threshold_old) and (winrate_baseline >= threshold_base) and (winrate_heur >= threshold_heur)
    return passed


def main_rl_loop(iterations=5, num_games=3000, window_size=3, target_moves = 100000):
    current_version, current_master = get_latest_master()
    candidate_model = 'models/model_candidate.pt'
    print(f"\n[*] Starting reincforcement learning loop")
    print(f"[*] Current Master: {current_master} (Version: {current_version})")
    for i in range(iterations):
        print(f"\n=============================================")
        print(f"           Iteration of RL {i + 1}/{iterations}            ")
        print(f"=============================================")

        model_to_play = load_cnn_model(current_master, model_4_structure)
        generate_bootstrapping_data(model_to_play,  **(generate_bootstrapping_data_params()| {'num_games': num_games}))
        while True:
            training_data_path, num_winning_moves = get_sliding_window_data(window_size=window_size)

            if num_winning_moves >= target_moves:
                print(f">>> {num_winning_moves} moves found in {window_size} files. Starting the training")
                break
            else:
                print(f">>> Not enough moves in the window {num_winning_moves}/{target_moves}")
                print(f">>> Generating {num_games} new games to gather more moves")
                generate_bootstrapping_data(model_to_play, num_games=num_games)

        run_rl_training(
            model_path=current_master,
            data_path=training_data_path,
            save_path=candidate_model,
            **run_rl_training_params()
        )

        is_better = evaluate_candidate(candidate_model, current_master, **evaluate_candidate_params())

        if is_better:
            current_version += 1
            new_master_path = f'models/model_{current_version}_rl.pt'

            print(f">>> Succes! Candidate passed evaluation")
            print(f">>> Saving new master as: {new_master_path}")
            shutil.copy(candidate_model, new_master_path)
            current_master = new_master_path
        else:
            print(">>> Candidate Failed to pass evaluation")

    print("\n[Iteration finished] - check file", current_master)


def get_latest_master(models_dir="models"):
    files = glob.glob(f"{models_dir}/model_*.pt")
    highest_version = 0
    latest_model_path = f"{models_dir}/model_4.pt"

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