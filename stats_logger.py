import csv
import os
import time
from datetime import datetime
from utils import load_config


def log_iteration_stats(
        iteration: int,
        master_before: str,
        master_after: str,
        epoch_losses: list,
        eval_results: dict,
        buffer_samples: int,
        files_used: int,
        warmup_ratio: float,
        epochs: int,
        lr: float,
        iter_start_time: float,
        log_dir: str = "logs"
) -> None:
    # 1. Tworzymy unikalną nazwę pliku z datą i godziną
    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_log_{current_time_str}.csv"

    # 2. Upewniamy się, że folder 'logs' istnieje
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    cfg = load_config()
    loop_cfg = cfg.get("main_rl_loop", {})
    gen_cfg = cfg.get("generator", {})
    cnn_cfg = cfg.get("CNN", {})
    eval_cfg = cfg.get("evaluate_candidate", {})

    row_dict = {
        # 1. Czas i wersje modeli
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "iteration": iteration,
        "master_before": master_before,
        "master_after": master_after,
        "passed": eval_results["passed"],
        "iter_time_sec": round(time.time() - iter_start_time, 1),

        # 2. Wyniki treningu (Loss) i ewaluacji (Winrate)
        "loss_start": round(epoch_losses[0], 4) if epoch_losses else None,
        "loss_end": round(epoch_losses[-1], 4) if epoch_losses else None,
        "winrate_master": eval_results["winrate_master"],
        "winrate_base": eval_results["winrate_base"],
        "winrate_heur": eval_results["winrate_heur"],

        # 3. Faktyczny stan bufora i parametry epoki w tej iteracji
        "buffer_samples": buffer_samples,
        "files_used": files_used,
        "warmup_ratio": warmup_ratio,
        "epochs": epochs,
        "lr": lr,
        "batch_size": cnn_cfg.get("batch_size"),

        # 4. Parametry z config.yaml (Generator & Loop)
        "cfg_num_games": loop_cfg.get("num_games"),
        "cfg_target_moves": loop_cfg.get("target_moves"),
        "cfg_window_size": loop_cfg.get("window_size"),
        "cfg_only_winners": loop_cfg.get("only_winners"),
        "gen_simulations": gen_cfg.get("num_simulations"),
        "gen_c_puct": gen_cfg.get("c_puct"),
        "gen_temp": gen_cfg.get("temp"),
        "gen_temp_moves": gen_cfg.get("temp_moves"),
        "gen_opening_k": gen_cfg.get("opening_moves_k"),

        # 5. Parametry z config.yaml (Ewaluacja)
        "eval_baseline_model": eval_cfg.get("baseline_model_path"),
        "eval_simulations": eval_cfg.get("num_simulations"),
        "eval_c_puct": eval_cfg.get("c_puct"),
        "eval_games_old": eval_cfg.get("num_games_old"),
        "eval_games_base": eval_cfg.get("num_games_base"),
        "eval_games_heur": eval_cfg.get("num_games_heur"),
        "thr_old": eval_cfg.get("threshold_old"),
        "thr_base": eval_cfg.get("threshold_base"),
        "thr_heur": eval_cfg.get("threshold_heur"),
    }

    file_exists = os.path.isfile(log_path)
    with open(log_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)