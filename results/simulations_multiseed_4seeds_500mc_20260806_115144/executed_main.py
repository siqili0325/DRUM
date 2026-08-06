"""
Usage:
    python main.py                        # defalut 4 parallel workers
    python main.py --n_workers 10         # 10 parallel workers
    python main.py --seeds 1 2 3          # run specific seeds only
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import copy
import os
import sys
import shutil
import argparse
import traceback
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from scipy.optimize import minimize as scipy_minimize

from helper import simulate_data_linear_confounder, F_hat_Network,Baseline_NN, train_f_hat_new, train_dro_model, train_baseline_model_new
from DRUM import fit_drum, predict_drum, cross_fit_debiased, train_Fx_estimator, predict_final_debiased
                

DEFAULT_CONFIG = {

    'N_SOURCE': 5000,
    'N_TARGET': 1000,
    'DIM_X': 15,
    'DIM_A': 5,
    'DIM_EPSILON': 4,
    'LR_F': 1e-5,
    'EPOCHS_F': 100,
    'LR_G': 1e-5,
    'EPOCHS_G': 300,
    'L_SAMPLES': 256,
    'LR_Fx': 1e-5,
    'LR_Omega': 1e-5,
    'EPOCHS_OMEGA': 200,
    'EPOCHS_Fx': 200,
    'DELTA': 0.3,
    'LR_PRIMAL': 2e-4,
    'LR_DUAL': 1e-4,
    'FINETUNE_STEPS': 80,
    'HIDDEN_DIM': 16,
    'NOISE_DIM': 32,
    'LR_GS': 5e-4,
    'NUM_EPOCHS_ENG': 500,
    'BATCH_SIZE': 128,
    'TRAIN_SEEDS': list(range(1, 5)),
    'MC_SEED': 9999,                  
    'N_MC': 500,                       
    'N_WORKERS': 5,                 
    'PERTURBATION_SCALES': [0.5, 1.0, 1.5],
}

# All 14 method names in a canonical order
METHOD_NAMES = [
    'DRUM (plug-in)', 'DRUM',
    'naive', 'duchi',
]

def f_bar_confounded_2(X, A):

    dim_A = A.shape[1]
    dim_X = X.shape[1]
    n = A.shape[0]

    linear_A = 0.1 * np.sum(A, axis=1, keepdims=True)
    quad_A = 0.1 * np.sum(A**2, axis=1, keepdims=True)

    xa_interact = np.zeros((n, 1))
    for i in range(min(dim_A, dim_X)):
        xa_interact += 0.1 * (X[:, i:i+1] * A[:, i:i+1])

    sign_interact = np.zeros((n, 1))
    for i in range(min(dim_A, dim_X)):
        sign_interact += 0.2 * np.sign(A[:, i:i+1]) * (X[:, i:i+1]**2)

    if dim_A >= 2:
        xa_interact += 0.3 * (X[:, 0:1] * A[:, 1:2])
        xa_interact += 0.3 * (X[:, 1:2] * A[:, 0:1])

    return linear_A + quad_A + xa_interact + sign_interact


def generate_A_local_shift(X, d0, dim_A, perturbation_scale=1.0):

    n = X.shape[0]
    beta = d0['beta'] * np.random.uniform(
        -perturbation_scale, perturbation_scale,
        size=(X.shape[1], dim_A))
    deterministic = X @ beta
    noise = np.random.multivariate_normal(np.zeros(dim_A), d0['cov'], n)
    return deterministic + d0['noise_scale'] * noise


def draw_mc_components(d0, dim_X, dim_A, n_target, n_mc_samples, seed):

    np.random.seed(seed)
    directions = []
    noises = []
    for _ in range(n_mc_samples):
        D = np.random.uniform(-1, 1, size=(dim_X, dim_A))
        noise = np.random.multivariate_normal(
            np.zeros(dim_A), d0['cov'], n_target)
        directions.append(D)
        noises.append(noise)
    return directions, noises


def generate_mc_test_sets_nested(X_target, d0, f_bar_func,
                                 directions, noises, perturbation_scale):

    test_sets = []
    for D, noise in zip(directions, noises):
        beta_perturbed = d0['beta'] * (perturbation_scale * D)
        deterministic = X_target @ beta_perturbed
        A_test = deterministic + d0['noise_scale'] * noise
        Y_mean = f_bar_func(X_target, A_test)
        test_sets.append({
            'x': X_target.copy(),
            'y_noiseless': Y_mean.flatten(),
        })
    return test_sets


def generate_mc_test_sets(X_target, d0, n_target, dim_X, dim_A,
                          f_bar_func, n_mc_samples, perturbation_scale, seed):

    np.random.seed(seed)
    test_sets = []
    for _ in range(n_mc_samples):
        A_test = generate_A_local_shift(X_target, d0, dim_A,
                                        perturbation_scale=perturbation_scale)
        Y_mean = f_bar_func(X_target, A_test)
        test_sets.append({
            'x': X_target.copy(),
            'y_noiseless': Y_mean.flatten(),
        })
    return test_sets
    

def train_weighted_erm(model, X, y, w, X_val, y_val, epochs, lr, bs=128):
    opt = optim.Adam(model.parameters(), lr=lr)
    xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    wt = torch.tensor(w, dtype=torch.float32).view(-1, 1)
    xv = torch.tensor(X_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    ds = torch.utils.data.TensorDataset(xt, yt, wt)
    ld = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True,
                                     generator=torch.Generator().manual_seed(42))
    best_v, best_s = float('inf'), None
    for _ in range(epochs):
        model.train()
        for xb, yb, wb in ld:
            loss = (wb * (model(xb) - yb)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = ((model(xv) - yv)**2).mean().item()
        if v < best_v:
            best_v, best_s = v, copy.deepcopy(model.state_dict())
    if best_s:
        model.load_state_dict(best_s)
    model.eval()
    return model, best_v


def train_erm(model, X, y, X_val, y_val, epochs, lr, bs=128):
    opt = optim.Adam(model.parameters(), lr=lr)
    xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    xv = torch.tensor(X_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    ds = torch.utils.data.TensorDataset(xt, yt)
    ld = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True,
                                     generator=torch.Generator().manual_seed(42))
    best_v, best_s = float('inf'), None
    for _ in range(epochs):
        model.train()
        for xb, yb in ld:
            loss = ((model(xb) - yb)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = ((model(xv) - yv)**2).mean().item()
        if v < best_v:
            best_v, best_s = v, copy.deepcopy(model.state_dict())
    if best_s:
        model.load_state_dict(best_s)
    model.eval()
    return model, best_v


def train_dro_mse(model, X, y, X_val, y_val, epochs, lr, rho, bs=128):
    opt = optim.Adam(model.parameters(), lr=lr)
    xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    xv = torch.tensor(X_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    ds = torch.utils.data.TensorDataset(xt, yt)
    ld = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True,
                                     generator=torch.Generator().manual_seed(42))
    best_v, best_s = float('inf'), None
    for _ in range(epochs):
        model.train()
        for xb, yb in ld:
            per = (model(xb) - yb)**2
            m_l, s_l = per.mean(), per.std()
            loss = m_l + rho * s_l / np.sqrt(per.size(0)) if s_l > 1e-8 else m_l
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = ((model(xv) - yv)**2).mean().item()
        if v < best_v:
            best_v, best_s = v, copy.deepcopy(model.state_dict())
    if best_s:
        model.load_state_dict(best_s)
    model.eval()
    return model, best_v


def run_single_seed(train_seed, cfg):

    torch.set_num_threads(2)

    seed_dir = os.path.join(cfg['BASE_DIR'], f'seed_{train_seed}')
    os.makedirs(seed_dir, exist_ok=True)
    log_path = os.path.join(seed_dir, 'log.txt')

    def log(msg):
        ts = datetime.now().strftime('%H:%M:%S')
        line = f"[seed={train_seed} {ts}] {msg}"
        print(line, flush=True)
        with open(log_path, 'a') as f:
            f.write(line + '\n')

    try:
        log("Starting pipeline")

        np.random.seed(train_seed)
        torch.manual_seed(train_seed)

        log("Generating data...")
        source_df, target_df, _, d0_source, d0_target = \
            simulate_data_linear_confounder(
                n_source=cfg['N_SOURCE'], n_target=cfg['N_TARGET'],
                dim_X=cfg['DIM_X'], dim_A=cfg['DIM_A'],
                f_bar_func=f_bar_confounded_2, seed=train_seed)

        source_var = np.var(source_df['y'].values)
        src_train, src_val = train_test_split(
            source_df, test_size=0.2, random_state=train_seed)

        Xs_tr = src_train.filter(regex='^x_').values
        As_tr = src_train.filter(regex='^a_').values
        Ys_tr = src_train['y'].values
        Xs_vl = src_val.filter(regex='^x_').values
        Ys_vl = src_val['y'].values
        Xt = target_df.filter(regex='^x_').values

        log(f"  Source: train={len(Xs_tr)}, val={len(Xs_vl)}, "
            f"target={len(Xt)}, source_var={source_var:.4f}")

        g_seed = torch.Generator().manual_seed(train_seed)
        BS = cfg['BATCH_SIZE']

        source_train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(Xs_tr, dtype=torch.float32),
            torch.tensor(As_tr, dtype=torch.float32),
            torch.tensor(Ys_tr, dtype=torch.float32).view(-1, 1))
        source_train_loader = torch.utils.data.DataLoader(
            source_train_dataset, batch_size=BS, shuffle=True, generator=g_seed)

        source_val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(Xs_vl, dtype=torch.float32),
            torch.tensor(src_val.filter(regex='^a_').values, dtype=torch.float32),
            torch.tensor(Ys_vl, dtype=torch.float32).view(-1, 1))
        source_val_loader = torch.utils.data.DataLoader(
            source_val_dataset, batch_size=BS, shuffle=True,
            generator=torch.Generator().manual_seed(train_seed))

        target_loader = torch.utils.data.DataLoader(
            torch.tensor(Xt, dtype=torch.float32),
            batch_size=BS, shuffle=True,
            generator=torch.Generator().manual_seed(train_seed))

        log("Training f_hat (Stage 1)...")
        f_hat = F_hat_Network(cfg['DIM_X'], cfg['DIM_A'])
        trained_f_hat = train_f_hat_new(f_hat, source_train_loader, source_val_loader,
            cfg['EPOCHS_F'], cfg['LR_F'])

        log("Training DRUM conditional (Stage 2)......")
        m_robust_C, g_source_C, g_worst_C = fit_drum(
            Xs_tr, As_tr, Xt, trained_f_hat,
            conditional=True, delta=cfg['DELTA'],
            num_layer=2, hidden_dim=cfg['HIDDEN_DIM'],
            noise_dim=cfg['NOISE_DIM'], lr_gs=cfg['LR_GS'],
            lr_primal=cfg['LR_PRIMAL'], lr_dual=cfg['LR_DUAL'],
            num_epochs_eng=cfg['NUM_EPOCHS_ENG'],
            batch_size_eng=128, n_samples=cfg['L_SAMPLES'],
            finetune_steps=cfg['FINETUNE_STEPS'],
            finetune_L=64, grad_clip=2.0, verbose=False)

        X_pooled = np.vstack([source_df.filter(regex='^x_').values, Xt])

        log("Training debiased estimator (conditional)...")
        F_hat_C = cross_fit_debiased(
            source_df, target_df,
            preliminary_generator=g_worst_C,
            preliminary_source_generator=g_source_C,
            conditional=True,
            dim_x=cfg['DIM_X'], dim_a=cfg['DIM_A'],
            epochs_f=cfg['EPOCHS_F'], lr_f=cfg['LR_F'],
            epochs_omega=cfg['EPOCHS_OMEGA'], lr_omega=cfg['LR_Omega'],
            epochs_g=cfg['EPOCHS_G'], lr_g=cfg['LR_G'],
            l_samples=cfg['L_SAMPLES'],
            delta=cfg['DELTA'], lr_primal=cfg['LR_PRIMAL'],
            lr_dual=cfg['LR_DUAL'],
            finetune_steps=cfg['FINETUNE_STEPS'])
        Fx_estimator_C = train_Fx_estimator(
            F_hat_C, X_pooled, cfg['EPOCHS_Fx'], cfg['LR_Fx'], BS)

        log("Training ERM baseline...")
        baseline_net = Baseline_NN(cfg['DIM_X'])
        trained_baseline = train_baseline_model_new(
            baseline_net, source_train_loader, source_val_loader, 20, 0.001)

        log("Training DRO (Duchi) baseline...")
        duchi_net = Baseline_NN(cfg['DIM_X'])
        trained_duchi = train_dro_model(
            duchi_net, source_train_loader, source_val_loader, 50, 5e-04, 0.25)

        trained_f_hat.eval()
        trained_baseline.eval()
        trained_duchi.eval()
        Fx_estimator_C.eval()


        log(f"Evaluating on {cfg['N_MC']} MC test sets × "
            f"{len(cfg['PERTURBATION_SCALES'])} scales...")

        mc_directions, mc_noises = draw_mc_components(
            d0_target, cfg['DIM_X'], cfg['DIM_A'],
            cfg['N_TARGET'], cfg['N_MC'], seed=cfg['MC_SEED'])

        summary_rows = []

        for scale in cfg['PERTURBATION_SCALES']:
            log(f"  scale={scale}...")

            mc_tests = generate_mc_test_sets_nested(
                Xt, d0_target, f_bar_confounded_2,
                mc_directions, mc_noises, scale)

            raw_nmse = {name: [] for name in METHOD_NAMES}

            for test in mc_tests:
                x_t = torch.tensor(test['x'], dtype=torch.float32)
                y_true = test['y_noiseless']

                with torch.no_grad():
                    pred_engression = m_robust_C(x_t.numpy()).flatten()
                    pred_debiased_L = predict_final_debiased(
                        x_t, Fx_estimator_C)
                    pred_naive = trained_baseline(x_t).numpy().flatten()
                    pred_duchi = trained_duchi(x_t).numpy().flatten()

                preds = {
                    'DRUM (plug-in)':  np.asarray(pred_engression).flatten(),
                    'DRUM':  np.asarray(pred_debiased_L).flatten(),
                    'naive':       pred_naive,
                    'duchi':       pred_duchi,
                }

                for name in METHOD_NAMES:
                    mse = mean_squared_error(y_true, preds[name])
                    raw_nmse[name].append(mse / source_var)

            raw_df = pd.DataFrame(raw_nmse)
            raw_df.to_csv(
                os.path.join(seed_dir, f'mc_raw_scale_{scale:.1f}.csv'),
                index=False)

            row = {'perturbation_scale': scale, 'source_var': source_var}
            for name in METHOD_NAMES:
                vals = raw_nmse[name]
                row[f'worst_{name}'] = max(vals)
                row[f'mean_{name}'] = np.mean(vals)
                row[f'median_{name}'] = np.median(vals)
                row[f'q95_{name}'] = np.percentile(vals, 95)
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(seed_dir, 'summary.csv'), index=False)

        log("Pipeline complete")
        return (train_seed, True, summary_df)

    except Exception as e:
        error_msg = f"FAILED: {e}\n{traceback.format_exc()}"
        log(error_msg)
        return (train_seed, False, error_msg)


def aggregate_across_seeds(base_dir, cfg):

    summaries = []
    completed_seeds = []

    for seed in cfg['TRAIN_SEEDS']:
        path = os.path.join(base_dir, f'seed_{seed}', 'summary.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['train_seed'] = seed
            summaries.append(df)
            completed_seeds.append(seed)

    if not summaries:
        print("ERROR: No completed seeds found.")
        return None

    print(f"\nAggregating {len(summaries)} / {len(cfg['TRAIN_SEEDS'])} seeds: "
          f"{completed_seeds}")

    all_df = pd.concat(summaries, ignore_index=True)

    all_df.to_csv(os.path.join(base_dir, 'all_seeds_long.csv'), index=False)

    n_seeds = len(completed_seeds)
    agg_rows = []

    for scale in cfg['PERTURBATION_SCALES']:
        mask = all_df['perturbation_scale'] == scale
        sub = all_df[mask]

        if len(sub) < n_seeds:
            print(f"  WARNING: scale={scale} has only {len(sub)}/{n_seeds} seeds")

        row = {'perturbation_scale': scale, 'n_seeds': len(sub)}

        for metric_prefix in ['worst', 'mean', 'median', 'q95']:
            for name in METHOD_NAMES:
                col = f'{metric_prefix}_{name}'
                vals = sub[col].values
                row[f'{col}_mean'] = np.mean(vals)
                row[f'{col}_std'] = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                row[f'{col}_se'] = (np.std(vals, ddof=1) / np.sqrt(len(vals))
                                    if len(vals) > 1 else 0.0)
                row[f'{col}_min'] = np.min(vals)
                row[f'{col}_max'] = np.max(vals)

        agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(os.path.join(base_dir, 'aggregate_summary.csv'), index=False)

    return agg_df

def print_summary_table(base_dir):
    """Print a formatted table of aggregate results."""
    agg_path = os.path.join(base_dir, 'aggregate_summary.csv')
    if not os.path.exists(agg_path):
        return
    agg = pd.read_csv(agg_path)

    print("\n" + "=" * 100)
    print("AGGREGATE RESULTS: WORST-CASE NORMALIZED MSE  (mean ± s.e. across seeds)")
    print("=" * 100)

    header = f"{'Scale':<8}"
    for name in METHOD_NAMES:                      
        header += f"  {name:<24}"
    print(header)
    print("-" * (8 + 26 * len(METHOD_NAMES)))

    for _, row in agg.iterrows():
        line = f"{row['perturbation_scale']:<8.1f}"
        for name in METHOD_NAMES:                
            m = row[f'worst_{name}_mean']
            se = row[f'worst_{name}_se']
            line += f"  {m:.3f} ± {se:.3f}         "
        print(line)


def main():
    parser = argparse.ArgumentParser(
        description="DRUM — Multi-Seed Parallel Experiment")
    parser.add_argument('--n_workers', type=int, default=None,
                        help='Number of parallel workers (default: from config)')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Specific training seeds to run (default: 1-10)')
    parser.add_argument('--n_mc', type=int, default=None,
                        help='MC test samples per scale (default: from config)')
    parser.add_argument('--aggregate_only', action='store_true',
                        help='Skip training, just aggregate existing results')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Existing results directory (for --aggregate_only)')
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    if args.n_mc is not None:
        cfg['N_MC'] = args.n_mc
    if args.n_workers is not None:
        cfg['N_WORKERS'] = args.n_workers
    if args.seeds is not None:
        cfg['TRAIN_SEEDS'] = args.seeds

    if args.aggregate_only and args.results_dir:
        base_dir = args.results_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_dir = os.path.join(
            'results',
            f'simulations_multiseed_{len(cfg["TRAIN_SEEDS"])}seeds_'
            f'{cfg["N_MC"]}mc_{timestamp}')
        os.makedirs(base_dir, exist_ok=True)

    cfg['BASE_DIR'] = base_dir

    if not args.aggregate_only:
        shutil.copy2(os.path.abspath(__file__),
                     os.path.join(base_dir, f"executed_{os.path.basename(__file__)}"))

        with open(os.path.join(base_dir, 'experiment_config.txt'), 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("DRUM Simulations — MULTI-SEED EXPERIMENT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Command: {' '.join(sys.argv)}\n\n")
            for k, v in sorted(cfg.items()):
                f.write(f"{k}: {v}\n")

        print("=" * 70)
        print(f"DRUM — Multi-Seed Experiment")
        print(f"  Seeds: {cfg['TRAIN_SEEDS']}")
        print(f"  MC samples: {cfg['N_MC']} per scale")
        print(f"  MC seed (fixed): {cfg['MC_SEED']}")
        print(f"  Perturbation scales: {cfg['PERTURBATION_SCALES']}")
        print(f"  Workers: {cfg['N_WORKERS']}")
        print(f"  Output: {base_dir}")
        print("=" * 70)

        completed = []
        failed = []

        if cfg['N_WORKERS'] == 1:

            for seed in cfg['TRAIN_SEEDS']:
                result = run_single_seed(seed, cfg)
                if result[1]:
                    completed.append(result[0])
                else:
                    failed.append((result[0], result[2]))
        else:
            with ProcessPoolExecutor(max_workers=cfg['N_WORKERS']) as executor:
                futures = {
                    executor.submit(run_single_seed, seed, cfg): seed
                    for seed in cfg['TRAIN_SEEDS']
                }
                for future in as_completed(futures):
                    seed_id = futures[future]
                    try:
                        train_seed, success, payload = future.result()
                        if success:
                            completed.append(train_seed)
                            print(f"\n>>> Seed {train_seed} COMPLETED "
                                  f"({len(completed)}/{len(cfg['TRAIN_SEEDS'])})")
                        else:
                            failed.append((train_seed, payload))
                            print(f"\n>>> Seed {train_seed} FAILED: "
                                  f"{payload[:200]}")
                    except Exception as e:
                        failed.append((seed_id, str(e)))
                        print(f"\n>>> Seed {seed_id} EXCEPTION: {e}")

        print(f"\n{'=' * 70}")
        print(f"PARALLEL EXECUTION COMPLETE")
        print(f"  Completed: {sorted(completed)}")
        if failed:
            print(f"  Failed: {[f[0] for f in failed]}")
        print(f"{'=' * 70}")

    print("\nAggregating results across seeds...")
    agg_df = aggregate_across_seeds(base_dir, cfg)

    if agg_df is not None:
        print_summary_table(base_dir)

    print(f"\nAll outputs in: {base_dir}")
    print("Done.")


if __name__ == '__main__':
    main()