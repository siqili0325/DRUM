import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from engression import engression # Install the package first https://github.com/xwshen51/engression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import wasserstein_distance
import math
import copy 
from datetime import datetime
import sys
import os

from sklearn.model_selection import train_test_split
from fun import simulate_data_linear_confounder, generate_conditional_A_linear, F_hat_Network, Generator_Network, Baseline_NN, train_f_hat, train_f_hat_new, train_generator, train_baseline_model, train_baseline_model_new, predict_m_star, train_dro_model, cross_fit_f_w_g, train_Fx_estimator, train_debiased_generator, predict_final_debiased, fit_engression_dro, fit_engression_dro_penalty_grid, diagnose_dro_issue


def f_bar_confounded_2(X, A):
    dim_A = A.shape[1]
    dim_X = X.shape[1]
    n = A.shape[0]

    linear_A = 0.1 * np.sum(A, axis=1, keepdims=True)
    quad_A = 0.1 * np.sum(A**2, axis=1, keepdims=True)

    linear_X = 0.1 * np.sum(X, axis=1, keepdims=True)
    quad_X = 0.1 * np.sum(X**2, axis=1, keepdims=True)

    xa_interact = np.zeros((n, 1))
    for i in range(min(dim_A, dim_X)):
        xa_interact += 0.1 * (X[:, i:i+1] * A[:, i:i+1])

    sign_interact = np.zeros((n, 1))
    for i in range(min(dim_A, dim_X)):
        sign_interact += 0.2 * np.sign(A[:, i:i+1]) * (X[:, i:i+1]**2)
    
    if dim_A >= 2:
        xa_interact += 0.3 * (X[:, 0:1] * A[:, 1:2])
        xa_interact += 0.3 * (X[:, 1:2] * A[:, 0:1])

    F = linear_A + quad_A + xa_interact + sign_interact
    return F


def generate_A_local_shift(X, d0, dim_A, perturbation_type='coefficient', perturbation_scale=1.0):
    n = X.shape[0]
    
    if perturbation_type == 'coefficient':
        beta = d0['beta'] * np.random.uniform(-perturbation_scale, perturbation_scale, size=(X.shape[1], dim_A))
        deterministic = X @ beta
        
    elif perturbation_type == 'nonlinear':
        beta = d0['beta']
        deterministic = X @ beta
        quad_coef = np.random.uniform(-0.2, 0.2)
        deterministic += quad_coef * (X[:, 0:1] ** 2)
        
    elif perturbation_type == 'shift':
        beta = d0['beta']
        deterministic = X @ beta
        shift = np.random.uniform(-0.5, 0.5) * X[:, 0:1]
        deterministic += shift
    
    noise = np.random.multivariate_normal(np.zeros(dim_A), d0['cov'], n)
    A = deterministic + d0['noise_scale'] * noise
    
    return A


def evaluate_models_on_scale(
    models_dict, x_target_tensor, d0_target, f_bar_func,
    n_target, dim_X, dim_A, dim_epsilon, l_samples, source_ref2,
    perturbation_scale, n_mc_samples,
    x_shift_base, x_scale_base, x_mu_perturb_range, x_sigma_perturb_range,
    strength_range, noise_scale_params,
    local_strength, global_loc_shift_range, global_scale_range,
    x_perturbation=False,
    seed=None
):
    mc_test_sets = unified_monte_carlo_evaluation_enhanced(
        X_target=x_target_tensor.numpy(),
        d0=d0_target,
        n_target=n_target,
        dim_X=dim_X,
        dim_A=dim_A,
        f_bar_func=f_bar_func,
        n_mc_samples=n_mc_samples,
        alpha=1.0,  # Fixed at pure local
        perturbation_scale=perturbation_scale,
        x_shift_base=x_shift_base,
        x_scale_base=x_scale_base,
        x_mu_perturb_range=x_mu_perturb_range,
        x_sigma_perturb_range=x_sigma_perturb_range,
        strength_range=strength_range,
        noise_scale_params=noise_scale_params,
        local_strength=local_strength,
        global_loc_shift_range=global_loc_shift_range,
        global_scale_range=global_scale_range,
        x_perturbation=x_perturbation
    )
    
    mc_results = []
    for test_set in mc_test_sets:
        challenge_df = test_set['data']
        x_challenge = torch.tensor(challenge_df.filter(regex='^x_').values, dtype=torch.float32)
        y_challenge_noiseless = challenge_df['y_noiseless'].values
        
        with torch.no_grad():
            m_star_preds = predict_m_star(
                x_challenge, models_dict['f_hat'], models_dict['generator'],
                dim_A, dim_epsilon, l_samples=l_samples
            )
            m_robust_preds = models_dict['m_robust'](x_challenge).numpy()
            Fx_preds = predict_final_debiased(x_challenge, models_dict['Fx_estimator'])
            Fx_preds_L = predict_final_debiased(x_challenge, models_dict['Fx_estimator_L'])
            baseline_naive_preds = models_dict['baseline'](x_challenge).numpy()
            baseline_duchi_preds = models_dict['duchi'](x_challenge).numpy()
        
        mc_results.append({
            'mse_m_star': mean_squared_error(y_challenge_noiseless, m_star_preds),
            'mse_m_engression': mean_squared_error(y_challenge_noiseless, m_robust_preds),
            'mse_Fx': mean_squared_error(y_challenge_noiseless, Fx_preds),
            'mse_Fx_L': mean_squared_error(y_challenge_noiseless, Fx_preds_L),
            'mse_naive': mean_squared_error(y_challenge_noiseless, baseline_naive_preds),
            'mse_duchi': mean_squared_error(y_challenge_noiseless, baseline_duchi_preds),
            'normalized_mse_m_star': mean_squared_error(y_challenge_noiseless, m_star_preds) / source_ref2,
            'normalized_mse_m_engression': mean_squared_error(y_challenge_noiseless, m_robust_preds) / source_ref2,
            'normalized_mse_Fx': mean_squared_error(y_challenge_noiseless, Fx_preds) / source_ref2,
            'normalized_mse_Fx_L': mean_squared_error(y_challenge_noiseless, Fx_preds_L) / source_ref2,
            'normalized_mse_naive': mean_squared_error(y_challenge_noiseless, baseline_naive_preds) / source_ref2,
            'normalized_mse_duchi': mean_squared_error(y_challenge_noiseless, baseline_duchi_preds) / source_ref2,
        })
    
    return {
        'worst_mse_proposed': max(r['mse_m_star'] for r in mc_results),
        'worst_mse_engression': max(r['mse_m_engression'] for r in mc_results),
        'worst_mse_debiased': max(r['mse_Fx'] for r in mc_results),
        'worst_mse_debiased_L': max(r['mse_Fx_L'] for r in mc_results),
        'worst_mse_naive': max(r['mse_naive'] for r in mc_results),
        'worst_mse_duchi': max(r['mse_duchi'] for r in mc_results),
        'mean_mse_proposed': np.mean([r['mse_m_star'] for r in mc_results]),
        'mean_mse_engression': np.mean([r['mse_m_engression'] for r in mc_results]),
        'mean_mse_debiased': np.mean([r['mse_Fx'] for r in mc_results]),
        'mean_mse_debiased_L': np.mean([r['mse_Fx_L'] for r in mc_results]),
        'mean_mse_naive': np.mean([r['mse_naive'] for r in mc_results]),
        'mean_mse_duchi': np.mean([r['mse_duchi'] for r in mc_results]),
        'worst_mse_proposed_norm': max(r['normalized_mse_m_star'] for r in mc_results),
        'worst_mse_engression_norm': max(r['normalized_mse_m_engression'] for r in mc_results),
        'worst_mse_debiased_norm': max(r['normalized_mse_Fx'] for r in mc_results),
        'worst_mse_debiased_L_norm': max(r['normalized_mse_Fx_L'] for r in mc_results),
        'worst_mse_naive_norm': max(r['normalized_mse_naive'] for r in mc_results),
        'worst_mse_duchi_norm': max(r['normalized_mse_duchi'] for r in mc_results),
        'mean_mse_proposed_norm': np.mean([r['normalized_mse_m_star'] for r in mc_results]),
        'mean_mse_engression_norm': np.mean([r['normalized_mse_m_engression'] for r in mc_results]),
        'mean_mse_debiased_norm': np.mean([r['normalized_mse_Fx'] for r in mc_results]),
        'mean_mse_debiased_L_norm': np.mean([r['normalized_mse_Fx_L'] for r in mc_results]),
        'mean_mse_naive_norm': np.mean([r['normalized_mse_naive'] for r in mc_results]),
        'mean_mse_duchi_norm': np.mean([r['normalized_mse_duchi'] for r in mc_results]),
    }


def unified_monte_carlo_evaluation_enhanced(
    X_target, d0, n_target, dim_X, dim_A, f_bar_func,
    n_mc_samples=100,
    alpha=1.0,
    perturbation_scale=1.0,
    x_shift_base=0.1,
    x_scale_base=1.1,
    x_mu_perturb_range=(-0.2, 0.3), 
    x_sigma_perturb_range=(0.8, 1.2),
    strength_range=(0.6, 1.4),   
    noise_scale_params=(2, 2),
    local_strength=1.5,
    global_loc_shift_range=(0.5, 3.0),
    global_scale_range=(1.5, 2.5),
    x_perturbation=False,
    seed=None
):
    if seed is not None:
        np.random.seed(seed)
    
    test_sets = []
    
    for i in range(n_mc_samples):
        if x_perturbation:
            x_mu_perturb = np.random.uniform(*x_mu_perturb_range)
            x_mu_mc = x_shift_base + x_mu_perturb
            x_sigma_perturb = np.random.uniform(*x_sigma_perturb_range)
            x_sigma_mc = x_scale_base * x_sigma_perturb
            X_test = np.random.randn(n_target, dim_X) * x_sigma_mc + x_mu_mc
        else:
            X_test = X_target
        
        A_local = generate_A_local_shift(
            X_test, d0, dim_A, 
            perturbation_type='coefficient',
            perturbation_scale=perturbation_scale 
        )

        # Global (only used if alpha < 1)
        loc_shift = np.random.uniform(*global_loc_shift_range) * np.random.choice([-0.6, 1])
        scale = np.random.uniform(*global_scale_range)
        global_mean = d0['mean'] + loc_shift * np.ones(dim_A)
        global_cov = d0['cov'] * scale
        A_global = np.random.multivariate_normal(global_mean, global_cov, n_target)
        
        A_test = alpha * A_local + (1 - alpha) * A_global
    
        Y_mean_test = f_bar_func(X_test, A_test)
        Y_test = Y_mean_test + np.random.randn(n_target, 1) * 0.1
        
        df_X = pd.DataFrame(X_test, columns=[f'x_{j}' for j in range(X_test.shape[1])])
        df_Y = pd.DataFrame(Y_test, columns=['y'])
        df_Y_noiseless = pd.DataFrame(Y_mean_test, columns=['y_noiseless'])
        test_df = pd.concat([df_X, df_Y, df_Y_noiseless], axis=1)
        
        test_sets.append({
            'data': test_df,
            'alpha': alpha,
            'perturbation_scale': perturbation_scale,
            'iteration': i
        })
    
    return test_sets


def create_experiment_folder(base_dir, dim_A):

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"dimA_{dim_A}_{timestamp}"
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    print(f"Created experiment folder: {folder_path}")
    return folder_path


def save_executed_script(folder_path):

    script_path = os.path.abspath(__file__)
    script_name = os.path.basename(script_path)
    
    dest_path = os.path.join(folder_path, f"executed_{script_name}")
    
    import shutil
    shutil.copy2(script_path, dest_path)
    
    metadata_path = os.path.join(folder_path, "execution_metadata.txt")
    with open(metadata_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("EXECUTION METADATA\n")
        f.write("="*60 + "\n")
        f.write(f"Execution timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Script name: {script_name}\n")
        f.write(f"Script path: {script_path}\n")
        f.write(f"Python version: {sys.version}\n")
        f.write(f"Working directory: {os.getcwd()}\n")
    
    print(f"Executed script saved to: {dest_path}")
    print(f"Execution metadata saved to: {metadata_path}")


def create_comparison_plots_scale(all_results_df, results_folder):
    
    methods = ['proposed', 'engression', 'debiased', 'debiased_L', 'naive', 'duchi']
    method_labels = {
        'proposed': 'FlexDRO-Global',
        'engression': 'FlexDRO-Local',
        'debiased': 'FlexDRO-Debiased (Global)',
        'debiased_L': 'FlexDRO-Debiased (Local)',
        'naive': 'Baseline: ERM',
        'duchi': 'Baseline: DRO (Duchi & Namkoong)'
    }
    colors = {
        'proposed': '#1f77b4',
        'engression': '#ff7f0e',
        'debiased': '#2ca02c',
        'debiased_L': '#F5A9B6',
        'naive': '#d62728',
        'duchi': '#9467bd'
    }
    markers = {'proposed': 'o', 'engression': '*', 'debiased': 'd', 'debiased_L': 'h', 'naive': 's', 'duchi': '^'}
    
    df_sorted = all_results_df.sort_values('perturbation_scale')
    
    # === WORST-CASE MSE vs PERTURBATION SCALE ===
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method in methods:
        col = f'worst_mse_{method}_norm'
        ax.plot(df_sorted['perturbation_scale'], df_sorted[col], 
               f'{markers[method]}-', 
               label=method_labels[method],
               color=colors[method],
               linewidth=2, markersize=10)
    
    ax.set_xscale('log')
    ax.set_xlabel('Perturbation Scale of MC Samples', fontsize=18)
    ax.set_ylabel('Normalized Worst-case MSE', fontsize=18)
    ax.set_title('Worst-case MSE vs Perturbation Scale', fontsize=18)
    ax.legend(fontsize=18, loc='best')
    ax.grid(True, alpha=0.3, which='both')
    ax.tick_params(axis='both', which='major', labelsize=16, length=5, width=1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'worst_case_mse_vs_scale.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(results_folder, 'worst_case_mse_vs_scale.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method in methods:
        col = f'mean_mse_{method}_norm'
        ax.plot(df_sorted['perturbation_scale'], df_sorted[col], 
               f'{markers[method]}-', 
               label=method_labels[method],
               color=colors[method],
               linewidth=2, markersize=10)
    
    ax.set_xscale('log')
    ax.set_xlabel('Perturbation Scale of MC Samples', fontsize=18)
    ax.set_ylabel('Normalized Mean MSE', fontsize=18)
    ax.set_title('Mean MSE vs Perturbation Scale', fontsize=18)
    ax.legend(fontsize=18, loc='best')
    ax.grid(True, alpha=0.3, which='both')
    ax.tick_params(axis='both', which='major', labelsize=16, length=5, width=1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'mean_mse_vs_scale.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(results_folder, 'mean_mse_vs_scale.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    scales = df_sorted['perturbation_scale'].values

    for idx, (metric, title) in enumerate([('worst', 'Worst-case'), ('mean', 'Mean')]):
        ax = axes[idx]
        for method in methods:
            col = f'{metric}_mse_{method}_norm'
            ax.plot(df_sorted['perturbation_scale'], df_sorted[col], 
                f'{markers[method]}-', 
                label=method_labels[method],
                color=colors[method],
                linewidth=2, markersize=10)
        
        ax.set_xscale('log')
        ax.set_xticks(scales)
        ax.set_xticklabels([str(s) for s in scales])
        ax.set_xlabel('Perturbation Scale of MC Samples', fontsize=18)
        ax.set_ylabel(f'Normalized {title} MSE', fontsize=18)
        ax.grid(True, alpha=0.3, which='both')
        ax.tick_params(axis='both', which='major', labelsize=16, length=5, width=1.0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=18,
            bbox_to_anchor=(0.5, -0.02), frameon=True, framealpha=0.9,
            edgecolor='#cccccc')

    plt.tight_layout(rect=[0, 0.18, 1, 1])
    plt.savefig(os.path.join(results_folder, 'mse_comparison_vs_scale.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(results_folder, 'mse_comparison_vs_scale.png'), dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    scales = df_sorted['perturbation_scale'].values
    x_positions = np.arange(len(scales))
    
    for idx, (metric, title) in enumerate([('worst', 'Worst-case'), ('mean', 'Mean')]):
        ax = axes[idx]
        for method in methods:
            col = f'{metric}_mse_{method}_norm'
            ax.plot(x_positions, df_sorted[col].values, 
                   f'{markers[method]}-', 
                   label=method_labels[method],
                   color=colors[method],
                   linewidth=2, markersize=10)
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(s) for s in scales])
        ax.set_xlabel('Perturbation Scale of MC Samples', fontsize=18)
        ax.set_ylabel(f'Normalized {title} MSE', fontsize=18)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=16, length=5, width=1.0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=18,
            bbox_to_anchor=(0.5, -0.02), frameon=True, framealpha=0.9,
            edgecolor='#cccccc')
    
    plt.tight_layout(rect=[0, 0.18, 1, 1])
    plt.savefig(os.path.join(results_folder, 'mse_comparison_vs_scale_even.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(results_folder, 'mse_comparison_vs_scale_even.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to: {results_folder}")


def main():

    N_SOURCE_SAMPLES = 5000
    N_TARGET_SAMPLES = 1000
    DIM_X = 15
    DIM_A = 5
    DIM_EPSILON = 4
    
    LR_F = 1e-5
    EPOCHS_F = 100
    
    LR_G = 1e-5
    EPOCHS_G = 300
    L_SAMPLES = 256
    
    LR_Fx = 1e-5
    LR_Omega = 1e-5
    EPOCHS_OMEGA = 200
    EPOCHS_Fx = 300
    
    DELTA = 0.3
    LR_PRIMAL = 2e-4
    LR_DUAL = 1e-4
    FINETUNE_STEPS = 80
    HIDDEN_DIM = 16
    NOISE_DIM = 32
    LR_GS = 5e-4
    NUM_EPOCHS_ENG = 500
    
    BATCH_SIZE = 128
    SEED = 1
    N_MC_SAMPLES = 100
    
    PERTURBATION_SCALES = [0.6, 1.0, 1.4, 1.8]
    
    X_SHIFT_BASE = 0.1
    X_SCALE_BASE = 1.1
    X_MU_PERTURB_RANGE = (-0.2, 0.3)
    X_SIGMA_PERTURB_RANGE = (0.8, 1.2)
    STRENGTH_RANGE = (0.8, 1.6)
    NOISE_SCALE_PARAMS = (3, 0.25)
    LOCAL_STRENGTH = 0.5
    GLOBAL_LOC_SHIFT_RANGE = (0.5, 2.0) 
    GLOBAL_SCALE_RANGE = (1.5, 2.5)
    
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    base_results_dir = "results"
    os.makedirs(base_results_dir, exist_ok=True)
    results_folder = create_experiment_folder(base_results_dir, DIM_A)
    
    save_executed_script(results_folder)

    all_results = []
    
    print("="*80)
    print(f"CONFOUNDER EXPERIMENT: DIM_A = {DIM_A}")
    print(f"Perturbation scales: {PERTURBATION_SCALES}")
    print(f"Results folder: {results_folder}")
    print("="*80)
    
    print("\nSimulating data...")
    source_df, target_df_full, source_cov_A, d0_source, d0_target = simulate_data_linear_confounder(
        n_source=N_SOURCE_SAMPLES, n_target=N_TARGET_SAMPLES,
        dim_X=DIM_X, dim_A=DIM_A, f_bar_func=f_bar_confounded_2,
        seed=SEED
    )
    

    g_seed = torch.Generator()
    g_seed.manual_seed(SEED)
    
    source_train_df, source_val_df = train_test_split(source_df, test_size=0.2, random_state=SEED)
    
    source_train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(source_train_df.filter(regex='^x_').values, dtype=torch.float32),
        torch.tensor(source_train_df.filter(regex='^a_').values, dtype=torch.float32),
        torch.tensor(source_train_df['y'].values, dtype=torch.float32).view(-1, 1)
    )
    source_train_loader = torch.utils.data.DataLoader(
        source_train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g_seed
    )
    
    source_val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(source_val_df.filter(regex='^x_').values, dtype=torch.float32),
        torch.tensor(source_val_df.filter(regex='^a_').values, dtype=torch.float32),
        torch.tensor(source_val_df['y'].values, dtype=torch.float32).view(-1, 1)
    )
    source_val_loader = torch.utils.data.DataLoader(
        source_val_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g_seed
    )
    
    target_loader = torch.utils.data.DataLoader(
        torch.tensor(target_df_full.filter(regex='^x_').values, dtype=torch.float32),
        batch_size=BATCH_SIZE, shuffle=True, generator=g_seed
    )
    
    source_ref2 = np.var(source_df['y'].values)
    
    print("\nTraining models...")
    
    # Proposed (Global)
    f_hat = F_hat_Network(DIM_X, DIM_A)
    generator = Generator_Network(DIM_EPSILON, DIM_A)
    trained_f_hat = train_f_hat_new(f_hat, source_train_loader, source_val_loader, EPOCHS_F, LR_F)
    trained_generator = train_generator(generator, trained_f_hat, target_loader,
                                       EPOCHS_G, LR_G, DIM_X, DIM_A, DIM_EPSILON, L_SAMPLES)
    
    # Engression (Local)
    X_source_np = source_train_df.filter(regex='^x_').values
    A_source_np = source_train_df.filter(regex='^a_').values
    X_target_np = target_df_full.filter(regex='^x_').values

    m_robust, g_source_eng, g_worst_eng = fit_engression_dro(
        X_source=X_source_np,
        A_source=A_source_np,
        X_target=X_target_np,
        f_hat=trained_f_hat,
        num_layer=2, hidden_dim=HIDDEN_DIM, noise_dim=NOISE_DIM, lr_gs=LR_GS,
        lr_primal=LR_PRIMAL, lr_dual=LR_DUAL, grad_clip=2.0,
        num_epochs=NUM_EPOCHS_ENG, batch_size=128,
        n_samples=L_SAMPLES,
        delta=DELTA, finetune_steps=FINETUNE_STEPS, finetune_lr=1e-5, finetune_L=64, verbose=True
    ) 

    X_pooled = np.vstack([source_df.filter(regex='^x_').values, X_target_np])

    # Debiased local
    generator_for_debiasing_L = copy.deepcopy(g_worst_eng)
    F_hat_L = cross_fit_f_w_g(source_df, target_df_full, target_loader, generator_for_debiasing_L, DIM_X, DIM_A, DIM_EPSILON, L_SAMPLES,
    EPOCHS_G, LR_G, EPOCHS_F, EPOCHS_OMEGA, LR_F, LR_Omega, 
    HIDDEN_DIM, NOISE_DIM, LR_GS, LR_PRIMAL, LR_DUAL, NUM_EPOCHS_ENG, DELTA, FINETUNE_STEPS,
    local=True)
    final_Fx_estimator_L = train_Fx_estimator(F_hat_L, X_pooled, EPOCHS_Fx, LR_Fx, BATCH_SIZE)

    # Debiased global
    generator_for_debiasing = copy.deepcopy(trained_generator)
    F_hat = cross_fit_f_w_g(source_df, target_df_full, target_loader, generator_for_debiasing, DIM_X, DIM_A, DIM_EPSILON, L_SAMPLES, 
    EPOCHS_G, LR_G, EPOCHS_F, EPOCHS_OMEGA, LR_F, LR_Omega, local=False)
    final_Fx_estimator = train_Fx_estimator(F_hat, X_pooled, EPOCHS_Fx, LR_Fx, BATCH_SIZE)
    
    # Baselines
    baseline_net = Baseline_NN(DIM_X)
    trained_baseline = train_baseline_model_new(baseline_net, source_train_loader, source_val_loader, 20, 0.001)
    duchi_net = Baseline_NN(DIM_X)
    trained_duchi = train_dro_model(duchi_net, source_train_loader, source_val_loader, 50, 5e-04, 0.25)
    
    # Set to eval mode
    trained_f_hat.eval()
    trained_generator.eval()
    trained_baseline.eval()
    trained_duchi.eval()
    final_Fx_estimator.eval()
    final_Fx_estimator_L.eval()
    
    # Pack models
    models_dict = {
        'f_hat': trained_f_hat,
        'generator': trained_generator,
        'm_robust': m_robust,
        'Fx_estimator': final_Fx_estimator,
        'Fx_estimator_L': final_Fx_estimator_L,
        'baseline': trained_baseline,
        'duchi': trained_duchi
    }
    
    x_target_tensor = torch.tensor(X_target_np, dtype=torch.float32)
    
    # ==========================================================================
    # Evaluate on ALL perturbation scales (alpha fixed at 1.0)
    # ==========================================================================
    print("\nEvaluating on multiple perturbation scales ...")
    
    for scale in PERTURBATION_SCALES:
        print(f"  scale = {scale}...", end=" ")
        
        results = evaluate_models_on_scale(
            models_dict, x_target_tensor, d0_target, f_bar_confounded_2,
            N_TARGET_SAMPLES, DIM_X, DIM_A, DIM_EPSILON, L_SAMPLES, source_ref2,
            perturbation_scale=scale,
            n_mc_samples=N_MC_SAMPLES,
            x_shift_base=X_SHIFT_BASE,
            x_scale_base=X_SCALE_BASE,
            x_mu_perturb_range=X_MU_PERTURB_RANGE,
            x_sigma_perturb_range=X_SIGMA_PERTURB_RANGE,
            strength_range=STRENGTH_RANGE,
            noise_scale_params=NOISE_SCALE_PARAMS,
            local_strength=LOCAL_STRENGTH,
            global_loc_shift_range=GLOBAL_LOC_SHIFT_RANGE,
            global_scale_range=GLOBAL_SCALE_RANGE,
            seed=SEED + int(scale * 10)
        )
        
        results['perturbation_scale'] = scale
        all_results.append(results)
        
        print(f"Worst MSE - Global: {results['worst_mse_proposed_norm']:.2f}, "
              f"Global-debiased: {results['worst_mse_debiased_norm']:.2f}, "
              f"Local: {results['worst_mse_engression_norm']:.2f}, "
              f"Local-debiased: {results['worst_mse_debiased_L_norm']:.2f}, "
              f"Naive: {results['worst_mse_naive_norm']:.2f}")
    
    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv(os.path.join(results_folder, 'results_all.csv'), index=False)
    print(f"\nResults saved to: {results_folder}")
    
    print("\nCreating comparison plots...")
    create_comparison_plots_scale(all_results_df, results_folder)
    
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY (varying perturbation scale)")
    print("="*80)
    print(f"{'Scale':<10} {'Proposed-G':<12} {'Proposed-L':<12} {'Debiased-G':<12} {'Debiased-L':<12}  {'Naive':<12} {'Duchi':<12} {'Best':<20}")
    print("-" * 90)
    
    for _, row in all_results_df.iterrows():
        methods_mse = {
            'Proposed-G': row['worst_mse_proposed_norm'],
            'Proposed-L': row['worst_mse_engression_norm'],
            'Debiased-Global': row['worst_mse_debiased_norm'],
            'Debiased-Local': row['worst_mse_debiased_L_norm'],
            'Naive': row['worst_mse_naive_norm'],
            'Duchi': row['worst_mse_duchi_norm']
        }
        best = min(methods_mse, key=methods_mse.get)
        
        print(f"{row['perturbation_scale']:<10.1f} {methods_mse['Proposed-G']:<12.2f} {methods_mse['Proposed-L']:<12.2f} "
          f"{methods_mse['Debiased-Global']:<15.2f} {methods_mse['Debiased-Local']:<15.2f} "
          f"{methods_mse['Naive']:<12.2f} {methods_mse['Duchi']:<12.2f} {best:<20}")
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print(f"All results saved to: {results_folder}")
    print("="*80)


if __name__ == '__main__':
    main()