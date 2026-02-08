import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from datetime import datetime
import sys
from engression import engression  

def sample_A_with_grad(eng, X, L):

    A_list = []

    for _ in range(L):
        A_list.append(eng.model(X))  
    A = torch.stack(A_list, dim=1)   
    N, Ls, dim_a = A.shape
    A_flat = A.reshape(N*Ls, dim_a)  
    return A, A_flat

def energy_gap(eng_candidate, eng_source, X_s, A_s, beta=1.0):

    A_pred1 = eng_candidate.model(X_s)  
    A_pred2 = eng_candidate.model(X_s)
    term1_c = torch.norm(A_s - A_pred1, dim=1).mean()
    term2_c = 0.5 * torch.norm(A_pred1 - A_pred2, dim=1).mean()
    en_cand = term1_c - term2_c

    with torch.no_grad():
        A_src1 = eng_source.model(X_s)
        A_src2 = eng_source.model(X_s)
        term1_s = torch.norm(A_s - A_src1, dim=1).mean()
        term2_s = 0.5 * torch.norm(A_src1 - A_src2, dim=1).mean()
        en_src = term1_s - term2_s

    return en_cand - en_src

def energy_gap_stabilized(eng_candidate, eng_source, X_s, A_s, M=16):

    term1_c_samples = []
    for _ in range(M):
        with torch.no_grad():
            A_pred_c = eng_candidate.model(X_s)
        term1_c_samples.append(torch.norm(A_s - A_pred_c, dim=1))
    
    term1_c = torch.stack(term1_c_samples).detach().mean(dim=0).mean()

    A_pred_grad = eng_candidate.model(X_s)
    term1_c_grad = torch.norm(A_s - A_pred_grad, dim=1).mean()
    
    term1_c = term1_c + (term1_c_grad - term1_c).detach()

    A_pred1_c = eng_candidate.model(X_s)
    A_pred2_c = eng_candidate.model(X_s)
    term2_c = 0.5 * torch.norm(A_pred1_c - A_pred2_c, dim=1).mean()
    en_cand = term1_c - term2_c

    with torch.no_grad():
        A_src1 = eng_source.model(X_s)
        A_src2 = eng_source.model(X_s)
        term1_s = torch.norm(A_s - A_src1, dim=1).mean()
        term2_s = 0.5 * torch.norm(A_src1 - A_src2, dim=1).mean()
        en_src = term1_s - term2_s

    return en_cand - en_src

def energy_gap_paired(eng_candidate, eng_source, X_s, A_s, M=32):

    term1_c_samples = []
    term1_s_samples = []
    
    for i in range(M):
        seed = 42 + i
        
        torch.manual_seed(seed)
        with torch.no_grad():
            A_pred_c = eng_candidate.model(X_s)
        term1_c_samples.append(torch.norm(A_s - A_pred_c, dim=1))
        
        torch.manual_seed(seed)
        with torch.no_grad():
            A_pred_s = eng_source.model(X_s)
        term1_s_samples.append(torch.norm(A_s - A_pred_s, dim=1))
    
    term1_c = torch.stack(term1_c_samples).mean(dim=0).mean()
    term1_s = torch.stack(term1_s_samples).mean(dim=0).mean()
    
    term2_c_samples = []
    term2_s_samples = []
    
    for i in range(M):
        seed1 = 1000 + 2*i
        seed2 = 1000 + 2*i + 1
        
        torch.manual_seed(seed1)
        with torch.no_grad():
            A_c1 = eng_candidate.model(X_s)
        torch.manual_seed(seed2)
        with torch.no_grad():
            A_c2 = eng_candidate.model(X_s)
        term2_c_samples.append(torch.norm(A_c1 - A_c2, dim=1))
        
        torch.manual_seed(seed1)
        with torch.no_grad():
            A_s1 = eng_source.model(X_s)
        torch.manual_seed(seed2)
        with torch.no_grad():
            A_s2 = eng_source.model(X_s)
        term2_s_samples.append(torch.norm(A_s1 - A_s2, dim=1))
    
    term2_c = 0.5 * torch.stack(term2_c_samples).mean(dim=0).mean()
    term2_s = 0.5 * torch.stack(term2_s_samples).mean(dim=0).mean()
    
    en_cand = term1_c - term2_c
    en_src = term1_s - term2_s
    
    return en_cand - en_src

def energy_gap_paired_with_grad(eng_candidate, eng_source, X_s, A_s, M=32):

    device = X_s.device
    N = X_s.shape[0]
    term1_c_samples = []
    term1_s_samples = []
    
    for i in range(M):
        seed = 42 + i
        
        torch.manual_seed(seed)
        with torch.no_grad():
            A_pred_c = eng_candidate.model(X_s)
        term1_c_samples.append(torch.norm(A_s - A_pred_c, dim=1))
        
        torch.manual_seed(seed)
        with torch.no_grad():
            A_pred_s = eng_source.model(X_s)
        term1_s_samples.append(torch.norm(A_s - A_pred_s, dim=1))
    
    term1_c_stable = torch.stack(term1_c_samples).mean(dim=0).mean()
    term1_s_stable = torch.stack(term1_s_samples).mean(dim=0).mean()
    
    grad_seed =42
    
    torch.manual_seed(grad_seed)
    A_pred_c_grad = eng_candidate.model(X_s)
    term1_c_grad = torch.norm(A_s - A_pred_c_grad, dim=1).mean()
    
    term1_c = term1_c_stable.detach() + (term1_c_grad - term1_c_grad.detach())
    term1_s = term1_s_stable
    
    term2_c_samples = []
    term2_s_samples = []
    
    for i in range(M):
        seed1 = 1000 + 2*i
        seed2 = 1000 + 2*i + 1
        
        torch.manual_seed(seed1)
        with torch.no_grad():
            A_c1 = eng_candidate.model(X_s)
        torch.manual_seed(seed2)
        with torch.no_grad():
            A_c2 = eng_candidate.model(X_s)
        term2_c_samples.append(torch.norm(A_c1 - A_c2, dim=1))
        
        torch.manual_seed(seed1)
        with torch.no_grad():
            A_s1 = eng_source.model(X_s)
        torch.manual_seed(seed2)
        with torch.no_grad():
            A_s2 = eng_source.model(X_s)
        term2_s_samples.append(torch.norm(A_s1 - A_s2, dim=1))
    
    term2_c_stable = 0.5 * torch.stack(term2_c_samples).mean(dim=0).mean()
    term2_s_stable = 0.5 * torch.stack(term2_s_samples).mean(dim=0).mean()
    
    torch.manual_seed(7777)
    A_c1_grad = eng_candidate.model(X_s)
    torch.manual_seed(7778)
    A_c2_grad = eng_candidate.model(X_s)
    term2_c_grad = 0.5 * torch.norm(A_c1_grad - A_c2_grad, dim=1).mean()
    
    term2_c = term2_c_stable.detach() + (term2_c_grad - term2_c_grad.detach())
    term2_s = term2_s_stable
    
    en_cand = term1_c - term2_c
    en_src = term1_s - term2_s
    
    gap = en_cand - en_src
    
    return gap

def target_objective(eng_g, f_hat, X_t, L):

    N, dim_x = X_t.shape
    A, A_flat = sample_A_with_grad(eng_g, X_t, L) 
    # repeat X accordingly
    X_rep = X_t.unsqueeze(1).expand(-1, L, -1).reshape(N*L, dim_x) 
    y = f_hat(X_rep, A_flat).view(N, L)     
    inner = y.mean(dim=1)      
    return (inner ** 2).mean()   

def find_worst_case_g_primal_dual(
    g_source, f_hat, X_source, A_source, Y_source, X_target,
    delta,
    num_steps=800,
    L=64,             
    lr_primal=1e-4,      
    lr_dual=1e-2,     
    device="cpu",
    grad_clip=1.0,       
    log_every=50,
    return_history=False,
):

    g_worst = copy.deepcopy(g_source)
    g_worst.model.train()
    f_hat.eval()  

    X_s = torch.as_tensor(X_source, dtype=torch.float32, device=device)
    A_s = torch.as_tensor(A_source, dtype=torch.float32, device=device)
    X_t = torch.as_tensor(X_target, dtype=torch.float32, device=device)

    opt_g = torch.optim.Adam(g_worst.model.parameters(), lr=lr_primal)

    lam = torch.tensor(0.0, device=device)

    hist = {"obj": [], "gap": [], "lam": [], "lagr": []}

    for step in range(1, num_steps + 1):
        opt_g.zero_grad()

        obj = target_objective(g_worst, f_hat, X_t, L=L)
        gap = energy_gap(g_worst, g_source, X_s, A_s)
        lagr = obj + lam * (gap - delta)

        lagr.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(g_worst.model.parameters(), grad_clip)
        opt_g.step()

        with torch.no_grad():
            lam += lr_dual * (gap - delta)
            lam.clamp_(min=0.0)

        if step % log_every == 0 or step == 1:
            print(f"[g-opt step {step:04d}] "
                  f"obj={obj.item():.4f}  gap={gap.item():.4f}  "
                  f"λ={lam.item():.4f}  L={lagr.item():.4f}")
        if return_history:
            hist["obj"].append(obj.item())
            hist["gap"].append(gap.item())
            hist["lam"].append(lam.item())
            hist["lagr"].append(lagr.item())

    g_worst.model.eval()
    return g_worst


def find_worst_case_g_primal_dual_v2(
    g_source, f_hat, X_source, A_source, X_target,
    delta,
    num_steps=800,
    L=64,
    lr_primal=1e-4,
    lr_dual=1e-4,
    device="cpu",
    grad_clip=1.0,
    log_every=20,
    M_gap=16
):
    g_worst = copy.deepcopy(g_source)
    g_worst.model.train()
    f_hat.eval()

    X_s = torch.as_tensor(X_source, dtype=torch.float32, device=device)
    A_s = torch.as_tensor(A_source, dtype=torch.float32, device=device)
    X_t = torch.as_tensor(X_target, dtype=torch.float32, device=device)

    opt_g = torch.optim.Adam(g_worst.model.parameters(), lr=lr_primal)
    lam = torch.tensor(0.1, device=device) # Start with small positive value

    for step in range(1, num_steps + 1):
        opt_g.zero_grad()

        obj = target_objective(g_worst, f_hat, X_t, L=L)
        gap = energy_gap_stabilized(g_worst, g_source, X_s, A_s, M=M_gap)
        
        lagr = obj + lam.detach() * (gap - delta)

        lagr.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(g_worst.model.parameters(), grad_clip)
        opt_g.step()

        with torch.no_grad():
            new_gap = energy_gap_stabilized(g_worst, g_source, X_s, A_s, M=M_gap)
            lam += lr_dual * (new_gap - delta)
            lam.clamp_(min=0.0)

        if step % log_every == 0 or step == 1:
            print(f"[g-opt step {step:04d}] "
                  f"obj={obj.item():.4f}  gap={new_gap.item():.4f}  "
                  f"λ={lam.item():.4f}  L={lagr.item():.4f}")

    g_worst.model.eval()
    return g_worst


def _val_energy(eng, X_val, A_val):
    """Two-sample energy on a held-out split."""
    Xv = torch.as_tensor(X_val, dtype=torch.float32)
    Av = torch.as_tensor(A_val, dtype=torch.float32)
    with torch.no_grad():
        A1 = eng.model(Xv)                 # stochastic pass 1
        A2 = eng.model(Xv)                 # stochastic pass 2
        term1 = torch.norm(Av - A1, dim=1).mean()
        term2 = 0.5 * torch.norm(A1 - A2, dim=1).mean()
        return (term1 - term2).item()


def fit_engression_dro(
    X_source, A_source, X_target, f_hat,
    num_layer=2, hidden_dim=100, noise_dim=100, lr_gs=1e-4,
    lr_primal=1e-4, lr_dual=1e-3, num_epochs=500, batch_size=128, device="cpu",
    n_samples=128, delta=0.1,  # for robust predictor + DRO radius
    finetune_steps=400, finetune_lr=1e-4, finetune_L=64, 
    grad_clip = 1.0,
    verbose=True,
    val_frac=0.2
):
    
    N = X_source.shape[0]
    rng = np.random.RandomState(42)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_val = int(np.round(val_frac * N))
    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]

    X_tr = torch.as_tensor(X_source[tr_idx], dtype=torch.float32)
    A_tr = torch.as_tensor(A_source[tr_idx], dtype=torch.float32)
    X_va = X_source[val_idx]
    A_va = A_source[val_idx]

    g_source = engression(
        x=X_tr, y=A_tr,
        classification=False,
        num_layer=num_layer, hidden_dim=hidden_dim, noise_dim=noise_dim, out_act=None,
        add_bn=True, resblock=False, beta=0.8,
        lr=lr_gs, num_epochs=num_epochs, batch_size=batch_size,
        device=device, standardize=False, verbose=verbose
    )

    val_en = _val_energy(g_source, X_va, A_va)
    if verbose:
        print(f"[Engression] held-out validation energy: {val_en:.4f}  "
              f"(val_frac={val_frac}, epochs={num_epochs}, hd={hidden_dim}, noise_dim={noise_dim})")

    g_worst = find_worst_case_g_primal_dual_v2(
        g_source=g_source, 
        f_hat=f_hat, 
        X_source=X_source, 
        A_source=A_source, 
        X_target=X_target, 
        delta=delta, 
        num_steps=finetune_steps, 
        L=finetune_L, 
        lr_primal=lr_primal, 
        lr_dual=lr_dual, 
        grad_clip=grad_clip)

    @torch.no_grad()
    def m_robust(X):
        X = X if isinstance(X, torch.Tensor) else torch.as_tensor(X, dtype=torch.float32)
        N, dim_x = X.shape

        A = []
        for _ in range(n_samples):
            A.append(g_worst.model(X))          
        A = torch.stack(A, dim=1)               
        X_rep = X.unsqueeze(1).expand(-1, n_samples, -1).reshape(N*n_samples, dim_x)
        A_flat = A.reshape(N*n_samples, -1)
        y = f_hat(X_rep, A_flat).view(N, n_samples).mean(dim=1)
        return y 

    return m_robust, g_source, g_worst


def diagnose_dro_issue(g_source, g_worst, f_hat, X_source, A_source, X_target, delta):

    print("\n" + "="*80)
    print("DRO DIAGNOSTIC")
    print("="*80)
    
    X_s = torch.as_tensor(X_source, dtype=torch.float32)
    A_s = torch.as_tensor(A_source, dtype=torch.float32)
    X_t = torch.as_tensor(X_target, dtype=torch.float32)
    
    with torch.no_grad():
        A_worst_1 = g_worst.model(X_s)
        A_worst_2 = g_worst.model(X_s)
        term1_worst = torch.norm(A_s - A_worst_1, dim=1).mean()
        term2_worst = 0.5 * torch.norm(A_worst_1 - A_worst_2, dim=1).mean()
        en_worst = term1_worst - term2_worst
        
        A_source_1 = g_source.model(X_s)
        A_source_2 = g_source.model(X_s)
        term1_source = torch.norm(A_s - A_source_1, dim=1).mean()
        term2_source = 0.5 * torch.norm(A_source_1 - A_source_2, dim=1).mean()
        en_source = term1_source - term2_source
        
        gap = en_worst - en_source
        
        print(f"\nEnergy Distance Analysis:")
        print(f"  Energy(g_worst):  {en_worst.item():.6f}")
        print(f"  Energy(g_source): {en_source.item():.6f}")
        print(f"  Gap = Energy(worst) - Energy(source): {gap.item():.6f}")
        print(f"  Delta (constraint): {delta}")
        print(f"  Constraint violated? {gap.item() > delta} (gap - delta = {gap.item() - delta:.6f})")
        
        N_t, dim_x = X_t.shape
        L = 32 
        
        A_worst_samples = []
        for _ in range(L):
            A_worst_samples.append(g_worst.model(X_t))
        A_worst_stacked = torch.stack(A_worst_samples, dim=1)
        X_rep = X_t.unsqueeze(1).expand(-1, L, -1).reshape(N_t*L, dim_x)
        A_worst_flat = A_worst_stacked.reshape(N_t*L, -1)
        y_worst = f_hat(X_rep, A_worst_flat).view(N_t, L)
        obj_worst = (y_worst.mean(dim=1) ** 2).mean()
        
        print(f"\nObjective at g_worst:")
        print(f"  E_X[(E_eps f_hat(X, g_worst(X,eps)))^2] = {obj_worst.item():.6f}")
        
        A_source_samples = []
        for _ in range(L):
            A_source_samples.append(g_source.model(X_t))
        A_source_stacked = torch.stack(A_source_samples, dim=1)
        A_source_flat = A_source_stacked.reshape(N_t*L, -1)
        y_source = f_hat(X_rep, A_source_flat).view(N_t, L)
        obj_source = (y_source.mean(dim=1) ** 2).mean()
        
        print(f"\nObjective at g_source:")
        print(f"  E_X[(E_eps f_hat(X, g_source(X,eps)))^2] = {obj_source.item():.6f}")
        print(f"  Improvement: {obj_worst.item() - obj_source.item():.6f}")
        
        print(f"\nDistribution comparison:")
        A_worst_check = g_worst.model(X_t[:100])
        A_source_check = g_source.model(X_t[:100])
        
        print(f"  g_worst mean: {A_worst_check.mean(dim=0).numpy()}")
        print(f"  g_source mean: {A_source_check.mean(dim=0).numpy()}")
        print(f"  L2 distance between means: {torch.norm(A_worst_check.mean(dim=0) - A_source_check.mean(dim=0)).item():.6f}")
        
        print(f"\nIs g_worst different from g_source?")
        total_diff = 0
        for p_w, p_s in zip(g_worst.model.parameters(), g_source.model.parameters()):
            total_diff += torch.norm(p_w - p_s).item()
        print(f"  Total parameter difference: {total_diff:.6f}")
        if total_diff < 1e-5:
            print(f"  WARNING: g_worst is IDENTICAL to g_source!")
        
        return {
            'gap': gap.item(),
            'delta': delta,
            'constraint_violated': gap.item() > delta,
            'obj_worst': obj_worst.item(),
            'obj_source': obj_source.item(),
            'param_diff': total_diff
        }

def fit_engression_dro_penalty_grid(
    X_source, A_source, X_target, f_hat, lambda_grid, delta,
    num_layer=2, hidden_dim=100, noise_dim=100, lr_gs=1e-4,
    num_epochs=500, batch_size=128, val_frac=0.2,
    finetune_steps=400, lr_primal=1e-4, finetune_L=64, 
    grad_clip=1.0, M_gap=16,
    n_samples=128, device="cpu", verbose=True
):

    N = X_source.shape[0]
    rng = np.random.RandomState(42)
    idx = np.arange(N)
    n_val = int(np.round(val_frac * N))
    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]

    X_tr = torch.as_tensor(X_source[tr_idx], dtype=torch.float32, device=device)
    A_tr = torch.as_tensor(A_source[tr_idx], dtype=torch.float32, device=device)
    X_va = X_source[val_idx]
    A_va = A_source[val_idx]
    
    g_source = engression(
        x=X_tr, y=A_tr,
        classification=False,
        num_layer=num_layer, hidden_dim=hidden_dim, noise_dim=noise_dim, out_act=None,
        add_bn=True, resblock=False, beta=0.8, # finetuned
        lr=lr_gs, num_epochs=num_epochs, batch_size=batch_size,
        device=device, standardize=False, verbose=verbose
    )
    val_en = _val_energy(g_source, X_va, A_va)
    if verbose:
        print(f"[Engression] held-out validation energy: {val_en:.4f}")
    
    print("\n" + "="*80)
    print(f"Starting Penalty Method Grid Search (delta={delta})")
    print(f"Lambda grid: {lambda_grid}")
    print("="*80)

    best_g_worst = None
    best_obj = float('inf')
    best_lambda = None
    
    X_s_full = torch.as_tensor(X_source, dtype=torch.float32, device=device)
    A_s_full = torch.as_tensor(A_source, dtype=torch.float32, device=device)
    X_t_full = torch.as_tensor(X_target, dtype=torch.float32, device=device)

    for lam in lambda_grid:
        print(f"\n--- Training with fixed lambda = {lam} ---")
        
        current_g_worst = find_worst_case_g_penalty_method(
            g_source=g_source, f_hat=f_hat, 
            X_source=X_s_full, A_source=A_s_full, X_target=X_t_full,
            delta=delta,
            fixed_lambda=lam,
            num_steps=finetune_steps,
            L=finetune_L,
            lr_primal=lr_primal,
            device=device,
            grad_clip=grad_clip,
            M_gap=M_gap
        )

        with torch.no_grad():
            final_gap = energy_gap_paired(current_g_worst, g_source, X_s_full, A_s_full, M=M_gap)
            final_obj = target_objective(current_g_worst, f_hat, X_t_full, L=finetune_L)
        
        print(f"--- Result for lambda={lam}: Final obj={final_obj.item():.4f}, Final gap={final_gap.item():.4f}")

        if final_gap.item() <= delta:
            print(f"  Constraint SATISFIED (gap {final_gap.item():.4f} <= delta {delta})")
            if final_obj.item() < best_obj:
                print(f"  *** New BEST model found (obj {final_obj.item():.4f} < {best_obj:.4f}) ***")
                best_obj = final_obj.item()
                best_g_worst = current_g_worst
                best_lambda = lam
            else:
                print(f"  Constraint satisfied, but obj {final_obj.item():.4f} is not better than best {best_obj:.4f}")
        else:
            print(f"  Constraint FAILED (gap {final_gap.item():.4f} > delta {delta})")

    if best_g_worst is None:
        print("\n" + "!"*80)
        print("WARNING: No model satisfied the constraint. Returning g_source.")
        print("Try increasing lambda values in the grid or increasing finetune_steps.")
        print("!"*80)
        best_g_worst = g_source
    else:
        print("\n" + "="*80)
        print(f"Grid Search Complete. Best model found with lambda = {best_lambda}")
        print(f"Best final objective: {best_obj:.4f}")
        print("="*80)

    @torch.no_grad()
    def m_robust(X):
        X = X if isinstance(X, torch.Tensor) else torch.as_tensor(X, dtype=torch.float32)
        N, dim_x = X.shape

        A = []
        for _ in range(n_samples):
            A.append(best_g_worst.model(X))
        A = torch.stack(A, dim=1)
        X_rep = X.unsqueeze(1).expand(-1, n_samples, -1).reshape(N*n_samples, dim_x)
        A_flat = A.reshape(N*n_samples, -1)
        y = f_hat(X_rep, A_flat).view(N, n_samples).mean(dim=1)
        return y 

    return m_robust, g_source, best_g_worst


def cross_fit_f_w_g(source_df, target_df, target_loader, preliminary_generator, dim_x, dim_a, dim_epsilon, l_samples, EPOCHS_G, LR_G, EPOCHS_F, EPOCHS_OMEGA, LR_F, LR_Omega, 
hidden_dim = False,noise_dim= False,lr_gs= False,lr_primal= False,lr_dual=False,num_epochs_eng=False, delta=False,finetune_steps=False,
seed=42, local=False):

    kf = KFold(n_splits=3, shuffle=True, random_state=seed)
    folds = list(kf.split(source_df))
    (train_idx1, val_idx1), (train_idx2, val_idx2), (train_idx3, val_idx3) = folds

    Tkf = KFold(n_splits=3, shuffle=True, random_state=seed)
    Tfolds = list(Tkf.split(target_df))
    (Ttrain_idx1, Tval_idx1), (Ttrain_idx2, Tval_idx2), (Ttrain_idx3, Tval_idx3) = Tfolds

    out_of_sample_omegas = torch.zeros(len(source_df), 1)

    out_of_sample_residuals_preds = torch.zeros(len(source_df), 1)

    out_of_sample_f_hat_preds = torch.zeros(len(source_df), 1)

    X_s_full = torch.tensor(source_df.filter(regex='^x_').values, dtype=torch.float32)
    A_s_full = torch.tensor(source_df.filter(regex='^a_').values, dtype=torch.float32)
    Y_s_full = torch.tensor(source_df['y'].values, dtype=torch.float32).view(-1, 1)
    dim_x = X_s_full.shape[1]
    dim_a = A_s_full.shape[1]

    X_s_val, A_s_val, Y_s_val = X_s_full[train_idx1], A_s_full[train_idx1], Y_s_full[train_idx1]
    X_s_train, A_s_train, Y_s_train = X_s_full[val_idx1], A_s_full[val_idx1], Y_s_full[val_idx1]
    f_hat_fold_1 = F_hat_Network(dim_x, dim_a)
    train_dataset_fold = torch.utils.data.TensorDataset(X_s_train, A_s_train, Y_s_train)
    train_loader_fold = torch.utils.data.DataLoader(train_dataset_fold, batch_size=128, shuffle=True)
    trained_f_hat_1 = train_f_hat(f_hat_fold_1, train_loader_fold, EPOCHS_F, LR_F)
 
    with torch.no_grad():
        trained_f_hat_1.eval()
        f_hat_preds_val = trained_f_hat_1(X_s_full[val_idx2], A_s_full[val_idx2])
    out_of_sample_f_hat_preds[val_idx2]=f_hat_preds_val # store small data
    out_of_sample_residuals_preds[val_idx2]=Y_s_full[val_idx2]-f_hat_preds_val # store small data

    X_s_val, A_s_val, Y_s_val = X_s_full[train_idx2], A_s_full[train_idx2], Y_s_full[train_idx2]
    X_s_train, A_s_train, Y_s_train = X_s_full[val_idx2], A_s_full[val_idx2], Y_s_full[val_idx2]
    f_hat_fold_2 = F_hat_Network(dim_x, dim_a)
    train_dataset_fold = torch.utils.data.TensorDataset(X_s_train, A_s_train, Y_s_train)
    train_loader_fold = torch.utils.data.DataLoader(train_dataset_fold, batch_size=128, shuffle=True)
    trained_f_hat_2 = train_f_hat(f_hat_fold_2, train_loader_fold, EPOCHS_F, LR_F)

    with torch.no_grad():
        trained_f_hat_2.eval()
        f_hat_preds_val = trained_f_hat_2(X_s_full[val_idx3], A_s_full[val_idx3])
    out_of_sample_f_hat_preds[val_idx3]=f_hat_preds_val
    out_of_sample_residuals_preds[val_idx3]=Y_s_full[val_idx3]-f_hat_preds_val

    X_s_val, A_s_val, Y_s_val = X_s_full[train_idx3], A_s_full[train_idx3], Y_s_full[train_idx3]
    X_s_train, A_s_train, Y_s_train = X_s_full[val_idx3], A_s_full[val_idx3], Y_s_full[val_idx3]
    f_hat_fold_3 = F_hat_Network(dim_x, dim_a)
    train_dataset_fold = torch.utils.data.TensorDataset(X_s_train, A_s_train, Y_s_train)
    train_loader_fold = torch.utils.data.DataLoader(train_dataset_fold, batch_size=128, shuffle=True)
    trained_f_hat_3 = train_f_hat(f_hat_fold_3, train_loader_fold, EPOCHS_F, LR_F)

    with torch.no_grad():
        trained_f_hat_3.eval()
        f_hat_preds_val = trained_f_hat_3(X_s_full[val_idx1], A_s_full[val_idx1])
    out_of_sample_f_hat_preds[val_idx1]=f_hat_preds_val
    out_of_sample_residuals_preds[val_idx1]=Y_s_full[val_idx1]-f_hat_preds_val

    X_s_val, A_s_val, Y_s_val = X_s_full[train_idx1], A_s_full[train_idx1], Y_s_full[train_idx1]
    X_s_train, A_s_train, Y_s_train = X_s_full[val_idx1], A_s_full[val_idx1], Y_s_full[val_idx1]
    X_t_temp = torch.tensor(target_df.filter(regex='^x_').values, dtype=torch.float32)
    X_t_val=X_t_temp[Ttrain_idx1]
    X_t_train=X_t_temp[Tval_idx1]

    def generate_A(generator, X_tensor, local):
        
        with torch.no_grad():
            if local:
                A = generator.model(X_tensor)
            else:
                A = generator(torch.randn(len(X_tensor), dim_epsilon))
        return A
    
    A_t = generate_A(preliminary_generator,X_t_train,local)
    X_combined = torch.cat([X_s_train, X_t_train], dim=0)
    A_combined = torch.cat([A_s_train, A_t], dim=0)
    labels_combined = torch.cat([torch.ones(len(X_s_train), 1), torch.zeros(len(X_t_train), 1)], dim=0)

    omega_classifier_fold1 = Omega_Classifier(dim_x, dim_a)
    optimizer_o1 = optim.Adam(omega_classifier_fold1.parameters(), lr=LR_Omega)
    loss_fn_o1 = nn.BCELoss()
    for _ in range(EPOCHS_OMEGA): 
        optimizer_o1.zero_grad()
        p_preds = omega_classifier_fold1(X_combined, A_combined)
        loss_o = loss_fn_o1(p_preds, labels_combined)
        loss_o.backward()
        optimizer_o1.step()

    with torch.no_grad():
        omega_classifier_fold1.eval()
        p_val = omega_classifier_fold1(X_s_full[val_idx2], A_s_full[val_idx2])
        omega_weights = ((1 - p_val) / (p_val + 1e-8))
        omega_weights = omega_weights / omega_weights.mean()
        out_of_sample_omegas[val_idx2] = omega_weights

    X_s_val, A_s_val, Y_s_val = X_s_full[train_idx2], A_s_full[train_idx2], Y_s_full[train_idx2]
    X_s_train, A_s_train, Y_s_train = X_s_full[val_idx2], A_s_full[val_idx2], Y_s_full[val_idx2]
    X_t_temp = torch.tensor(target_df.filter(regex='^x_').values, dtype=torch.float32)
    X_t_val=X_t_temp[Ttrain_idx2]
    X_t_train=X_t_temp[Tval_idx2]

    A_t = generate_A(preliminary_generator,X_t_train,local)
    X_combined = torch.cat([X_s_train, X_t_train], dim=0)
    A_combined = torch.cat([A_s_train, A_t], dim=0)
    labels_combined = torch.cat([torch.ones(len(X_s_train), 1), torch.zeros(len(X_t_train), 1)], dim=0)

    omega_classifier_fold2 = Omega_Classifier(dim_x, dim_a)
    optimizer_o2 = optim.Adam(omega_classifier_fold2.parameters(), lr=LR_Omega)
    loss_fn_o2 = nn.BCELoss() #Binary Cross-Entropy Loss
    for _ in range(EPOCHS_OMEGA):
        optimizer_o2.zero_grad()
        p_preds = omega_classifier_fold2(X_combined, A_combined)
        loss_o = loss_fn_o2(p_preds, labels_combined)
        loss_o.backward()
        optimizer_o2.step()

    with torch.no_grad():
        omega_classifier_fold2.eval()
        p_val = omega_classifier_fold2(X_s_full[val_idx3], A_s_full[val_idx3])
        omega_weights = ((1 - p_val) / (p_val + 1e-8))
        omega_weights = omega_weights / omega_weights.mean()
        out_of_sample_omegas[val_idx3] = omega_weights
    
    X_s_val, A_s_val, Y_s_val = X_s_full[train_idx3], A_s_full[train_idx3], Y_s_full[train_idx3]
    X_s_train, A_s_train, Y_s_train = X_s_full[val_idx3], A_s_full[val_idx3], Y_s_full[val_idx3]
    X_t_temp = torch.tensor(target_df.filter(regex='^x_').values, dtype=torch.float32)
    X_t_val=X_t_temp[Ttrain_idx3]
    X_t_train=X_t_temp[Tval_idx3]

    A_t = generate_A(preliminary_generator,X_t_train,local)
    X_combined = torch.cat([X_s_train, X_t_train], dim=0)
    A_combined = torch.cat([A_s_train, A_t], dim=0)
    labels_combined = torch.cat([torch.ones(len(X_s_train), 1), torch.zeros(len(X_t_train), 1)], dim=0)

    omega_classifier_fold3 = Omega_Classifier(dim_x, dim_a)
    optimizer_o3 = optim.Adam(omega_classifier_fold3.parameters(), lr=LR_Omega)
    loss_fn_o3 = nn.BCELoss() 
    for _ in range( EPOCHS_OMEGA): 
        optimizer_o3.zero_grad()
        p_preds = omega_classifier_fold3(X_combined, A_combined)
        loss_o = loss_fn_o3(p_preds, labels_combined)
        loss_o.backward()
        optimizer_o3.step()

    with torch.no_grad():
        omega_classifier_fold3.eval()
        p_val = omega_classifier_fold3(X_s_full[val_idx1], A_s_full[val_idx1]) 
        omega_weights = ((1 - p_val) / (p_val + 1e-8))
        omega_weights = omega_weights / omega_weights.mean()
        out_of_sample_omegas[val_idx1] = omega_weights

    fhat_temp = out_of_sample_f_hat_preds[val_idx1]
    resi_temp = out_of_sample_residuals_preds[val_idx1]
    omegas_temp = out_of_sample_omegas[val_idx1]

    X_s_val, A_s_val, Y_s_val = X_s_full[train_idx1], A_s_full[train_idx1], Y_s_full[train_idx1]
    X_s_train, A_s_train, Y_s_train = X_s_full[val_idx1], A_s_full[val_idx1], Y_s_full[val_idx1]
    train_dataset_fold = torch.utils.data.TensorDataset(X_s_train, A_s_train, Y_s_train)
    train_loader_fold = torch.utils.data.DataLoader(train_dataset_fold, batch_size=128, shuffle=True)
    X_t_temp = torch.tensor(target_df.filter(regex='^x_').values, dtype=torch.float32)
    X_t_val1=X_t_temp[Ttrain_idx1]
    X_t_train1=X_t_temp[Tval_idx1]

    if local:
        _,_,trained_generator_debiased_1 = fit_engression_dro_debiased(
        X_source=X_s_train.numpy(),
        A_source=A_s_train.numpy(),
        X_target=X_t_train1.numpy(),
        f_hat=trained_f_hat_2,          
        cross_fit_omegas=omegas_temp,    
        cross_fit_residuals=resi_temp, 
        pretrained_g_source=preliminary_generator, 
        delta=delta,
        num_layer=2, hidden_dim=hidden_dim, noise_dim=noise_dim,
        lr_gs=lr_gs, lr_primal=lr_primal, lr_dual=lr_dual,
        num_epochs=num_epochs_eng, 
        finetune_steps=finetune_steps,
        finetune_lr=1e-4,
        finetune_L=64,
        grad_clip=1.0,
        n_samples=128,
        verbose=True
    )
    else:
        generator_1 = Generator_Network(dim_epsilon, dim_a)
        trained_generator_debiased_1 = train_debiased_generator(generator_1,omegas_temp,resi_temp,trained_f_hat_2,X_s_train,X_t_train1,EPOCHS_G,LR_G,dim_a,dim_epsilon,l_samples)
    
    fhat_temp = out_of_sample_f_hat_preds[val_idx2]
    resi_temp = out_of_sample_residuals_preds[val_idx2]
    omegas_temp = out_of_sample_omegas[val_idx2]

    X_s_val, A_s_val, Y_s_val = X_s_full[train_idx2], A_s_full[train_idx2], Y_s_full[train_idx2]
    X_s_train, A_s_train, Y_s_train = X_s_full[val_idx2], A_s_full[val_idx2], Y_s_full[val_idx2]
    train_dataset_fold = torch.utils.data.TensorDataset(X_s_train, A_s_train, Y_s_train)
    train_loader_fold = torch.utils.data.DataLoader(train_dataset_fold, batch_size=128, shuffle=True)
    X_t_temp = torch.tensor(target_df.filter(regex='^x_').values, dtype=torch.float32)
    X_t_val2=X_t_temp[Ttrain_idx2]
    X_t_train2=X_t_temp[Tval_idx2]

    if local:
        _,_,trained_generator_debiased_2 = fit_engression_dro_debiased(
        X_source=X_s_train.numpy(),
        A_source=A_s_train.numpy(),
        X_target=X_t_train2.numpy(),
        f_hat=trained_f_hat_3,           
        cross_fit_omegas=omegas_temp,    
        cross_fit_residuals=resi_temp,  
        pretrained_g_source=preliminary_generator, 
        delta=delta,
        num_layer=2, hidden_dim=hidden_dim, noise_dim=noise_dim,
        lr_gs=lr_gs, lr_primal=lr_primal, lr_dual=lr_dual,
        num_epochs=num_epochs_eng, 
        finetune_steps=finetune_steps,
        finetune_lr=1e-4,
        finetune_L=64,
        grad_clip=1.0,
        n_samples=128,
        verbose=True
    )
    else:
        generator_2 = Generator_Network(dim_epsilon, dim_a)
        trained_generator_debiased_2 = train_debiased_generator(generator_2,omegas_temp,resi_temp,trained_f_hat_3,X_s_train,X_t_train2,EPOCHS_G,LR_G,dim_a,dim_epsilon,l_samples)

    fhat_temp = out_of_sample_f_hat_preds[val_idx3]
    resi_temp = out_of_sample_residuals_preds[val_idx3]
    omegas_temp = out_of_sample_omegas[val_idx3]

    X_s_val, A_s_val, Y_s_val = X_s_full[train_idx3], A_s_full[train_idx3], Y_s_full[train_idx3]
    X_s_train, A_s_train, Y_s_train = X_s_full[val_idx3], A_s_full[val_idx3], Y_s_full[val_idx3]
    train_dataset_fold = torch.utils.data.TensorDataset(X_s_train, A_s_train, Y_s_train)
    train_loader_fold = torch.utils.data.DataLoader(train_dataset_fold, batch_size=128, shuffle=True)
    X_t_temp = torch.tensor(target_df.filter(regex='^x_').values, dtype=torch.float32)
    X_t_val3=X_t_temp[Ttrain_idx3]
    X_t_train3=X_t_temp[Tval_idx3]

    if local:
        _,_,trained_generator_debiased_3 = fit_engression_dro_debiased(
        X_source=X_s_train.numpy(),
        A_source=A_s_train.numpy(),
        X_target=X_t_train3.numpy(),
        f_hat=trained_f_hat_1,           
        cross_fit_omegas=omegas_temp,   
        cross_fit_residuals=resi_temp,
        pretrained_g_source=preliminary_generator, 
        delta=delta,
        num_layer=2, hidden_dim=hidden_dim, noise_dim=noise_dim,
        lr_gs=lr_gs, lr_primal=lr_primal, lr_dual=lr_dual,
        num_epochs=num_epochs_eng, 
        finetune_steps=finetune_steps,
        finetune_lr=1e-4,
        finetune_L=64,
        grad_clip=1.0,
        n_samples=128,
        verbose=True
    )
    else:
        generator_3 = Generator_Network(dim_epsilon, dim_a)
        trained_generator_debiased_3 = train_debiased_generator(generator_3,omegas_temp,resi_temp,trained_f_hat_1,X_s_train,X_t_train3,EPOCHS_G,LR_G,dim_a,dim_epsilon,l_samples)
    
    with torch.no_grad():
        if local:
            trained_generator_debiased_2.model.eval()
            X_t_subset = X_t_temp[Tval_idx1]
            A_t_samples = torch.stack([trained_generator_debiased_2.model(X_t_subset) for _ in range(l_samples)], dim=1)
            A_t1 = torch.mean(A_t_samples, dim=1)
        else:
            trained_generator_debiased_2.eval()
            epsilon = torch.randn(len(X_t_temp[Tval_idx1]), l_samples, dim_epsilon)
            A_t_samples = trained_generator_debiased_2(epsilon)
            A_t1 = torch.mean(A_t_samples, dim=1)
    
    with torch.no_grad():
        if local:
            trained_generator_debiased_3.model.eval()
            X_t_subset = X_t_temp[Tval_idx2]
            A_t_samples = torch.stack([trained_generator_debiased_3.model(X_t_subset) for _ in range(l_samples)], dim=1)
            A_t2 = torch.mean(A_t_samples, dim=1)
        else:
            trained_generator_debiased_3.eval()
            epsilon = torch.randn(len(X_t_temp[Tval_idx2]), l_samples, dim_epsilon)
            A_t_samples = trained_generator_debiased_3(epsilon)
            A_t2 = torch.mean(A_t_samples, dim=1)
    
    with torch.no_grad():
        if local:
            trained_generator_debiased_1.model.eval()
            X_t_subset = X_t_temp[Tval_idx3]
            A_t_samples = torch.stack([trained_generator_debiased_1.model(X_t_subset) for _ in range(l_samples)], dim=1)
            A_t3 = torch.mean(A_t_samples, dim=1)
        else:
            trained_generator_debiased_1.eval()
            epsilon = torch.randn(len(X_t_temp[Tval_idx3]), l_samples, dim_epsilon)
            A_t_samples = trained_generator_debiased_1(epsilon)
            A_t3 = torch.mean(A_t_samples, dim=1)

    n_source = len(source_df)
    n_target = len(target_df)
    rho = n_source / n_target

    X_s = torch.tensor(source_df.filter(regex='^x_').values, dtype=torch.float32)
    A_s = torch.tensor(source_df.filter(regex='^a_').values, dtype=torch.float32)
    Y_s = torch.tensor(source_df['y'].values, dtype=torch.float32).view(-1, 1)
    X_t = torch.tensor(target_df.filter(regex='^x_').values, dtype=torch.float32)

    S_pooled = torch.cat([torch.ones(n_source, 1), torch.zeros(n_target, 1)], dim=0)
    X_pooled = torch.cat([X_s, X_t], dim=0)
    A_pooled = torch.cat([A_s, A_t], dim=0)
    Y_pooled = torch.cat([Y_s, torch.zeros(n_target, 1)], dim=0)

    residuals_pooled = torch.cat([out_of_sample_residuals_preds, torch.zeros(n_target, 1)], dim=0)

    omegas = np.array(out_of_sample_omegas, dtype=float)
    omega_scaled = (omegas - omegas.min()) / (omegas.max() - omegas.min())
    omega_scaled = torch.tensor(omega_scaled, dtype=torch.float32)

    omega_pooled = torch.cat([omega_scaled, torch.zeros(n_target, 1)], dim=0)

    out_of_sample_f_hat_preds = torch.zeros(n_target, 1)

    with torch.no_grad():
        out_of_sample_f_hat_preds[Tval_idx1] = trained_f_hat_3(X_t[Tval_idx1], A_t1) 
        out_of_sample_f_hat_preds[Tval_idx2] = trained_f_hat_1(X_t[Tval_idx2], A_t2) 
        out_of_sample_f_hat_preds[Tval_idx3] = trained_f_hat_2(X_t[Tval_idx3], A_t3)

        f_hat_pooled = torch.cat([torch.zeros(n_source, 1), out_of_sample_f_hat_preds], dim=0)
        F = S_pooled * omega_pooled * residuals_pooled + rho * (1 - S_pooled) * f_hat_pooled

    return F


def train_Fx_estimator(F_hat, X_pooled, epochs, lr, batch_size):

    print("\n--- Training Final Debiased Estimator ---")
    dim_x = X_pooled.shape[1]
   
    class FinalEstimator(torch.nn.Module):
        def __init__(self, dim_x):
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(dim_x, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1)
            )
        
        def forward(self, x):
            return self.network(x)
    
    final_estimator = FinalEstimator(dim_x)
    
    if not isinstance(X_pooled, torch.Tensor):
        X_pooled = torch.tensor(X_pooled, dtype=torch.float32)
    if not isinstance(F_hat, torch.Tensor):
        F_hat = torch.tensor(F_hat, dtype=torch.float32)
    if len(F_hat.shape) == 1:
        F_hat = F_hat.unsqueeze(1)
    
    dataset = torch.utils.data.TensorDataset(X_pooled, F_hat)
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    optimizer = torch.optim.Adam(final_estimator.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    final_estimator.train()
    
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        
        for x_batch, f_hat_batch in data_loader:

            f_pred = final_estimator(x_batch)
            loss = loss_fn(f_pred, f_hat_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1}/{epochs}], MSE Loss: {avg_loss:.4f}")
    
    print("--- Final Debiased Estimator Training Complete ---")
    
    final_estimator.eval()
    
    return final_estimator


def train_debiased_generator(generator_model, cross_fit_omegas, cross_fit_residuals, f_hat_model,   X_s_tensor, X_t_tensor, epochs, lr, dim_a, dim_epsilon, l_samples, batch_size=64):

    print("\n--- Training Debiased Generator ---")
    optimizer = optim.Adam(generator_model.parameters(), lr=lr)

    f_hat_model.eval()
    generator_model.train()
    for param in f_hat_model.parameters(): 
        param.requires_grad = False
    
    source_dataset_cf = torch.utils.data.TensorDataset(X_s_tensor,cross_fit_omegas,cross_fit_residuals)
    source_loader_cf = torch.utils.data.DataLoader(source_dataset_cf, batch_size=batch_size,shuffle=True)
    target_loader = torch.utils.data.DataLoader(X_t_tensor, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        for (source_batch_cf, target_batch) in zip(source_loader_cf, target_loader):
            x_s, omega_batch, residual_batch = source_batch_cf
            x_t = target_batch
            
            inner_exp_t = torch.mean(f_hat_model(
                x_t.unsqueeze(1).expand(-1, l_samples, -1).reshape(-1, x_t.shape[-1]), 
                generator_model(torch.randn(x_t.shape[0], l_samples, dim_epsilon)).reshape(-1, dim_a)
            ).view(x_t.shape[0], l_samples), dim=1)
            original_loss = torch.mean(inner_exp_t**2)
            
            with torch.no_grad(): # g_bar is fixed!
                inner_exp_s = torch.mean(f_hat_model(
                    x_s.unsqueeze(1).expand(-1, l_samples, -1).reshape(-1, x_s.shape[-1]), 
                    generator_model(torch.randn(x_s.shape[0], l_samples, dim_epsilon)).reshape(-1, dim_a)
                ).view(x_s.shape[0], l_samples), dim=1)
            
            correction_term = 2 * torch.mean(omega_batch * inner_exp_s.view(-1, 1) * residual_batch)
            total_loss = original_loss + correction_term

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch [{epoch+1}/{epochs}], Total loss: {total_loss:.4f}")
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    print(f"GLOBAL - Final original_loss: {original_loss.item():.4f}, correction: {correction_term.item():.4f}")

    print("Debiased generator trained.")
    return generator_model


def fit_engression_dro_debiased(
    X_source, A_source, X_target, f_hat,
    cross_fit_omegas,
    cross_fit_residuals,
    pretrained_g_source=None,
    num_layer=2, hidden_dim=100, noise_dim=100, lr_gs=1e-4,
    lr_primal=1e-4, lr_dual=1e-3, num_epochs=500, batch_size=128, device="cpu",
    n_samples=128, delta=0.1,
    finetune_steps=400, finetune_lr=1e-4, finetune_L=64, 
    grad_clip=1.0,
    verbose=True
):

    X_s = torch.as_tensor(X_source, dtype=torch.float32, device=device)
    A_s = torch.as_tensor(A_source, dtype=torch.float32, device=device)
    X_t = torch.as_tensor(X_target, dtype=torch.float32, device=device)

    if pretrained_g_source is not None:
        g_source = copy.deepcopy(pretrained_g_source)
        if verbose:
            print("Using pre-trained g_source")
    else:
        if verbose:
            print("\n--- Fitting g_source via engression ---")
            g_source = engression(
                x=X_s, y=A_s,
                classification=False,
                num_layer=num_layer, hidden_dim=hidden_dim, noise_dim=noise_dim, out_act=None,
                add_bn=True, resblock=False, beta=0.8,
                lr=lr_gs, num_epochs=num_epochs, batch_size=batch_size,
                device=device, standardize=False, verbose=verbose
            )

    if verbose:
        print("\n--- Finding g_worst with debiased objective ---")
    
    g_worst = find_worst_case_g_debiased(
        g_source=g_source, 
        f_hat=f_hat, 
        X_source=X_source, 
        A_source=A_source, 
        X_target=X_target,
        cross_fit_omegas=cross_fit_omegas,
        cross_fit_residuals=cross_fit_residuals,
        delta=delta, 
        num_steps=finetune_steps, 
        L=finetune_L, 
        lr_primal=lr_primal, 
        lr_dual=lr_dual, 
        grad_clip=grad_clip,
        device=device,
        verbose=verbose
    )

    @torch.no_grad()
    def m_robust(X):
        X = X if isinstance(X, torch.Tensor) else torch.as_tensor(X, dtype=torch.float32)
        N, dim_x = X.shape
        A = []
        for _ in range(n_samples):
            A.append(g_worst.model(X))
        A = torch.stack(A, dim=1)
        X_rep = X.unsqueeze(1).expand(-1, n_samples, -1).reshape(N*n_samples, dim_x)
        A_flat = A.reshape(N*n_samples, -1)
        y = f_hat(X_rep, A_flat).view(N, n_samples).mean(dim=1)
        return y 

    return m_robust, g_source, g_worst


def find_worst_case_g_debiased(
    g_source, f_hat, 
    X_source, A_source, X_target,
    cross_fit_omegas,
    cross_fit_residuals,
    delta,
    num_steps=400,
    L=64,
    lr_primal=1e-4,
    lr_dual=1e-4,
    device="cpu",
    grad_clip=1.0,
    M_gap=16,
    log_every=20,
    verbose=True
):

    g_worst = copy.deepcopy(g_source)
    g_worst.model.train()
    f_hat.eval()

    X_s = torch.as_tensor(X_source, dtype=torch.float32, device=device)
    A_s = torch.as_tensor(A_source, dtype=torch.float32, device=device)
    X_t = torch.as_tensor(X_target, dtype=torch.float32, device=device)
    omegas = torch.as_tensor(cross_fit_omegas, dtype=torch.float32, device=device).view(-1)
    residuals = torch.as_tensor(cross_fit_residuals, dtype=torch.float32, device=device).view(-1)

    opt_g = torch.optim.Adam(g_worst.model.parameters(), lr=lr_primal)
    lam = 1.0
    
    N_s, dim_x = X_s.shape
    N_t = X_t.shape[0]
    dim_a = A_s.shape[1]

    for step in range(1, num_steps + 1):
        opt_g.zero_grad()

        A_t_samples = torch.stack([g_worst.model(X_t) for _ in range(L)], dim=1)
        X_t_rep = X_t.unsqueeze(1).expand(-1, L, -1).reshape(N_t * L, dim_x)
        A_t_flat = A_t_samples.reshape(N_t * L, dim_a)
        inner_exp_t = f_hat(X_t_rep, A_t_flat).view(N_t, L).mean(dim=1)
        loss_original = (inner_exp_t ** 2).mean()
        
        with torch.no_grad():
            A_s_samples = torch.stack([g_worst.model(X_s) for _ in range(L)], dim=1)
            X_s_rep = X_s.unsqueeze(1).expand(-1, L, -1).reshape(N_s * L, dim_x)
            A_s_flat = A_s_samples.reshape(N_s * L, dim_a)
            inner_exp_s = f_hat(X_s_rep, A_s_flat).view(N_s, L).mean(dim=1)
        
        loss_correction = 2 * (omegas * inner_exp_s * residuals).mean()
        loss_debiased = loss_original + loss_correction
        
        gap = energy_gap_stabilized(g_worst, g_source, X_s, A_s, M=M_gap)

        lagrangian = loss_debiased + lam * (gap - delta)
        lagrangian.backward()
        
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(g_worst.model.parameters(), grad_clip)
        opt_g.step()

        with torch.no_grad():
            new_gap = energy_gap_stabilized(g_worst, g_source, X_s, A_s, M=M_gap)
            lam = lam + lr_dual * (new_gap - delta)
            lam = max(0.0, lam) 

        if verbose and (step % log_every == 0 or step == 1):
            print(f"  [step {step:04d}] loss={loss_debiased.item():.4f} "
                  f"(orig={loss_original.item():.4f}, corr={loss_correction.item():.4f}) "
                  f"gap={new_gap.item():.4f} λ={lam.item():.4f}")

    print(f"LOCAL - Final original_loss: {loss_original.item():.4f}, correction: {loss_correction.item():.4f}")


    g_worst.model.eval()
    return g_worst

def predict_final_debiased(x_data, final_estimator):

    final_estimator.eval()
    with torch.no_grad():
        predictions = final_estimator(x_data)
    return predictions.numpy()

## Adapted from https://github.com/namkoong-lab/robustopt
def project_onto_chi_square_ball(w, rho, tol=1e-10):
    assert rho > 0
    rho = float(rho)
    
    w_sort = np.sort(w)[::-1] 
    w_sort_cumsum = w_sort.cumsum()
    w_sort_sqr_cumsum = np.square(w_sort).cumsum()
    nn = float(w_sort.shape[0])

    def solve_inner_eta(w_s, w_sc, n, lam):
        fs = w_s - (w_sc - (1. + lam * n)) / (np.arange(n) + 1.)
        ind = (fs > 0).sum() - 1
        if ind < 0:
            return (w_sc[-1] - (1. + lam * n)) / n, n - 1
        return (1 / (ind + 1.)) * (w_sc[ind] - (1. + lam * n)), ind

    lam_min = 0.0
   
    lam_max = max(0.0, (1/nn) * (nn * w_sort[0] / np.sqrt(2. * rho + 1.) - 1.))
    lam_init_max = lam_max if lam_max > 0 else 1.0

    if lam_max <= 0:
        eta, _ = solve_inner_eta(w_sort, w_sort_cumsum, nn, 0.)
        p = w - eta
        p[p < 0] = 0.
        return p

    for _ in range(100): 
        lam = 0.5 * (lam_max + lam_min)
        if lam == lam_min or lam == lam_max: break
        eta, ind = solve_inner_eta(w_sort, w_sort_cumsum, nn, lam)
        
        if ind < 0: 
             thresh = 0
        else:
             thresh = 0.5 * nn * (w_sort_sqr_cumsum[ind] - 2. * eta * w_sort_cumsum[ind] + eta**2 * (ind + 1.))

        if thresh > (rho + 0.5) * (1 + lam * nn)**2:
            lam_min = lam
        else:
            lam_max = lam

    lam = 0.5 * (lam_max + lam_min)
    eta, _ = solve_inner_eta(w_sort, w_sort_cumsum, nn, lam)
    p = w - eta
    p[p < 0] = 0
    return (1. / (1. + lam * nn)) * p

def train_dro_model(model, train_loader, val_loader, epochs, lr, rho):

    print(f"\n--- Training DRO Baseline (rho={rho}) ---")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss(reduction='none')

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for x_batch, a_batch, y_batch in train_loader:
            y_pred = model(x_batch)
            
            per_sample_losses = loss_fn(y_pred, y_batch.view(-1, 1))
            
            w = per_sample_losses.detach().numpy().flatten()
            p_worst_case = project_onto_chi_square_ball(w, rho)
            p_worst_case_t = torch.tensor(p_worst_case, dtype=torch.float32)

            robust_loss = torch.dot(p_worst_case_t, per_sample_losses.flatten())
            
            optimizer.zero_grad()
            robust_loss.backward()
            optimizer.step()
            total_train_loss += robust_loss.item()
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_batch, a_batch, y_batch in val_loader:
                y_pred = model(x_batch)
                val_loss = torch.nn.functional.mse_loss(y_pred, y_batch.view(-1, 1))
                total_val_loss += val_loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1}/{epochs}], Robust Train Loss: {avg_train_loss:.4f}, Val MSE: {avg_val_loss:.4f}")
            
    print("--- DRO Training Complete ---")
    return model

class F_hat_Network(nn.Module):
    def __init__(self, dim_x, dim_a):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_x + dim_a, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x, a):
        xa_input = torch.cat([x, a], dim=1)
        return self.network(xa_input)

class Generator_Network(nn.Module):

    def __init__(self, dim_epsilon, dim_a):
        super().__init__()
        self.dim_epsilon = dim_epsilon
        self.dim_a = dim_a
        self.target_mean = 0.0
        self.target_std = 3.0
        
        self.network = nn.Sequential(
            nn.Linear(dim_epsilon, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, dim_a)
        )
    
    def forward(self, epsilon):
        output = self.network(epsilon)

        batch_mean = output.mean(dim=0, keepdim=True).detach()
        batch_std = output.std(dim=0, keepdim=True).detach() + 1e-6
    
        standardized = (output - batch_mean) / batch_std
        final_output = self.target_mean + standardized * self.target_std
        
        return final_output

class Omega_Classifier(nn.Module):

    def __init__(self, dim_x, dim_a):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_x + dim_a, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x, a):
        xa_input = torch.cat([x, a], dim=1)
        return self.network(xa_input)
        
def train_f_hat(f_hat_model, data_loader, epochs, lr):

    print("\n--- Train f_hat network ---")
    optimizer = optim.Adam(f_hat_model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    f_hat_model.train()

    for epoch in range(epochs):
        total_loss = 0
        for x_batch, a_batch, y_batch in data_loader:

            y_pred = f_hat_model(x_batch, a_batch)
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1}/{epochs}], MSE Loss: {avg_loss:.4f}")
    
    print("--- f_hat network Complete ---")
    return f_hat_model

def train_f_hat_new(f_hat_model, train_loader, val_loader, epochs, lr):
    print("\n--- Train f_hat network ---")
    optimizer = optim.Adam(f_hat_model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    for epoch in range(epochs):
        f_hat_model.train()
        total_train_loss = 0
        for x_batch, a_batch, y_batch in train_loader:
            y_pred = f_hat_model(x_batch, a_batch)
            loss = loss_fn(y_pred, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()

        f_hat_model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_batch, a_batch, y_batch in val_loader:
                y_pred = f_hat_model(x_batch, a_batch)
                loss = loss_fn(y_pred, y_batch)
                total_val_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    print("--- f_hat network Complete ---")
    return f_hat_model

def train_generator(generator_model, f_hat_model, data_loader, epochs, lr, dim_x, dim_a, dim_epsilon, l_samples):

    print("\n--- Train g network ---")
    optimizer = optim.Adam(generator_model.parameters(), lr=lr)
    
    # Freeze the f_hat network's weights
    f_hat_model.eval()
    for param in f_hat_model.parameters():
        param.requires_grad = False
        
    generator_model.train()

    for epoch in range(epochs):
        total_loss_g = 0
        for x_batch in data_loader:
            batch_size = x_batch.shape[0]
            
            epsilon = torch.randn(batch_size, l_samples, dim_epsilon)
            generated_a = generator_model(epsilon)
  
            repeated_x = x_batch.unsqueeze(1).expand(-1, l_samples, -1)
            
            flat_x = repeated_x.reshape(-1, dim_x)
            flat_a = generated_a.reshape(-1, dim_a)
            
            f_hat_preds = f_hat_model(flat_x, flat_a)
            
            preds_reshaped = f_hat_preds.view(batch_size, l_samples)
            inner_expectation = torch.mean(preds_reshaped, dim=1)
            
            loss_main = torch.mean(inner_expectation**2)

            gen_mean = generated_a.mean()
            gen_std = generated_a.std()
            mean_penalty = 0.001 * gen_mean**2  
            std_penalty = 0.001 * (gen_std - 3.0)**2 
            
            max_penalty = 0.01 * torch.relu(generated_a.abs() - 10).mean()
            loss = loss_main + mean_penalty + std_penalty + max_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss_g += loss.item()

        avg_loss_g = total_loss_g / len(data_loader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1}/{epochs}], Generator Loss: {avg_loss_g:.4f}")
            
    print("--- g network Complete ---")
    return generator_model

class Baseline_NN(torch.nn.Module):

    def __init__(self, dim_x):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(dim_x, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.network(x)

def train_baseline_model(baseline_model, data_loader, epochs, lr):
    """Trains the baseline NN."""
    print("\n--- Training Baseline NN ---")
    optimizer = torch.optim.Adam(baseline_model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    baseline_model.train()

    for epoch in range(epochs):
        total_loss = 0
        for x_batch, _, y_batch in data_loader:
            y_pred = baseline_model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"  Baseline Epoch [{epoch+1}/{epochs}], MSE Loss: {total_loss / len(data_loader):.4f}")
    print("--- Baseline Training Complete ---")
    return baseline_model


def train_baseline_model_new(baseline_model, train_loader, val_loader, epochs, lr):
    """Trains the baseline NN."""
    print("\n--- Training Baseline NN ---")
    optimizer = torch.optim.Adam(baseline_model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    baseline_model.train()

    for epoch in range(epochs):

        baseline_model.train()
        total_train_loss = 0
        for x_batch, _, y_batch in train_loader: 
            y_pred = baseline_model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        baseline_model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_batch, _, y_batch in val_loader: 
                y_pred = baseline_model(x_batch)
                loss = loss_fn(y_pred, y_batch)
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        if (epoch + 1) % 10 == 0:
            print(f"  Baseline Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    print("--- Baseline Training Complete ---")
    return baseline_model

def predict_m_star(x_data, f_hat_model, generator_model, dim_a, dim_epsilon, l_samples):
  
    f_hat_model.eval()
    generator_model.eval()
    
    with torch.no_grad():
        num_samples = x_data.shape[0]
        epsilon = torch.randn(num_samples, l_samples, dim_epsilon)
        generated_a = generator_model(epsilon)
        repeated_x = x_data.unsqueeze(1).expand(-1, l_samples, -1)
        flat_x = repeated_x.reshape(-1, x_data.shape[1])
        flat_a = generated_a.reshape(-1, dim_a)
        f_hat_preds = f_hat_model(flat_x, flat_a)
        preds_reshaped = f_hat_preds.view(num_samples, l_samples)
        final_predictions = torch.mean(preds_reshaped, dim=1)

    return final_predictions.numpy()

    f_hat_model.eval()
    generator_model.eval()
    
    with torch.no_grad():
        num_samples = x_data.shape[0]
        epsilon = torch.randn(num_samples, l_samples, dim_epsilon)
        generated_a = generator_model(epsilon)
        repeated_x = x_data.unsqueeze(1).expand(-1, l_samples, -1)
        flat_x = repeated_x.reshape(-1, x_data.shape[1])
        flat_a = generated_a.reshape(-1, dim_a)
        f_hat_preds = f_hat_model(flat_x, flat_a)
        preds_reshaped = f_hat_preds.view(num_samples, l_samples)
        final_predictions = torch.mean(preds_reshaped, dim=1)

    return final_predictions.numpy()

DEFAULT_BETA_MATRIX = np.array([
    [1.0, 0.5, 0.3, 0.2, 0.1],   # X_0
    [0.8, 1.0, 0.4, 0.1, 0.2],   # X_1
    [0.5, 0.3, 1.0, 0.5, 0.3],   # X_2
    [0.3, 0.2, 0.6, 1.0, 0.4],   # X_3
    [0.2, 0.4, 0.2, 0.3, 1.0],   # X_4
    [0.1, 0.1, 0.3, 0.2, 0.5],   # X_5
    [0.0, 0.2, 0.1, 0.4, 0.3],   # X_6
    [0.1, 0.0, 0.2, 0.1, 0.2],   # X_7
    [0.0, 0.2, 0.0, 0.2, 0.1],   # X_8
    [0.0, 0.3, 0.0, 0.0, 0.1],   # X_9
    [0.0, 0.0, 0.0, 0.0, 0.0],   # 10
    [0.0, 0.0, 0.0, 0.0, 0.0],   # 11
    [0.0, 0.0, 0.0, 0.0, 0.0],   # 12
    [0.0, 0.0, 0.0, 0.0, 0.0],   # 13
    [0.0, 0.0, 0.0, 0.0, 0.0],   # 14
    [0.0, 0.0, 0.0, 0.0, 0.0]   
])

def simulate_data_linear_confounder(
    n_source, n_target, dim_X, dim_A, f_bar_func, 
    beta_matrix=None,
    source_strength=1.0, 
    target_strength=1.0,
    source_noise_scale=0.8, #source A|X
    target_noise_scale=1.2, #target A|X
    x_shift=0.1,  # Target X distribution shift in mean (original mean 0)
    x_scale=1.1, # Target X distribution shift in variance
    a_shift=0.1,  # Target A distribution shift in mean original mean 0)
    seed=42
):

    np.random.seed(seed)

    if beta_matrix is None:
        beta_matrix = DEFAULT_BETA_MATRIX
    
    beta = beta_matrix[:dim_X, :dim_A]
    
    X_source = np.random.randn(n_source, dim_X)
    cov_A_source = np.eye(dim_A) 
    mean_A_source = np.zeros(dim_A)
    
    A_source = generate_conditional_A_linear(
        X_source, dim_A, cov_A_source, mean_A_source, beta_matrix,
        strength=source_strength,
        noise_scale=source_noise_scale
    )

    y_mean_source = f_bar_func(X_source, A_source)
    noise_source = np.random.randn(n_source, 1) * 0.05
    Y_source = y_mean_source + noise_source

    
    # ==================== TARGET DATA ====================
    X_target = np.random.randn(n_target, dim_X) * x_scale + x_shift
    cov_A_target = np.eye(dim_A)
    mean_A_target = np.zeros(dim_A) + a_shift
    
    A_target = generate_conditional_A_linear(
        X_target, dim_A, cov_A_target, mean_A_target, beta_matrix,
        strength=target_strength,
        noise_scale=target_noise_scale
    )
    
    y_mean_target = f_bar_func(X_target, A_target)
    noise_target = np.random.randn(n_target, 1) * 0.1
    Y_target = y_mean_target + noise_target
    

    def create_dataframe(X, A, Y):
        df_X = pd.DataFrame(X, columns=[f'x_{i}' for i in range(dim_X)])
        df_A = pd.DataFrame(A, columns=[f'a_{i}' for i in range(dim_A)])
        df_Y = pd.DataFrame(Y, columns=['y'])
        return pd.concat([df_X, df_A, df_Y], axis=1)
    
    source_df = create_dataframe(X_source, A_source, Y_source)
    target_df_full = create_dataframe(X_target, A_target, Y_target)
    

    # ==================== STORE DISTRIBUTION INFO ====================
    d0_source = {
        'mean': mean_A_source,
        'cov': cov_A_source,
        'beta': beta.copy(),  # Store the actual beta used
        'strength': source_strength,
        'noise_scale': source_noise_scale,
        'type': 'source'
    }
    
    d0_target = {
        'mean': mean_A_target,
        'cov': cov_A_target,
        'beta': beta.copy(),
        'strength': target_strength,
        'noise_scale': target_noise_scale,
        'type': 'target'
    }
    
    return source_df, target_df_full, cov_A_source, d0_source, d0_target


def generate_conditional_A_linear(X, dim_A, cov_A, mean_A, beta_matrix, 
                                   strength=1.0, noise_scale=1.0):

    n, dim_X = X.shape
    beta = beta_matrix[:dim_X, :dim_A]
    deterministic = strength * (X @ beta)
    noise = np.random.multivariate_normal(mean_A, cov_A * noise_scale**2, n)
    return deterministic + noise