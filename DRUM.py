import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold

from engression import engression as fit_engression
from engression.models import StoNet
from helper import F_hat_Network, train_f_hat

class DRUMGenerator:

    def __init__(self, engressor, conditional: bool):
        self.eng = engressor
        self.conditional = conditional

    def sample(self, X_or_N, L=1):

        if self.conditional:
            X = X_or_N
            samples = [self.eng.model(X) for _ in range(L)]
        else:
            N = X_or_N if isinstance(X_or_N, int) else X_or_N.shape[0]
            samples = [self.eng.model(N) for _ in range(L)]
        return torch.stack(samples, dim=1)          # [N, L, dim_a]

    def sample_flat(self, X_or_N, L=1):

        A = self.sample(X_or_N, L)
        N, Ls, dim_a = A.shape
        return A, A.reshape(N * Ls, dim_a)

    @property
    def model(self):
        return self.eng.model

    def train_mode(self):
        self.eng.model.train()

    def eval_mode(self):
        self.eng.model.eval()

    def parameters(self):
        return self.eng.model.parameters()

    def deepcopy(self):
        return DRUMGenerator(copy.deepcopy(self.eng), self.conditional)

def fit_source_generator(
    X_source, A_source, conditional,
    num_layer=2, hidden_dim=100, noise_dim=100,
    lr=5e-4, num_epochs=500, batch_size=128,
    beta=0.8, val_frac=0.2, device="cpu", verbose=True,
):

    N = X_source.shape[0]
    rng = np.random.RandomState(42)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_val = int(np.round(val_frac * N))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    X_tr = torch.as_tensor(X_source[tr_idx], dtype=torch.float32, device=device)
    A_tr = torch.as_tensor(A_source[tr_idx], dtype=torch.float32, device=device)

    if conditional:
        eng = fit_engression(
            x=X_tr, y=A_tr, classification=False,
            num_layer=num_layer, hidden_dim=hidden_dim,
            noise_dim=noise_dim, out_act=None,
            add_bn=True, resblock=False, beta=beta,
            lr=lr, num_epochs=num_epochs, batch_size=batch_size,
            device=device, standardize=False, verbose=verbose,
        )
    else:
        eng = _fit_marginal_engression(
            A_tr, num_layer=num_layer, hidden_dim=hidden_dim,
            noise_dim=noise_dim, beta=beta, lr=lr,
            num_epochs=num_epochs, batch_size=batch_size,
            device=device, verbose=verbose,
        )

    g_source = DRUMGenerator(eng, conditional=conditional)

    val_en = _val_energy(
        g_source, X_source[val_idx], A_source[val_idx], device=device,
    )
    if verbose:
        tag = "conditional" if conditional else "marginal"
        print(f"[Source generator ({tag})] held-out energy: {val_en:.4f}")

    return g_source, val_en


def _fit_marginal_engression(
    A_train, num_layer, hidden_dim, noise_dim,
    beta, lr, num_epochs, batch_size, device, verbose,
):

    from engression.engression import Engressor
    from engression.loss_func import energy_loss_two_sample

    dim_a = A_train.shape[1]
    N = A_train.shape[0]

    eng = Engressor(
        in_dim=0, out_dim=dim_a, classification=False,
        num_layer=num_layer, hidden_dim=hidden_dim,
        noise_dim=noise_dim, out_act=None,
        add_bn=True, resblock=False, beta=beta,
        lr=lr, num_epochs=num_epochs, batch_size=batch_size,
        standardize=False, device=device, check_device=False,
        verbose=verbose,
    )
    eng.model.train()
    optimizer = torch.optim.Adam(eng.model.parameters(), lr=lr)

    A_train = A_train.to(device)

    if batch_size is None or batch_size >= N:
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            A_hat1 = eng.model(N)          
            A_hat2 = eng.model(N)        
            losses = energy_loss_two_sample(A_train, A_hat1, A_hat2, beta=beta, verbose=True)
            loss = losses[0]
            loss.backward()
            optimizer.step()
            if verbose and (epoch == 0 or (epoch + 1) % 100 == 0):
                print(f"  [Marginal eng epoch {epoch+1}/{num_epochs}] "
                      f"energy={loss.item():.4f}")
    else:
        
        indices = torch.arange(N, device=device)
        for epoch in range(num_epochs):
            perm = torch.randperm(N, device=device)
            for start in range(0, N, batch_size):
                batch_idx = perm[start:start + batch_size]
                A_batch = A_train[batch_idx]
                bs = A_batch.shape[0]

                optimizer.zero_grad()
                A_hat1 = eng.model(bs)
                A_hat2 = eng.model(bs)
                losses = energy_loss_two_sample(
                    A_batch, A_hat1, A_hat2, beta=beta, verbose=True)
                loss = losses[0]
                loss.backward()
                optimizer.step()

            if verbose and (epoch == 0 or (epoch + 1) % 100 == 0):
                print(f"  [Marginal eng epoch {epoch+1}/{num_epochs}] "
                      f"energy={loss.item():.4f}")

    eng.model.eval()
    return eng


def _val_energy(g_wrap, X_val, A_val, device="cpu"):

    Xv = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    Av = torch.as_tensor(A_val, dtype=torch.float32, device=device)
    N = Av.shape[0]

    with torch.no_grad():
        if g_wrap.conditional:
            A1 = g_wrap.eng.model(Xv)
            A2 = g_wrap.eng.model(Xv)
        else:
            A1 = g_wrap.eng.model(N)
            A2 = g_wrap.eng.model(N)
        term1 = torch.norm(Av - A1, dim=1).mean()
        term2 = 0.5 * torch.norm(A1 - A2, dim=1).mean()
    return (term1 - term2).item()


def energy_gap_unified(g_candidate, g_source, X_s, A_s, M=16):

    N = A_s.shape[0]

    term1_c_samples = []
    for _ in range(M):
        with torch.no_grad():
            A_c = _forward(g_candidate, X_s, N)
        term1_c_samples.append(torch.norm(A_s - A_c, dim=1))
    term1_c_stable = torch.stack(term1_c_samples).mean(dim=0).mean()

    A_c_grad = _forward(g_candidate, X_s, N)
    term1_c_grad = torch.norm(A_s - A_c_grad, dim=1).mean()
    term1_c = term1_c_stable.detach() + (term1_c_grad - term1_c_grad.detach())

    A_c1 = _forward(g_candidate, X_s, N)
    A_c2 = _forward(g_candidate, X_s, N)
    term2_c = 0.5 * torch.norm(A_c1 - A_c2, dim=1).mean()
    en_cand = term1_c - term2_c

    with torch.no_grad():
        A_s1 = _forward(g_source, X_s, N)
        A_s2 = _forward(g_source, X_s, N)
        term1_s = torch.norm(A_s - A_s1, dim=1).mean()
        term2_s = 0.5 * torch.norm(A_s1 - A_s2, dim=1).mean()
        en_src = term1_s - term2_s

    return en_cand - en_src


def _forward(g_wrap, X, N):

    if g_wrap.conditional:
        return g_wrap.eng.model(X)
    else:
        return g_wrap.eng.model(N)


def target_objective(g_wrap, f_hat, X_t, L=64):

    N, dim_x = X_t.shape
    A, A_flat = g_wrap.sample_flat(X_t, L)      
    X_rep = X_t.unsqueeze(1).expand(-1, L, -1).reshape(N * L, dim_x)
    y = f_hat(X_rep, A_flat).view(N, L)
    inner = y.mean(dim=1)
    return (inner ** 2).mean()

def find_worst_case_generator(
    g_source, f_hat, X_source, A_source, X_target,
    delta, conditional,
    num_steps=400, L=64,
    lr_primal=1e-4, lr_dual=1e-4,
    grad_clip=2.0, M_gap=16,
    device="cpu", log_every=20,
):

    g_worst = g_source.deepcopy()
    g_worst.train_mode()
    f_hat.eval()

    X_s = torch.as_tensor(X_source, dtype=torch.float32, device=device)
    A_s = torch.as_tensor(A_source, dtype=torch.float32, device=device)
    X_t = torch.as_tensor(X_target, dtype=torch.float32, device=device)

    opt_g = torch.optim.Adam(g_worst.parameters(), lr=lr_primal)
    lam = torch.tensor(0.1, device=device)

    for step in range(1, num_steps + 1):
        opt_g.zero_grad()

        obj = target_objective(g_worst, f_hat, X_t, L=L)
        gap = energy_gap_unified(g_worst, g_source, X_s, A_s, M=M_gap)

        lagr = obj + lam.detach() * (gap - delta)
        lagr.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(g_worst.parameters(), grad_clip)
        opt_g.step()

        with torch.no_grad():
            new_gap = energy_gap_unified(g_worst, g_source, X_s, A_s, M=M_gap)
            lam = lam + lr_dual * (new_gap - delta)
            lam.clamp_(min=0.0)

        if step % log_every == 0 or step == 1:
            tag = "cond" if conditional else "marg"
            print(f"  [{tag} step {step:04d}/{num_steps}] "
                  f"obj={obj.item():.4f}  gap={new_gap.item():.6e}  "
                  f"lam={lam.item():.6e}")

    with torch.no_grad():
        final_gap = energy_gap_unified(
            g_worst, g_source, X_s, A_s, M=32).item()
    print(f"  [DIAG] delta={delta:.4f}  final_gap={final_gap:.6e}  "
          f"satisfied={final_gap <= delta}")

    g_worst.eval_mode()
    return g_worst


def predict_drum(X, f_hat, g_worst, L=256, device="cpu"):

    f_hat.eval()
    g_worst.eval_mode()
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    N, dim_x = X.shape

    with torch.no_grad():
        A, A_flat = g_worst.sample_flat(X, L)
        X_rep = X.unsqueeze(1).expand(-1, L, -1).reshape(N * L, dim_x)
        preds = f_hat(X_rep, A_flat).view(N, L).mean(dim=1)

    return preds.cpu().numpy()


def fit_drum(
    X_source, A_source, X_target, f_hat,
    conditional=True, delta=0.3,
    num_layer=2, hidden_dim=100, noise_dim=100,
    lr_gs=5e-4, num_epochs_eng=500, batch_size_eng=128, beta=0.8,
    lr_primal=2e-4, lr_dual=1e-4, finetune_steps=400, finetune_L=64,
    grad_clip=2.0, M_gap=16,
    n_samples=256,
    val_frac=0.2, device="cpu", verbose=True,
):
    
    g_source, val_en = fit_source_generator(
        X_source, A_source, conditional=conditional,
        num_layer=num_layer, hidden_dim=hidden_dim, noise_dim=noise_dim,
        lr=lr_gs, num_epochs=num_epochs_eng, batch_size=batch_size_eng,
        beta=beta, val_frac=val_frac, device=device, verbose=verbose,
    )

    g_worst = find_worst_case_generator(
        g_source, f_hat, X_source, A_source, X_target,
        delta=delta, conditional=conditional,
        num_steps=finetune_steps, L=finetune_L,
        lr_primal=lr_primal, lr_dual=lr_dual,
        grad_clip=grad_clip, M_gap=M_gap,
        device=device, log_every=20 if verbose else finetune_steps,
    )

    @torch.no_grad()
    def m_robust(X):
        return predict_drum(X, f_hat, g_worst, L=n_samples, device=device)

    return m_robust, g_source, g_worst


class Omega_Classifier(nn.Module):

    def __init__(self, dim_x, dim_a):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_x + dim_a, 64), 
            nn.ReLU(),
            nn.Linear(64, 32), 
            nn.ReLU(),
            nn.Linear(32, 1), 
            nn.Sigmoid() 
        )
    def forward(self, x, a):
        xa_input = torch.cat([x, a], dim=1)
        return self.network(xa_input)

def cross_fit_debiased(
    source_df, target_df,
    preliminary_generator,        
    preliminary_source_generator, 
    conditional,
    dim_x, dim_a,
    epochs_f=100, lr_f=1e-5,
    epochs_omega=200, lr_omega=1e-5,
    epochs_g=300, lr_g=1e-5,
    l_samples=256,
    delta=0.3, lr_primal=2e-4, lr_dual=1e-4, finetune_steps=80,
    lam_init=0.1,
    seed=42,
):

    kf = KFold(n_splits=3, shuffle=True, random_state=seed)
    folds_s = list(kf.split(source_df))
    (_, val_idx1), (_, val_idx2), (_, val_idx3) = folds_s

    Tkf = KFold(n_splits=3, shuffle=True, random_state=seed)
    folds_t = list(Tkf.split(target_df))
    (_, Tval_idx1), (_, Tval_idx2), (_, Tval_idx3) = folds_t

    X_s = torch.tensor(source_df.filter(regex='^x_').values, dtype=torch.float32)
    A_s = torch.tensor(source_df.filter(regex='^a_').values, dtype=torch.float32)
    Y_s = torch.tensor(source_df['y'].values, dtype=torch.float32).view(-1, 1)
    X_t = torch.tensor(target_df.filter(regex='^x_').values, dtype=torch.float32)

    n_source = len(source_df)
    n_target = len(target_df)

    out_of_sample_omegas = torch.zeros(n_source, 1)
    out_of_sample_residuals = torch.zeros(n_source, 1)
    out_of_sample_f_hat_preds = torch.zeros(len(source_df), 1)

    f_hat_1 = F_hat_Network(dim_x, dim_a)
    ds1 = torch.utils.data.TensorDataset(X_s[val_idx1], A_s[val_idx1], Y_s[val_idx1])
    loader1 = torch.utils.data.DataLoader(ds1, batch_size=128, shuffle=True)
    trained_f_hat_1 = train_f_hat(f_hat_1, loader1, epochs_f, lr_f)
    with torch.no_grad():
        trained_f_hat_1.eval()
        preds = trained_f_hat_1(X_s[val_idx2], A_s[val_idx2])
    out_of_sample_residuals[val_idx2] = Y_s[val_idx2] - preds

    f_hat_2 = F_hat_Network(dim_x, dim_a)
    ds2 = torch.utils.data.TensorDataset(X_s[val_idx2], A_s[val_idx2], Y_s[val_idx2])
    loader2 = torch.utils.data.DataLoader(ds2, batch_size=128, shuffle=True)
    trained_f_hat_2 = train_f_hat(f_hat_2, loader2, epochs_f, lr_f)
    with torch.no_grad():
        trained_f_hat_2.eval()
        preds = trained_f_hat_2(X_s[val_idx3], A_s[val_idx3])
    out_of_sample_residuals[val_idx3] = Y_s[val_idx3] - preds

    f_hat_3 = F_hat_Network(dim_x, dim_a)
    ds3 = torch.utils.data.TensorDataset(X_s[val_idx3], A_s[val_idx3], Y_s[val_idx3])
    loader3 = torch.utils.data.DataLoader(ds3, batch_size=128, shuffle=True)
    trained_f_hat_3 = train_f_hat(f_hat_3, loader3, epochs_f, lr_f)
    with torch.no_grad():
        trained_f_hat_3.eval()
        preds = trained_f_hat_3(X_s[val_idx1], A_s[val_idx1])
    out_of_sample_residuals[val_idx1] = Y_s[val_idx1] - preds


    def _generate_A_from_preliminary(X_tensor):
        with torch.no_grad():
            return _sample_A(preliminary_generator, X_tensor, conditional)

    A_gen_1 = _generate_A_from_preliminary(X_t[Tval_idx1])
    X_comb_1 = torch.cat([X_s[val_idx1], X_t[Tval_idx1]], dim=0)
    A_comb_1 = torch.cat([A_s[val_idx1], A_gen_1], dim=0)
    labels_1 = torch.cat([torch.ones(len(val_idx1), 1), torch.zeros(len(Tval_idx1), 1)])
    omega_clf_1 = Omega_Classifier(dim_x, dim_a)
    opt_o1 = optim.Adam(omega_clf_1.parameters(), lr=lr_omega)
    loss_fn_o = nn.BCELoss()
    for _ in range(epochs_omega):
        opt_o1.zero_grad()
        loss_fn_o(omega_clf_1(X_comb_1, A_comb_1), labels_1).backward()
        opt_o1.step()
    with torch.no_grad():
        omega_clf_1.eval()
        p = omega_clf_1(X_s[val_idx2], A_s[val_idx2])
        w = (1 - p) / (p + 1e-8)
        out_of_sample_omegas[val_idx2] = w / w.mean()

    A_gen_2 = _generate_A_from_preliminary(X_t[Tval_idx2])
    X_comb_2 = torch.cat([X_s[val_idx2], X_t[Tval_idx2]], dim=0)
    A_comb_2 = torch.cat([A_s[val_idx2], A_gen_2], dim=0)
    labels_2 = torch.cat([torch.ones(len(val_idx2), 1), torch.zeros(len(Tval_idx2), 1)])
    omega_clf_2 = Omega_Classifier(dim_x, dim_a)
    opt_o2 = optim.Adam(omega_clf_2.parameters(), lr=lr_omega)
    for _ in range(epochs_omega):
        opt_o2.zero_grad()
        loss_fn_o(omega_clf_2(X_comb_2, A_comb_2), labels_2).backward()
        opt_o2.step()
    with torch.no_grad():
        omega_clf_2.eval()
        p = omega_clf_2(X_s[val_idx3], A_s[val_idx3])
        w = (1 - p) / (p + 1e-8)
        out_of_sample_omegas[val_idx3] = w / w.mean()

    A_gen_3 = _generate_A_from_preliminary(X_t[Tval_idx3])
    X_comb_3 = torch.cat([X_s[val_idx3], X_t[Tval_idx3]], dim=0)
    A_comb_3 = torch.cat([A_s[val_idx3], A_gen_3], dim=0)
    labels_3 = torch.cat([torch.ones(len(val_idx3), 1), torch.zeros(len(Tval_idx3), 1)])
    omega_clf_3 = Omega_Classifier(dim_x, dim_a)
    opt_o3 = optim.Adam(omega_clf_3.parameters(), lr=lr_omega)
    for _ in range(epochs_omega):
        opt_o3.zero_grad()
        loss_fn_o(omega_clf_3(X_comb_3, A_comb_3), labels_3).backward()
        opt_o3.step()
    with torch.no_grad():
        omega_clf_3.eval()
        p = omega_clf_3(X_s[val_idx1], A_s[val_idx1])
        w = (1 - p) / (p + 1e-8)
        out_of_sample_omegas[val_idx1] = w / w.mean()

    omegas_1, residuals_1 = _compute_generator_nuisances(
        trained_f_hat_2, omega_clf_2, X_s, A_s, Y_s, val_idx1)
    g_debiased_1 = _train_debiased_generator_fold(
        initial_generator=preliminary_generator,
        source_reference_generator=preliminary_source_generator,
        f_hat_model=trained_f_hat_2,
        X_source=X_s[val_idx1], A_source=A_s[val_idx1],
        X_target=X_t[Tval_idx1],
        omegas=omegas_1, residuals=residuals_1,
        conditional=conditional, dim_a=dim_a, l_samples=l_samples,
        epochs_g=epochs_g, lr_g=lr_g,
        delta=delta, lr_primal=lr_primal, lr_dual=lr_dual,
        finetune_steps=finetune_steps, lam_init=lam_init)

    omegas_2, residuals_2 = _compute_generator_nuisances(
        trained_f_hat_3, omega_clf_3, X_s, A_s, Y_s, val_idx2)
    g_debiased_2 = _train_debiased_generator_fold(
        initial_generator=preliminary_generator,
        source_reference_generator=preliminary_source_generator,
        f_hat_model=trained_f_hat_3,
        X_source=X_s[val_idx2], A_source=A_s[val_idx2],
        X_target=X_t[Tval_idx2],
        omegas=omegas_2, residuals=residuals_2,
        conditional=conditional, dim_a=dim_a, l_samples=l_samples,
        epochs_g=epochs_g, lr_g=lr_g,
        delta=delta, lr_primal=lr_primal, lr_dual=lr_dual,
        finetune_steps=finetune_steps, lam_init=lam_init)

    omegas_3, residuals_3 = _compute_generator_nuisances(
        trained_f_hat_1, omega_clf_1, X_s, A_s, Y_s, val_idx3)
    g_debiased_3 = _train_debiased_generator_fold(
        initial_generator=preliminary_generator,
        source_reference_generator=preliminary_source_generator,
        f_hat_model=trained_f_hat_1,
        X_source=X_s[val_idx3], A_source=A_s[val_idx3],
        X_target=X_t[Tval_idx3],
        omegas=omegas_3, residuals=residuals_3,
        conditional=conditional, dim_a=dim_a, l_samples=l_samples,
        epochs_g=epochs_g, lr_g=lr_g,
        delta=delta, lr_primal=lr_primal, lr_dual=lr_dual,
        finetune_steps=finetune_steps, lam_init=lam_init)


    def _mc_prediction(f_hat_model, X_sub, g_wrap, L):
        """Average f_hat over L generator draws."""
        B, dx = X_sub.shape
        A = g_wrap.sample(X_sub, L)
        X_rep = X_sub.unsqueeze(1).expand(-1, L, -1).reshape(B * L, dx)
        A_flat = A.reshape(B * L, -1)
        preds = f_hat_model(X_rep, A_flat)
        return preds.view(B, L).mean(dim=1, keepdim=True)

    out_of_sample_f_preds_t = torch.zeros(n_target, 1)

    with torch.no_grad():
        trained_f_hat_3.eval()
        g_debiased_2.eval_mode()
        out_of_sample_f_preds_t[Tval_idx1] = _mc_prediction(
            trained_f_hat_3, X_t[Tval_idx1], g_debiased_2, l_samples)

    with torch.no_grad():
        trained_f_hat_1.eval()
        g_debiased_3.eval_mode()
        out_of_sample_f_preds_t[Tval_idx2] = _mc_prediction(
            trained_f_hat_1, X_t[Tval_idx2], g_debiased_3, l_samples)

    with torch.no_grad():
        trained_f_hat_2.eval()
        g_debiased_1.eval_mode()
        out_of_sample_f_preds_t[Tval_idx3] = _mc_prediction(
            trained_f_hat_2, X_t[Tval_idx3], g_debiased_1, l_samples)

    rho = n_source / n_target

    omegas = out_of_sample_omegas.detach().clone()
    omega_range = omegas.max() - omegas.min()
    if omega_range > 1e-8:
        omega_source = (omegas - omegas.min()) / omega_range
    else:
        omega_source = torch.ones_like(omegas)

    S = torch.cat([torch.ones(n_source, 1), torch.zeros(n_target, 1)])
    omega_pooled = torch.cat([omega_source, torch.zeros(n_target, 1)])
    resid_pooled = torch.cat([out_of_sample_residuals, torch.zeros(n_target, 1)])
    fpred_pooled = torch.cat([torch.zeros(n_source, 1), out_of_sample_f_preds_t])

    F = S * omega_pooled * resid_pooled + rho * (1 - S) * fpred_pooled

    return F


def _sample_A(g_wrap, X, conditional):
    """Draw a single sample of A from the generator."""
    if conditional:
        return g_wrap.eng.model(X)
    else:
        return g_wrap.eng.model(X.shape[0])


def _compute_generator_nuisances(f_hat, omega_clf, X_s, A_s, Y_s, idx):

    f_hat.eval()
    omega_clf.eval()
    with torch.no_grad():
        preds = f_hat(X_s[idx], A_s[idx])
        residuals = Y_s[idx] - preds

        p = omega_clf(X_s[idx], A_s[idx])
        omegas = (1.0 - p) / (p + 1e-8)
        omegas = omegas / omegas.mean().clamp_min(1e-8)
    return omegas, residuals


def _train_debiased_generator_fold(
    initial_generator, source_reference_generator,
    f_hat_model, X_source, A_source, X_target,
    omegas, residuals,
    conditional, dim_a, l_samples,
    epochs_g, lr_g,
    delta, lr_primal, lr_dual, finetune_steps, lam_init,
):

    g_deb = initial_generator.deepcopy()
    g_deb.train_mode()

    f_hat_model.eval()
    for p in f_hat_model.parameters():
        p.requires_grad = False

    source_reference_generator.eval_mode()
    for p in source_reference_generator.parameters():
        p.requires_grad = False

    lam = float(lam_init)
    optimizer = optim.Adam(g_deb.parameters(), lr=lr_primal or lr_g)
    delta_val = float(delta) if delta else 0.1
    L = l_samples or 64
    num_steps = finetune_steps or 200
    lr_dual_val = float(lr_dual) if lr_dual else 1e-4

    N_s = X_source.shape[0]
    N_t = X_target.shape[0]
    omegas_flat = omegas.view(-1)
    residuals_flat = residuals.view(-1)

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()

        A_t = g_deb.sample(X_target, L)                 
        X_t_rep = X_target.unsqueeze(1).expand(-1, L, -1).reshape(N_t * L, -1)
        A_t_flat = A_t.reshape(N_t * L, dim_a)
        inner_t = f_hat_model(X_t_rep, A_t_flat).view(N_t, L).mean(dim=1)
        loss_term1 = (inner_t ** 2).mean()

        with torch.no_grad():
            A_s_gen = initial_generator.sample(X_source, L)
            X_s_rep = X_source.unsqueeze(1).expand(-1, L, -1).reshape(N_s * L, -1)
            A_s_flat = A_s_gen.reshape(N_s * L, dim_a)
            inner_s = f_hat_model(X_s_rep, A_s_flat).view(N_s, L).mean(dim=1)

        loss_term2 = 2.0 * (
            omegas_flat.detach() * inner_s * residuals_flat.detach()
        ).mean()

        loss_debiased = loss_term1 + loss_term2

        M_gap = 32
        n_sub = min(500, N_s)
        idx = torch.randperm(N_s)[:n_sub]
        X_sub = X_source[idx]
        A_sub = A_source[idx]

        gw1 = torch.stack(
            [_forward(g_deb, X_sub, n_sub) for _ in range(M_gap)], dim=1)
        gw2 = torch.stack(
            [_forward(g_deb, X_sub, n_sub) for _ in range(M_gap)], dim=1)
        cross_term = torch.norm(A_sub.unsqueeze(1) - gw1, dim=2).mean()
        gen_self = 0.5 * torch.norm(gw1 - gw2, dim=2).mean()

        with torch.no_grad():
            gs1 = torch.stack(
                [_forward(source_reference_generator, X_sub, n_sub)
                 for _ in range(M_gap)], dim=1)
            gs2 = torch.stack(
                [_forward(source_reference_generator, X_sub, n_sub)
                 for _ in range(M_gap)], dim=1)
            data_self = (
                torch.norm(A_sub.unsqueeze(1) - gs1, dim=2).mean()
                - 0.5 * torch.norm(gs1 - gs2, dim=2).mean()
            )

        gap = (cross_term - gen_self) - data_self
        lagrangian = loss_debiased + lam * (gap - delta_val)

        lagrangian.backward()
        torch.nn.utils.clip_grad_norm_(g_deb.parameters(), 1.0)
        optimizer.step()

        gap_val = gap.detach().item()
        with torch.no_grad():
            lam = max(0.0, lam + lr_dual_val * (gap_val - delta_val))

        if step % 20 == 0 or step == 1:
            tag = "cond" if conditional else "marg"
            print(f"      [{tag} deb step {step:04d}] "
                  f"t1={loss_term1.item():.4f}  "
                  f"t2={loss_term2.item():.4f}  "
                  f"gap={gap_val:.6f}  lam={lam:.6f}")

    g_deb.eval_mode()
    return g_deb

def train_Fx_estimator(F_hat, X_pooled, epochs=300, lr=1e-5, batch_size=128):
    """Regress pseudo-outcomes F on X. Identical to debias_new2.py."""
    dim_x = X_pooled.shape[1] if hasattr(X_pooled, 'shape') else X_pooled.size(1)

    class FinalEstimator(nn.Module):
        def __init__(self, dx):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(dx, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            return self.network(x)

    model = FinalEstimator(dim_x)

    if not isinstance(X_pooled, torch.Tensor):
        X_pooled = torch.tensor(X_pooled, dtype=torch.float32)
    if not isinstance(F_hat, torch.Tensor):
        F_hat = torch.tensor(F_hat, dtype=torch.float32)
    if F_hat.dim() == 1:
        F_hat = F_hat.unsqueeze(1)

    ds = torch.utils.data.TensorDataset(X_pooled, F_hat)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    print("\n--- Training Final Debiased Estimator ---")
    model.train()
    for epoch in range(epochs):
        total = 0.0
        nb = 0
        for xb, fb in loader:
            loss = loss_fn(model(xb), fb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1}/{epochs}], MSE: {total / nb:.4f}")

    print("--- Final Debiased Estimator Complete ---")
    model.eval()
    return model


def predict_final_debiased(X, estimator):
    """Predict from the final debiased regressor."""
    estimator.eval()
    X = torch.as_tensor(X, dtype=torch.float32)
    with torch.no_grad():
        return estimator(X).cpu().numpy()