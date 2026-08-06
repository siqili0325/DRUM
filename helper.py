import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.optim as optim

DEFAULT_BETA_MATRIX = np.array([
    [1.0, 0.5, 0.3, 0.2, 0.1,   0.4, 0.2, 0.1, 0.3, 0.2, 0.4, 0.2, 0.1, 0.3, 0.2],   
    [0.4, 1.0, 0.4, 0.1, 0.2,   0.3, 0.5, 0.2, 0.1, 0.1, 0.3, 0.5, 0.2, 0.1, 0.1],  
    [0.5, 0.3, 1.0, 0.5, 0.3,   0.2, 0.1, 0.4, 0.2, 0.3, 0.2, 0.1, 0.4, 0.2, 0.3],  
    [0.3, 0.2, 0.6, 1.0, 0.4,   0.1, 0.3, 0.5, 0.1, 0.2, 0.1, 0.3, 0.5, 0.1, 0.2],
    [0.2, 0.4, 0.2, 0.3, 1.0,   0.3, 0.2, 0.1, 0.5, 0.4, 0.3, 0.2, 0.1, 0.5, 0.4], 
    [0.1, 0.1, 0.3, 0.2, 0.5,   1.0, 0.4, 0.3, 0.2, 0.1, 1.0, 0.4, 0.3, 0.2, 0.1], 
    [0.0, 0.2, 0.1, 0.4, 0.3,   0.3, 1.0, 0.2, 0.4, 0.2, 0.3, 1.0, 0.2, 0.4, 0.2], 
    [0.1, 0.0, 0.2, 0.1, 0.2,   0.2, 0.3, 1.0, 0.1, 0.3, 0.2, 0.3, 1.0, 0.1, 0.3], 
    [0.0, 0.2, 0.0, 0.2, 0.1,   0.1, 0.2, 0.3, 1.0, 0.2, 0.1, 0.2, 0.3, 1.0, 0.2], 
    [0.0, 0.3, 0.0, 0.0, 0.1,   0.2, 0.1, 0.2, 0.3, 1.0, 0.2, 0.1, 0.2, 0.3, 1.0],
    [0.3, 0.2, 0.2, 0.0, 0.4,   0.1, 0.0, 0.1, 0.2, 0.3, 0.1, 0.0, 0.1, 0.2, 0.3],
    [0.1, 0.4, 0.0, 0.3, 0.1,   0.2, 0.1, 0.0, 0.1, 0.2, 0.2, 0.1, 0.0, 0.1, 0.2],
    [0.0, 0.2, 0.3, 0.1, 0.2,   0.0, 0.2, 0.1, 0.0, 0.1, 0.0, 0.2, 0.1, 0.0, 0.1],
    [0.2, 0.0, 0.1, 0.4, 0.0,   0.1, 0.0, 0.2, 0.1, 0.0, 0.1, 0.0, 0.2, 0.1, 0.0], 
    [0.1, 0.3, 0.2, 0.0, 0.3,   0.0, 0.1, 0.0, 0.2, 0.1, 0.0, 0.1, 0.0, 0.2, 0.1],
    [0.0, 0.0, 0.0, 0.0, 0.0,   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
])

def simulate_data(n_source, n_target, dim_X, dim_A, f_bar_func, seed=42):
    np.random.seed(seed)

    X_source = np.random.randn(n_source, dim_X)
    
    cov_A_source = np.random.randn(dim_A, dim_A)
    cov_A_source = np.dot(cov_A_source, cov_A_source.transpose())
    mean_A_source = np.zeros(dim_A)
    A_source = np.random.multivariate_normal(mean_A_source, cov_A_source, n_source)

    y_mean_source = f_bar_func(X_source, A_source)
    noise_source = np.random.randn(n_source, 1) * 0.1
    Y_source = y_mean_source + noise_source

    target_mean_X = np.random.rand(dim_X) * 0.5
    target_cov_X_scale = np.diag(np.random.uniform(0.8, 1.2, dim_X))
    X_target = np.random.randn(n_target, dim_X) @ target_cov_X_scale + target_mean_X
    
    target_mean_A = np.random.rand(dim_A) * 0.3
    cov_A_target = np.random.randn(dim_A, dim_A)
    cov_A_target = np.dot(cov_A_target, cov_A_target.transpose())
    A_target = np.random.multivariate_normal(target_mean_A, cov_A_target, n_target)
    
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

    d0_source = {
        'mean': mean_A_source,
        'cov': cov_A_source,
        'type': 'source'
    }
    
    d0_target = {
        'mean': target_mean_A,
        'cov': cov_A_target,
        'type': 'target'
    }
    
    return source_df, target_df_full, cov_A_source, d0_source, d0_target


def simulate_data_conditional(n_source, n_target, dim_X, dim_A, f_bar_func, seed=42,conditional_type='linear', source_strength=0.5, target_strength=0.8,source_noise_scale=1.0, target_noise_scale=1.0,source_X_indices=None,target_X_indices=None):

    np.random.seed(seed)
    
    X_source = np.random.randn(n_source, dim_X)
    
    if source_X_indices is None:
        source_X_indices = list(range(dim_A))
    if target_X_indices is None:
        target_X_indices = source_X_indices
    
    cov_A_source = np.random.randn(dim_A, dim_A)
    cov_A_source = np.dot(cov_A_source, cov_A_source.transpose())
    mean_A_source = np.zeros(dim_A)
    
    A_source = generate_conditional_A(
        X_source, dim_A, cov_A_source, mean_A_source,
        conditional_type=conditional_type,
        strength=source_strength,
        noise_scale=source_noise_scale,
        X_indices=source_X_indices
    )
    
    y_mean_source = f_bar_func(X_source, A_source)
    noise_source = np.random.randn(n_source, 1) * 0.1
    Y_source = y_mean_source + noise_source
    
    target_mean_X = np.random.rand(dim_X) * 0.5
    target_cov_X_scale = np.diag(np.random.uniform(0.8, 1.2, dim_X))
    X_target = np.random.randn(n_target, dim_X) @ target_cov_X_scale + target_mean_X
    
    cov_A_target = np.random.randn(dim_A, dim_A)
    cov_A_target = np.dot(cov_A_target, cov_A_target.transpose())
    target_mean_A = np.random.rand(dim_A) * 0.3
    
    A_target = generate_conditional_A(
        X_target, dim_A, cov_A_target, target_mean_A,
        conditional_type=conditional_type,
        strength=target_strength, 
        noise_scale=target_noise_scale,
        X_indices=target_X_indices 
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
    
    d0_source = {
        'mean': mean_A_source,
        'cov': cov_A_source,
        'type': 'source',
        'conditional_type': conditional_type,
        'strength': source_strength,
        'noise_scale': source_noise_scale,
        'X_indices': source_X_indices
    }
    
    d0_target = {
        'mean': target_mean_A,
        'cov': cov_A_target,
        'type': 'target',
        'conditional_type': conditional_type,
        'strength': target_strength,
        'noise_scale': target_noise_scale,
        'X_indices': target_X_indices
    }
    
    return source_df, target_df_full, cov_A_source, d0_source, d0_target


def generate_conditional_A_linear(X, dim_A, cov_A, mean_A, beta_matrix, 
                                   strength=1.0, noise_scale=1.0):
    """
    Generate A conditional on X using linear relationship.
    A = strength * (X @ beta) + noise
    """
    n, dim_X = X.shape
    beta = beta_matrix[:dim_X, :dim_A]
    deterministic = strength * (X @ beta)
    noise = np.random.multivariate_normal(mean_A, cov_A * noise_scale**2, n)
    return deterministic + noise


def generate_conditional_A_nonlinear(X, dim_A, cov_A, mean_A, beta_matrix, 
                                      strength=1.0, noise_scale=1.0):
    
    n, dim_X = X.shape
    beta = beta_matrix[:dim_X, :dim_A]
    linear_part = X @ beta
    nonlinear_part = np.zeros((n, dim_A))

    for j in range(dim_A):

        for i in range(min(5, dim_X - 1)):
            nonlinear_part[:, j] += 0.1 * beta[i, j] * X[:, i] * X[:, i + 1]
        for i in range(min(5, dim_X)):
            nonlinear_part[:, j] += 0.1 * beta[i, j] * np.sign(X[:, i]) * X[:, i]**2

    deterministic = strength * (linear_part + nonlinear_part)
    noise = np.random.multivariate_normal(mean_A, cov_A * noise_scale**2, n)
    return deterministic + noise


def simulate_data_linear_confounder(
    n_source, n_target, dim_X, dim_A, f_bar_func, 
    beta_matrix=None,
    source_strength=1.0, 
    target_strength=1.0,
    source_noise_scale=0.8, 
    target_noise_scale=1.2, 
    x_shift=0.1,  
    x_scale=1.1, 
    a_shift=0.1,
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
    
    d0_source = {
        'mean': mean_A_source,
        'cov': cov_A_source,
        'beta': beta.copy(), 
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

def simulate_data_nonlinear_confounder(
    n_source, n_target, dim_X, dim_A, f_bar_func, 
    beta_matrix=None,
    source_strength=1.0, 
    target_strength=1.0,
    source_noise_scale=0.3,
    target_noise_scale=1.2,
    x_shift=0.1,
    x_scale=1.1,
    a_shift=0.1,
    seed=42
):
    np.random.seed(seed)

    if beta_matrix is None:
        beta_matrix = DEFAULT_BETA_MATRIX
    
    beta = beta_matrix[:dim_X, :dim_A]

    X_source = np.random.randn(n_source, dim_X)
    cov_A_source = np.eye(dim_A)
    mean_A_source = np.zeros(dim_A)
    
    A_source = generate_conditional_A_nonlinear(
        X_source, dim_A, cov_A_source, mean_A_source, beta_matrix,
        strength=source_strength,
        noise_scale=source_noise_scale
    )

    y_mean_source = f_bar_func(X_source, A_source)
    noise_source = np.random.randn(n_source, 1) * 0.05
    Y_source = y_mean_source + noise_source

    X_target = np.random.randn(n_target, dim_X) * x_scale + x_shift
    cov_A_target = np.eye(dim_A)
    mean_A_target = np.zeros(dim_A) + a_shift
    
    A_target = generate_conditional_A_nonlinear(
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

    d0_source = {
        'mean': mean_A_source,
        'cov': cov_A_source,
        'beta': beta.copy(),
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

def train_baseline_model_new(baseline_model, train_loader, val_loader, epochs, lr):

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