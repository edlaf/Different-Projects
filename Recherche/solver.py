import numpy as np
import cvxpy as cp
import plotly.graph_objects as go
from scipy.stats import t
from scipy.special import gamma
from tqdm import tqdm


def generate_matrix(dist_name, m, n):
    if dist_name == 'Uniform':
        A = np.sqrt(12)*(np.random.rand(m, n) - 0.5) / np.sqrt(n)
        
    elif dist_name == 'Gaussian':
        A = np.random.randn(m, n) / np.sqrt(n)
        
    elif dist_name == 'Rademacher':
        A = np.random.choice([-1, 1], size=(m, n)) / np.sqrt(n)
        
    elif dist_name == 'Laplace':
        A = np.random.laplace(loc=0, scale=1/np.sqrt(2*n), size=(m, n))
        
    elif dist_name == 'Exponential':
        A = (np.random.exponential(scale=1, size=(m, n)) - 1) / np.sqrt(n)
        
    elif dist_name == 'Gamma(shape=2)':
        shape = 2
        A = (np.random.gamma(shape, scale=1, size=(m, n)) - shape) / np.sqrt(shape * n)
        
    elif dist_name == 'Beta(2,5)':
        alpha, beta_ = 2, 5
        mean = alpha/(alpha+beta_)
        var = alpha*beta_/(((alpha+beta_)**2)*(alpha+beta_+1))
        A = (np.random.beta(alpha, beta_, size=(m, n)) - mean) / np.sqrt(var * n)
        
    elif dist_name == 'Chi-square(df=4)':
        df = 4
        A = (np.random.chisquare(df, size=(m, n)) - df) / np.sqrt(2*df*n)
        
    elif dist_name == 'Student t(df=5)':
        df = 5
        var_t = df/(df-2)
        A = np.random.standard_t(df, size=(m, n)) / np.sqrt(var_t * n)
        
    elif dist_name == 'Weibull(k=1.5)':
        k = 1.5
        from scipy.special import gamma
        mean_w = gamma(1 + 1/k)
        var_w = gamma(1 + 2/k) - mean_w**2
        A = (np.random.weibull(k, size=(m, n)) - mean_w) / np.sqrt(var_w * n)
        
    elif dist_name == 'Logistic':
        var_log = (np.pi**2)/3
        A = np.random.logistic(loc=0, scale=1, size=(m, n)) / np.sqrt(var_log * n)
        
    elif dist_name == 'Poisson(λ=1)':
        lam = 1
        A = (np.random.poisson(lam, size=(m, n)) - lam) / np.sqrt(lam * n)
        
    elif dist_name == 'Pareto(α=3)':
        alpha = 3
        mean_p = alpha/(alpha-1)
        var_p = alpha/(alpha-2) - mean_p**2
        A = (np.random.pareto(alpha, size=(m, n)) + 1 - mean_p) / np.sqrt(var_p * n)
        
    elif dist_name == 'Student-t(df=3)':
        from scipy.stats import t
        df = 3
        X = t.rvs(df=df, size=(m, n))
        A = X / (np.std(X) * np.sqrt(n))
        
    elif dist_name == 'Permutation':
        cols = np.random.choice(n, size=m, replace=False)
        A = np.zeros((m, n), dtype=int)
        for i, col in enumerate(cols):
            A[i, col] = 1
            
    elif dist_name == 'Householder':
        v = np.random.randn(n)
        v /= np.linalg.norm(v)
        H = np.eye(n) - 2*np.outer(v, v)
        A = H[:m, :]
        
    elif dist_name == 'QR':
        Q, _ = np.linalg.qr(np.random.randn(n, m), mode='reduced')
        A = Q.T
        
    elif dist_name == 'SVD':
        U, _, _ = np.linalg.svd(np.random.randn(n, n), full_matrices=False)
        A = U[:m, :]
        
    elif dist_name == 'Power (GS)':
        X = np.random.randn(m, n)
        Q = np.zeros((m, n))
        for i in range(m):
            v = X[i, :].copy()
            for j in range(i):
                v -= np.dot(Q[j, :], X[i, :]) * Q[j, :]
            Q[i, :] = v / np.linalg.norm(v)
        A = Q
        
    else:
        A = np.random.randn(m, n) / np.sqrt(n)
    
    return A

def sol_l1(n=1000, m=10, num_trials=50, pour_sparce = 60):
    n_sparce = int(n*pour_sparce/100)
    dist_list = [
        'Uniform', 'Gaussian', 'Rademacher', 'Laplace',
        'Exponential', 'Gamma(shape=2)', 'Beta(2,5)', 'Chi-square(df=4)',
        'Student t(df=5)', 'Weibull(k=1.5)', 'Logistic', 'Poisson(λ=1)',
        'Pareto(α=3)', 'Student-t(df=3)', 'Power (GS)', 'Householder', 'QR', 'SVD', 'Permutation'
    ]
    
    results_mean = []
    results_min = []
    results_max = []
    results_support = []
    
    for dist_name in tqdm(dist_list):
        errors = []
        for _ in range(num_trials):
            A = generate_matrix(dist_name, m, n)
            x = np.zeros(n)
            nonzero_indices = np.random.choice(n, size=n_sparce, replace=False)
            x[nonzero_indices] = np.random.randn(n_sparce)
            x_true = x
            b = A @ x_true
            x = cp.Variable(n)
            constraints = [A @ x == b]
            objective = cp.Minimize(cp.norm1(x))
            prob = cp.Problem(objective, constraints)
            
            try:
                prob.solve()  
            except cp.SolverError:
                prob.solve(solver=cp.SCS)
            
            x_hat = x.value
            x_hat[np.abs(x_hat) < 1e-8] = 0
            if x_hat is not None:
                err = np.linalg.norm(x_hat - x_true, 1)
                support_ = np.abs(np.count_nonzero(x_hat)-np.count_nonzero(x_true))
            else:
                err = np.nan
            errors.append(err)
        valid_errors = [e for e in errors if not np.isnan(e)]
        if len(valid_errors) == 0:
            mean_err = np.nan
            min_err  = np.nan
            max_err  = np.nan
        else:
            mean_err = np.mean(valid_errors)
            min_err  = np.min(valid_errors)
            max_err  = np.max(valid_errors)
            support  = np.mean(support_)
        
        results_mean.append(mean_err)
        results_min.append(min_err)
        results_max.append(max_err)
        results_support.append(support)
    
    fig = go.Figure(data=[go.Table(
        header=dict(values=["Distribution", "Mean L1 Error", "Min L1 Error", "Max L1 Error"," Support"],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[
            dist_list,
            [f"{val:.4e}" if not np.isnan(val) else "nan" for val in results_mean],
            [f"{val:.4e}" if not np.isnan(val) else "nan" for val in results_min],
            [f"{val:.4e}" if not np.isnan(val) else "nan" for val in results_max],
            [f"{val:.4e}" if not np.isnan(val) else "nan" for val in results_support]
        ],
        fill_color='lavender',
        align='left')
    )])
    
    fig.update_layout(title=f"L1 Minimization Error for {num_trials} trials, A in R^({m}x{n})")
    fig.show()
    fig_err = go.Figure()
    fig_err.add_trace(go.Scatter(
        x=dist_list,
        y=results_mean,
        name = "diff_mean"
    ))
    fig_err.add_trace(go.Scatter(
        x=dist_list,
        y=results_min,
        name = "diff_min"
    ))
    fig_err.add_trace(go.Scatter(
        x=dist_list,
        y=results_max,
        name = "diff_max"
    ))
    fig_err.update_layout(
        title="Comparaison des erreurs L1 (moyenne, min, max)",
        xaxis_title="Méthode",
        yaxis_title="Erreur L1"
    )
    fig_err.show()
    fig_sup = go.Figure()
    fig_sup.add_trace(go.Bar(
        x=dist_list,
        y=results_support,
        name="Mean Support Difference"
    ))
    fig_sup.update_layout(
        title="Comparaison du support (écart moyen en nombre d'éléments non nuls)",
        xaxis_title="Méthode",
        yaxis_title="Écart moyen de support"
    )
    fig_sup.show()
    return dist_list, results_mean, results_min, results_max
