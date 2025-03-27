import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
from scipy.stats import t
from scipy.special import gamma
import time
import ipywidgets as widgets
from ipywidgets import interactive_output, VBox, HBox

cache_results = {}

def generator_(n=10000, m=10):
    matrice_tab = []
    name_tab = []
    time_tab = []

    start = time.time()
    A = np.sqrt(12)*(np.random.rand(m, n) - 1/2*np.ones((m, n)))/np.sqrt(n)
    end = time.time()
    matrice_tab.append(A)
    name_tab.append('Uniform')
    time_tab.append(end-start)
    
    start = time.time()
    A = np.random.randn(m, n) / np.sqrt(n)
    matrice_tab.append(A)
    name_tab.append('Gaussian')
    time_tab.append(end-start)

    start = time.time()
    A = np.random.choice([-1, 1], size=(m, n)) / np.sqrt(n)
    end = time.time()
    matrice_tab.append(A)
    name_tab.append('Rademacher')
    time_tab.append(end-start)

    # start = time.time()
    # A = (np.random.binomial(1, 0.5, size=(m, n)) - 0.5) / (np.sqrt(0.25 * n))
    # end = time.time()
    # matrice_tab.append(A)
    # name_tab.append('Bernoulli')
    # time_tab.append(end-start)

    start = time.time()
    A = np.random.laplace(loc=0, scale=1/np.sqrt(2*n), size=(m, n))
    end = time.time()
    matrice_tab.append(A)
    name_tab.append('Laplace')
    time_tab.append(end-start)
    
    start = time.time()
    A = (np.random.exponential(scale=1, size=(m, n)) - 1) / np.sqrt(1 * n)
    end = time.time()
    matrice_tab.append(A); name_tab.append('Exponential')
    time_tab.append(end-start)

    shape = 2
    start = time.time()
    A = (np.random.gamma(shape, scale=1, size=(m, n)) - shape) / np.sqrt(shape * n)
    end = time.time()
    matrice_tab.append(A); name_tab.append('Gamma(shape=2)')
    time_tab.append(end-start)

    alpha, beta = 2, 5
    mean = alpha/(alpha+beta)
    var = alpha*beta/(((alpha+beta)**2)*(alpha+beta+1))
    start = time.time()
    A = (np.random.beta(alpha, beta, size=(m, n)) - mean) / np.sqrt(var * n)
    end = time.time()
    matrice_tab.append(A); name_tab.append('Beta(2,5)')
    time_tab.append(end-start)

    df = 4
    A = (np.random.chisquare(df, size=(m, n)) - df) / np.sqrt(2*df*n)
    end = time.time()
    matrice_tab.append(A); name_tab.append('Chi-square(df=4)')
    time_tab.append(end-start)

    df = 5
    var_t = df/(df-2)
    start = time.time()
    A = np.random.standard_t(df, size=(m, n)) / np.sqrt(var_t * n)
    end = time.time()
    matrice_tab.append(A); name_tab.append('Student t(df=5)')
    time_tab.append(end-start)

    k = 1.5
    mean = gamma(1+1/k)
    var = gamma(1+2/k) - mean**2
    start = time.time()
    A = (np.random.weibull(k, size=(m, n)) - mean) / np.sqrt(var * n)
    end = time.time()
    matrice_tab.append(A); name_tab.append('Weibull(k=1.5)')
    time_tab.append(end-start)

    var_log = (np.pi**2)/3
    start = time.time()
    A = np.random.logistic(loc=0, scale=1, size=(m, n)) / np.sqrt(var_log * n)
    end = time.time()
    matrice_tab.append(A); name_tab.append('Logistic')
    time_tab.append(end-start)

    lam = 1
    start = time.time()
    A = (np.random.poisson(lam, size=(m, n)) - lam) / np.sqrt(lam * n)
    end = time.time()
    matrice_tab.append(A); name_tab.append('Poisson(λ=1)')
    time_tab.append(end-start)

    alpha = 3
    mean = alpha/(alpha-1)
    var = (alpha/(alpha-2)) - mean**2
    start = time.time()
    A = (np.random.pareto(alpha, size=(m, n)) + 1 - mean) / np.sqrt(var * n)
    end = time.time()
    matrice_tab.append(A); name_tab.append('Pareto(α=3)')
    time_tab.append(end-start)

    start = time.time()
    X = t.rvs(df=3, size=(m, n))
    A = X / (np.std(X) * np.sqrt(n))
    end = time.time()
    matrice_tab.append(A)
    name_tab.append('Student-t(df=3)')
    time_tab.append(end-start)
    return matrice_tab, name_tab, time_tab


def compare_(n=10000, m=10, nb_sim = 1000):
    mark = True
    for _ in tqdm(range (nb_sim)):
        matrice_tab, name_tab, time_tab = generator_(n=n, m=m)
        if mark:
            erreur = [[] for _ in range(len(name_tab))]
            time = [[] for _ in range(len(name_tab))]
            mark = False
        for i in range (len(matrice_tab)):
            A = matrice_tab[i]
            print(A)
            prod = np.dot(A,A.T)
            diag = np.diagonal(prod)-1
            masque = np.eye(prod.shape[0], dtype=bool)
            elements_hors_diagonale = prod[~masque]
            erreur[i].append(np.mean(np.abs(np.concatenate([elements_hors_diagonale, diag]))))
            time[i].append(time_tab[i])
    erreur = [np.mean(np.array(erreur[i])) for i in range (len(erreur))]
    time   = [np.mean(np.array(time[i])) for i in range (len(time))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = name_tab, y = erreur,mode='markers', marker=dict(size = 20, color = 'darkblue')))
    fig.update_layout(
                title="Comparing the different densities",
                xaxis_title="Density",
                yaxis_title="L1 Loss"
            )
    fig.show()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = name_tab, y = time,mode='markers', marker=dict(size = 20, color = 'darkblue')))
    fig.update_layout(
                title="Comparing the different densities (time generation)",
                xaxis_title="Density",
                yaxis_title="Generation time"
            )
    fig.show()
    
def is_orthogonal(A, tol=1e-8):
    m, _ = A.shape
    return np.allclose(np.eye(m), A.T @ A, atol=tol)


def compare_(n=10000, m=10, nb_sim = 1000):
    mark = True
    for _ in tqdm(range (nb_sim)):
        matrice_tab, name_tab, time_tab = generator_(n=n, m=m)
        if mark:
            erreur = [[] for _ in range(len(name_tab))]
            time = [[] for _ in range(len(name_tab))]
            mark = False
        for i in range (len(matrice_tab)):
            A = matrice_tab[i]
            prod = np.dot(A,A.T)
            diag = np.diagonal(prod)-1
            masque = np.eye(prod.shape[0], dtype=bool)
            elements_hors_diagonale = prod[~masque]
            erreur[i].append(np.mean(np.abs(np.concatenate([elements_hors_diagonale, diag]))))
            time[i].append(time_tab[i])
    erreur = [np.mean(np.array(erreur[i])) for i in range (len(erreur))]
    time   = [np.mean(np.array(time[i])) for i in range (len(time))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = name_tab, y = erreur,mode='markers', marker=dict(size = 20, color = 'darkblue')))
    fig.update_layout(
                title="Comparing the different densities",
                xaxis_title="Density",
                yaxis_title="L1 Loss"
            )
    fig.show()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = name_tab, y = time,mode='markers', marker=dict(size = 20, color = 'darkblue')))
    fig.update_layout(
                title="Comparing the different densities (time generation)",
                xaxis_title="Density",
                yaxis_title="Generation time"
            )
    fig.show()
    
def visu_pro(A):
    prod = np.dot(A,A.T)
    diag = np.diagonal(prod)
    masque = np.eye(prod.shape[0], dtype=bool)
    elements_hors_diagonale = prod[~masque]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(diag)),y=diag))
    fig.update_layout(
                title="Diagonale",
                xaxis_title="valeur",
                yaxis_title="elements",
                plot_bgcolor='#D3D3D3',
                paper_bgcolor='#D3D3D3',
                xaxis=dict(showgrid=True, gridcolor='#808080'),
                yaxis=dict(showgrid=True, gridcolor='#808080')
            )
    fig.show()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(elements_hors_diagonale)),y=elements_hors_diagonale))
    fig.update_layout(
                title="Hors diagonale",
                xaxis_title="valeur",
                yaxis_title="elements"
            )
    fig.show()
    
def compare_g(n, m, nb_sim=100, visu = True):
    global cache_results
    key = (n, m)
    if key in cache_results:
        erreur_moy, temps_moy, name_tab = cache_results[key]
    else:
        mark = True
        for _ in range(nb_sim):
            matrice_tab, name_tab, time_tab = generator_(n=n, m=m)
            if mark:
                erreur = [[] for _ in range(len(name_tab))]
                temps  = [[] for _ in range(len(name_tab))]
                mark = False
            for i in range(len(matrice_tab)):
                A = matrice_tab[i]
                prod = np.dot(A, A.T)
                diag = np.diagonal(prod) - 1
                masque = np.eye(prod.shape[0], dtype=bool)
                elements_hors_diagonale = prod[~masque]
                erreur[i].append(np.mean(np.abs(np.concatenate([elements_hors_diagonale, diag]))))
                temps[i].append(time_tab[i])
        erreur_moy = [np.mean(np.array(erreur[i])) for i in range(len(erreur))]
        temps_moy  = [np.mean(np.array(temps[i])) for i in range(len(temps))]
        cache_results[key] = (erreur_moy, temps_moy, name_tab)
    if visu:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=name_tab, y=erreur_moy, mode='markers',
                                marker=dict(size=20, color='darkblue')))
        fig1.update_layout(
            title="Comparaison des distributions (L1 Loss moyen)",
            xaxis_title="Distribution",
            yaxis_title="L1 Loss"
        )
        fig1.show()
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=name_tab, y=temps_moy, mode='markers',
                                marker=dict(size=20, color='darkblue')))
        fig2.update_layout(
            title="Comparaison des distributions (Temps de génération moyen)",
            xaxis_title="Distribution",
            yaxis_title="Temps (s)"
        )
        fig2.show()
    
def compare(n_values,m_values):
    for n in tqdm(n_values, desc="n values"):
        for m in m_values:
            compare_g(n, m, visu = False)
    n_slider = widgets.SelectionSlider(
        options=n_values,
        value=n_values[2],
        description='n'
    )
    m_slider = widgets.SelectionSlider(
        options=m_values,
        value=m_values[2],
        description='m'
    )
    out = interactive_output(compare_g, {'n': n_slider, 'm': m_slider})
    display(VBox([out, HBox([n_slider, m_slider])]))
    
def get_data(n, m, nb_simu=100):
    global cache_results
    key = (n, m)
    if key in cache_results:
        return cache_results[key]
    else:
        accumulated_errors = None
        accumulated_times = None
        for _ in range(nb_simu):
            matrice_tab, name_tab, time_tab = generator_(n=n, m=m)
            errors = []
            for A in matrice_tab:
                prod = np.dot(A, A.T)
                diag = np.diagonal(prod) - 1
                masque = np.eye(prod.shape[0], dtype=bool)
                hors_diag = prod[~masque]
                errors.append(np.mean(np.abs(np.concatenate([hors_diag, diag]))))
            if accumulated_errors is None:
                accumulated_errors = np.array(errors)
                accumulated_times = np.array(time_tab)
            else:
                accumulated_errors += np.array(errors)
                accumulated_times += np.array(time_tab)
        
        error_moyenne = list(accumulated_errors / nb_simu)
        temps_moyens = list(accumulated_times / nb_simu)
        cache_results[key] = name_tab, error_moyenne, temps_moyens
        return name_tab, error_moyenne, temps_moyens


def show_error(n, m):
    name_tab, erreur_list, time_tab = get_data(n, m)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=name_tab, 
        y=erreur_list, 
        mode='markers', 
        marker=dict(size=20, color='darkblue')
    ))
    fig.update_layout(
        title="Comparing the different densities (L1 Loss)",
        xaxis_title="Density",
        yaxis_title="L1 Loss"
    )
    fig.show()

def show_time(n, m):
    """Affiche le graphique du temps de génération."""
    name_tab, erreur_list, time_tab = get_data(n, m)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=name_tab, 
        y=time_tab, 
        mode='markers', 
        marker=dict(size=20, color='darkblue')
    ))
    fig.update_layout(
        title="Comparing the different densities (time generation)",
        xaxis_title="Density",
        yaxis_title="Generation time"
    )
    fig.show()

def compare_two_sliders(n_values, m_values):
    for n in tqdm(n_values, desc="n values"):
        for m in m_values:
            get_data(n, m)
    n_slider_err = widgets.SelectionSlider(
        options=n_values,
        value=n_values[2],
        description='n (Err)'
    )
    m_slider_err = widgets.SelectionSlider(
        options=m_values,
        value=m_values[2],
        description='m (Err)'
    )
    out_err = interactive_output(
        show_error, 
        {'n': n_slider_err, 'm': m_slider_err}
    )
    n_slider_time = widgets.SelectionSlider(
        options=n_values,
        value=n_values[2],
        description='n (Time)'
    )
    m_slider_time = widgets.SelectionSlider(
        options=m_values,
        value=m_values[2],
        description='m (Time)'
    )
    out_time = interactive_output(
        show_time, 
        {'n': n_slider_time, 'm': m_slider_time}
    )
    
    box_err = VBox([
        out_err, 
        HBox([n_slider_err, m_slider_err])
    ])
    box_time = VBox([
        out_time, 
        HBox([n_slider_time, m_slider_time])
    ])
    
    display(box_err, box_time)


