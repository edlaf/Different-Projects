import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
import time
import Rand_Mat as RM
import ipywidgets as widgets
from ipywidgets import interactive_output, VBox, HBox

def generator_(n = 1000, m = 10, Squared = True):
    if Squared:
        tab_time = []
        name_tab = []
        matrice_tab = []
        
        start = time.time()
        cols = np.random.choice(n, size=m, replace=False)
        P = np.zeros((m, n), dtype=int)
        for i, col in enumerate(cols):
            P[i, col] = 1
        end = time.time()
        tab_time.append(end-start)
        name_tab.append('Permutation')
        matrice_tab.append(P)

        start = time.time()
        v = np.random.randn(n)
        v /= np.linalg.norm(v)
        P = np.eye(n) - 2*np.outer(v, v)
        end = time.time()
        tab_time.append(end-start); name_tab.append('Householder')
        matrice_tab.append(P)

        start = time.time()
        Q, _ = np.linalg.qr(np.random.randn(n, m), mode='reduced')
        Q = np.transpose(Q)
        end = time.time()
        tab_time.append(end-start); name_tab.append('QR')
        matrice_tab.append(Q)

        start = time.time()
        U, _, _ = np.linalg.svd(np.random.randn(n, n), full_matrices=False)
        end = time.time()
        tab_time.append(end-start); name_tab.append('SVD')
        matrice_tab.append(U)

        start = time.time()
        X = np.random.randn(m, n)
        Q = np.zeros((m, n))
        for i in range(m):
            v = X[i, :].copy()
            for j in range(i):
                # projection de la ième ligne sur la jème ligne déjà orthonormée
                v -= np.dot(Q[j, :], X[i, :]) * Q[j, :]
            Q[i, :] = v / np.linalg.norm(v)
        end = time.time()
        tab_time.append(end-start); name_tab.append('Power (GS)')
        matrice_tab.append(Q)

        # start = time.time()
        # X = np.random.randn(n, n)
        # lr = 0.1
        # for _ in range(50):
        #     grad = X @ (X.T @ X - np.eye(n))
        #     X -= lr * grad
        # Q, _ = np.linalg.qr(X)
        # end = time.time()
        # tab_time.append(end-start); name_tab.append('Gradient descent')
        # matrice_tab.append(Q)
    
    return matrice_tab, name_tab, tab_time



def compare_(n=10000, m=10, nb_sim = 1000, Squared = True):
    mark = True
    for _ in tqdm(range (nb_sim)):
        matrice_tab, name_tab, time_tab = generator_(n=n, m=m, Squared=Squared)
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
    
cache_results = {}
cache_results_RM = {}


def get_data(n, m, nb_simu=100, Squared=True):
    global cache_results
    key = (n, m, nb_simu, Squared)
    if key in cache_results:
        return cache_results[key]
    else:
        accumulated_errors = None
        accumulated_times = None
        for _ in range(nb_simu):
            matrice_tab, name_tab, time_tab = generator_(n=n, m=m, Squared=Squared)
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
        cache_results[key] = (name_tab, error_moyenne, temps_moyens)
        return name_tab, error_moyenne, temps_moyens


def get_data_RM(n, m, nb_simu=100, Squared=True):
    global cache_results_RM
    key = (n, m, nb_simu, Squared)
    if key in cache_results_RM:
        return cache_results_RM[key]
    else:
        accumulated_errors = None
        accumulated_times = None
        for _ in range(nb_simu):
            matrice_tab, name_tab, time_tab = RM.generator_(n=n, m=m)
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
        cache_results_RM[key] = (name_tab, error_moyenne, temps_moyens)
        return name_tab, error_moyenne, temps_moyens

def compare_table_combined(n, m_tab, nb_simu=1000, Squared=True):
    results = {}
    method_names = []
    
    for m in tqdm(m_tab):
        names_det, error_det, _ = get_data(n, m, nb_simu=nb_simu, Squared=Squared)
        names_rand, error_rand, _ = get_data_RM(n, m, nb_simu=nb_simu, Squared=Squared)
        
        if not results:
            for name in names_det:
                results[name] = []
            for name in names_rand:
                results[name] = []
            method_names = names_det + names_rand
        
        for i, name in enumerate(names_det):
            results[name].append(error_det[i])
        for i, name in enumerate(names_rand):
            results[name].append(error_rand[i])
    
    fig = go.Figure()
    for method in method_names:
        fig.add_trace(go.Scatter(x=m_tab, y=results[method], mode='lines+markers', name=method))
    fig.update_layout(
        title=f"Précision (L1 Loss) en fonction de m pour n = {n}",
        xaxis_title="m",
        yaxis_title="L1 Loss (précision)",
        xaxis_type="log"
    )
    fig.show()


def interactive_compare_combined(n_tab, m_tab, nb_simu=100, Squared=True):
    for n in n_tab:
        for m in tqdm(m_tab):
            get_data(n, m, nb_simu=nb_simu, Squared=Squared)
            get_data_RM(n, m, nb_simu=nb_simu, Squared=Squared)
    n_slider = widgets.SelectionSlider(
        options=n_tab,
        value=n_tab[len(n_tab)//2],
        description='n'
    )
    out = interactive_output(lambda n: compare_table_combined(n, m_tab, nb_simu, Squared), {'n': n_slider})
    display(VBox([n_slider, out]))

def compare_table_combined_time(n, m_tab, nb_simu=1000, Squared=True):
    results = {}
    method_names = []
    
    for m in tqdm(m_tab, desc="m values"):
        names_det, _, time_det = get_data(n, m, nb_simu=nb_simu, Squared=Squared)
        names_rand, _, time_rand = get_data_RM(n, m, nb_simu=nb_simu, Squared=Squared)
        
        if not results:
            for name in names_det:
                results[name] = []
            for name in names_rand:
                results[name] = []
            method_names = names_det + names_rand
        
        for i, name in enumerate(names_det):
            results[name].append(time_det[i])
        for i, name in enumerate(names_rand):
            results[name].append(time_rand[i])
    
    fig = go.Figure()
    for method in method_names:
        fig.add_trace(go.Scatter(x=m_tab, y=results[method], mode='lines+markers', name=method))
    fig.update_layout(
        title=f"Temps de génération en fonction de m pour n = {n}",
        xaxis_title="m",
        yaxis_title="Temps de génération (s)",
        yaxis_type="log",
        xaxis_type="log"
    )
    fig.show()


def interactive_compare_combined_time(n_tab, m_tab, nb_simu=1000, Squared=True):
    n_slider = widgets.SelectionSlider(
        options=n_tab,
        value=n_tab[len(n_tab)//2],
        description='n'
    )
    out = interactive_output(lambda n: compare_table_combined_time(n, m_tab, nb_simu, Squared), {'n': n_slider})
    display(VBox([n_slider, out]))




