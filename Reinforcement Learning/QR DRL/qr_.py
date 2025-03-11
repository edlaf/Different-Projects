import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import matplotlib.pyplot as plt
pd.set_option("display.max_rows", 100)



'''
Market Simulation using a model similar to the Queue Reactive model

Possibility to interact with the market at each step and to create an agent for high frequency strategies using reinforcement learning

It is just a model, and was not meant to reproduce every caracteristics of a real market.
'''

class One_step:
    def __init__(self, tab_cancel, tab_order, tab_add):
        self.n_limit = int(len(tab_add) / 2)
        self.intensities_order  = tab_order
        self.intensities_add    = tab_add
        self.intensities_cancel = tab_cancel
    
    def next_step(self, state, bid, ask, next_size_cancel, next_size_order):
        """
        Modèle purement Poissonien dont l'intensité dépend de l'imbalance.
        Retourne le prochain événement possible :
          - time_f   : temps jusqu'au prochain événement
          - side_f   : côté ('A' pour ask, 'B' pour bid)
          - limit_f  : indice de la limite où se produit l'événement
          - action_f : type d'événement ('Order', 'Add' ou 'Cancel')
        """
        times_order  = []
        action_order = ['A', 'B']
        print(len(self.intensities_order))
        b = False
        times_order.append(np.random.exponential(self.intensities_order[0](state)*(ask[0]*10+1)))
        times_order.append(np.random.exponential(self.intensities_order[1](state)*(bid[0]*10+1)))
        for i in range(len(self.intensities_order)):
            if action_order[i] == 'A' and ask[0] < next_size_order:
                times_order[0] = np.infty
            elif action_order[i] == 'B' and bid[0] < next_size_order:
                times_order[1] = np.infty

        times_order  = np.array(times_order)
        order_idx = np.argmin(times_order)
        print(order_idx)
        print(times_order)
        action_order_chosen = action_order[order_idx]
        time_order = times_order[order_idx]
        
        times_add  = []
        action_add = ['A' for _ in range(self.n_limit)] + ['B' for _ in range(self.n_limit)]
        for i in range(len(self.intensities_add)):
            if action_add[i] == 'A':
                times_add.append(np.random.exponential(self.intensities_add[i](state)*(ask[i]*10+1)))
            if action_add[i] == 'B':
                index = i - self.n_limit
                times_add.append(np.random.exponential(self.intensities_add[i](state)*(bid[index]*10+1)))
        times_add  = np.array(times_add)
        add_idx = np.argmin(times_add)
        limit_add  = add_idx % self.n_limit
        action_add_chosen = action_add[add_idx]
        time_add = times_add[add_idx]
        
        times_cancel  = []
        action_cancel = ['A' for _ in range(self.n_limit)] + ['B' for _ in range(self.n_limit)]
        for i in range(len(self.intensities_cancel)):
            if i < self.n_limit:
                if i == self.n_limit - 1 or ask[i] < next_size_cancel:
                    times_cancel.append(np.infty)
                else:
                    times_cancel.append(np.random.exponential(self.intensities_cancel[i](state)*(ask[i]*10+1)))
            else:
                index = i - self.n_limit
                if index == self.n_limit - 1 or bid[index] < next_size_cancel:
                    times_cancel.append(np.infty)
                else:
                    times_cancel.append(np.random.exponential(self.intensities_cancel[i](state)*(bid[index]*10+1)))
        times_cancel  = np.array(times_cancel)
        cancel_idx = np.argmin(times_cancel)
        limit_cancel  = cancel_idx % self.n_limit
        action_cancel_chosen = action_cancel[cancel_idx]
        time_cancel = times_cancel[cancel_idx]
        
        times = np.array([time_order, time_add, time_cancel])
        actions = np.array([action_order_chosen, action_add_chosen, action_cancel_chosen])
        event_idx = np.argmin(times)
        time_f = times[event_idx]
        side_f = actions[event_idx]
        if event_idx == 0:
            limit_f = 0
            action_f = 'Order'
        elif event_idx == 1:
            limit_f = limit_add
            action_f = 'Add'
        else:
            limit_f = limit_cancel
            action_f = 'Cancel'
        return time_f, side_f, limit_f, action_f

class Qr:
    def __init__(self, tab_cancel, tab_order, tab_add, price_0, tick, theta, nb_of_action,
                 liquidy_last_lim, size_max, lambda_event, event_prob):
        """
        Paramètres :
          - tab_cancel, tab_order, tab_add : fonctions d'intensité
          - price_0      : prix initial
          - tick         : tick
          - theta        : paramètre de mean-reversion
          - nb_of_action : nombre d'étapes de la simulation
          - liquidy_last_lim : liquidité du dernier niveau
          - size_max     : [size_max_add, size_max_cancel, size_max_order]
          - lambda_event : temps moyen d'un événement d'actualité
          - event_prob   : probabilité d'un événement
        """
        self.n_limit = int(len(tab_add) / 2)
        self.bid = [0 for _ in range(self.n_limit)]
        self.ask = [0 for _ in range(self.n_limit)]
        self.time = 0
        self.df_evolution = pd.DataFrame(columns=['Time', 'Limit', 'Side', 'Action', 'Price', 'Size',
                                                    'Bid_1', 'Ask_1', 'Bid_2', 'Ask_2', 'Bid_3', 'Ask_3', 'Obs'])
        self.steps = One_step(tab_cancel, tab_order, tab_add)
        self.price = price_0
        self.tick = tick
        self.nb_of_action = nb_of_action
        self.theta = theta
        self.state = 0
        self.liquidy_last = liquidy_last_lim
        self.size_max_add = size_max[0]
        self.size_max_cancel = size_max[1]
        self.size_max_order = size_max[2]
        self.event_prob = event_prob
        self.lambda_event = lambda_event
        self.length_event = 0
        self.is_event = False
        self.price_0 = price_0
        
    def intiate_market(self, initial_ask, initial_bid):
        """Initialise le marché avec les tailles initiales."""
        self.bid = initial_bid.copy()
        self.ask = initial_ask.copy()
        self.time = 0
        self.df_evolution = self.df_evolution.iloc[0:0]
        self.df_evolution.loc[len(self.df_evolution)] = [self.time, 'N/A', 'N/A', 'N/A', self.price, 'N/A',
                                                        self.bid[0], self.ask[0], self.bid[1], self.ask[1],
                                                        self.bid[2], self.ask[2], 'Opening']
    
    def state_(self):
        """
        Calcule l'imbalance en privilégiant le niveau disponible le plus pertinent.
        Par construction, self.ask[2] et self.bid[2] devraient rester égaux à liquidy_last.
        """
        if (self.ask[1] + self.bid[1]) == 0:
            return (self.ask[2] - self.bid[2]) / (self.ask[2] + self.bid[2])
        if (self.ask[0] + self.bid[0]) == 0:
            return (self.ask[1] - self.bid[1]) / (self.ask[1] + self.bid[1])
        return (self.ask[0] - self.bid[0]) / (self.ask[0] + self.bid[0])
    
    def step(self):
        """Simule une étape du marché et met à jour le carnet d'ordres."""
        # np.random.seed(42)
        if not self.is_event:
            if np.random.uniform() > self.event_prob:
                next_size_add = np.random.randint(1, self.size_max_add)
                next_size_cancel = np.random.randint(1, self.size_max_cancel)
                next_size_order = np.random.randint(1, self.size_max_order)
                time_f, side_f, limit_f, action_f = self.steps.next_step(
                    self.state_(), self.bid, self.ask, next_size_cancel, next_size_order)
                tab_next_step = [-1 for _ in range(13)]
                tab_next_step[-1] = 'N/A'
            else:
                self.is_event = True
                self.length_event = np.random.poisson(self.lambda_event[np.random.randint(len(self.lambda_event))]) + 1
                next_size_add = np.random.randint(1, max(1, self.size_max_add - 2))
                next_size_cancel = np.random.randint(1, max(1, self.size_max_cancel - 2))
                next_size_order = np.random.randint(2, self.size_max_order)
                time_f, side_f, limit_f, action_f = self.steps.next_step(
                    self.state_(), self.bid, self.ask, next_size_cancel, next_size_order)
                time_f = time_f / self.length_event
                tab_next_step = [-1 for _ in range(13)]
                tab_next_step[-1] = 'Start_event'
                self.length_event -= 1
        else:
            if self.length_event == 0:
                self.is_event = False
                next_size_add = np.random.randint(1, self.size_max_add)
                next_size_cancel = np.random.randint(1, self.size_max_cancel)
                next_size_order = np.random.randint(1, self.size_max_order)
                time_f, side_f, limit_f, action_f = self.steps.next_step(
                    self.state_(), self.bid, self.ask, next_size_cancel, next_size_order)
                tab_next_step = [-1 for _ in range(13)]
                tab_next_step[-1] = 'End_event'
            else:
                next_size_add = np.random.randint(1, max(1, self.size_max_add + 1))
                next_size_cancel = np.random.randint(1, max(1, self.size_max_cancel + 1))
                next_size_order = np.random.randint(1, max(1, self.size_max_order - 2))
                time_f, side_f, limit_f, action_f = self.steps.next_step(
                    self.state_(), self.bid, self.ask, next_size_cancel, next_size_order)
                time_f = time_f / self.length_event
                tab_next_step = [-1 for _ in range(13)]
                tab_next_step[-1] = 'In_event'
                self.length_event -= 1
        
        if side_f == 'A': # Ask
            current_ask = self.ask[limit_f]
            if action_f == 'Order':
                tab_next_step[5] = next_size_order
                next_ask = current_ask - next_size_order
                tab_next_step[4] = self.price + self.tick / 2
                tab_next_step[0] = self.time + time_f
                tab_next_step[1] = limit_f
                tab_next_step[2] = side_f
                tab_next_step[3] = action_f
                if next_ask == 0: # Ask
                    tab_next_step[-1] = 'new_limit'
                    if np.random.uniform(0,1) > self.theta:
                        self.price += self.tick
                        self.ask[0] = self.ask[1]
                        self.ask[1] = self.ask[2]
                        self.ask[2] = self.liquidy_last
                        self.bid[2] = self.bid[1]
                        self.bid[1] = self.bid[0]
                        self.bid[0] = 0
                    # else:
                    #     self.price -= self.tick
                    #     self.ask[2] = self.ask[1]
                    #     self.ask[1] = self.ask[0]
                        self.ask[0] = 0
                    #     self.bid[0] = self.bid[1]
                    #     self.bid[1] = self.bid[2]
                    #     self.bid[2] = self.liquidy_last
                else:
                    self.ask[limit_f] -= next_size_order
            elif action_f == 'Cancel':
                tab_next_step[5] = next_size_cancel
                next_ask = current_ask - next_size_cancel
                tab_next_step[4] = self.price + (1+limit_f)*self.tick / 2
                tab_next_step[0] = self.time + time_f
                tab_next_step[2] = side_f
                tab_next_step[3] = action_f
                tab_next_step[1] = limit_f
                if next_ask == 0 and limit_f == 0:
                    tab_next_step[-1] = 'new_limit'
                    if np.random.uniform(0,1) > self.theta:
                        self.price += self.tick
                        self.ask[0] = self.ask[1]
                        self.ask[1] = self.ask[2]
                        self.ask[2] = self.liquidy_last
                        self.bid[2] = self.bid[1]
                        self.bid[1] = self.bid[0]
                        self.bid[0] = 0
                        print(self.bid[0])
                    else:
                        self.ask[0] = 0
                    #     self.price -= self.tick
                    #     self.ask[2] = self.ask[1]
                    #     self.ask[1] = self.ask[0]
                        # self.ask[0] = 0
                    #     self.bid[0] = self.bid[1]
                    #     self.bid[1] = self.bid[2]
                    #     self.bid[2] = self.liquidy_last
                else:
                    self.ask[limit_f] -= next_size_cancel
            elif action_f == 'Add':
                tab_next_step[5] = next_size_add
                next_ask = current_ask + next_size_add
                tab_next_step[4] = self.price + (1+limit_f)*self.tick / 2
                tab_next_step[0] = self.time + time_f
                tab_next_step[2] = side_f
                tab_next_step[3] = action_f
                tab_next_step[1] = limit_f
                self.ask[limit_f] += next_size_add
            
            tab_next_step[-6] = self.ask[0]
            tab_next_step[-7] = self.bid[0]
            tab_next_step[-5] = self.bid[1]
            tab_next_step[-4] = self.ask[1]
            tab_next_step[-2] = self.ask[2]
            tab_next_step[-3] = self.bid[2]
        
        if side_f == 'B':  # Côté Bid
            current_bid = self.bid[limit_f]
            if action_f == 'Order':
                tab_next_step[5] = next_size_order
                next_bid = current_bid - next_size_order
                tab_next_step[4] = self.price - self.tick / 2
                tab_next_step[0] = self.time + time_f
                tab_next_step[1] = limit_f
                tab_next_step[2] = side_f
                tab_next_step[3] = action_f
                if next_bid == 0:
                    tab_next_step[-1] = 'new_limit'
                    if np.random.uniform(0,1) < self.theta:
                        # self.price += self.tick
                        # self.ask[0] = self.ask[1]
                        # self.ask[1] = self.ask[2]
                        # self.ask[2] = self.liquidy_last
                        # self.bid[2] = self.bid[1]
                        # self.bid[1] = self.bid[0]
                        self.bid[0] = 0
                        butututu = 0
                    else:
                        self.price -= self.tick
                        self.ask[2] = self.ask[1]
                        self.ask[1] = self.ask[0]
                        self.ask[0] = 0
                        self.bid[0] = self.bid[1]
                        self.bid[1] = self.bid[2]
                        self.bid[2] = self.liquidy_last
                else:
                    self.bid[limit_f] -= next_size_order
                    
            elif action_f == 'Cancel':
                tab_next_step[5] = next_size_cancel
                next_bid = current_bid - next_size_cancel
                tab_next_step[4] = self.price - (1+limit_f)*self.tick / 2
                tab_next_step[0] = self.time + time_f
                tab_next_step[2] = side_f
                tab_next_step[3] = action_f
                tab_next_step[1] = limit_f
                if next_bid == 0 and limit_f == 0:
                    tab_next_step[-1] = 'new_limit'
                    if np.random.uniform(0,1) < self.theta:
                        # self.price += self.tick
                        # self.ask[0] = self.ask[1]
                        # self.ask[1] = self.ask[2]
                        # self.ask[2] = self.liquidy_last
                        # self.bid[2] = self.bid[1]
                        # self.bid[1] = self.bid[0]
                        self.bid[0] = 0
                        butututut = 0
                    else:
                        self.price -= self.tick
                        self.ask[2] = self.ask[1]
                        self.ask[1] = self.ask[0]
                        self.ask[0] = 0
                        self.bid[0] = self.bid[1]
                        self.bid[1] = self.bid[2]
                        self.bid[2] = self.liquidy_last
                else:
                    self.bid[limit_f] -= next_size_cancel
            elif action_f == 'Add':
                tab_next_step[5] = next_size_add
                next_bid = current_bid + 1
                tab_next_step[4] = self.price - (1+limit_f)*self.tick / 2
                tab_next_step[0] = self.time + time_f
                tab_next_step[2] = side_f
                tab_next_step[3] = action_f
                tab_next_step[1] = limit_f
                self.bid[limit_f] += next_size_add
            
            tab_next_step[-6] = self.ask[0]
            tab_next_step[-7] = self.bid[0]
            tab_next_step[-5] = self.bid[1]
            tab_next_step[-4] = self.ask[1]
            tab_next_step[-2] = self.ask[2]
            tab_next_step[-3] = self.bid[2]
        
        self.time += time_f
        return tab_next_step
    
    def run_market(self, initial_ask, initial_bid):
        """Lance une simulation complète du marché."""
        self.intiate_market(initial_ask, initial_bid)
        for i in range(self.nb_of_action):
            self.df_evolution.loc[len(self.df_evolution)] = self.step()
        return self.df_evolution
    
    def visu(self, initial_ask, initial_bid, price_only):
        """Visualise la simulation du marché."""
        df = self.run_market(initial_ask, initial_bid)
        df_1 = df[df['Limit'] == 0]
        df_price = df_1[df_1['Action'] == 'Order']
        df_2 = df[df['Limit'] == 1]
        df_3 = df[df['Limit'] == 2]
        df_4 = df[df['Obs'].isin(['In_event', 'End_event', 'Start_event'])]
        
        fig = go.Figure()
        if not price_only:
            fig.add_trace(go.Scatter(x=df_1['Time'], y=df_1['Price'], mode='markers', name="Limit_1",
                                     marker=dict(size=5, color="red", opacity=0.7)))
            fig.add_trace(go.Scatter(x=df_2['Time'], y=df_2['Price'], mode='markers', name="Limit_2",
                                     marker=dict(size=4, color="orange", opacity=0.6)))
            fig.add_trace(go.Scatter(x=df_3['Time'], y=df_3['Price'], mode='markers', name="Limit_3",
                                     marker=dict(size=3, color="gold", opacity=0.5)))
            fig.add_trace(go.Scatter(x=df_4['Time'], y=100*np.ones(len(df_4)), mode='markers', name="EVENT",
                                     marker=dict(size=4, color="black", opacity=0.8)))
        fig.add_trace(go.Scatter(x=df_price['Time'], y=df_price['Price'], mode='lines', name="Sell_Price",
                                 line=dict(width=2, color='darkred')))
        fig.update_layout(
            title="Market Simulation",
            xaxis_title="Time",
            yaxis_title="Price",
            plot_bgcolor='#D3D3D3',
            paper_bgcolor='#D3D3D3',
            xaxis=dict(showgrid=True, gridcolor='#808080'),
            yaxis=dict(showgrid=True, gridcolor='#808080')
        )
        fig.show()
    
    
    def run_and_trade_market(self, initial_ask, initial_bid):
        self.intiate_market(initial_ask, initial_bid)

class TradingAgent:
    def __init__(self, transaction_cost_cancel=0, transaction_cost_market=0):
        self.position = 0                # 0 ou 1 (nombre d'actifs détenus)
        self.cash = 1000.0
        self.entry_price = None
        # Ordres ADD actifs séparés par côté
        self.order_active_bid = None     # pour l'achat (ADD sur le bid)
        self.order_active_ask = None     # pour la vente (ADD sur le ask)
        self.order_price = None          # prix auquel l'ordre a été placé
        self.order_volume = 1            # taille de l'ordre (fixée à 1)
        # Quantités enregistrées lors du placement de l'ordre
        self.recorded_bid_volume = None  
        self.recorded_ask_volume = None  
        self.number_ask_volume_traded = None
        self.number_bid_volume_traded = None
        self.number_actif_init = 100

    def decide_action(self, market_state):
        """
        Stratégie de décision de l'agent.
        
        Si l'agent n'a pas d'actif (position == 0) :
          - S'il n'a pas d'ordre actif sur le bid (order_active_bid est None),
            actions possibles : "add_bid", "order_ask", "do nothing"
          - Sinon, actions possibles : "cancel_bid", "order_ask", "do nothing"
        
        Si l'agent possède un actif (position == 1) :
          - S'il n'a pas d'ordre actif sur le ask (order_active_ask est None),
            actions possibles : "order_bid", "add_ask", "do nothing"
          - Sinon, actions possibles : "cancel_ask", "order_bid", "do nothing"
        """
        if self.position == 0:
            if self.order_active_bid is None:
                actions = ["order_ask", "do nothing", "order_bid"] #["add_bid", "order_ask", "do nothing"]
            else:
                actions = ["cancel_bid", "order_ask", "do nothing"]
                print('ca ne doit jamais arriver')
        else:
            if self.order_active_ask is None:
                actions = ["order_ask", "order_bid","do nothing"] #["order_bid", "add_ask", "do nothing"]
            else:
                actions = ["cancel_ask", "order_bid", "do nothing"]
                print('ca ne doit jamais arriver')
        return np.random.choice(actions)

class QrWithAgent(Qr):  # Inhère de votre classe Qr existante
    def execute_agent_action(self, action, agent):
        """
        Exécute l'action de l'agent en répliquant la logique de mise à jour du carnet de step().
        Les tailles d'ordre sont fixées à 1.
        
        Pour un agent acheteur (position == 0) :
          - "order_ask" : consommer 1 unité du côté ask (achat immédiat)
          - "add_bid"   : ajouter 1 unité sur le bid (placer un ordre ADD)
          - "cancel_bid": retirer 1 unité du bid (annuler l'ordre ADD)
        
        Pour un agent vendeur (position == 1) :
          - "order_bid" : consommer 1 unité du côté bid (vente immédiate)
          - "add_ask"   : ajouter 1 unité sur le ask (placer un ordre ADD)
          - "cancel_ask": retirer 1 unité du ask (annuler l'ordre ADD)
        """
        next_size = 1
        time_f = 0  # action immédiate
        
        if agent.position == 0:
            if action == "order_ask":
                # Achat immédiat : consommer 1 unité du côté ask
                if self.ask[0] >= next_size:
                    self.ask[0] -= next_size
                else:
                    if np.random.uniform(0,1) > self.theta:
                        self.price += self.tick
                        self.ask[0] = self.ask[1]
                        self.ask[1] = self.ask[2]
                        self.ask[2] = self.liquidy_last
                        self.bid[2] = self.bid[1]
                        self.bid[1] = self.bid[0]
                        self.bid[0] = 0
                agent.position += 1
                agent.cash -= (self.price + self.tick/2)
                agent.order_active_bid = None
            elif action == "add_bid":
                # Placer un ordre ADD sur le bid
                self.bid[0] += next_size
                agent.order_active_bid = "add_bid"
                agent.order_price = self.price + self.tick/2
            elif action == "cancel_bid":
                if self.bid[0] >= next_size:
                    self.bid[0] -= next_size
                agent.order_active_bid = None
        else:
            if action == "order_bid":
                # Vente immédiate : consommer 1 unité du côté bid
                if self.bid[0] >= next_size:
                    self.bid[0] -= next_size
                else:
                    if np.random.uniform(0,1) < self.theta:
                        self.price += self.tick
                        self.ask[0] = self.ask[1]
                        self.ask[1] = self.ask[2]
                        self.ask[2] = self.liquidy_last
                        self.bid[2] = self.bid[1]
                        self.bid[1] = self.bid[0]
                        self.bid[0] = 0
                agent.position -= 1
                agent.cash += (self.price - self.tick/2)
                agent.order_active_ask = None
            elif action == "add_ask":
                # Placer un ordre ADD sur le ask
                self.ask[0] += next_size
                agent.order_active_ask = "add_ask"
                agent.order_price = self.price - self.tick/2
            elif action == "cancel_ask":
                if self.ask[0] >= next_size:
                    self.ask[0] -= next_size
                agent.order_active_ask = None
                
    def execute_agent_action_order(self, action, agent):
        next_size = 1
        time_f = 0
    
        if action == "order_ask":
            if self.ask[0] >= next_size:
                self.ask[0] -= next_size
            else:
                if np.random.uniform(0,1) > self.theta:
                    self.price += self.tick
                    self.ask[0] = self.ask[1]
                    self.ask[1] = self.ask[2]
                    self.ask[2] = self.liquidy_last
                    self.bid[2] = self.bid[1]
                    self.bid[1] = self.bid[0]
                    self.bid[0] = 0
            agent.position += 1
            agent.cash -= (self.price + self.tick/2)
            agent.order_active_bid = None
        if action == "order_bid":
            if self.bid[0] >= next_size:
                self.bid[0] -= next_size
            else:
                if np.random.uniform(0,1) < self.theta:
                    self.price += self.tick
                    self.ask[0] = self.ask[1]
                    self.ask[1] = self.ask[2]
                    self.ask[2] = self.liquidy_last
                    self.bid[2] = self.bid[1]
                    self.bid[1] = self.bid[0]
                    self.bid[0] = 0
            agent.position -= 1
            agent.cash += (self.price - self.tick/2)
            agent.order_active_ask = None

    def steplog(self, initial_ask, initial_bid, agent, cash_initial=1000.0):
        """
        Exécute la simulation du marché avec interaction de l'agent et enregistre un DataFrame enrichi.
        Colonnes d'origine (13) auxquelles on ajoute :
          - 'PnL'           : net_worth - cash_initial, où net_worth = agent.cash + agent.position * self.price
          - 'NB_asset'      : nombre d'actifs détenus (agent.position)
          - 'Limit_add_bid' : 1 si un ordre ADD actif sur le bid, sinon 0
          - 'Limit_add_ask' : 1 si un ordre ADD actif sur le ask, sinon 0
          - 'Agent_Action'  : l'action de l'agent si il intervient ce move, "NA" sinon.
          
        Tous les 20 moves, l'agent intervient (aucun appel à self.step()).
        De plus, si une nouvelle limite est déclenchée et que l'agent possède un ordre ADD,
        l'ordre est annulé et cette annulation est notée dans la même ligne de log.
        """
        self.intiate_market(initial_ask, initial_bid)
        df_columns = ['Time', 'Limit', 'Side', 'Action', 'Price', 'Size',
                      'Bid_1', 'Ask_1', 'Bid_2', 'Ask_2', 'Bid_3', 'Ask_3', 'Obs',
                      'PnL', 'NB_asset', 'Limit_add_bid', 'Limit_add_ask', 'Agent_Action']
        df_log = pd.DataFrame(columns=df_columns)
        
        # Réinitialisation du suivi des ordres de l'agent
        agent.order_active_bid = None
        agent.order_active_ask = None
        agent.order_price = None
        agent.recorded_bid_volume = None
        agent.recorded_ask_volume = None

        last_agent_time = self.time

        # Enregistrement de l'état initial
        net_worth = agent.cash + (agent.number_actif_init + agent.position) * self.price
        pnl = net_worth - cash_initial - agent.number_actif_init * self.price_0
        nb_asset = agent.number_actif_init + agent.position
        limit_add_bid = 1 if agent.order_active_bid is not None else 0
        limit_add_ask = 1 if agent.order_active_ask is not None else 0
        init_row = [self.time, 'N/A', 'N/A', 'N/A', self.price, 'N/A',
                    self.bid[0], self.ask[0], self.bid[1], self.ask[1],
                    self.bid[2], self.ask[2], 'Opening', pnl, nb_asset, limit_add_bid, limit_add_ask, "NA"]
        df_log.loc[len(df_log)] = init_row
        for i in range(self.nb_of_action):
            # Exécuter un move de marché et récupérer la ligne de base
            if i % 20 == 0:

                # Move agent : on n'appelle PAS self.step() pour ce move
                market_state = {"price": self.price, "bid": self.bid.copy(), "ask": self.ask.copy(), "time": self.time}
                agent_action = agent.decide_action(market_state)
                self.execute_agent_action(agent_action, agent)
                self.execute_agent_action_order(agent_action, agent)
                last_agent_time = self.time  # mise à jour de la dernière action

                # Lors d'un ADD, enregistrer la quantité actuelle
                if agent.order_active_bid == "add_bid":
                    agent.recorded_bid_volume = self.bid[0]
                if agent.order_active_ask == "add_ask":
                    agent.recorded_ask_volume = self.ask[0]

                net_worth = agent.cash + (agent.number_actif_init + agent.position) * self.price
                pnl = net_worth - cash_initial - agent.number_actif_init * self.price_0
                nb_asset = agent.number_actif_init + agent.position
                limit_add_bid = 1 if agent.order_active_bid is not None else 0
                limit_add_ask = 1 if agent.order_active_ask is not None else 0

                # Définir Side et Obs en fonction de l'action
                if agent.position == 0:
                    if agent_action == "order_ask":
                        agent_side = "A"
                    else:
                        agent_side = "B"
                else:
                    if agent_action == "order_bid":
                        agent_side = "B"
                    else:
                        agent_side = "A"

                if agent_action.startswith("cancel"):
                    agent_obs = "Cancel"
                elif agent_action.startswith("order"):
                    agent_obs = "Order"
                elif agent_action.startswith("add"):
                    agent_obs = "Add"
                else:
                    agent_obs = "Nothing"
                
                size = 1
                agent_row = [
                    self.time,         
                    0,                 
                    agent_side,        
                    agent_obs,         
                    self.price,        
                    size,              
                    self.bid[0],       
                    self.ask[0],       
                    self.bid[1],       
                    self.ask[1],       
                    self.bid[2],       
                    self.ask[2],       
                    "Agent",           
                    pnl,               
                    nb_asset,          
                    limit_add_bid,     
                    limit_add_ask,     
                    agent_action       
                ]
                df_log.loc[len(df_log)] = agent_row
                continue
            else:
                step_row = self.step()  # liste de 13 éléments
                if step_row[-1] == 'new_limit':
                    #### ATTENTION CAS OU CELA VIDE LA LIMITE N'est PAs fait mais c'est giga giga giga rare e vrai nan....
                    # TRAITER LE CAS OU L'ACHAT CONSOMME NOTRE ADD
                    # 1) Log de l'événement de nouvelle limite
                    net_worth = agent.cash + (agent.number_actif_init + agent.position) * self.price
                    pnl = net_worth - cash_initial - agent.number_actif_init * self.price_0
                    new_limit_row = step_row.copy() + [pnl, agent.position, 0, 0, "NA"]
                    df_log.loc[len(df_log)] = new_limit_row

                    # 2) Si l'agent possède un ordre ADD actif, log d'une ligne auto_cancel
                    auto_cancel = False
                    auto_msgs = []
                    if agent.order_active_bid is not None:
                        agent.order_active_bid = None
                        auto_cancel = True
                        side = 'B'
                        auto_msgs.append("Auto Cancel Bid")
                        if self.bid[0] > 1:
                            self.bid[0] -= 1
                        elif self.bid[1] > 1:
                            self.bid[1] -= 1
                    if agent.order_active_ask is not None:
                        agent.order_active_ask = None
                        auto_cancel = True
                        auto_msgs.append("Auto Cancel Ask")
                        side = 'A'
                        if self.ask[0] > 1:
                            self.ask[0] -= 1
                        elif self.ask[1] > 1:
                            self.ask[1] -= 1
                    if auto_cancel:
                        auto_msg = " & ".join(auto_msgs) + " due to new limit"
                        net_worth = agent.cash + (agent.number_actif_init + agent.position) * self.price
                        pnl = net_worth - cash_initial - agent.number_actif_init * self.price_0
                        tubu = [self.time,         
                                0,                 
                                side,        
                                'Cancel',         
                                self.price,        
                                1,              
                                self.bid[0],       
                                self.ask[0],       
                                self.bid[1],       
                                self.ask[1],       
                                self.bid[2],       
                                self.ask[2], 'Agent']
                        auto_cancel_row = tubu + [pnl, agent.position, 0, 0, auto_msg]
                        df_log.loc[len(df_log)] = auto_cancel_row
                    continue
                # Move de marché normal
                net_worth = agent.cash + (agent.number_actif_init + agent.position) * self.price
                pnl = net_worth - cash_initial - agent.number_actif_init * self.price_0
                nb_asset = agent.number_actif_init + agent.position
                limit_add_bid = 1 if agent.order_active_bid is not None else 0
                limit_add_ask = 1 if agent.order_active_ask is not None else 0
                market_row = step_row + [pnl, nb_asset, limit_add_bid, limit_add_ask, "NA"]
                df_log.loc[len(df_log)] = market_row

        self.df_evolution = df_log
        return df_log
    
    def steplog_without_agent(self, initial_ask, initial_bid, agent, cash_initial=1000.0):
        """
        Exécute la simulation du marché avec interaction de l'agent et enregistre un DataFrame enrichi.
        Colonnes d'origine (13) auxquelles on ajoute :
          - 'PnL'           : net_worth - cash_initial, où net_worth = agent.cash + agent.position * self.price
          - 'NB_asset'      : nombre d'actifs détenus (agent.position)
          - 'Limit_add_bid' : 1 si un ordre ADD actif sur le bid, sinon 0
          - 'Limit_add_ask' : 1 si un ordre ADD actif sur le ask, sinon 0
          - 'Agent_Action'  : l'action de l'agent si il intervient ce move, "NA" sinon.
          
        Tous les 20 moves, l'agent intervient (aucun appel à self.step()).
        De plus, si une nouvelle limite est déclenchée et que l'agent possède un ordre ADD,
        l'ordre est annulé et cette annulation est notée dans la même ligne de log.
        """
        self.intiate_market(initial_ask, initial_bid)
        df_columns = ['Time', 'Limit', 'Side', 'Action', 'Price', 'Size',
                      'Bid_1', 'Ask_1', 'Bid_2', 'Ask_2', 'Bid_3', 'Ask_3', 'Obs',
                      'PnL', 'NB_asset', 'Limit_add_bid', 'Limit_add_ask', 'Agent_Action']
        df_log = pd.DataFrame(columns=df_columns)
        
        # Réinitialisation du suivi des ordres de l'agent
        agent.order_active_bid = None
        agent.order_active_ask = None
        agent.order_price = None
        agent.recorded_bid_volume = None
        agent.recorded_ask_volume = None

        last_agent_time = self.time

        # Enregistrement de l'état initial
        net_worth = agent.cash + (agent.number_actif_init + agent.position) * self.price
        pnl = net_worth - cash_initial - agent.number_actif_init * self.price_0
        nb_asset = agent.number_actif_init + agent.position
        limit_add_bid = 1 if agent.order_active_bid is not None else 0
        limit_add_ask = 1 if agent.order_active_ask is not None else 0
        init_row = [self.time, 'N/A', 'N/A', 'N/A', self.price, 'N/A',
                    self.bid[0], self.ask[0], self.bid[1], self.ask[1],
                    self.bid[2], self.ask[2], 'Opening', pnl, nb_asset, limit_add_bid, limit_add_ask, "NA"]
        df_log.loc[len(df_log)] = init_row
        for i in range(self.nb_of_action):
            # Exécuter un move de marché et récupérer la ligne de base
            if True:
                step_row = self.step()  # liste de 13 éléments
                net_worth = agent.cash + (agent.number_actif_init + agent.position) * self.price
                pnl = net_worth - cash_initial - agent.number_actif_init * self.price_0
                nb_asset = agent.number_actif_init + agent.position
                limit_add_bid = 1 if agent.order_active_bid is not None else 0
                limit_add_ask = 1 if agent.order_active_ask is not None else 0
                market_row = step_row + [pnl, nb_asset, limit_add_bid, limit_add_ask, "NA"]
                df_log.loc[len(df_log)] = market_row
        self.df_evolution = df_log
        return df_log

class Price_impact():
    def __init__(self, tab_cancel, tab_order, tab_add, price_0, tick, theta, nb_of_action,
                liquidy_last_lim, size_max, lambda_event, event_prob):
        self.qr_agent = QrWithAgent(tab_cancel, tab_order, tab_add, price_0, tick, theta, nb_of_action,
                liquidy_last_lim, size_max, lambda_event, event_prob)
        self.qr_without_agent = QrWithAgent(tab_cancel, tab_order, tab_add, price_0, tick, theta, nb_of_action,
                liquidy_last_lim, size_max, lambda_event, event_prob)
        
    def visu_agent_act(self, initial_ask, initial_bid, agent, cash_initial=1000.0, price_only = True):
        df_without_agent = self.qr_without_agent.steplog_without_agent(initial_ask, initial_bid, agent, cash_initial)
        # print(df_without_agent.head())
        df_with_agent = self.qr_agent.steplog(initial_ask, initial_bid, agent, cash_initial)
        df = df_without_agent
        print()
        print(df_with_agent.head())
        df_1 = df[df['Limit'] == 0]
        df_price = df_1[df_1['Action'] == 'Order']
        
        print(df_price.head())
        df_2 = df[df['Limit'] == 1]
        df_3 = df[df['Limit'] == 2]
        df_4 = df[df['Obs'].isin(['In_event', 'End_event', 'Start_event'])]
        df = df_with_agent
        df_11 = df[df['Limit'] == 0]
        df_price_2 = df_11[df_11['Action'] == 'Order']
        df_22 = df[df['Limit'] == 1]
        df_33 = df[df['Limit'] == 2]
        df_44 = df[df['Obs'].isin(['In_event', 'End_event', 'Start_event'])]
        
        print(df_price['Time'])
        print(df_price['Price'])
        
        print(df_price_2['Time'])
        print(df_price_2['Price'])
        
        print(len(df_with_agent))
        print(len(df_without_agent))
        
        fig = go.Figure()
        if not price_only:
            fig.add_trace(go.Scatter(x=df_1['Time'], y=df_1['Price'], mode='markers', name="Limit_1",
                                    marker=dict(size=5, color="red", opacity=0.7)))
            fig.add_trace(go.Scatter(x=df_2['Time'], y=df_2['Price'], mode='markers', name="Limit_2",
                                    marker=dict(size=4, color="orange", opacity=0.6)))
            fig.add_trace(go.Scatter(x=df_3['Time'], y=df_3['Price'], mode='markers', name="Limit_3",
                                    marker=dict(size=3, color="gold", opacity=0.5)))
            fig.add_trace(go.Scatter(x=df_4['Time'], y=100*np.ones(len(df_4)), mode='markers', name="EVENT",
                                    marker=dict(size=4, color="black", opacity=0.8)))
            fig.add_trace(go.Scatter(x=df_price['Time'], y=df_price['Price'], mode='lines', name="Sell_Price",
                                line=dict(width=2, color='darkred')))
        fig.add_trace(go.Scatter(x=df_without_agent['Time'], y=df_with_agent['Price'], mode='lines', name="Sell_Price_no_agent",
                                line=dict(width=2, color='darkred')))
        if not price_only:
            fig.add_trace(go.Scatter(x=df_11['Time'], y=df_11['Price'], mode='markers', name="Limit_1_agent",
                                    marker=dict(size=5, color="red", opacity=0.7)))
            fig.add_trace(go.Scatter(x=df_22['Time'], y=df_22['Price'], mode='markers', name="Limit_2_agent",
                                    marker=dict(size=4, color="orange", opacity=0.6)))
            fig.add_trace(go.Scatter(x=df_33['Time'], y=df_33['Price'], mode='markers', name="Limit_3_agent",
                                    marker=dict(size=3, color="gold", opacity=0.5)))
            fig.add_trace(go.Scatter(x=df_44['Time'], y=100*np.ones(len(df_44)), mode='markers', name="EVENT_agent",
                                    marker=dict(size=4, color="black", opacity=0.8)))
            fig.add_trace(go.Scatter(x=df_price_2['Time'], y=df_price_2['Price'], mode='lines', name="Sell_Price_agent",
                                line=dict(width=2, color='darkblue')))
        fig.add_trace(go.Scatter(x=df_with_agent['Time'], y=df_with_agent['Price'], mode='lines', name="Sell_Price_agent",
                                line=dict(width=2, color='darkblue')))
        fig.update_layout(
            title="Market Simulation Agent vs No Agent",
            xaxis_title="Time",
            yaxis_title="Price",
            plot_bgcolor='#D3D3D3',
            paper_bgcolor='#D3D3D3',
            xaxis=dict(showgrid=True, gridcolor='#808080'),
            yaxis=dict(showgrid=True, gridcolor='#808080')
        )
        fig.show()

class MarketEnv:
    def __init__(self, simulation, agent, initial_ask, initial_bid, nb_steps):
        self.simulation = simulation
        self.agent = agent
        self.initial_ask = initial_ask
        self.initial_bid = initial_bid
        self.nb_steps = nb_steps
        self.current_step = 0
    
    def reset(self):
        self.simulation.intiate_market(self.initial_ask, self.initial_bid)
        self.agent.position = 0
        self.agent.order_active = None
        self.agent.cash = 1000.0
        self.agent.entry_price = None
        # Initialisation des attributs pour le suivi de l'ordre
        self.agent.order_quantity = 0
        self.agent.order_price = None
        self.agent.order_bid_level = None
        self.agent.order_ask_level = None
        self.current_step = 0
        return self.get_state()
    
    def get_state(self):
        state = np.array([
            self.simulation.price,
            self.simulation.time,
            self.simulation.bid[0],
            self.simulation.bid[1],
            self.simulation.bid[2],
            self.simulation.ask[0],
            self.simulation.ask[1],
            self.simulation.ask[2],
            self.agent.position
        ], dtype=np.float32)
        return state
    
    def step(self, action):
        """
        Pour une étape :
          - On fait évoluer le marché d'un move.
          - À chaque move, on vérifie si un ordre actif a été consommé.
          - Tous les 20 moves, l'agent prend une décision (limit_buy, cancel_buy, limit_sell, cancel_sell ou do_nothing).
          - Le reward correspond à la variation de la valeur nette de l'agent.
        """
        prev_net = self.agent.cash + self.agent.position * self.simulation.price
        prev_position = self.agent.position

        # Faire évoluer le marché d'un move
        _ = self.simulation.step()
        self.current_step += 1

        # Vérifier l'exécution d'un ordre actif
        if self.agent.order_active == "buy":
            # Si le niveau du bid a diminué par rapport au moment du placement,
            # on considère que l'ordre a été consommé.
            if self.simulation.bid[0] < self.agent.order_bid_level:
                # Exécution de l'ordre d'achat
                self.agent.position = 1
                # On déduit le prix d'achat de son cash
                self.agent.cash -= self.agent.order_price
                # On efface l'ordre actif
                self.agent.order_active = None
                self.agent.order_quantity = 0
                self.agent.order_price = None
                self.agent.order_bid_level = None
        elif self.agent.order_active == "sell":
            # Pour la vente, on considère que l'ordre est exécuté si le niveau du ask a diminué
            if self.simulation.ask[0] < self.agent.order_ask_level:
                self.agent.position = 0
                self.agent.cash += self.agent.order_price
                self.agent.order_active = None
                self.agent.order_quantity = 0
                self.agent.order_price = None
                self.agent.order_ask_level = None

        # Tous les 20 moves, l'agent prend une décision
        if self.current_step % 20 == 0:
            if self.agent.position == 0:
                mapping = {0:"do_nothing", 1:"limit_buy", 2:"cancel_buy"}
            else:
                mapping = {0:"do_nothing", 1:"limit_sell", 2:"cancel_sell"}
            action_name = mapping.get(action, "do_nothing")
            self.simulation.execute_agent_action(action_name, self.agent)

        new_net = self.agent.cash + self.agent.position * self.simulation.price
        reward = new_net - prev_net
        state = self.get_state()
        done = self.current_step >= self.nb_steps
        return state, reward, done, {}
    
    def step(self, initial_ask, initial_bid, agent, cash_initial=1000.0):
        """
        Exécute la simulation du marché avec interaction de l'agent et enregistre un DataFrame enrichi.
        Colonnes d'origine (13) auxquelles on ajoute :
          - 'PnL'           : net_worth - cash_initial, où net_worth = agent.cash + agent.position * self.price
          - 'NB_asset'      : nombre d'actifs détenus (agent.position)
          - 'Limit_add_bid' : 1 si un ordre ADD actif sur le bid, sinon 0
          - 'Limit_add_ask' : 1 si un ordre ADD actif sur le ask, sinon 0
          - 'Agent_Action'  : l'action de l'agent si il intervient ce move, "NA" sinon.
        
        Tous les 20 moves, l'agent intervient (aucun appel à self.step()).
        De plus, si une nouvelle limite est déclenchée et que l'agent possède un ordre ADD,
        l'ordre est annulé et cette annulation est notée dans la même ligne de log.
        """
        self.intiate_market(initial_ask, initial_bid)

        
        # Réinitialisation du suivi des ordres de l'agent
        agent.order_active_bid = None
        agent.order_active_ask = None
        agent.order_price = None
        agent.recorded_bid_volume = None
        agent.recorded_ask_volume = None

        last_agent_time = self.time

        # Enregistrement de l'état initial
        net_worth = agent.cash + agent.position * self.price
        pnl = net_worth - cash_initial

        for i in range(self.nb_of_action):
            # Exécuter un move de marché et récupérer la ligne de base
            if i % 20 == 0:

                # Move agent : on n'appelle PAS self.step() pour ce move
                market_state = {"price": self.price, "bid": self.bid.copy(), "ask": self.ask.copy(), "time": self.time}
                agent_action = agent.decide_action(market_state)
                self.execute_agent_action(agent_action, agent)


                # Lors d'un ADD, enregistrer la quantité actuelle
                if agent.order_active_bid == "add_bid":
                    agent.recorded_bid_volume = self.bid[0]
                if agent.order_active_ask == "add_ask":
                    agent.recorded_ask_volume = self.ask[0]

                net_worth = agent.cash + agent.position * self.price
                pnl = net_worth - cash_initial

                continue
            else:
                step_row = self.step()
                if step_row[-1] == 'new_limit':
                    #### ATTENTION CAS OU CELA VIDE LA LIMITE N'est PAs fait mais c'est giga giga giga rare e vrai nan....
                    # TRAITER LE CAS OU L'ACHAT CONSOMME NOTRE ADD
                    # 1) Log de l'événement de nouvelle limite
                    net_worth = agent.cash + agent.position * self.price
                    pnl = net_worth - cash_initial
                    new_limit_row = step_row.copy() + [pnl, agent.position, 0, 0, "NA"]

                    # 2) Si l'agent possède un ordre ADD actif, log d'une ligne auto_cancel
                    auto_cancel = False
                    auto_msgs = []
                    if agent.order_active_bid is not None:
                        agent.order_active_bid = None
                        auto_cancel = True
                        side = 'B'
                        auto_msgs.append("Auto Cancel Bid")
                        if self.bid[0] > 1:
                            self.bid[0] -= 1
                        elif self.bid[1] > 1:
                            self.bid[1] -= 1
                    if agent.order_active_ask is not None:
                        agent.order_active_ask = None
                        auto_cancel = True
                        auto_msgs.append("Auto Cancel Ask")
                        side = 'A'
                        if self.ask[0] > 1:
                            self.ask[0] -= 1
                        elif self.ask[1] > 1:
                            self.ask[1] -= 1
                    if auto_cancel:
                        auto_msg = " & ".join(auto_msgs) + " due to new limit"
                        net_worth = agent.cash + agent.position * self.price
                        pnl = net_worth - cash_initial

                    continue
                # Move de marché normal
                net_worth = agent.cash + agent.position * self.price
                pnl = net_worth - cash_initial
                nb_asset = agent.position


        self.df_evolution = df_log
        return df_log


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

def main():
    def order_bid(x):
        return 0.018*(x**4+1)
    def order_ask(x):
        return 0.018*(x**4+1)
    def tabs_order():
        return [order_ask, order_bid]
    def cancel_bid_1(x):
        return 0.3*(x**2+1)
    def cancel_ask_1(x):
        return 0.3*(x**2+1)
    def cancel_bid_2(x):
        return 0.7*(x**4+1)
    def cancel_ask_2(x):
        return 0.7*(x**4+1)
    def cancel_bid_3(x):
        return 1*(x**4+1)
    def cancel_ask_3(x):
        return 1*(x**4+1)
    def tabs_cancel():
        return [cancel_ask_1, cancel_ask_2, cancel_ask_3, cancel_bid_1, cancel_bid_2, cancel_bid_3]
    def add_bid_1(x):
        return 0.2*(x**4+1)
    def add_ask_1(x):
        return 0.2*(x**4+1)
    def add_bid_2(x):
        return 0.2*(x**2+1)
    def add_ask_2(x):
        return 0.3*(x**2+1)
    def add_bid_3(x):
        return 0.7*(x**2+1)
    def add_ask_3(x):
        return 0.7*(x**2+1)
    def tabs_add():
        return [add_ask_1, add_ask_2, add_ask_3, add_bid_1, add_bid_2, add_bid_3]
    
    tab_add = tabs_add()
    tab_cancel = tabs_cancel()
    tab_order = tabs_order()
    intensity_cancel = tab_cancel
    intensity_order  = tab_order
    intensity_add    = tab_add
    
    price_0 = 100.0
    tick = 0.5
    theta = 0.5
    nb_of_action = 100
    liquidy_last_lim = 50
    size_max = [5, 4, 8]
    lambda_event = [10 for i in range(34)] + [100 for i in range(15)] + [1000]
    event_prob = 1/200
    
    initial_ask = [10, 20, 30]
    initial_bid = [10, 20, 30]
    
    simulation = QrWithAgent(intensity_cancel, intensity_order, intensity_add,
                            price_0, tick, theta, nb_of_action, liquidy_last_lim,
                            size_max, lambda_event, event_prob)
    agent = TradingAgent()
    
    nb_steps = nb_of_action
    env = MarketEnv(simulation, agent, initial_ask, initial_bid, nb_steps)
    
    state_dim = 9
    action_dim = 4
    lr = 1e-3
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    num_episodes = 25000
    batch_size = 32
    replay_capacity = 10000
    target_update = 10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    q_network = QNetwork(state_dim, action_dim).to(device)
    target_network = QNetwork(state_dim, action_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(replay_capacity)
    
    def select_action(state, epsilon):
        if random.random() < epsilon:
            return random.randrange(action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = q_network(state_tensor)
            return q_values.argmax().item()
    
    episode_rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
                
                q_values = q_network(states).gather(1, actions)
                with torch.no_grad():
                    next_q_values = target_network(next_states).max(1)[0].unsqueeze(1)
                target = rewards + gamma * next_q_values * (1 - dones)
                
                loss = F.mse_loss(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)
        print(f"Episode {episode+1}/{num_episodes} - Total Reward: {total_reward:.2f} - Epsilon: {epsilon:.2f}")
        if (episode+1) % target_update == 0:
            target_network.load_state_dict(q_network.state_dict())
    random_final_rewards = []
    for _ in range(1000):
        state = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            random_action = random.randrange(action_dim)
            state, reward, done, _ = env.step(random_action)
            total_reward += reward
        random_final_rewards.append(total_reward)
    avg_random_price = np.mean(random_final_rewards)
    print("reward final moyen (stratégie aléatoire) :", avg_random_price)
    plt.plot(episode_rewards)
    plt.plot(np.arange(0,num_episodes,1), np.ones(num_episodes)*avg_random_price)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Évolution des récompenses par épisode")
    plt.show()


    window_size = 20
    rolling_avg = np.convolve(
        episode_rewards, 
        np.ones(window_size) / window_size, 
        mode='valid'
    )

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(0,num_episodes,1), np.ones(num_episodes)*avg_random_price)
    plt.plot(
        np.arange(window_size - 1, len(episode_rewards)), 
        rolling_avg, 
        color='red', 
        label=f"Rolling Average (window={window_size})"
    )
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Évolution des récompenses par épisode (avec moyenne glissante)")
    plt.legend()
    plt.show()


def dataframe():
    def order_bid(x):
        return 0.008*(x**2+1)
    def order_ask(x):
        return 0.008*(x**2+1)
    def tabs_order():
        return [order_ask, order_bid]
    def cancel_bid_1(x):
        return 0.15*(x**2+1)
    def cancel_ask_1(x):
        return 0.13*(x**2+1)
    def cancel_bid_2(x):
        return 0.7*(x**4+1)
    def cancel_ask_2(x):
        return 0.7*(x**4+1)
    def cancel_bid_3(x):
        return 1*(x**4+1)
    def cancel_ask_3(x):
        return 1*(x**4+1)
    def tabs_cancel():
        return [cancel_ask_1, cancel_ask_2, cancel_ask_3, cancel_bid_1, cancel_bid_2, cancel_bid_3]
    def add_bid_1(x):
        return 0.005*(x**4+1)
    def add_ask_1(x):
        return 0.005*(x**4+1)
    def add_bid_2(x):
        return 0.2*(x**2+1)
    def add_ask_2(x):
        return 0.3*(x**2+1)
    def add_bid_3(x):
        return 0.7*(x**2+1)
    def add_ask_3(x):
        return 0.7*(x**2+1)
    def tabs_add():
        return [add_ask_1, add_ask_2, add_ask_3, add_bid_1, add_bid_2, add_bid_3]
    
    tab_add = tabs_add()
    tab_cancel = tabs_cancel()
    tab_order = tabs_order()
    intensity_cancel = tab_cancel
    intensity_order  = tab_order
    intensity_add    = tab_add

    price_0 = 100.25
    tick = 0.5
    theta = 0.5
    nb_of_action = 10000  # nombre d'étapes de la simulation
    liquidy_last_lim = 50
    size_max = [5, 4, 8]
    lambda_event = [10 for i in range(34)] + [100 for i in range(15)] + [1000]
    event_prob = 1/200
    initial_ask = [10, 20, 30]
    initial_bid = [10, 20, 30]
    
    simulation = QrWithAgent(intensity_cancel, intensity_order, intensity_add,
                             price_0, tick, theta, nb_of_action, liquidy_last_lim,
                             size_max, lambda_event, event_prob)
    agent = TradingAgent()
    
    # Exécuter la simulation avec log interactif
    df_enhanced = simulation.steplog(initial_ask, initial_bid, agent)
    
    # Afficher le DataFrame final pour vérifier l'évolution du PnL et des ordres de l'agent
    return df_enhanced



def Agent_VS_no_Agent():
    def order_bid(x):
        return 0.008*(x**6+1)
    def order_ask(x):
        return 0.008*(x**6+1)
    def tabs_order():
        return [order_ask, order_bid]
    def cancel_bid_1(x):
        return 0.015*(x**2+1)
    def cancel_ask_1(x):
        return 0.013*(x**2+1)
    def cancel_bid_2(x):
        return 0.7*(x**4+1)
    def cancel_ask_2(x):
        return 0.7*(x**4+1)
    def cancel_bid_3(x):
        return 1*(x**4+1)
    def cancel_ask_3(x):
        return 1*(x**4+1)
    def tabs_cancel():
        return [cancel_ask_1, cancel_ask_2, cancel_ask_3, cancel_bid_1, cancel_bid_2, cancel_bid_3]
    def add_bid_1(x):
        return 0.01*(x**4+1)
    def add_ask_1(x):
        return 0.01*(x**4+1)
    def add_bid_2(x):
        return 0.2*(x**2+1)
    def add_ask_2(x):
        return 0.3*(x**2+1)
    def add_bid_3(x):
        return 0.7*(x**2+1)
    def add_ask_3(x):
        return 0.7*(x**2+1)
    def tabs_add():
        return [add_ask_1, add_ask_2, add_ask_3, add_bid_1, add_bid_2, add_bid_3]
    
    tab_add = tabs_add()
    tab_cancel = tabs_cancel()
    tab_order = tabs_order()
    intensity_cancel = tab_cancel
    intensity_order  = tab_order
    intensity_add    = tab_add

    price_0 = 100.25
    tick = 0.5
    theta = 0.5
    nb_of_action = 10000  # nombre d'étapes de la simulation
    liquidy_last_lim = 50
    size_max = [5, 4, 8]
    lambda_event = [10 for i in range(34)] + [100 for i in range(15)] + [1000]
    event_prob = 1/200
    initial_ask = [10, 20, 30]
    initial_bid = [10, 20, 30]
    
    simulation = Price_impact(intensity_cancel, intensity_order, intensity_add,
                             price_0, tick, theta, nb_of_action, liquidy_last_lim,
                             size_max, lambda_event, event_prob)
    agent = TradingAgent()
    
    # Exécuter la simulation avec log interactif
    simulation.visu_agent_act(initial_ask, initial_bid, agent)