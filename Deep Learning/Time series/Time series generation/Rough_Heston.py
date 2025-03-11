import numpy as np
import plotly.graph_objects as go

'''
Implementation of the Rough-Hestion model by M.Rosenbaum and Jim Gatheral
'''

class Rough_Heston:
    def __init__(self, Hurst_coef, T, Nb_steps, mu, alpha, nu, mean, v_0, S_0):
        '''
        takes ==>
                Hurst_coef : Hurst_coef of the rough Heston model
                T          : Duration of the simulation
                Nb_steps   : Number of steps in the simulation
                mu         : mean of the rough OU for the volatility
                alpha      : average return to mean
                nu         : volatility of the volatility
                mean       : drift of the asset
                S_0        : initial price
                v_0        : initial volatility
        '''
        
        self.hurst_coef   = Hurst_coef
        self.horizon_time = T
        self.nb_step      = Nb_steps
        self.mu           = mu
        self.alpha        = alpha
        self.nu           = nu
        self.mean         = mean
        self.dt           = T/(Nb_steps-1)
        self.S_0          = S_0
        self.log_vol_0    = np.log(v_0)
    
    def vol(self):
        log_vol_tab = [self.log_vol_0]
        gauss       = np.random.normal(size = (self.nb_step-1))
        for i in range (self.nb_step-1):
            log_vol_tab.append(log_vol_tab[-1] - self.alpha*(log_vol_tab[-1]-self.mu)*self.dt + self.nu*gauss[i]*self.dt**(self.hurst_coef/2))
        return np.exp(np.array(log_vol_tab))
    
    def r_heston(self):
        vol     = self.vol()
        tab_sim = [self.S_0]
        gauss   = np.random.normal(size = (self.nb_step-1))
        for i in range (self.nb_step-1):
            tab_sim.append(tab_sim[-1] * (1 + self.mean * self.dt + gauss[i] * vol[i] * np.sqrt(self.dt)))
        return np.array(tab_sim), vol
    
    def visu(self):
        price, vol = self.r_heston()
        time = np.linspace(0, self.horizon_time, self.nb_step)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time, y=price, mode='lines', name="Stock-Price", line=dict(width = 0.85, color = 'darkred')))
        fig.update_layout(
            title="Rough-Heston",
            xaxis_title="Time",
            yaxis_title="Price",
            plot_bgcolor='#D3D3D3',
            paper_bgcolor='#D3D3D3',
            xaxis=dict(showgrid=True, gridcolor='#808080'),
            yaxis=dict(showgrid=True, gridcolor='#808080')
        )
        fig.show()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time, y=vol, mode='lines', name="Volatilty", line=dict(width = 0.85,color = 'darkblue')))
        fig.update_layout(
            title="Rough-Heston Volatility",
            xaxis_title="Time",
            yaxis_title="Volatility",
            plot_bgcolor='#D3D3D3',
            paper_bgcolor='#D3D3D3',
            xaxis=dict(showgrid=True, gridcolor='#808080'),
            yaxis=dict(showgrid=True, gridcolor='#808080')
        )
        fig.show()