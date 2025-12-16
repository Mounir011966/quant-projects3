import numpy as np

class HestonSimulator:
    def __init__(self, S0=100, v0=0.04, r=0.0, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7, T=1/12, dt=1/252):
        """
        S0: Prix initial
        v0: Variance initiale
        r: Taux sans risque (souvent 0 pour le hedging pur)
        kappa, theta, sigma: Paramètres Heston
        rho: Corrélation prix-vol
        T: Maturité de l'option 
        dt: Pas de temps 
        """
        self.params = (S0, v0, r, kappa, theta, sigma, rho)
        self.T = T
        self.dt = dt
        self.N_steps = int(T / dt)

    def simulate_paths(self, n_paths=10000):
        S0, v0, r, kappa, theta, sigma, rho = self.params
    
        Z1 = np.random.normal(size=(n_paths, self.N_steps))
        Z2_uncorr = np.random.normal(size=(n_paths, self.N_steps))
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2_uncorr  # Cholesky 
        
        S = np.zeros((n_paths, self.N_steps + 1))
        v = np.zeros((n_paths, self.N_steps + 1))
        S[:, 0] = S0
        v[:, 0] = v0
        

        for t in range(self.N_steps):

            v_curr = np.maximum(v[:, t], 0)
            S_curr = S[:, t]
   
            dv = kappa * (theta - v_curr) * self.dt + sigma * np.sqrt(v_curr * self.dt) * Z2[:, t]
            v[:, t+1] = v_curr + dv
            

            dS = r * S_curr * self.dt + np.sqrt(v_curr * self.dt) * S_curr * Z1[:, t]
            S[:, t+1] = S_curr + dS
            
        return S, v

if __name__ == "__main__":

    sim = HestonSimulator()
    S, v = sim.simulate_paths(5)
