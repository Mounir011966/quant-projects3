import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm


from src.simulator import HestonSimulator
from src.modeldeeplearning import DeepHedger


def train(n_epochs=200, n_paths=2000, transaction_cost=0.005):
    
    
   
    K = 100.0
    T = 1/12 

    sim = HestonSimulator(T=T)
    model = DeepHedger()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    

    losses = []
    
    for epoch in tqdm(range(n_epochs)):
        
        S_np, v_np = sim.simulate_paths(n_paths)
        
        S = torch.FloatTensor(S_np)
        v = torch.FloatTensor(v_np)
        
        hedging_pnl = torch.zeros(n_paths)
        total_cost = torch.zeros(n_paths)

        prev_delta = torch.zeros(n_paths, 1) 
        
        for t in range(sim.N_steps):

            current_S = S[:, t].unsqueeze(1)
            current_v = v[:, t].unsqueeze(1)
            time_left = torch.full((n_paths, 1), T - t*sim.dt)
            
            # Standardisation 
            log_moneyness = torch.log(current_S / K)
            
            # Input vector: [Log(S/K), Time, Vol, Prev_Delta]
            features = torch.cat([log_moneyness * 100, time_left * 100, current_v * 100, prev_delta], dim=1)
            current_delta = model(features)
            cost = transaction_cost * current_S * torch.abs(current_delta - prev_delta)
            total_cost += cost.squeeze()
            
            
            next_S = S[:, t+1]
            price_change = next_S - current_S.squeeze()
            hedging_pnl += current_delta.squeeze() * price_change
            
            prev_delta = current_delta

        # C. On a vendu un Call, on doit payer (S-K)+
        final_S = S[:, -1]
        payoff = torch.relu(final_S - K)
        
        # PnL Total = Gains de Hedging - Coûts - Payoff à payer
        total_pnl = hedging_pnl - total_cost - payoff
        
       
        risk_aversion = 1.0
        utility = -torch.exp(-risk_aversion * total_pnl)
        loss = torch.mean(-utility) # On minimise la perte d'utilité

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        losses.append(loss.item())


    return model, losses

if __name__ == "__main__":
    trained_model, history = train(n_epochs=50)
    torch.save(trained_model.state_dict(), "deep_hedger.pth")
   