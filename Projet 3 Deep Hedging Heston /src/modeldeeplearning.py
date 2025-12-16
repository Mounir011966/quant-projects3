import torch
import torch.nn as nn

class DeepHedger(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32):
        """
        Input Features (4 dimensions):
        1. Log-Moneyness log(S/K) : Pour normaliser le prix
        2. Time to Maturity (T-t)
        3. Volatilité courante (v_t)
        4. Delta précédent (position actuelle)
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),                # Non-linéarité standard
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1), # Sortie: Le Delta cible
            nn.Sigmoid()              # Astuce: Force le Delta entre 0 et 1 (Call Spread / Call)
        )

    def forward(self, x):
        return self.net(x)