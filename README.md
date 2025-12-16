# quant-projects3
Academic-inspired personal project, building on stochastic calculus &amp; financial derivatives courses (KTH/Centrale).

Deep Hedging with Transaction Costs
The goal of this project is to compare a Deep Learning model against the traditional Black-Scholes formula for hedging options.

The problem with Black-Scholes is that it assumes trading is free. In the real world, transaction costs exist. If you follow the formula perfectly, you over-trade and lose money on fees.

I built a model using PyTorch that learns to hedge a Call option under the Heston model. The AI is penalized for trading too frequently, forcing it to find a balance between risk management and cost reduction.

Results: The AI learned a "wait-and-see" strategy. Instead of rebalancing continuously like Black-Scholes, it holds its position until price movements justify the cost. In my simulations with 1% transaction fees, this approach reduced total losses by roughly 30%.
