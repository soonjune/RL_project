# RL_project

environment: n-chain

## algorithms

1. HDQN
   - score:
2. MCTS
   - score: average around 1050
3. Bootstrapped_DQN with 10 heads (seed 32) nchain
   - BoostrappedDQN Readme file 보고 baseline 설치. tensorflow 버전에 유의(1.15 필요)
   - 수업시간에 배운 NoisyDQN 및 BayesBackpropDQN, MNFDQN 비교 가능  
   - nchain with 100 states (starting from 1, s2)
   - score 10 reached at episode 5 and forgets but soon returns (slower)
   - DQN first reaches 10 at episode 200 forgets at 900