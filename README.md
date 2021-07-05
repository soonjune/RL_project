# RL_project

environment: n-chain

## algorithms

1. HDQN
2. Vanilla DQN
3. Bootstrapped_DQN with 10 heads (seed 32) nchain
   - BoostrappedDQN Readme file 보고 baseline 설치. tensorflow 버전에 유의(1.15 필요)
   - 수업시간에 배운 NoisyDQN 및 BayesBackpropDQN, MNFDQN 비교 가능  
   - nchain with 100 states (starting from 1, s2)
   - score 10 reached at episode 5 and forgets but soon returns (slower)
   - DQN first reaches 10 at episode 200 forgets at 900
4. UCB
   - application of UCB using mean and std of DQN heads
5. TDU
   - uncertainty estimation using temporal difference uncertainty
  
## how to make virtual environment and run Bootstrapped DQN
```
python3 -m venv rl
source rl/bin/activate
cd RL_project/BootstrappedDQN
mkdir graphs
mkdir graphs/mean graphs/std
cd ..
python3 -m qlearn.toys.main_nchain --agent BootstrappedDQN --cuda 0 --input-dim 20 --double-q 0 --ucb 1 --max-episodes 2500 —seed 1
```
