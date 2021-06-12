# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import os
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Test DQN
def test(args, env, dqn, cnt=0, k=0):
    rewards = []

    # Test performance over several episodes
    done = True
    dqn.online_net.eval()
    # dqn.online_net.freeze_noise()
    # initialize q_vals
    dqn.left_vals = []
    dqn.right_vals = []
    for _ in range(args.evaluation_episodes):
        while True:
            if done:
                state, reward_sum, done = env.reset(), 0, False
            if args.agent == 'VariationalDQN':
                action = dqn.act(state[None], sample=False)
            elif args.agent in ['NoisyDQN', 'BayesBackpropDQN', 'MNFDQN']:
                action = dqn.act(state[None], eval=True)
            elif args.agent == 'DQN':
                action = dqn.act(state[None])
            elif args.agent == 'BootstrappedDQN':
                action = dqn.act(state[None]) 
                   # Choose an action greedily
            state, reward, done, _ = env.step(int(action))  # Step
            reward_sum += reward

            if done:
                rewards.append(reward_sum)
                break

    left_mean, left_std = [], []
    right_mean, right_std = [], []
    lscaler = MinMaxScaler()
    rscaler = MinMaxScaler()
    dqn.left_vals = lscaler.fit_transform(dqn.left_vals)
    dqn.right_vals = rscaler.fit_transform(dqn.right_vals)

    for left_pred, right_pred in zip(dqn.left_vals, dqn.right_vals):
        left_mean.append(np.mean(left_pred))
        left_std.append(np.std(left_pred))
        right_mean.append(np.mean(right_pred))
        right_std.append(np.std(right_pred))

    # print(left_mean)
    # print(right_mean)

    if args.agent == "BootstrappedDQN" and (cnt < 42 or cnt % 100 == 0) or ((left_std[0] < right_std[0]) and cnt < 200):
        plt.plot([i for i in range(1,args.input_dim+9)],left_mean, 'bo', markersize=2, label='left')
        plt.plot([i for i in range(1,args.input_dim+9)], right_mean, 'ro', markersize=2, label='right')
        plt.ylabel('mean q_val')
        plt.xlabel('steps')
        plt.legend()
        if args.ucb:
            plt.title(f"Episode {cnt}: UCB (mean+std)")
        else:
            plt.title(f"Episode {cnt}: After training {k}th head")
        plt.savefig(f'./graphs/mean/mean_{cnt}')
        plt.close('all')

        plt.plot([i for i in range(1,args.input_dim+9)], left_std, 'bo', markersize=2, label='left')
        plt.plot([i for i in range(1,args.input_dim+9)], right_std, 'ro', markersize=2, label='right')
        plt.ylabel('q_val std')
        plt.xlabel('steps')
        plt.legend()
        if args.ucb:
            plt.title(f"Episode {cnt}: UCB (mean+std)")
        else:
            plt.title(f"Episode {cnt}: After training {k}th head")
        plt.savefig(f'./graphs/std/std_{cnt}')
        plt.close('all')

    env.close()

    # return average reward
    return sum(rewards) / len(rewards)
