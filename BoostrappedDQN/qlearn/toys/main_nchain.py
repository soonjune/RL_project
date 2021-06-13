# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse
import time
from datetime import datetime
import random
import numpy as np
import math
from collections import Counter
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from baselines.common.schedules import LinearSchedule
from baselines.deepq.replay_buffer import ReplayBuffer

from qlearn.toys.agent import Agent
from qlearn.toys.bootstrapped_agent import BootstrappedAgent
from qlearn.toys.bayes_backprop_agent import BayesBackpropAgent
from qlearn.toys.noisy_agent import NoisyAgent
from qlearn.toys.mnf_agent import MNFAgent
from qlearn.envs.nchain import NChainEnv
# from qlearn.toys.memory import ReplayBuffer
from qlearn.toys.test import test
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='DQN')
parser.add_argument('--seed', type=int, default=510, help='Random seed')
parser.add_argument('--cuda', type=int, default=1, help='use cuda')
parser.add_argument('--max-steps', type=int, default=int(50e6), metavar='STEPS', help='Number of training steps')

parser.add_argument('--evaluation-episodes', type=int, default=1, metavar='N',
                    help='Number of evaluation episodes to average over')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--replay_buffer_size', type=int, default=int(10000), metavar='CAPACITY',
                    help='Experience replay memory capacity')
parser.add_argument('--learning-freq', type=int, default=10, metavar='k', help='Frequency of sampling from memory')
parser.add_argument("--learning-starts", type=int, default=32, help="number of iterations after which learning starts")
parser.add_argument('--discount', type=float, default=0.999, metavar='GAMMA', help='Discount factor')
parser.add_argument('--target-update-freq', type=int, default=100, metavar='TAU',
                    help='Number of steps after which to update target network')
parser.add_argument('--lr', type=float, default=0.001, metavar='ETA', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='EPSILON', help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--input-dim', type=int, default=8, help='the length of chain environment')
parser.add_argument('--evaluation-interval', type=int, default=10, metavar='STEPS',
                    help='Number of training steps between evaluations')
parser.add_argument('--nheads', type=int, default=10, help='number of heads in Bootstrapped DQN')
parser.add_argument('--agent', type=str, default='DQN', help='type of agent')
parser.add_argument('--final-exploration', type=float, default=0.1, help='last value of epsilon')
parser.add_argument('--final-exploration-step', type=float, default=1000, help='horizon of epsilon schedule')
parser.add_argument('--max-episodes', type=int, default=int(2e3), metavar='EPISODES',
                    help='Number of training episodes')
parser.add_argument('--hidden_dim', type=int, default=int(16), help='number of hidden unit used in normalizing flows')
parser.add_argument('--n-hidden', type=int, default=int(0), help='number of hidden layer used in normalizing flows')
parser.add_argument('--n-flows-q', type=int, default=int(1),
                    help='number of normalizing flows using for the approximate posterior q')
parser.add_argument('--n-flows-r', type=int, default=int(1),
                    help='number of normalizing flows using for auxiliary posterior r')
parser.add_argument('--logdir', type=str, default='logs', help='log directory')
parser.add_argument('--double-q', type=int, default=1, help='whether or not to use Double DQN')
parser.add_argument('--ucb', type=int, default=0, help='whether or not to use UCB')
parser.add_argument('--use-tdu', type=int, default=0, help='whether or not to use TDU')


# Setup
args = parser.parse_args()
assert args.agent in ['DQN', 'BootstrappedDQN', 'NoisyDQN', 'BayesBackpropDQN', 'MNFDQN']

print(' ' * 26 + 'Options')
for k, v in vars(args).items():
    print(' ' * 26 + k + ': ' + str(v))

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Environment
env = NChainEnv(args.input_dim)
action_space = env.action_space.n

# Log
date = time.strftime('%Y-%m-%d.%H%M')
run_dir = '{}/{}-{}-{}'.format(args.logdir, 'Nchain', args.agent, date)

log = SummaryWriter(run_dir)
print('Writing logs to {}'.format(run_dir))

# Agent
if args.agent == 'BootstrappedDQN':
    dqn = BootstrappedAgent(args, env)
elif args.agent == 'NoisyDQN':
    dqn = NoisyAgent(args, env)
elif args.agent == 'BayesBackpropDQN':
    dqn = BayesBackpropAgent(args, env)
elif args.agent == 'MNFDQN':
    dqn = MNFAgent(args, env)
else:
    dqn = Agent(args, env)

replay_buffer = ReplayBuffer(args.replay_buffer_size)
# mem = ReplayBuffer(args.memory_capacity)

# schedule of epsilon annealing
exploration = LinearSchedule(args.final_exploration_step, args.final_exploration, 1)

# import pdb
# pdb.set_trace()


# graph_rewards = []
# Training loop
dqn.online_net.train()
timestamp, ten_count = 0, 0
for episode in range(args.max_episodes):

    epsilon = exploration.value(episode)
    for i in range(args.input_dim):
        dqn.tdu[i] = [[], []]
    state, done = env.reset(), False
    if args.agent == 'BootstrappedDQN':
        k = random.randrange(args.nheads)
    elif args.agent == 'VariationalDQN':
        dqn.online_net.freeze_noise()
    elif args.agent == 'BayesBackpropDQN':
        dqn.online_net.reset_noise()
    elif args.agent == 'MNFDQN':
        dqn.online_net.reset_noise()
    while not done:
        timestamp += 1

        if args.agent == 'BootstrappedDQN':
            action = dqn.act_single_head(state[None], k)
        elif args.agent in ['NoisyDQN', 'BayesBackpropDQN', 'MNFDQN']:
            action = dqn.act(state[None], eval=False)
        elif args.agent == 'DQN':
            action = dqn.act_e_greedy(state[None], epsilon=epsilon)

        next_state, reward, done, _ = env.step(int(action))

        # tdu estimation
        states = Variable(dqn.FloatTensor(torch.from_numpy(state)))
        actions = action
        next_states = Variable(dqn.FloatTensor(torch.from_numpy(next_state)))
        rewards = torch.as_tensor(reward, dtype=dqn.FloatTensor.dtype).view(-1, 1)
        terminals = torch.as_tensor(done, dtype=dqn.FloatTensor.dtype).view(-1, 1)

        td_errors = []
        state_values = dqn.online_net(states)
        s_primes = dqn.target_net(next_states)
        for i in range(args.nheads):
            state_action_values = state_values[i][actions]
            next_state_values = s_primes[i].max(0)[0]
            target_state_action_values = rewards + (1 - terminals) * args.discount * next_state_values.view(-1, 1)
            td_errors.append((target_state_action_values.detach() - state_action_values).item())
        tdu = np.std(td_errors)
        # print(sum(state), action, tdu)
            # Store the transition in memory
        replay_buffer.add(state, action, reward+tdu, next_state, float(done))

        # Move to the next state
        state = next_state
        #
        if timestamp % args.target_update_freq == 0:
            dqn.update_target_net()

    if timestamp > args.learning_starts and timestamp % args.learning_freq == 0:
        obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(args.batch_size)
        loss = dqn.learn(obses_t, actions, rewards, obses_tp1, dones, k)
        log.add_scalar('loss', loss, timestamp)

    # if episode % 10 == 0:
    #     visited = []
    #     for transition in replay_buffer.memory:
    #         visited.append(transition.state.sum())
    #     print(Counter(visited))

    avg_reward = test(args, env, dqn, episode)  # Test
    # if avg_reward == 10.0:
    #     ten_count += 1
    #     if ten_count > 50:
    #         break
    # graph_rewards.append(avg_reward)

    if episode > 4:
        print('episode: ' + str(episode) + ', Avg. reward: ' + str(round(avg_reward, 4)))

# plt.plot(graph_rewards, 'bo')
# plt.ylabel('avg. rewards')
# plt.xlabel('episodes')
# plt.savefig(f'avg_{args.agent}')
