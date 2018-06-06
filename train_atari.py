import argparse
import random

import torch

from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from dqn.wrappers import *


def parse_args():
    parser = argparse.ArgumentParser("DQN experiments for Atari games")
    parser.add_argument("--seed", type=int, default=42, help="which seed to use")
    # Environment
    parser.add_argument("--env", type=str, default="PongNoFrameskip-v4", help="name of the game")
    # Core DQN parameters
    parser.add_argument("--replay-buffer-size", type=int, default=int(1e6), help="replay buffer size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--num-steps", type=int, default=int(1e6),
                        help="total number of steps to run the environment for")
    parser.add_argument("--batch-size", type=int, default=32, help="number of transitions to optimize at the same time")
    parser.add_argument("--learning-starts", type=int, default=10000, help="number of steps before learning starts")
    parser.add_argument("--learning-freq", type=int, default=1,
                        help="number of iterations between every optimization step")
    parser.add_argument("--target-update-freq", type=int, default=1000,
                        help="number of iterations between every target network update")
    parser.add_argument("--use-double-dqn", type=bool, default=True, help="use double deep Q-learning")
    # e-greedy exploration parameters
    parser.add_argument("--eps-start", type=float, default=1.0, help="e-greedy start threshold")
    parser.add_argument("--eps-end", type=float, default=0.02, help="e-greedy end threshold")
    parser.add_argument("--eps-fraction", type=float, default=0.1, help="fraction of num-steps")
    # Reporting
    parser.add_argument("--print-freq", type=int, default=10, help="print frequency.")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    assert "NoFrameskip" in args.env, "Require environment with no frameskip"
    env = gym.make(args.env)
    env.seed(args.seed)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = WarpFrame(env)
    env = PyTorchFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, 4)

    replay_buffer = ReplayBuffer(args.replay_buffer_size)

    agent = DQNAgent(
        env.observation_space,
        env.action_space,
        replay_buffer,
        use_double_dqn=args.use_double_dqn,
        lr=args.lr,
        batch_size=args.batch_size,
        gamma=args.gamma
    )

    eps_timesteps = args.eps_fraction * float(args.num_steps)
    episode_rewards = [0.0]
    loss = [0.0]

    state = env.reset()
    for t in range(args.num_steps):
        fraction = min(1.0, float(t) / eps_timesteps)
        eps_threshold = args.eps_start + fraction * (args.eps_end - args.eps_start)
        sample = random.random()
        if sample > eps_threshold:
            action = agent.act(np.array(state))
        else:
            action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        agent.memory.add(state, action, reward, next_state, float(done))
        state = next_state

        episode_rewards[-1] += reward
        if done:
            state = env.reset()
            episode_rewards.append(0.0)

        if t > args.learning_starts and t % args.learning_freq == 0:
            agent.optimise_td_loss()

        if t > args.learning_starts and t % args.target_update_freq == 0:
            agent.update_target_network()

        num_episodes = len(episode_rewards)
        if done and args.print_freq is not None and len(episode_rewards) % args.print_freq == 0:
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            print("********************************************************")
            print("steps: {}".format(t))
            print("episodes: {}".format(num_episodes))
            print("mean 100 episode reward: {}".format(mean_100ep_reward))
            print("% time spent exploring: {}".format(int(100 * eps_threshold)))
            print("********************************************************")
