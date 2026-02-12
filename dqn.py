import torch
from torch import nn
import torch.nn.functional as F
import gymnasium as gym
from img import ImageEnv
from cnn import CNNActionValue
from replay_buffer import ReplayBuffer
import itertools
import yaml
import random
import os
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import argparse


DATE_FORMAT = "%m-%d %H:%M:%S"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# To save plots as images
matplotlib.use("Agg")

device = "cuda" if torch.cuda.is_available() else "cpu"


# Deep Q-Learning Agent
class DQN:
    def __init__(self, hyperparameter_set):
        self.hyperparameter_set = hyperparameter_set
        with open("hyperparameters.yml", "r") as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        # Hyperparameters
        self.env_id = hyperparameters["env_id"]
        self.replay_buffer_size = hyperparameters[
            "replay_buffer_size"
        ]  # size of replay buffer
        self.mini_batch_size = hyperparameters[
            "mini_batch_size"
        ]  # size of the training data set samples from replay buffer
        self.epsilon_init = hyperparameters["epsilon_init"]  # 1 = 100% random actions
        self.epsilon_decay = hyperparameters["epsilon_decay"]  # epsilon decay rate
        self.epsilon_min = hyperparameters["epsilon_min"]  # minimum epsilon value
        self.network_sync_rate = hyperparameters[
            "network_sync_rate"
        ]  # number of steps the agent takes before syncing the policy
        self.discount_factor_g = hyperparameters[
            "discount_factor_g"
        ]  # discount rate (gamma)
        self.learning_rate_a = hyperparameters[
            "learning_rate_a"
        ]  # learning rate (alpha)

        # Neural Network
        self.loss_fn = nn.MSELoss()  # NN Loss Function: Mean Squared Error
        self.optimizer = None  # NN Optimizer

        # Paths
        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}.pt")
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}.png")

    def run(self, is_training=True, render=False):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(message)
            with open(self.LOG_FILE, "w") as file:
                file.write(message + "\n")
        env = gym.make(
            self.env_id, render_mode="human" if render else None, continuous=False
        )
        env = ImageEnv(env)

        state_dim = (4, 84, 84)
        action_dim = env.action_space.n

        policy_net = CNNActionValue(state_dim[0], action_dim).to(device)

        if is_training:
            # Create the target network
            target_net = CNNActionValue(state_dim[0], action_dim).to(device)

            # Resuming training
            if os.path.exists(self.MODEL_FILE):
                print(
                    f"{start_time.strftime(DATE_FORMAT)}: Resuming training from checkpoint"
                )
                policy_net.load_state_dict(torch.load(self.MODEL_FILE))
                target_net.load_state_dict(torch.load(self.MODEL_FILE))
                epsilon = self.epsilon_min
            else:
                # Sync if starting fresh and initialize epsilon
                target_net.load_state_dict(policy_net.state_dict())
                epsilon = self.epsilon_init

            # Initialize replay buffer
            replay_buffer = ReplayBuffer(self.replay_buffer_size)

            # Policy network optimizer
            self.optimizer = torch.optim.Adam(
                policy_net.parameters(), lr=self.learning_rate_a
            )

            # Tracked for syncing policy and target network
            step_count = 0

            # List to keep track of epsilon decay
            epsilon_history = []

            # Track best reward
            best_reward = -999999999
            rewards_per_episode = []
        else:
            # Load saved policy
            policy_net.load_state_dict(torch.load(self.MODEL_FILE))

            # Evaluate Model
            policy_net.eval()

        # Keep training until we are satisfied with results or need to tweak hyperparameters
        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            terminated = False
            episode_reward = 0.0

            while not terminated:
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        # tensor([1, 2, 3, ...]) ===> tensor([[1, 2, 3, ...]])
                        action = (
                            policy_net(state.unsqueeze(dim=0)).squeeze().argmax().item()
                        )

                # processing:
                new_state, reward, terminated, _, info = env.step(action)

                # accumulate reward
                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)
                action = torch.tensor(action, dtype=torch.long, device=device)

                if is_training:
                    replay_buffer.append((state, action, new_state, reward, terminated))
                    step_count += 1

                # If we have enough experiences
                if len(replay_buffer) > self.mini_batch_size:
                    mini_batch = replay_buffer.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_net, target_net)

                    # Decay epsilon
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                    # copy policy network to target network after a certain number of steps
                    if step_count > self.network_sync_rate:
                        target_net.load_state_dict(policy_net.state_dict())
                        step_count = 0

                # move to the new state
                state = new_state

                # when game is over, break
                if terminated:
                    break

            rewards_per_episode.append(episode_reward)

            # Save model when new best reward is obtained
            if is_training:
                if episode_reward > best_reward:
                    message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(message)
                    with open(self.LOG_FILE, "a") as file:
                        file.write(message + "\n")

                    torch.save(policy_net.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

    # Calculates target and trains the policy network
    def optimize(self, mini_batch, policy_net, target_net):
        # Process the batch all at once rather than one experience at a time

        # Transpose the list of experiences and separate at each element
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # Stack tensors to create batch tensors
        # tensor([[1,2,3]])
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            # Calculate target Q-values
            target_q = (
                rewards
                + (1 - terminations)
                * self.discount_factor_g
                * target_net(new_states).max(dim=1)[0]
            )

        # Calculate Q-values from current policy
        current_q = (
            policy_net(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        )

        # Compute loss for the whole minibatch
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()  # Compute gradients (backpropagation)
        self.optimizer.step()  # Update network parameters i.e. weights and biases

    def save_graph(self, rewards_per_episode, epsilon_history):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        # Plot avg rewards (Y-axis) vs episodes (X-Axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x - 99) : (x + 1)])
        ax1.plot(mean_rewards)
        ax1.set_ylabel("Mean Rewards")
        ax1.set_xlabel("Episodes")

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        ax2.plot(epsilon_history)
        ax2.set_ylabel("Epsilon Decay")
        ax2.set_xlabel("Episodes")

        plt.tight_layout()
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test model.")
    parser.add_argument("hyperparameters", help="")
    parser.add_argument("--train", help="Training mode", action="store_true")
    args = parser.parse_args()
    dqn = DQN(hyperparameter_set=args.hyperparameters)

    if args.train:
        dqn.run(is_training=True)
    else:
        dqn.run(is_training=True, render=True)
