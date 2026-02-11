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

device = "cuda" if torch.cuda.is_available() else "cpu"


class DQN:
    def __init__(self, hyperparameter_set):
        with open("hyperparameters.yml", "r") as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.replay_memory_size = hyperparameters["replay_memory_size"]
        self.mini_batch_size = hyperparameters["mini_batch_size"]
        self.epsilon_init = hyperparameters["epsilon_init"]
        self.epsilon_decay = hyperparameters["epsilon_decay"]
        self.epsilon_min = hyperparameters["epsilon_min"]

    def run(self, is_training=True, render=False):
        env = gym.make(
            "CarRacing-v3", render_mode="human" if render else None, continuous=False
        )
        env = ImageEnv(env)

        state_dim = (4, 84, 84)
        action_dim = env.action_space.n

        policy = CNNActionValue(state_dim[0], action_dim).to(device)

        if is_training:
            replay_buffer = ReplayBuffer(self.replay_memory_size)

            epsilon = self.epsilon_init

        rewards_per_episode = []
        epsilon_history = []

        # keep training until we are satisfied with results
        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            terminated = False
            episode_reward = 0.0

            while not terminated:
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    #action = torch.tensor(action, dtype=torch.float, device=device)
                else:
                    with torch.no_grad():
                        # tensor([1, 2, 3, ...]) ===> tensor([[1, 2, 3, ...]])
                        action = policy(state.unsqueeze(dim=0)).squeeze().argmax()

                # processing:
                new_state, reward, terminated, _, info = env.step(action.item())

                # accumulate reward
                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    replay_buffer.append((state, action, new_state, reward, terminated))

                # move to the new state
                state = new_state

                # when game is over, break
                if terminated:
                    break

            rewards_per_episode.append(episode_reward)

            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)

if __name__ == '__main__':
    agent = DQN('car_racing3')
    agent.run(is_training=True, render=True)