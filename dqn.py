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
        self.network_sync_rate = hyperparameters["network_sync_rate"]
        self.discount_factor_g = hyperparameters["discount_factor_g"]
        self.learning_rate_a = hyperparameters["learning_rate_a"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None
        self.replay_buffer_size = hyperparameters["replay_buffer_size"] # size of replay buffer

    def run(self, is_training=True, render=False):
        env = gym.make(
            "CarRacing-v3", render_mode="human" if render else None, continuous=False
        )
        env = ImageEnv(env)

        state_dim = (4, 84, 84)
        action_dim = env.action_space.n

        policy_net = CNNActionValue(state_dim[0], action_dim).to(device)

        if is_training:
            # Initialize epsilon and replay buffer
            replay_buffer = ReplayBuffer(self.replay_buffer_size)
            epsilon = self.epsilon_init
            target_net = CNNActionValue(state_dim[0], action_dim).to(device)
            # sync target network to policy network
            target_net.load_state_dict(policy_net.state_dict())

            step_count = 0

            # policy network optimizer
            self.optimizer = torch.optim.Adam(
                policy_net.parameters(), lr=self.learning_rate_a
            )

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
                    # action = torch.tensor(action, dtype=torch.float, device=device)
                else:
                    with torch.no_grad():
                        # tensor([1, 2, 3, ...]) ===> tensor([[1, 2, 3, ...]])
                        action = policy_net(state.unsqueeze(dim=0)).squeeze().argmax()

                # processing:
                new_state, reward, terminated, _, info = env.step(action.item())

                # accumulate reward
                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    replay_buffer.append((state, action, new_state, reward, terminated))
                    step_count += 1

                # move to the new state
                state = new_state

                # when game is over, break
                if terminated:
                    break

            rewards_per_episode.append(episode_reward)

            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)

            # if we have enough experiences
            if len(replay_buffer) > self.mini_batch_size:
                # sample from memory
                mini_batch = replay_buffer.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_net, target_net)

                # copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_net.load_state_dict(policy_net.state_dict())
                    step_count = 0

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
            target_q = rewards + (1-terminations) * self.discount_factor_g * target_net(new_states).max(dim=1)[0]
            
        # Calculate Q-values from current policy
        current_q = policy_net(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        
        # Compute loss for the whole minibatch
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients (backpropagation)
        self.optimizer.step()       # Update network parameters i.e. weights and biases


if __name__ == "__main__":
    agent = DQN("car_racing3")
    agent.run(is_training=True, render=True)
