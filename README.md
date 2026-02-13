# Teaching an Agent to Drive

Implementation of a Deep Q-Learning Network to play the Car Racing game environment from OpenAI Gymnasium.

## Understanding the problem

The Car Racing game is a racing environment that involves a closed-loop track, where an agent controls a racing car. The objective of the game is to cover as many in-bounds track tiles as possible in the shortest amount of time.

### Action Space

There are 5 possible actions when using a discrete version of this environment:

- 0: do nothing
- 1: steer right
- 2: steer left
- 3: gas
- 4: brake

### Observation Space

A top-down 96x96 RGB image of the car and race track.

### Rewards

The reward is -0.1 for every frame and +1000/N for every track tile visited, where N is the total number of tiles visited in the track.

## Frame Preprocessing

I decided to preprocess the raw frames (96x96 RGB) before being feeding them to the network. This includes the following:

1. **Crop**: 96x96 → 84x84 (remove black bar at bottom)
2. **Grayscale**: RGB → single channel (color is unnecessary)
3. **Normalize**: Pixel values scaled to 0-1
4. **Stack**: 4 consecutive frames stacked together so the agent has some context for actions
5. **Ignore**: Ignored first 50 frames (just a zoom-in on the car)

## Network Architecture

The Q-network is a CNN that takes 4 stacked 84x84 grayscale frames as input. Two convolutional layers (16 filters with 8x8 kernels, then 32 filters with 4x4 kernels) extract spatial features and reduce dimensionality from 84x84 to 9x9. The 9x9 output is then flattened and passed through a fully connected layer (256 units) before producing 5 Q-values — one for each discrete action (do nothing, steer left, steer right, gas, brake).

## Replay Buffer

In gameplay environments, consecutive frames are highly correlated because each frame looks nearly identical to the previous one. For example, in the Car Racing environment, the agent might be on a straight path for 100+ frames, all requiring the same action. Training on sequential data biases the gradients toward whatever situation is happening right now, giving the network a narrow, repetitive view. To address this, we use a replay buffer. The buffer stores the agent's past experiences (state, action, reward, next_state) and samples shuffled batches for training, breaking data correlation and improving training stability.

## Target Network

In Q-learning, we update our network using the Bellman equation:
```
Q(s, a) = reward + γ * max(Q(s', a'))
```

The problem is that when we use the same network to both predict Q-values and generate targets, updating the network means the targets shift too. Essentially, it becomes a moving target.

To solve this, DQN uses two networks:

| Network | Purpose | Updates |
|---------|---------|---------|
| Policy Network | Makes decisions, gets trained | Every step |
| Target Network | Provides stable Q-value targets | Every N steps (sync) |

By only syncing every N steps, we give the policy network a stable target to learn from.

## Epsilon-Greedy Exploration

The agent uses epsilon-greedy action selection to balance exploration and exploitation:

- With probability ε: take a random action (explore)
- With probability 1-ε: take the best action according to the policy (exploit)

Epsilon starts at 1.0 (100% random) and decays over time toward a minimum value (0.05), allowing the agent to explore early and exploit learned behavior later.

## Results & Lessons Learned

### Initial Attempt

My first training run plateaued around -800 mean reward after 200 episodes. While the agent occasionally achieved spikes as high as -57, performance was inconsistent and the average showed no upward trend.

**The issue:** My network_sync_rate was too low and the policy was chasing a constantly moving target, leading to unstable learning. 

## Project Structure
```
├── dqn.py              # DQN agent and training loop
├── cnn.py              # CNN architecture for Q-value prediction
├── replay_buffer.py    # Experience replay implementation
├── img.py              # Frame preprocessing wrapper
├── hyperparameters.yml # Training configurations
├── runs/               # Saved models, logs, and graphs
└── README.md
```


