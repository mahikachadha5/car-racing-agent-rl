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

## Network Architecture

The Q-network is a CNN that takes 4 stacked 84x84 grayscale frames as input. Two convolutional layers (16 filters with 8x8 kernels, then 32 filters with 4x4 kernels) extract spatial features and reduce dimensionality from 84x84 to 9x9. The 9x9 output is then flattened and passed through a fully connected layer (256 units) before producing 5 Q-values — one for each discrete action (do nothing, steer left, steer right, gas, brake).

## Replay Buffer

In gameplay environments, consecutive frames are highly correlated because each frame looks nearly identical to the previous one. For example, in the Car Racing environment, the agent might be on a straight path for 100+ frames, all requiring the same action. Training on sequential data biases the gradients toward whatever situation is happening right now, giving the network a narrow, repetitive view. To address this, we use a replay buffer. The buffer stores the agent's past experiences (state, action, reward, next_state) and samples shuffled batches for training, breaking data correlation and improving training stability.
