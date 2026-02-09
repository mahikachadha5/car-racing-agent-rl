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

## DQN
Because this problem consists of a very large observation space and a small discrete action space. A Deep Q-Network (DQN) is the algorithm of choice for obtaining a solution.
