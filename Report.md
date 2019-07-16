# Report on solving the Tennis collab-compet Unity Environment

This is a report on my experience to solve the tennis environment. The description is in the README. There you can read about downloading the environment, reward per timestep and so on.

## MADDPG - Multi-Agent Deep Deterministic Policiy Gradient

The [MADDPG](https://arxiv.org/abs/1706.02275) algorithm is similar to the [DDPG](https://arxiv.org/abs/1509.02971) that solved the reacher environment in my other github project `drl-reacher-continuous-control`. 

You can see a graph of the score and agents loaded with the trained weights in the `report.ipynb`.

## MADDPG Neural Network Architecture

### Actor Neural Network
The structure for the local and the target networks is the same:
- Linear Layer with input state dimension for a single agent mapped to 64 hidden units
- ReLu
- Linear Layer with 64 hidden units
- ReLu
- Linear Layer that maps the 64 hidden units to the action dimension for one agent

### Critic Neural Network
Again local and target network have the same structure
- Linear Layer with input state dimension of both agents states mapped to 64 hidden units
- ReLu
- The output of the ReLu and the combined actions dimension of both agents is combined
- Linear Layer to map the ( 64 + full actions dim) to 64 hidden units
- ReLu
- Linear layer that maps to a scalar output that represents the Q(s,a) value

### Hyperparameter

- Tau: 1e-2 (update factor for the target network parameters)
- Gamma: 0.95 , discount factor
- Replay buffer size: 1e5
- Batch size: 512
- Learning rate for critic and actor optimizer: 1e-4
- random seed: 48 ( don't tested that realy)

## Results

I managet to solve the environment with a mean of 0.5 for a 100 episode window after 3204 episodes. The saved weights and scores saved as a numpy npy file are in the `weights` folder. I loaded and ploted the results in the report notebook.

## Further research
I think that a prioritized experience replay buffer will perform better, because it takes some time for the exploration part of the agent to sample experience with higher play scores.
Also testing more hyperparameter iterations could be good. 
Interesting could also be to reduce the information the agent recieve after finding a faster setup.