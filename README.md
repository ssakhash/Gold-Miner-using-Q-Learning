# Gold-Miner-using-Q-Learning
A 5x5 environment is chosen for this project with gold mine in the location (5,1) and miner's home in the location (1,5). The miner starts at position (1,1) with no gold. The goal is to train the Reinforcement Algorithm model in such a way that the miner can carry a maximum of 3 gold at any point, and collects reward equivalent to the amount of gold being carried when it is dropped off at the home. The miner collects a gold anytime the miner is in the adjacent state to the gold mine and moves into the gold mine. In such cases, the new state is the same as the old state with an increment in the number of gold currently present in the miner's possession. The project uses Q-Learning algorithm to train the model using exploration/exploitation method in order to maximise the cumulative reward obtained by the Miner with a discount factor of 0.6. After running the simulation for the first 40 times, it was found that the miner learns effectively in an exploration/exploitation method of training.

<p align="center">
  <img src="https://github.com/ssakhash/Gold-Miner-using-Q-Learning/blob/main/Output.png" />
</p>
