#Initializing the Libraries
import numpy as np
from copy import copy, deepcopy

#Function initialized the Q-Table for RL with Gaussian Distribution (Mean, Standard Deviation) = (0, 1)
def initial_QTable():
    """
    Initialize the Q-table for reinforcement learning with a Gaussian distribution.
    
    The Q-table represents the action-value function, which estimates the expected future reward
    for taking a specific action in a given state. It is initialized with random values drawn from
    a Gaussian distribution with mean 0 and standard deviation 1.
    
    Returns:
        np.ndarray: The initialized Q-table with shape (4, 4, 5, 5).
    """

    np.random.seed(6)
    QTable = np.random.normal(0, 1, size = (4, 4, 5, 5))
    return QTable

#Function returns the Optimal Policy based on Index of Max Optimal Value
def policy_finder(policy_index):
    """
    Get the optimal policy based on the index of the maximum Q-value.
    
    The policy determines the action to take in a given state. It is derived from the Q-table by
    selecting the action with the highest expected future reward.
    
    Args:
        policy_index (int): The index of the maximum Q-value.
        
    Returns:
        str: The optimal policy ('Up', 'Down', 'Left', 'Right', or 'Unknown Policy').
    """

    policy_map = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}
    return policy_map.get(policy_index, "Unknown Policy")

#Function returns Reward given Current State, Gold at hand and Optimal Policy
def reward_finder(gold, x, y, optimal_policy):
    """
    Calculate the reward based on the current state, gold at hand, and optimal policy.
    
    The reward function determines the immediate reward received for taking an action in a given state.
    It encourages the agent to reach specific goal states and collect gold.
    
    Args:
        gold (int): The amount of gold at hand.
        x (int): The x-coordinate of the current state.
        y (int): The y-coordinate of the current state.
        optimal_policy (str): The optimal policy ('Up', 'Down', 'Left', 'Right').
        
    Returns:
        int: The reward value.
    """

    if gold > 0:
        if (x, y) == (1, 0) and optimal_policy == "Up":
            return gold
        elif (x, y) == (0, 1) and optimal_policy == "Left":
            return gold
    return 0

#Function returns the Next State in Regular Cases
def action_checker(x, y, optimal_policy):
    """
    Calculate the next state based on the current state and optimal policy.
    
    The next state is determined by applying the optimal policy to the current state. It represents
    the expected state the agent will transition to after taking the optimal action.
    
    Args:
        x (int): The x-coordinate of the current state.
        y (int): The y-coordinate of the current state.
        optimal_policy (str): The optimal policy ('Up', 'Down', 'Left', 'Right').
        
    Returns:
        tuple: The next state coordinates (x, y).
    """

    if optimal_policy == "Up" and x > 0:
        return x - 1, y
    elif optimal_policy == "Down" and x < 4:
        return x + 1, y
    elif optimal_policy == "Left" and y > 0:
        return x, y - 1
    elif optimal_policy == 'Right' and y < 4:
        return x, y + 1
    return x, y

#Function returns the Next State in Unique Cases
def next_state(gold, x, y, optimal_policy):
    if (x, y) in [(1, 0), (0, 1)] and gold > 0:
        if optimal_policy in ["Up", "Left"]:
            return 0, x, y
    elif (x, y) in [(4, 3), (3, 4)] and optimal_policy in ["Right", "Down"]:
        return min(gold + 1, 3), x, y
    return gold, *action_checker(x, y, optimal_policy)

#Function to Train Algorithm and obtain optimal Q-Table
def QLearning_algorithm(QTable):
    """
    Train the Q-learning algorithm to find the optimal Q-table.
    
    Q-learning is a model-free reinforcement learning algorithm that learns the optimal action-value
    function (Q-table) through iterative updates. It explores the environment and updates the Q-values
    based on the observed rewards and the estimated future rewards.
    
    Args:
        q_table (np.ndarray): The initial Q-table.
        gamma (float): The discount factor (default: 0.6).
        learning_rate (float): The learning rate (default: 0.9).
        max_episodes (int): The maximum number of episodes to run (default: 10000).
        convergence_threshold (int): The number of episodes to check for convergence (default: 10000).
        
    Returns:
        np.ndarray: The trained Q-table.
    """

    print("-----------------------Q-Learning Algorithm-----------------------")
    gamma, learning_rate = 0.6, 0.9
    counter = 0

    while True:
        gold, x, y = 0, 4, 0
        counter += 1
        QTable_Copy = deepcopy(QTable)

        for t in range(10000):
            QValue = [QTable[action_index][gold][x][y] for action_index in range(4)]
            randomness = np.random.rand()
            optimal_value = max(QValue) if randomness < 0.80 else QValue[3]
            policy_index = QValue.index(optimal_value)
            optimal_policy = policy_finder(policy_index)
            reward = reward_finder(gold, x, y, optimal_policy)
            gold_, x_, y_ = next_state(gold, x, y, optimal_policy)
            future_QValue = [QTable[future_action_index][gold_][x_][y_] for future_action_index in range(4)]
            future_optimal_value = max(future_QValue)
            QTable[policy_index][gold][x][y] += learning_rate * (reward + (gamma * future_optimal_value) - QTable[policy_index][gold][x][y])
            gold, x, y = gold_, x_, y_

        if np.array_equal(QTable_Copy, QTable):
            print("Q-Table Converges at Episode:", counter)
            break

        if counter % 10000 == 0:
            print("Episodes:", counter)

    return QTable

#Testing Function
def MDP(QTable):
    gold, x, y, gamma = 0, 4, 0, 0.6
    cummulative_reward = 0

    for t in range(1, 100):
        QValue = [QTable[action_index][gold][x][y] for action_index in range(4)]
        optimal_value = max(QValue)
        policy_index = QValue.index(optimal_value)
        optimal_policy = policy_finder(policy_index)
        reward = reward_finder(gold, x, y, optimal_policy)

        if ((x, y) == (1, 0) and gold > 0 and optimal_policy == "Up") or \
           ((x, y) == (0, 1) and gold > 0 and optimal_policy == "Left"):
            cummulative_reward += gold * (gamma ** t)

        gold, x, y = next_state(gold, x, y, optimal_policy)

        print(f'| Time: {t:02} | Next State: {(x, y)} | Optimal Policy: {optimal_policy}')

    print("Cummulative Reward:", cummulative_reward)


#Main Function
def main():
    QTable = initial_QTable()
    QTable1 = QLearning_algorithm(QTable)
    MDP(QTable1) 

if __name__=="__main__":
    main()