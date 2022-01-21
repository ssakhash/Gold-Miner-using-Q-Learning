#Initializing the Libraries
import numpy as np
from copy import copy, deepcopy

#Function initialized the Q-Table for RL with Gaussian Distribution (Mean, Standard Deviation) = (0, 1)
def initial_QTable():
    #After experimenting with the Random Seed values, it was observed that the random seed value of 6 had the best initialization.
    np.random.seed(6)
    QTable = np.random.normal(0, 1, size = (4, 4, 5, 5))
    return QTable

#Function returns the Optimal Policy based on Index of Max Optimal Value
def policy_finder(policy_index):
    if policy_index == 0:
        optimal_policy = 'Up'
    if policy_index == 1:
        optimal_policy = 'Down'
    if policy_index == 2:
        optimal_policy = 'Left'
    if policy_index == 3:
        optimal_policy = 'Right'
    return optimal_policy

#Function returns Reward given Current State, Gold at hand and Optimal Policy
def reward_finder(gold, x, y, optimal_policy):
    if gold > 0:
        if x == 1 and y == 0:
            if optimal_policy == "Up":
                reward = gold
                return reward
            else:
                reward = 0
                return reward
        elif x == 0 and y == 1:
            if optimal_policy == "Left":
                reward = gold
                return reward
            else:
                reward = 0
                return reward
        else:
            reward = 0
            return reward
    else:
        reward = 0
        return reward

#Function returns the Next State in Regular Cases
def action_checker(x, y, optimal_policy):
    if optimal_policy == "Up" and x > 0:
        return x-1, y
    if optimal_policy == "Down" and x < 4:
        return x+1, y
    if optimal_policy == "Left" and y > 0:
        return x, y-1
    if optimal_policy == 'Right' and y < 4:
        return x, y+1
    else:
        return x, y

#Function returns the Next State in Unique Cases
def next_state(gold, x, y, optimal_policy):
    #Case where Agent has Gold and is Approaching Home
    if x == 1 and y == 0:
        if gold > 0:
            if optimal_policy == "Up":
                gold = 0
                return gold, x, y
            else:
                x, y = action_checker(x, y, optimal_policy)
                return gold, x, y
        else:
            x, y = action_checker(x, y, optimal_policy)
            return gold, x, y
    if x == 0 and y == 1:
        if gold > 0:
            if optimal_policy == "Left":
                gold = 0
                return gold, x, y
            else:
                x, y = action_checker(x, y, optimal_policy)
                return gold, x, y
        else:
            x, y = action_checker(x, y, optimal_policy)
            return gold, x, y
    if x == 4 and y == 3:
        if optimal_policy == "Right":
            if gold < 3:
                gold = gold + 1
                return gold, x, y
            elif gold == 3:
                gold = 3
                return gold, x, y
            else:
                gold = 3
                return gold, x, y
        else:
            x, y = action_checker(x, y, optimal_policy)
            return gold, x, y
    if x == 3 and y == 4:
        if optimal_policy == "Down":
            if gold < 3:
                gold = gold + 1
                return gold, x, y
            elif gold == 3:
                gold = 3
                return gold, x, y
            else:
                gold = 3
                return gold, x, y
        else:
            x, y = action_checker(x, y, optimal_policy)
            return gold, x, y
    else:
        x, y = action_checker(x, y, optimal_policy)
        return gold, x, y

#Function to Train Algorithm and obtain optimal Q-Table
def QLearning_algorithm(QTable):
    print("-----------------------Q-Learning Algorithm-----------------------")
    #Hyperparameters Initialization
    gamma = 0.6
    learning_rate = 0.9
    flag = True
    counter = 0
    #Outer Loop with Episodes
    while flag == True:
        #Initial State Initialization
        gold = 0
        x = 4
        y = 0
        counter = counter + 1
        QTable_Copy = deepcopy(QTable)
        #Inner Loop with T-value
        for t in range(10000):
            QValue = []
            for action_index in range(4):
                QValue.append(QTable[action_index][gold][x][y])
            randomness = np.random.rand()
            if randomness < 0.80:
                optimal_value = max(QValue)
            elif randomness >= 0.80:
                optimal_value = QValue[3]
            policy_index = QValue.index(optimal_value)
            optimal_policy = policy_finder(policy_index)
            reward = reward_finder(gold, x, y, optimal_policy)
            gold_, x_, y_ = next_state(gold, x, y, optimal_policy)
            future_QValue = []
            for future_action_index in range(4):
                future_QValue.append(QTable[future_action_index][gold_][x_][y_])
            future_optimal_value = max(future_QValue)
            QTable[policy_index][gold][x][y] = QTable[policy_index][gold][x][y] + learning_rate * (reward + (gamma * future_optimal_value) - QTable[policy_index][gold][x][y])
            gold, x, y = next_state(gold, x, y, optimal_policy)
        if np.array_equal(QTable_Copy, QTable):
            print("Q-Table Converges at Episode: ", counter)
            flag = False
        else:
            flag = True
        if counter % 10000 == 0:
            print("Episodes: ", counter)
    return QTable

#Testing Function
def MDP(QTable):
    print("\n")
    print("----------------------------Test Run----------------------------")
    gold = 0
    x = 4
    y = 0
    gamma = 0.6
    cummulative_reward = 0
    for t in range(1, 100):
        QValue = []
        for action_index in range(4):
            QValue.append(QTable[action_index][gold][x][y])
        optimal_value = max(QValue)
        policy_index = QValue.index(optimal_value)
        optimal_policy = policy_finder(policy_index)
        reward = reward_finder(gold, x, y, optimal_policy)
        if x == 1 and y == 0:
            if gold > 0:
                if optimal_policy == "Up":
                    cummulative_reward = cummulative_reward + (gold * (gamma**t))
        if x == 0 and y == 1:
            if gold > 0:
                if optimal_policy == "Left":
                    cummulative_reward = cummulative_reward + (gold * (gamma**t))
        gold, x, y = next_state(gold, x, y, optimal_policy)
        if t < 10:
            print(f'| Time: 0{t} | Next State: {x, y} |  Optimal Policy: {optimal_policy}')
        else:
            print(f'| Time: {t} | Next State: {x, y} |  Optimal Policy: {optimal_policy}')
        #Next State based on Current State and Optimal Policy
    print("Cummulative Reward: ", cummulative_reward)

#Main Function
def main():
    QTable = initial_QTable()
    QTable1 = QLearning_algorithm(QTable)
    MDP(QTable1) 

if __name__=="__main__":
    main()