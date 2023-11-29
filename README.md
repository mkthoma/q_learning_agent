# Q Learning Agent

The Gridworld problem is a classic environment used in reinforcement learning to understand and implement algorithms such as Q-learning. It is a simple grid-based world where an agent navigates through different states, encountering rewards and penalties based on its actions.

In the Gridworld problem, the agent exists in a grid where each cell represents a state. The agent can move in the four cardinal directions: up, down, left, and right. Some cells may have positive rewards, negative rewards (penalties), or no reward at all. The agent's goal is to learn a policy that maximizes its cumulative reward over time.


### Key Components
- States: Each cell in the grid represents a state that the agent can be in.
- Actions: The agent can take actions to move to adjacent cells (up, down, left, right).
- Rewards: Each state has an associated reward. The agent aims to maximize its total reward.
- Transitions: Actions lead to transitions between states. The agent learns the consequences of its actions.

## Q-Learning Approach

Q-learning is a model-free reinforcement learning algorithm that enables the agent to learn a policy to maximize cumulative rewards. In the context of the Gridworld problem:

- Q-Values: Q-values represent the expected cumulative reward of taking a particular action in a specific state. The Q-value of a state-action pair is denoted as Q(s, a).
- Q-Table: The agent maintains a Q-table to store Q-values for all state-action pairs.
- Learning Process: The agent explores the environment, updating Q-values based on observed rewards and transitions. Q-values are updated using the Bellman equation: Q(s, a) = R + γ * max(Q(s', a')), where R is the immediate reward, γ is the discount factor, and max(Q(s', a')) is the maximum Q-value of the next state.
- Exploration vs. Exploitation: Q-learning balances exploration (trying new actions) and exploitation (choosing the best-known actions) to discover an optimal policy.

## Implementation Steps
1. Initialization: Initialize the Q-table with arbitrary values.
2. Exploration: Choose actions based on an exploration strategy (e.g., epsilon-greedy) to explore the environment.
3. Update Q-Values: Update Q-values based on observed rewards and transitions using the Bellman equation.
4. Policy Extraction: Extract the learned policy from the Q-table, choosing actions with the highest Q-values.
5. Repeat: Iterate through steps 2-4 until the agent converges to an optimal policy.

## Methods

- init Method:
    The init method is a special method in Python that is called when an instance of a class is created. It's used for initializing the attributes and parameters of the class. In the context of a reinforcement learning agent, this method is often used to set up the agent's parameters and data structures when it is created.

    ```python
    class QLearningAgent:
    def __init__(self, epsilon, alpha, gamma):
        self.epsilon = epsilon  # Exploration-exploitation trade-off parameter
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.qValues = {}       # Dictionary to store Q-values

    # Other initialization steps, if needed
    ```

- getQValue Method:
    The getQValue method is typically used to retrieve the Q-value for a given state-action pair. In the context of Q-learning, the Q-value represents the expected cumulative reward of taking a specific action in a specific state.

    ```python
    class QLearningAgent:
    # ...

    def getQValue(self, state, action):
        # Check if the Q-value is available, otherwise return a default value
        return self.qValues.get((state, action), 0.0)

    ```

- computeValueFromQValue Method:
    The computeValueFromQValue method is used to compute the state value from the Q-values. In Q-learning, the state value is the expected cumulative reward achievable from a given state by taking the best action. This method is essential for computing the state values during the learning process.

    ```python
    class QLearningAgent:
    # ...

    def computeValueFromQValues(self, state):
        possibleActions = get_possible_actions(state)
        if not possibleActions:
            return 0.0  # Terminal state
        # Return the maximum Q-value for the given state
        return max(self.getQValue(state, action) for action in possibleActions)

    ```    

- computeActionFromQValues Method:
    The computeActionFromQValues method is responsible for determining the best action to take in a given state based on the Q-values. In Q-learning, the agent chooses the action with the highest Q-value in a particular state. This method helps in extracting the optimal action for a given state during the decision-making process.

    ```python
    class QLearningAgent:
    # ...

    def computeActionFromQValues(self, state):
        possibleActions = get_possible_actions(state)
        if not possibleActions:
            return None  # Terminal state
        # Return the action with the highest Q-value for the given state
        return max(possibleActions, key=lambda action: self.getQValue(state, action))

    ```


- getAction Method:
    The getAction method is typically used during the decision-making process. It determines the action that the agent should take in a given state. This decision can be based on various strategies, such as choosing the action with the highest Q-value (exploitation) or exploring new actions with a certain probability (exploration).

    ```python
    class QLearningAgent:
    # ...

    def getAction(self, state):
        possibleActions = get_possible_actions(state)
        if not possibleActions:
            return None  # Terminal state
        # Exploration-exploitation trade-off
        if random() < self.epsilon:
            return random.choice(possibleActions)  # Exploration
        else:
            return self.computeActionFromQValues(state)  # Exploitation

    ```

- update Method:
    The update method is a crucial part of the Q-learning algorithm. It is used to update the Q-values based on observed rewards and transitions. This method incorporates the reward obtained after taking an action in a state and updates the Q-value accordingly. The update process is usually guided by the Bellman equation, which reflects the relationship between the current state's Q-value and the Q-values of the next state and the immediate reward.

    ```python
    class QLearningAgent:
    # ...

    def update(self, state, action, nextState, reward):
        # Update Q-value based on the Bellman equation
        currentQValue = self.getQValue(state, action)
        nextValue = self.computeValueFromQValues(nextState)
        newQValue = (1 - self.alpha) * currentQValue + self.alpha * (reward + self.gamma * nextValue)
        # Update the Q-value in the dictionary
        self.qValues[(state, action)] = newQValue

    ```

These methods collectively define the behavior and learning process of a Q-learning agent in a reinforcement learning environment. The implementation details of these methods will depend on the specific Q-learning agent class being used.

## Usage

You will need Pyhton 2.7 installed for the program to work. You can run it using the following command:
```
python gridworld.py -k 100 -a q -s 2000
```
where 
- `k`: This parameter (-k) specifies the number of episodes to run. In this case, it's set to 100, meaning the gridworld simulation will run for 100 episodes.

- `a q`: The -a flag indicates the type of agent to be used in the simulation. In this case, it's set to "q," suggesting that the Q-learning agent (qlearningAgents.QLearningAgent) will be used.

- `s`: The -s flag represents the speed of the animation during the simulation. The value after -s is set to 2000, indicating the speed factor. A higher value makes the simulation run faster, while a lower value slows it down.

Initial Stages:
![image](https://github.com/mkthoma/q_learning_agent/assets/135134412/97d77a3a-2b1c-4ab1-8309-b0a7a3a1125c)

Final Result:
![image](https://github.com/mkthoma/q_learning_agent/assets/135134412/5d7a1230-6115-4233-8bfb-ae86fa24c3ce)


### Training Logs

```
EPISODE 98 COMPLETE: RETURN WAS -0.282429536481

BEGINNING EPISODE: 99

Started in state: (0, 0)
Took action: north
Ended in state: (0, 1)
Got reward: 0.0

Started in state: (0, 1)
Took action: north
Ended in state: (0, 2)
Got reward: 0.0

Started in state: (0, 2)
Took action: east
Ended in state: (1, 2)
Got reward: 0.0

Started in state: (1, 2)
Took action: north
Ended in state: (1, 2)
Got reward: 0.0

Started in state: (1, 2)
Took action: south
Ended in state: (0, 2)
Got reward: 0.0

Started in state: (0, 2)
Took action: east
Ended in state: (0, 2)
Got reward: 0.0

Started in state: (0, 2)
Took action: north
Ended in state: (1, 2)
Got reward: 0.0

Started in state: (1, 2)
Took action: west
Ended in state: (1, 2)
Got reward: 0.0

Started in state: (1, 2)
Took action: south
Ended in state: (1, 2)
Got reward: 0.0

Started in state: (1, 2)
Took action: north
Ended in state: (1, 2)
Got reward: 0.0

Started in state: (1, 2)
Took action: south
Ended in state: (1, 2)
Got reward: 0.0

Started in state: (1, 2)
Took action: south
Ended in state: (1, 2)
Got reward: 0.0

Started in state: (1, 2)
Took action: east
Ended in state: (2, 2)
Got reward: 0.0

Started in state: (2, 2)
Took action: east
Ended in state: (3, 2)
Got reward: 0.0

Started in state: (3, 2)
Took action: exit
Ended in state: TERMINAL_STATE
Got reward: 1

EPISODE 99 COMPLETE: RETURN WAS 0.22876792455

BEGINNING EPISODE: 100

Started in state: (0, 0)
Took action: north
Ended in state: (0, 1)
Got reward: 0.0

Started in state: (0, 1)
Took action: south
Ended in state: (0, 0)
Got reward: 0.0

Started in state: (0, 0)
Took action: north
Ended in state: (0, 1)
Got reward: 0.0

Started in state: (0, 1)
Took action: north
Ended in state: (0, 2)
Got reward: 0.0

Started in state: (0, 2)
Took action: north
Ended in state: (0, 2)
Got reward: 0.0

Started in state: (0, 2)
Took action: west
Ended in state: (0, 2)
Got reward: 0.0

Started in state: (0, 2)
Took action: north
Ended in state: (0, 2)
Got reward: 0.0

Started in state: (0, 2)
Took action: east
Ended in state: (1, 2)
Got reward: 0.0

Started in state: (1, 2)
Took action: east
Ended in state: (1, 2)
Got reward: 0.0

Started in state: (1, 2)
Took action: south
Ended in state: (1, 2)
Got reward: 0.0

Started in state: (1, 2)
Took action: east
Ended in state: (1, 2)
Got reward: 0.0

Started in state: (1, 2)
Took action: west
Ended in state: (0, 2)
Got reward: 0.0

Started in state: (0, 2)
Took action: north
Ended in state: (0, 2)
Got reward: 0.0

Started in state: (0, 2)
Took action: east
Ended in state: (1, 2)
Got reward: 0.0

Started in state: (1, 2)
Took action: north
Ended in state: (1, 2)
Got reward: 0.0

Started in state: (1, 2)
Took action: north
Ended in state: (1, 2)
Got reward: 0.0

Started in state: (1, 2)
Took action: east
Ended in state: (2, 2)
Got reward: 0.0

Started in state: (2, 2)
Took action: east
Ended in state: (2, 2)
Got reward: 0.0

Started in state: (2, 2)
Took action: east
Ended in state: (3, 2)
Got reward: 0.0

Started in state: (3, 2)
Took action: exit
Ended in state: TERMINAL_STATE
Got reward: 1

EPISODE 100 COMPLETE: RETURN WAS 0.135085171767


AVERAGE RETURNS FROM START STATE: 0.340342971159

```


## Conclusion
The Gridworld problem provides a straightforward yet insightful environment for understanding and implementing Q-learning. By navigating a grid and learning from rewards, the agent can develop a strategy to optimize its actions, making Q-learning a fundamental algorithm in the field of reinforcement learning.
