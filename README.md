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
