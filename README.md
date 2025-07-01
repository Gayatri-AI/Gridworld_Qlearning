# Gridworld_Qlearning 
Technical Report: Q-Learning in Gridworld Environments

## Introduction
 This report describes an experiment where an artificial agent learns to navigate two different gridworlds using
 Q-Learning, a reinforcement learning algorithm. The agent learns by trial and error, balancing between trying
 new actions (exploration) and using what it already knows (exploitation). The goal is to reach a positive
 goal state while avoiding a negative terminal state, and the report explains how different settings affect the
 agent’s learning and performance. All results are supported by plots generated from the code and included
 in the accompanying document.
## Overview
 This experiment consists of two main parts, each with its own gridworld:
 • Part 1: A 5x5 grid with a positive goal (reward = 5), a negative terminal state (reward =-5), and
 some walls. The agent starts at (0,0).
 • Part 2: A larger, 5x11 grid where a tunnel joins two 5x5 grids with a positive goal (reward = 5), a
 negative terminal state (reward =-5), and more walls. The agent again starts at (0,0).
 In both parts, the agent can move North, South, East, or West. It gets a penalty (-1) if it tries to move
 outside the grid or into a wall. Part 2 uses a different method (softmax) for action selection compared to
 Part 1 (epsilon-greedy).
 Assumption: As there was no specific reward mentioned for obstacles in between. I have considered-1
 reward for them same as outside boundaries.
 
## Part 1: Results and Analysis
### Policy and Value Function Plots
 The agent was trained for 100,000 episodes. After training, two types of plots were generated for each value
 of gamma (which controls how much the agent cares about future rewards): 0.1, 0.5, and 0.9.
 1. Policy Map: Shows the best action (arrow) for each position. Walls are shown as black squares, the
 positive goal as a star, and the negative terminal state as a skull.
 2. Value Map: Shows how valuable each position is, according to the agent, using a heatmap. Higher
 values mean the agent expects more reward from that position.

### Plots and Interpretation
Figure 1: Policy Map, Gamma=0.1, Epsilon=0.1
Figure 2: Value Map, Gamma=0.1, Epsilon=0.1
Figure 3: Policy Map, Gamma=0.5, Epsilon=0.1
Figure 4: Value Map, Gamma=0.5, Epsilon=0.1
Figure 5: Policy Map, Gamma=0.9, Epsilon=0.1
Figure 6: Value Map, Gamma=0.9, Epsilon=0.1

### Comment on Results
 For gamma = 0.1, the agent mostly cares about immediate rewards. The value map is high only near the
 positive goal and negative near the bad terminal state. The policy map shows the agent quickly tries to reach
 the goal or avoid the bad state, sometimes taking risky shortcuts.
 For gamma =0.5 and 0.9, the agent cares more about future rewards. The value map spreads out, and the
 agent learns longer-term strategies, sometimes taking longer but safer paths to avoid walls and the negative
 terminal state.
 
 ###Gamma Variation
 Each gamma value (0.1, 0.5, 0.9) was tested with epsilon (exploration rate) fixed at 0.1. The above plots
 show how the agent’s strategy changes with gamma.
 
 ###Comment on Results
 Higher gamma (0.5, 0.9) leads to more cautious behavior. The agent avoids risky shortcuts and prefers paths
 that are longer but safer, as seen in the policy maps. The value maps show higher values spreading further
 from the goal, indicating the agent is planning ahead.
 3.3 Question 3: Steps to Goal vs. Epsilon
 For gamma = 0.9, the agent was trained with different epsilon values (0.1, 0.3, 0.5). The plot shows the
 average number of steps the agent took to reach the goal over the episodes.
 
Figure 7: Steps to Goal vs. Episode for Different Epsilon Values (Gamma=0.9)

 ###Comment on Results
 When epsilon is low (0.1), the agent mostly follows its learned policy and quickly learns a good path. The
 number of steps to the goal drops fast and stabilizes.
 When epsilon is higher (0.3 or 0.5), the agent explores more. This can help it find better paths, but early
 in training, it sometimes takes longer to reach the goal. Over time, all settings improve, but lower epsilon
 leads to faster, more stable learning.
 
 ## Part 2: Results and Analysis
 ###Policy and Value Function Plots
 The agent was trained for 200,000 episodes on a larger, more complex grid. Instead of epsilon-greedy, it used
 a softmax function to pick actions, which means it picks actions based on how good it thinks they are, but
 still sometimes tries less likely options.
 Again, policy and value maps were generated for gamma = 0.1, 0.5, and 0.9, with the softmax parameter
 (beta) set to 0.1.

Figure 8: Policy Map, Gamma=0.1, Beta=0.1
Figure 9: Value Map, Gamma=0.1, Beta=0.1
Figure 10: Policy Map, Gamma=0.5, Beta=0.1
Figure 11: Value Map, Gamma=0.5, Beta=0.1
Figure 12: Policy Map, Gamma=0.9, Beta=0.1
Figure 13: Value Map, Gamma=0.9, Beta=0.1

###Comment on Results
 With softmax, the agent explores more smoothly. The value maps show the agent learns to avoid the negative
 terminal state and head for the positive goal, even in the larger grid. The policy maps show arrows pointing
 toward the goal, avoiding walls and the negative state. Higher gamma leads to more cautious, long-term
 strategies, similar to Part 1.

## Gamma Variation with Softmax
 As before, higher gamma means the agent cares more about future rewards. The plots show that with higher
 gamma, the agent learns to take longer but safer paths, avoiding risks.
 
 ##Comment on Results
 The agent’s strategy becomes more cautious as gamma increases, similar to Part 1, but now in a more
 complex environment. The softmax exploration helps the agent find good paths without getting stuck in
 local optima. 
 
 ##Steps to Goal vs. Beta
 For gamma = 0.9, the agent was trained with different beta values (0.1, 0.3, 0.5). The plot shows the average
 number of steps to reach the goal over the episodes.
 Figure 14: Steps to Goal vs. Episode for Different Beta Values (Gamma=0.9)
 
 ##Comment on Results
 Lower beta (0.1) means the agent tries more random actions, which can help it discover new paths but also
 means it sometimes takes longer to reach the goal. Higher beta (0.5) means the agent sticks more to its best
 guess, so it learns faster but might get stuck in suboptimal paths if it doesn’t explore enough. A moderate
 beta (like 0.3) often gives a good balance between exploration and sticking to what works.

 
##Inference and Conclusion
 Looking at all the plots and results:
 • Gamma(Future Reward Importance): Higher gamma makes the agent plan ahead and take safer,
 sometimes longer paths. Lower gamma makes it focus on immediate rewards.
 • Epsilon/Beta (Exploration): Lower values make the agent stick to what it knows, learning quickly
 but possibly missing better paths. Higher values make it explore more, which can help find better paths
 but may slow down learning.
 • Environment Complexity: In the larger, more complex grid (Part 2), the agent needs more training
 to learn good strategies, but the overall patterns are similar to the smaller grid.
 • Exploration Strategy: Usingsoftmax (Part 2) gives smoother exploration compared to epsilon-greedy
 (Part 1), but both methods can find good policies if tuned well.
 6 Final Thoughts
 This assignment demonstrates how reinforcement learning agents can learn to navigate complex environments
 by balancing exploration and exploitation. The choice of parameters like gamma and exploration strategy
 (epsilon or beta) has a big impact on how well and how quickly the agent learns. The plots help visualize
 these effects and show how the agent’s strategy develops over time.
 
