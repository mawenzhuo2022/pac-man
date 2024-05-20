# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # This method performs the value iteration algorithm
        for iteration in range(self.iterations):
            # We start each iteration by gathering all states from our Markov Decision Process (MDP)
            states = self.mdp.getStates()

            # A temporary storage to hold the updated values for this iteration
            new_values = util.Counter()

            for state in states:
                # For terminal states, the value is always 0 as there are no further actions possible
                if not self.mdp.getPossibleActions(state):
                    new_values[state] = 0
                    continue

                # For non-terminal states, calculate the maximum Q-value across all possible actions
                max_q_value = max(self.computeQValueFromValues(state, action)
                                  for action in self.mdp.getPossibleActions(state))
                new_values[state] = max_q_value

            # Update the values with the newly computed values for the next iteration
            self.values = new_values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
        Compute the Q-value of a given action in a given state.
        The Q-value is calculated based on the value function stored in self.values.
        """

        # Initialize the total value to 0
        q_value = 0

        # Iterate over all possible next states given the current state and action
        for nextState, probability in self.mdp.getTransitionStatesAndProbs(state, action):
            # Calculate the contribution of each next state to the Q-value
            reward = self.mdp.getReward(state, action, nextState)
            discount_value = self.discount * self.getValue(nextState)
            q_value += probability * (reward + discount_value)

        return q_value

    def computeActionFromValues(self, state):
        """
        Determine the best action to take in a given state.

        This function selects the action that maximizes the Q-value, as computed from the
        current value function (self.values). If the state is terminal (no legal actions),
        the function returns None.
        """

        best_action = None
        highest_q_value = -float('inf')

        # Iterate over all possible actions in the current state
        for action in self.mdp.getPossibleActions(state):
            # Compute the Q-value for the current action
            q_value = self.computeQValueFromValues(state, action)

            # Update the best action if this action's Q-value is higher
            if q_value > highest_q_value:
                highest_q_value = q_value
                best_action = action

        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        """
        Perform asynchronous value iteration.

        In each iteration, only the value of one state is updated. This method is
        an alternative to the standard value iteration where all states are updated
        in each iteration.
        """

        states = self.mdp.getStates()  # Retrieve all states from the MDP

        # Iterate for the specified number of iterations
        for iteration in range(self.iterations):
            # Select a state based on the current iteration number
            state = states[iteration % len(states)]

            # Update the value of the state if it is not a terminal state
            if not self.mdp.isTerminal(state):
                max_value = max(self.computeQValueFromValues(state, action)
                                for action in self.mdp.getPossibleActions(state))

                # Update the value of the state with the highest Q-value
                self.values[state] = max_value


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        """
        Perform prioritized sweeping value iteration.

        This method first computes the predecessors for each state, and then uses
        a priority queue to focus on states that have the greatest difference
        between their current value and the maximum Q-value over all actions.
        """

        # Compute the predecessors of all states
        predecessors = {}
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue

            for action in self.mdp.getPossibleActions(state):
                for nextState, _ in self.mdp.getTransitionStatesAndProbs(state, action):
                    if nextState not in predecessors:
                        predecessors[nextState] = set()
                    predecessors[nextState].add(state)

        # Initialize a priority queue
        pq = util.PriorityQueue()

        # Populate the priority queue with all non-terminal states
        for state in [s for s in self.mdp.getStates() if not self.mdp.isTerminal(s)]:
            max_q_value = max(self.getQValue(state, action) for action in self.mdp.getPossibleActions(state))
            diff = abs(self.values[state] - max_q_value)
            pq.update(state, -diff)

        # Update states iteratively based on their priority
        for _ in range(self.iterations):
            if pq.isEmpty():
                break

            state = pq.pop()

            if not self.mdp.isTerminal(state):
                self.values[state] = max(self.getQValue(state, action) for action in self.mdp.getPossibleActions(state))

            # Update the priorities of the predecessors
            for predecessor in predecessors.get(state, []):
                max_q_value = max(
                    self.getQValue(predecessor, action) for action in self.mdp.getPossibleActions(predecessor))
                diff = abs(self.values[predecessor] - max_q_value)
                if diff > self.theta:
                    pq.update(predecessor, -diff)


