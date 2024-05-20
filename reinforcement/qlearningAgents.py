# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        """
        Initialize a Q-learning agent.

        This method sets up the agent by initializing its Q-values. The Q-values
        are stored in a Counter (an enhanced dictionary) from the util module,
        where keys are tuples of (state, action).
        """

        # Initialize the parent ReinforcementAgent class
        ReinforcementAgent.__init__(self, **args)

        # Create a Counter to store Q-values, indexed by (state, action) tuples
        self.q_values = util.Counter()

    def getQValue(self, state, action):
        """
        Returns the Q-value for a specific state-action pair.

        If the state-action pair has not been encountered before, it returns 0.0.
        Otherwise, it returns the Q-value from self.q_values.
        """

        # Check if the state-action pair has been encountered before
        if (state, action) not in self.q_values:
            # If not encountered, return a default value of 0.0
            return 0.0

        # If encountered, return the Q-value from the stored values
        return self.q_values[(state, action)]

    def computeValueFromQValues(self, state):
        """
        Returns the maximum Q-value for a given state over all legal actions.
        If there are no legal actions (e.g., in a terminal state), returns 0.0.
        """

        legalActions = self.getLegalActions(state)

        # If there are no legal actions, return 0.0 (e.g., terminal state)
        if not legalActions:
            return 0.0

        # Compute and return the maximum Q-value among all legal actions
        return max(self.getQValue(state, action) for action in legalActions)

    def computeActionFromQValues(self, state):
        """
        Compute the best action to take in a given state.

        Returns the action with the highest Q-value for the state. If there are no legal actions
        (as in a terminal state), it returns None.
        """

        legalActions = self.getLegalActions(state)

        # If there are no legal actions, return None (indicating a terminal state)
        if not legalActions:
            return None

        # Find the action with the highest Q-value
        bestAction = None
        maxQValue = float('-inf')
        for action in legalActions:
            qValue = self.getQValue(state, action)
            if qValue > maxQValue:
                maxQValue = qValue
                bestAction = action

        return bestAction

    def getAction(self, state):
        """
        Compute the action to take in the current state.

        With probability `self.epsilon`, a random action is chosen. Otherwise, the action
        with the best policy (highest Q-value) is selected. If there are no legal actions
        (terminal state), `None` is chosen as the action.
        """

        legalActions = self.getLegalActions(state)

        # If there are no legal actions (terminal state), return None
        if not legalActions:
            return None

        # Epsilon-greedy strategy
        if util.flipCoin(self.epsilon):
            # Choose a random action
            return random.choice(legalActions)
        else:
            # Choose the best action based on Q-values
            return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """
        Update the Q-value for a given state and action based on the observed transition
        and reward.

        This method is called by the parent class to observe a state-action-nextState-reward
        transition and to update the Q-value accordingly.
        """

        # Calculate the sample value based on whether nextState is present
        sample = reward
        if nextState:
            sample += self.discount * self.computeValueFromQValues(nextState)

        # Update the Q-value using the learning rate (alpha)
        self.q_values[(state, action)] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * sample


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
        Return the Q-value for a state-action pair.

        The Q-value is computed as the dot product of the feature vector for the state-action
        pair and the corresponding weights. This is a key concept in approximate Q-learning.
        """

        # Extract the features for the given state-action pair
        features = self.featExtractor.getFeatures(state, action)

        # Compute the dot product of features and their corresponding weights
        q_value = sum(features[feature] * self.weights[feature] for feature in features)

        return q_value

    def update(self, state, action, nextState, reward):
        """
        Update the weights based on the observed transition.

        This method adjusts the weights in the feature vector according to the difference
        between the observed reward and discounted future value, and the estimated Q-value.
        """

        # Calculate the difference (temporal difference error) between
        # (reward + discounted value of the next state) and the current Q-value
        diff = (reward + self.discount * self.getValue(nextState)) - self.getQValue(state, action)

        # Extract the features for the given state-action pair
        features = self.featExtractor.getFeatures(state, action)

        # Update the weights for each feature
        for feature in features:
            self.weights[feature] += self.alpha * diff * features[feature]


    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
