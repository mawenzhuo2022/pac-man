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

"""
THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING
A TUTOR OR CODE WRITTEN BY OTHER STUDENTS
- Candy Gao
"""

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
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            newQ = self.values.copy()

            for states in self.mdp.getStates():
                #check if at terminal states, reiter if not
                if self.mdp.isTerminal(states) == False:
                    optiA = self.computeActionFromValues(states)
                    newQ[states] = self.computeQValueFromValues(states, optiA)

            self.values = newQ



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        score = 0
        
        for s in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, s[0])
            score = score + s[1]*(reward + self.discount*self.values[s[0]])
        
        return score

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        optimal = None
        maxVal = float("-inf")
        
        for action in self.mdp.getPossibleActions(state):
            q = self.computeQValueFromValues(state, action)
            if maxVal < q:
                maxVal = q
                optimal = action
        
        return optimal

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
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        #maxVal = float("-inf")

        for i in range(self.iterations):
            cur = states[i % len(states)]
            if self.mdp.isTerminal(cur) == False:
                maxVal = float("-inf") #reinitialize max value for each action determination
                for action in self.mdp.getPossibleActions(cur):
                    q = self.computeQValueFromValues(cur, action)
                    if q >= maxVal:
                        maxVal = q
                self.values[cur] = maxVal



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
        "*** YOUR CODE HERE ***"
        pq = util.PriorityQueue()
        predecessors = util.Counter()

        for i in self.mdp.getStates():
            if self.mdp.isTerminal(i) == False: predecessors[i] = set()
        


        for i in self.mdp.getStates():
            if self.mdp.isTerminal(i): continue

            actions = self.mdp.getPossibleActions(i)
            for action in actions:
                nextMove = self.mdp.getTransitionStatesAndProbs(i, action)
                for j,p in nextMove:
                    if (p != 0) and (self.mdp.isTerminal(j) == False):
                        predecessors[j].add(i)

            optimal = self.computeActionFromValues(i)
            maxQ = self.computeQValueFromValues(i, optimal)
            pq.push(i, -1*abs(self.values[i]- maxQ))



        for x in range(self.iterations):
            if pq.isEmpty(): return
            #else:

            move = pq.pop()

            optimalV = self.computeActionFromValues(move)
            self.values[move] = self.computeQValueFromValues(move, optimalV)

            for child in predecessors[move]:
                opt = self.computeActionFromValues(child)
                maxiQ = self.computeQValueFromValues(child, opt)
                diff = abs(self.values[child] - maxiQ)
                if diff > self.theta: pq.update(child, -diff)

        



