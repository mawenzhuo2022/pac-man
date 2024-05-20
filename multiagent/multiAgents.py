# multiAgents.py
# --------------
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
import math

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Initialize empty lists
        ghost_dis = []
        food_dis = []

        # Calculate the Manhattan distance from new position to each ghost
        for ghost in newGhostStates:
            dis = manhattanDistance(newPos, ghost.configuration.pos)
            ghost_dis.append(dis)

        # Get all the food positions
        foodList = newFood.asList()

        # Calculate the Manhattan distance from new position to food position
        for food in foodList:
            dis = manhattanDistance(newPos, food)
            food_dis.append(dis)

        # get the smallest ghost distance if exists, otherwise set it to a large value
        if ghost_dis:
            closestGhost = min(ghost_dis)
        else:
            closestGhost = float('inf')

        # get the smallest one food distances if exists, otherwise set it to a large value
        if food_dis:
            closestFood = min(food_dis)
        else:
            closestFood = float('inf')

        # Get score
        score = successorGameState.getScore()

        # Deduct a large value from the score if a ghost is too close
        if closestGhost < 2:
            score = score - 10000

        # If the ghost is not too close, prioritize getting closer to food
        else:
            foodReward = 1.0 / closestFood
            score = score + foodReward

        return score




def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        bestScore, bestAction = self.minimax(gameState, 0, 0)
        return bestAction

    def minimax(self, gameState, depth, agentIndex):
        """
        Compute the best action using the minimax algorithm
        """
        if self.terminalTest(gameState, depth):
            return self.evaluationFunction(gameState), None

        if agentIndex == 0:  # Pacman's turn
            return self.maxValue(gameState, depth, agentIndex)
        else:  # Ghost's turn
            return self.minValue(gameState, depth, agentIndex)

    def maxValue(self, gameState, depth, agentIndex):
        """
        Calculate the best move for Pacman
        """
        actions = gameState.getLegalActions(agentIndex)
        bestAction = None
        bestValue = float('-inf')
        for action in actions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            value, _ = self.minimax(successorState, depth, (agentIndex + 1) % gameState.getNumAgents())
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestValue, bestAction

    def minValue(self, gameState, depth, agentIndex):
        """
        Calculate the best move for the Ghost
        """
        actions = gameState.getLegalActions(agentIndex)
        bestAction = None
        bestValue = float('inf')
        for action in actions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            if nextAgent == 0:  # Pacman's turn next
                value, _ = self.minimax(successorState, depth + 1, nextAgent)
            else:  # Another ghost's turn
                value, _ = self.minimax(successorState, depth, nextAgent)
            if value < bestValue:
                bestValue = value
                bestAction = action
        return bestValue, bestAction

    def terminalTest(self, gameState, depth):
        """
        Check if the game has ended
        """
        return depth == self.depth or gameState.isWin() or gameState.isLose()

    def evaluationFunction(self, currentGameState):
        """
        Determine the score of the current game state
        """
        from util import manhattanDistance

        # Extract useful game state info
        newPos = currentGameState.getPacmanPosition()
        newGhostStates = currentGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newFood = currentGameState.getFood()

        # Start with the base score
        score = currentGameState.getScore()

        # Subtract for being near a ghost (unless it's scared)
        for i, ghostState in enumerate(newGhostStates):
            distance = manhattanDistance(newPos, ghostState.getPosition())
            if distance <= 1 and newScaredTimes[i] == 0:
                return float('-inf')
            if newScaredTimes[i] > 0:
                score += 200 - distance

        # Add for proximity to food
        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        if foodDistances:
            score += 1.0 / min(foodDistances)

        return score

######################################################

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Initialize variables
        initial_depth = 0
        initial_agent = 0  # Pacman is the first agent

        # extreme values for initialization
        negative_infinity = float('-inf')
        positive_infinity = float('inf')

        # Begin the recursive alpha-beta pruning.
        _, bestAction = self.alphaBetaPruning(gameState, initial_depth, initial_agent, negative_infinity, positive_infinity)

        return bestAction

    def alphaBetaPruning(self, gameState, depth, agent, alpha, beta):
        """
        Recursive function to compute the best score and action using alpha-beta pruning.
        """

        # Return the game state and no action.
        if not gameState.getLegalActions(agent) or depth == self.depth:
            return self.evaluationFunction(gameState), None

        # Set the function (max or min) and initial best score.
        func = max if agent == 0 else min
        bestScore = float('-inf') if agent == 0 else float('inf')
        bestAction = None

        # Iterate through all legal actions
        for action in gameState.getLegalActions(agent):
            # Generate the successor game state after taking an action
            successor = gameState.generateSuccessor(agent, action)

            # Find the next agent. If the current agent is the last, the next one is Pacman; otherwise increment.
            nextAgent = (agent + 1) % gameState.getNumAgents()
            # If the next agent is Pacman, increase the depth.
            newDepth = depth if nextAgent else depth + 1

            # Recursively call alphaBetaPruning
            score, _ = self.alphaBetaPruning(successor, newDepth, nextAgent, alpha, beta)

            # Update the best score and action
            if (agent == 0 and score > bestScore) or (agent != 0 and score < bestScore):
                bestScore = score
                bestAction = action

            # Adjust the alpha or beta values
            if agent == 0:  # maximizing agent
                alpha = max(alpha, bestScore)
                if bestScore > beta:
                    break
            else:  # minimizing agents
                beta = min(beta, bestScore)
                if bestScore < alpha:
                    break

        # Return the best score and the action
        return bestScore, bestAction


############################################################

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        # get the best action for the current game state
        _, bestAction = self.expectMax(gameState, 0, 0)

        # Return the best action
        return bestAction

    def expectMax(self, gameState, depth, agent):
        """
        Computes the best score using expectimax recursively.
        """

        # Get all legal actions available
        legalActions = gameState.getLegalActions(agent)

        # Base case
        if not legalActions or depth == self.depth:
            return self.evaluationFunction(gameState), None

        # Determine the next agent's index
        nextAgent = (agent + 1) % gameState.getNumAgents()

        # Increment the depth
        newDepth = depth + 1 if nextAgent == 0 else depth

        if agent == 0:  # maximizing agent
            bestScore = float('-inf')  # Initialize best score to negative infinity.
            bestAction = None  # The best action leading to the best score.

            # Iterate through all possible actions
            for action in legalActions:
                # Get the resulting game state
                successor = gameState.generateSuccessor(agent, action)

                # Recursively call expectMax
                score, _ = self.expectMax(successor, newDepth, nextAgent)

                # Update the best score and action if this score is better.
                if score > bestScore:
                    bestScore = score
                    bestAction = action

            # Return Pacman's best score and action
            return bestScore, bestAction

        else:  # expectation agent
            # Sum the scores of all possible actions the ghost can take.
            totalScore = sum(self.expectMax(gameState.generateSuccessor(agent, action), newDepth, nextAgent)[0] for action in legalActions)

            # Calculate the expected score
            expectedScore = totalScore / len(legalActions)

            return expectedScore, None


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    def getGhostScore(position, ghostStates):
        """
        Compute the score based on the distance to the nearest ghost.
        """
        # If there are no ghost states, return a score of 0.
        if not ghostStates:
            return 0

        # Compute the negative exponential of the distance to the closest ghost.
        distances = [manhattanDistance(position, ghost.configuration.pos) for ghost in ghostStates]
        closest_distance = min(distances)
        score = math.exp(-closest_distance)
        return score

    def getFoodScore(position, foodGrid):
        """
        Compute the score based on the distance to the nearest food.
        """
        # Convert the food grid to a list of food positions.
        foodList = foodGrid.asList()

        # If there's no food left, return a score of 0.
        if not foodList:
            return 0

        # Compute the score based on the inverse of the distance to the closest food.
        distances = [manhattanDistance(position, food) for food in foodList]
        closest_distance = min(distances)
        score = 1 / (closest_distance + 0.1)
        return score

    # Get the current position of Pacman.
    position = currentGameState.getPacmanPosition()

    # Compute the evaluation score as the sum of the ghost and food scores.
    evaluation = getGhostScore(position, currentGameState.getGhostStates()) + getFoodScore(position,currentGameState.getFood())

    # Final score is a combination
    return currentGameState.getScore() + evaluation - 10 * len(currentGameState.getCapsules())


# Abbreviation
better = betterEvaluationFunction
