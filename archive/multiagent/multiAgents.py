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

"""
THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING
A TUTOR OR CODE WRITTEN BY OTHER STUDENTS
- Candy Gao
"""

from util import manhattanDistance
from game import Directions
import math
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
        #calculate distance from ghost and foodlist
        distanceGhost = [manhattanDistance(newPos, ghost.configuration.pos) for ghost in newGhostStates]
        #distanceGhost = manhattanDistance(newPos, newGhostStates[0].configuration.pos)
        distanceFood = [manhattanDistance(newPos, food) for food in newFood.asList()]
        
        if len(distanceGhost) > 0:
            closestGhost = float(min(distanceGhost))
        else: closestGhost = 1

        #if food exists
        if len(distanceFood) > 0: 
            closestFood = min(distanceFood)
            score = (0.5*closestGhost) / closestFood
        else:
            score = closestGhost

        return successorGameState.getScore() + score

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
        #call default state where depth is 0 and starts with pacman
        return self.miniMax(gameState, 0, 0)[1]

    def miniMax(self, gameState, depth, node_index):

        #break when recursion ends and no legal actions exist
        if len(gameState.getLegalActions(node_index)) == 0 or depth == self.depth:
            return self.evaluationFunction(gameState), None

        #call max when pacman plays, and min when ghost moves
        if node_index == 0:
            return self.maxVal(gameState, depth, node_index)
        else:
            return self.minVal(gameState, depth, node_index)
    
    def maxVal(self, gameState, depth, node_index):

        #initialize smallest max value possible and find all plausible moves
        maxV = float('-inf')
        move = None
        moveOptions = gameState.getLegalActions(node_index)

        #for each plausible move, find its children
        for i in moveOptions:
            nextMove = gameState.generateSuccessor(node_index, i)
            
            #if pacman, update depth of next recursion
            if node_index == gameState.getNumAgents()-1:
               prev = self.miniMax(nextMove, depth+1, 0)
            else:
               prev = self.miniMax(nextMove, depth, node_index+1)

            #update maximum value
            if maxV < prev[0]:
                maxV = prev[0] 
                move = i
        #return the score and movement for the node
        return maxV, move
    

    def minVal(self, gameState, depth, node_index):

        #initialize biggest min value possible and find all plausible moves
        minV = float('inf')
        move = None
        moveOptions = gameState.getLegalActions(node_index)
        
        #for each plausible move, find its children
        for j in moveOptions:
            nextMove = gameState.generateSuccessor(node_index, j)

            if node_index == gameState.getNumAgents()-1:
                prev = self.miniMax(nextMove, depth+1, 0)
            else:
                prev = self.miniMax(nextMove, depth, node_index+1)

            #update min value
            if minV > prev[0]: 
                minV = prev[0]
                move = j
        #return the score and movement for the node
        return minV, move


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #initial alpha and beta value as extremes
        alpha = float('-inf')
        beta = float('inf')
        #call default state where depth is 0 and starts with pacman
        return self.abprune(gameState, 0, 0, alpha, beta)[1]

    def abprune(self, gameState, depth, node_index, alpha, beta):

        if len(gameState.getLegalActions(node_index)) == 0 or depth == self.depth:
            return self.evaluationFunction(gameState), None

        if node_index == 0:
            return self.maxVal(gameState, depth, node_index, alpha, beta)
        else:
            return self.minVal(gameState, depth, node_index, alpha, beta)
    
    def maxVal(self, gameState, depth, node_index, alpha, beta):

        maxV = float('-inf')
        move = None
        moveOptions = gameState.getLegalActions(node_index)

        #for each possible move, call abprune and prev node
        for i in moveOptions:
            nextMove = gameState.generateSuccessor(node_index, i)
            
            #check if pacman playing
            if node_index == gameState.getNumAgents()-1:
               prev = self.abprune(nextMove, depth+1, 0, alpha, beta)
            else:
               prev = self.abprune(nextMove, depth, node_index+1, alpha, beta)

            #update max value and corresponding move
            if maxV < prev[0]: 
                maxV = prev[0]
                move = i

            #update alpha on max
            alpha = max(alpha, maxV)
            #break loop and prune child if max is greater than beta
            if maxV > beta: return maxV, move


        return maxV, move
    

    def minVal(self, gameState, depth, node_index, alpha, beta):

        minV = float('inf')
        move = None
        moveOptions = gameState.getLegalActions(node_index)
        
        #for each possible move, call abprune and prev node
        for j in moveOptions:
            nextMove = gameState.generateSuccessor(node_index, j)

            if node_index == gameState.getNumAgents()-1:
                prev = self.abprune(nextMove, depth+1, 0, alpha, beta)
            else:
                prev = self.abprune(nextMove, depth, node_index+1, alpha, beta)

            #update min value and corresponding move
            if minV > prev[0]: 
                minV = prev[0]
                move = j

            #update beta on min
            beta = min(beta, minV)
            #break loop and prune child if min is smaller than alpha
            if minV < alpha: return minV, move

        return minV, move

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
        #deault initial depth is 0 and pacman is playing
        return self.expectiMax(gameState, 0, 0)[1]

    def expectiMax(self, gameState, depth, node_index):

        if len(gameState.getLegalActions(node_index)) == 0 or depth == self.depth:
            return self.evaluationFunction(gameState), None

        if node_index == 0:
            return self.maxVal(gameState, depth, node_index)
        else:
            return self.minVal(gameState, depth, node_index)
    
    def maxVal(self, gameState, depth, node_index):

        #maxV as the smallest val by default
        maxV = float('-inf')
        move = None
        moveOptions = gameState.getLegalActions(node_index)

        #iterate through legal moves
        for i in moveOptions:
            nextMove = gameState.generateSuccessor(node_index, i)
            
            #check if pacman is playing
            if node_index == gameState.getNumAgents()-1:
               prev = self.expectiMax(nextMove, depth+1, 0)
            else:
               prev = self.expectiMax(nextMove, depth, node_index+1)

            # update max and move
            if maxV < prev[0]:
                maxV = prev[0] 
                move = i

        return maxV, move
    

    def minVal(self, gameState, depth, node_index):
        #min is based on combined universal distribution of previous nodes
        move = None
        expected = 0 #defult 0
        moveOptions = gameState.getLegalActions(node_index)
        
        #iterate through legal moves
        for j in moveOptions:
            nextMove = gameState.generateSuccessor(node_index, j)
            #check if pacman is playing
            if node_index == gameState.getNumAgents()-1:
                prev = self.expectiMax(nextMove, depth+1, 0)
            else:
                prev = self.expectiMax(nextMove, depth, node_index+1)
            
            #update the expected value by summing all possible moves
            expected += prev[0] * (1.0/len(moveOptions))

        return expected, move

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    measure score of each move by distance and count of food, ghost, and capsules
    prioritizes surviving by putting more weight on avoiding ghosts
    """
    "*** YOUR CODE HERE ***"

    Position = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    

    distanceGhost = [manhattanDistance(Position, ghost.configuration.pos) for ghost in newGhostStates]
    distanceFood = [manhattanDistance(Position, food) for food in newFood.asList()]
    closestFood = 1 #if no food is found, division is by 1
    closestGhost = 1 #if no ghost is found 

    #returns score for ghost, use log scale to minimize score on close ghost
    if len(distanceGhost) > 0:
        closestGhost = float(min(distanceGhost))
        ghostscore = math.log(closestGhost+0.1)

    #returns score for food, rewards close food
    if len(distanceFood) > 0: 
        closestFood = float(min(distanceFood))

    #count the amount of capsules
    capsules = len(currentGameState.getCapsules())
    
    # if no ghost found, go for food
    if len(distanceGhost) == 0:
        score = closestFood
    #else, low score for far food and close ghost
    else:
        score = (0.5*ghostscore)/closestFood




    return 0.5 * currentGameState.getScore() + score - (10*capsules)

# Abbreviation
better = betterEvaluationFunction
