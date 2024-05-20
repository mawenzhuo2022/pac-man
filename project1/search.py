# search.py
# ---------
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
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
import util
from util import Stack, Queue, PriorityQueue


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    #initial the stack to be the type of Stack
    stack = Stack()
    #push the initial state into stack
    initState = problem.getStartState()
    stack.push((initState, [], []))

    #run while loop when stack is not empty
    while not stack.isEmpty():
        #pop out the first item in stack
        state, path, vis = stack.pop()
        #return when meet the goal
        if problem.isGoalState(state):
            return path
        #run and go furthest or find "successor", do dfs
        for successor, action, stepcost in problem.getSuccessors(state):
            if successor not in vis:
                stack.push((successor, path + [action], vis + [state]))
    #return null when no solution
    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # initial the queue to be the type of Queue
    queue = Queue()
    # push the initial state into queue
    initState = problem.getStartState()
    queue.push((initState, [], []))
    # Set for visited states
    vis = set()

    # run while loop when queue is not empty
    while not queue.isEmpty():
        state, path, _ = queue.pop()
        #if reach the goal
        if problem.isGoalState(state):
            return path
        #add state when it is not visited
        if state not in vis:
            vis.add(state)
            # run and go widest or find "successor", do bfs
            for successor, action, stepcost in problem.getSuccessors(state):
                if successor not in vis:
                    queue.push((successor, path + [action], vis))

    # return null when no solution
    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    # Initialize the queue
    queue = PriorityQueue()
    initState = problem.getStartState()

    # Push the initial state into queue
    # (sequence, state, path, cost)
    queue.push((0, initState, [], 0), 0)
    # set visited state with cost
    vis = {}

    while not queue.isEmpty():
        sequenceNumber, state, path, cost = queue.pop()

        if problem.isGoalState(state):
            return path
        # If the state hasn't been visited yet, or found a cheaper path to this state,
        # process this state and update the cost.
        if state not in vis or cost < vis[state]:
            vis[state] = cost
            # For each neighboring state
            for successor, action, stepcost in problem.getSuccessors(state):
                newCost = cost + stepcost
                # If the successor hasn't been visited or found a cheaper path to the successor
                if successor not in vis or newCost < vis[successor]:
                    # Add the successor to the priority queue with new cost
                    # Create the new path
                    newPath = path + [action]
                    # Create the tuple to be pushed to the queue
                    itemToPush = (sequenceNumber, successor, newPath, newCost)
                    # Push the item onto the queue with the priority
                    queue.push(itemToPush, newCost)
                    sequenceNumber += 1  # Increment the sequence number
    return []  # Return empty list if no solution found
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # Initialize the priority queue with the starting state and its cost.
    queue = PriorityQueue()
    start = problem.getStartState()
    initialCost = heuristic(start, problem)
    # Push the initial state into queue
    queue.push((start, [], 0), initialCost)
    # Track the visited states to avoid revisiting them.
    vis = set()
    while not queue.isEmpty():
        state, path, cost = queue.pop()
        # If the current state is the goal, return the path
        if problem.isGoalState(state):
            return path
        # Process states that haven't been visited yet.
        if state not in vis:
            vis.add(state)
            for successor, action, stepcost in problem.getSuccessors(state):
                # Calculate the cost up to the current state and the estimated cost to the goal
                newCost = cost + stepcost
                heuristicCost = heuristic(successor, problem)
                totalCost = newCost + heuristicCost
                # Add successor states to the priority queue with the combined cost
                newPath = path + [action]
                queue.push((successor, newPath, newCost), totalCost)
    # Return an empty list if no solution is found.
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
