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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()  #获得合法动作
        
        
        #对合法动作评估，决定接下来走哪个方向
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)#选有最大值的后继
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        #生成best indices list,then randomly choose one of the best
        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        successorGameState = currentGameState.generatePacmanSuccessor(action)  #产生动作对应的状态 
        newPos = successorGameState.getPacmanPosition() #当前吃豆人位置
        newFood = successorGameState.getFood() #获得当前状态食物分布
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #Q1：reflex agent
        #aim: eat all fooos, aviod ghost,get high score
        
        min_Dist_food = -1  
        #Food = currentGameState.getFood()
        foodList= newFood.asList()
        
        for food in foodList:
            dist_food = util.manhattanDistance(newPos, food)
            if min_Dist_food == -1 or min_Dist_food - dist_food>=2:
                min_Dist_food = dist_food
          
        #distance to ghost
        ghostPos = successorGameState.getGhostPositions()#获取幽灵位置
        dist_ghost = 1
        
        
        for ghostState in ghostPos:
            dist = manhattanDistance(newPos,ghostState)
           
                       
            if dist <= 4 and dist !=0:
                dist_ghost = dist
 
        
        basicScore = successorGameState.getScore()
        foodScore = 10/min_Dist_food
        ghostScore = -12/dist_ghost
        
        totalScore = basicScore + foodScore + ghostScore 
        return totalScore
        

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        #self.depth 用来结束 min-value() 和 max-value() 互相递归的调用
        #self.evaluationFunction() 用来 评估得分
        # agentIndex == 0 表示吃豆人，>= 1 表示幽灵
       
        def minimax(agentIndex, gameState, depth):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            
            if agentIndex == 0: 
                actions = gameState.getLegalActions(agentIndex)
                #suc_State = gameState.generateSuccessor(agentIndex, action)
                
                return max(minimax(1, gameState.generateSuccessor(agentIndex, action), depth)for action in actions)
            
            elif agentIndex >= 1:
                actions = gameState.getLegalActions(agentIndex)
                number_agent = gameState.getNumAgents()
                sucAgent = agentIndex + 1
                if sucAgent == number_agent:
                    sucAgent = 0
                if sucAgent == 0:
                   depth += 1
                return min(minimax(sucAgent, gameState.generateSuccessor(agentIndex, action), depth) for action in actions)

        Max = float("-inf")
        f_action = Directions.WEST
        actions = gameState.getLegalActions(0)
        for action in actions:
            value = minimax(1, gameState.generateSuccessor(0, action), 0)
            if value > Max or Max == float("-inf"):
                Max = value
                f_action = action

        return f_action

 

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):

        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
   
        def ab_prune(agentIndex, gameState, depth, alpha, beta): 
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
              
            if agentIndex == 0:
                return max_value(agentIndex, gameState, depth, alpha, beta)
            elif agentIndex >= 1:
            
                return min_value(agentIndex, gameState, depth, alpha, beta)
               
            
        def max_value(agentIndex, gameState, depth, alpha, beta):
            value = float("-inf")
            states = gameState.getLegalActions(agentIndex)
            
            if not states:
                return self.evaluationFunction(gameState)
               
            for state in states:
                #suc_state = gameState.generateSuccessor(agentIndex, state)
                value = max(value, ab_prune(1, gameState.generateSuccessor(agentIndex, state), depth, alpha, beta))
                if value > beta:
                    return value
                alpha = max(alpha, value)
            return value
    
    
        def min_value(agentIndex, gameState, depth, alpha, beta):
            value = float("inf")
            states = gameState.getLegalActions(agentIndex)
            
            if not states:
                return self.evaluationFunction(gameState)
            
            number_agent = gameState.getNumAgents()
            beta_agent = agentIndex +1
            if beta_agent == number_agent:
                beta_agent = 0
            if beta_agent == 0:
                depth += 1 
                
            for state in states:
                
                #suc_state = gameState.generateSuccessor(agentIndex, state)
                value = min(value, ab_prune(beta_agent, gameState.generateSuccessor(agentIndex, state), depth, alpha, beta))
            
                if value < alpha:
                    return value
                beta = min(beta, value)
            return value 
    
        util = float("-inf")
        f_action = Directions.WEST
        states = gameState.getLegalActions(0)
        alpha = float('-inf')
        beta = float('inf')
        
        for f_state in states:
            ghost_value = ab_prune(1, gameState.generateSuccessor(0, f_state), 0, alpha, beta)
            
            if ghost_value > util or  util == float("-inf"):
                util = ghost_value
                f_action = f_state

            if util > beta:
                return util
            alpha = max(alpha, util)
                
        return f_action    
   
        
   
        
        #util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        
   
        
        def expectimax(agentIndex,gameState, depth):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
                        

            if agentIndex == 0:
                actions = gameState.getLegalActions(agentIndex)
                #suc_State = gameState.generateSuccessor(agentIndex, action)
                
                return max(expectimax(1, gameState.generateSuccessor(agentIndex, action), depth)for action in actions)
            
            elif agentIndex >= 1:
            #else:
                actions = gameState.getLegalActions(agentIndex)
                number_agent = gameState.getNumAgents()
                sucAgent = agentIndex + 1
                if sucAgent == number_agent:
                    sucAgent = 0
                if sucAgent == 0:
                    depth += 1
                return sum(expectimax(sucAgent, gameState.generateSuccessor(agentIndex, action), depth) for action in actions)

                
        Max = float("-inf")
        f_action = Directions.WEST
        actions = gameState.getLegalActions(0)
        for action in actions:
            value = expectimax(1, gameState.generateSuccessor(0, action), 0)
          
            if value > Max:
                Max = value
                f_action = action

        return f_action
        
        
        
        
       
        
        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    
    Position = currentGameState.getPacmanPosition()
    
    dist_Food = float('inf')
    Food = currentGameState.getFood()
    foodList= Food.asList()
    for food in foodList:
        dist_Food=  min(dist_Food, util.manhattanDistance(Position, food))
 
    dist_ghosts = 1
    ghost_States = currentGameState.getGhostPositions()
    for ghost_state in ghost_States:
        ghost_distance = util.manhattanDistance(Position, ghost_state)
        dist_ghosts += ghost_distance
    

    newCapsule = currentGameState.getCapsules()  
    number_capsules = len(newCapsule) 
    
    basicScore = currentGameState.getScore()
    foodScore = 1 / float(dist_Food)
    ghostScore = 1 / float(dist_ghosts)
    
    return basicScore + foodScore - ghostScore - number_capsules

    
   # util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
