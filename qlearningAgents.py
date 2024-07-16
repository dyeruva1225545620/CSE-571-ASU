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


import json
from game import *
from learningAgents import ReinforcementAgent
import copy
import pandas as pd
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
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.q_values = util.Counter()

        "*** YOUR CODE HERE ***"

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.q_values[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        #Terminal State
        if (len(self.getLegalActions(state)) == 0):
            return 0.0

        max_value = None
        for action in self.getLegalActions(state):
            if max_value is None or self.getQValue(state, action) > max_value:
                max_value = self.getQValue(state, action)
        assert max_value is not None
        return max_value

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        #Terminal State
        if (len(self.getLegalActions(state)) == 0):
            return None

        max_value = None
        max_actions = []
        for action in self.getLegalActions(state):
            if max_value is None or self.getQValue(state, action) > max_value:
                max_value = self.getQValue(state, action)
                max_actions = []
                max_actions.append(action)
        assert max_value is not None
        return random.choice(max_actions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if (len(legalActions) == 0):
            return None

        if (util.flipCoin(self.epsilon)):
            return random.choice(legalActions)

        return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        next_state_value = self.getValue(nextState)
        self.q_values[(state, action)] = ((1-self.alpha)*self.q_values[(state, action)]) \
            + self.alpha*(reward + (self.discount*next_state_value))

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
        self.forced_action = None
        QLearningAgent.__init__(self, **args)

    def getAction(self, state, use_forced=True):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        if (use_forced):
            if (self.forced_action is None):
                action = QLearningAgent.getAction(self,state)
                #print(f"assuming called by learningAgents and forced action is None, {action}")
                self.doAction(state, action)
                return action
            else:
                self.doAction(state, self.forced_action)
                #print(f"assuming called by learningAgents and forced action is not None, {self.forced_action}")
                return self.forced_action
        else:
            action = QLearningAgent.getAction(self,state)
            #print(f"assuming called by update method, {action}")
            return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        extractor = "SimpleExtractor"
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()
        self.feature_names = ["bias", "#-of-ghosts-1-step-away", "eats-food", "closest-food", "closest-ghost", "closest-capsule"]

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state, action)
        q_value = 0
        for feature_id in features:
            feature_value = features[feature_id]
            q_value += self.weights[feature_id]*feature_value
        return q_value

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        gamma = self.discount
        discounted_reward = gamma*self.getValue(nextState)
        update_value = -1*(self.getQValue(state, action) - (reward + discounted_reward))
        features = self.featExtractor.getFeatures(state, action)

        for feature_id in features:
            feature_value = features[feature_id]
            self.weights[feature_id] = self.weights[feature_id] + (self.alpha * update_value * feature_value)

        if (self.alpha != 0):
            weights_to_print = {feature_name: self.weights[feature_name] for feature_name in self.feature_names}
            print("Print Weights:", json.dumps(weights_to_print))

        q_value = self.getQValue(state, action)
        print("Q_value:", q_value)

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        print("final")
        PacmanQAgent.final(self, state)
        print("Game Score:", state.getScore())

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass


class ExtendedCounter(Counter):
    def __mul__(self, scalar_value):
        new_counter = ExtendedCounter()
        for key in self:
            new_counter[key] = float(self[key]*scalar_value)
        return new_counter

    @staticmethod
    def convert(counter):
        new_counter = ExtendedCounter()
        for key in counter:
            new_counter[key] = counter[key]
        return new_counter

class TrueOnlineSARSA(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        extractor = "SimpleExtractor"
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = ExtendedCounter()
        self.lambd = 0.5
        self.q_old = 0
        self.z = ExtendedCounter()
        self.feature_names = ["bias", "#-of-ghosts-1-step-away", "eats-food", "closest-food", "closest-ghost", "closest-capsule"]

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state, action)
        q_value = 0
        for feature_id in features:
            feature_value = features[feature_id]
            q_value += self.weights[feature_id]*feature_value
        return q_value

    def dot_prod(self, a, b):
        ans = 0.00
        for key in a:
            ans += a[key]*b[key]
        return ans

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        gamma = self.discount
        alpha = self.alpha
        lambd = self.lambd
        q_old = self.q_old
        #print(state.isWin(), state.isLose(), nextState.isWin(), nextState.isLose())

        x = ExtendedCounter.convert(self.featExtractor.getFeatures(state, action))
        #print("update start")
        #print("features", x)
        #print(f"action is {action}")
        a_prime = self.getAction(nextState, use_forced=False)
        #print(f"action_prime is {a_prime}")
        if (a_prime is None):
            assert nextState.isWin() or nextState.isLose()
        if (a_prime is None):
            x_prime = ExtendedCounter()
        else:
            x_prime = ExtendedCounter.convert(self.featExtractor.getFeatures(nextState, a_prime))
        q = self.dot_prod(self.weights, x)
        q_prime = self.dot_prod(self.weights, x_prime)
        tmp = reward + gamma*q_prime - q
        self.z = self.z*(gamma*lambd) + x*(1 - (alpha*gamma*lambd)*(self.dot_prod(self.z, x)))
        self.z = ExtendedCounter.convert(self.z)
        self.weights = self.weights + self.z*(alpha*(tmp + q - q_old)) - x*(alpha*(q - q_old))
        self.weights = ExtendedCounter.convert(self.weights)
        self.q_old = q_prime
        self.forced_action = a_prime
        if (a_prime is None):
            self.q_old = 0
            self.z = ExtendedCounter()

        if (self.alpha != 0):
            weights_to_print = {feature_name: self.weights[feature_name] for feature_name in self.feature_names}
            print("Print Weights:", json.dumps(weights_to_print))

        q_value = self.getQValue(state, action)
        print("Q_value:", q_value)
        print("alpha", alpha)

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        print("Game Score:", state.getScore())
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
