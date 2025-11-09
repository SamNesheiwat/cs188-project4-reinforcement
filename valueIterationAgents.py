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
        for _ in range(self.iterations):
            new_vals = util.Counter()

            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    new_vals[state] = 0
                    continue
                best_val = float("-inf")
                for act in self.mdp.getPossibleActions(state):
                    q_est = self.computeQValueFromValues(state, act)
                    if q_est > best_val:
                        best_val = q_est
                new_vals[state] = best_val if best_val != float("-inf") else 0
            self.values = new_vals



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
        total = 0.0
        for nxt, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, nxt)
            total += prob * (reward + self.discount * self.values[nxt])
        return total

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
            return None

        possible = self.mdp.getPossibleActions(state)
        best_act, best_val = None, float("-inf")

        for act in possible:
            q_val = self.computeQValueFromValues(state, act)
            if q_val > best_val:
                best_val, best_act = q_val, act

        return best_act

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
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
        preds = {}
        for s in self.mdp.getStates():
            preds[s] = set()
        for s in self.mdp.getStates():
            for act in self.mdp.getPossibleActions(s):
                for nxt, prob in self.mdp.getTransitionStatesAndProbs(s, act):
                    if prob > 0:
                        preds[nxt].add(s)

       
        pq = util.PriorityQueue()
        for s in self.mdp.getStates():
            if self.mdp.isTerminal(s):
                continue
            best_q = max([self.computeQValueFromValues(s, a)
                          for a in self.mdp.getPossibleActions(s)])
            diff = abs(self.values[s] - best_q)
            pq.update(s, -diff)

        for _ in range(self.iterations):
            if pq.isEmpty():
                break
            s = pq.pop()


            if not self.mdp.isTerminal(s):
                best_q = max([self.computeQValueFromValues(s, a)
                              for a in self.mdp.getPossibleActions(s)])
                self.values[s] = best_q

    
            for p in preds[s]:
                if self.mdp.isTerminal(p):
                    continue
                best_q = max([self.computeQValueFromValues(p, a)
                              for a in self.mdp.getPossibleActions(p)])
                diff = abs(self.values[p] - best_q)

                if diff > self.theta:
                    pq.update(p, -diff)

