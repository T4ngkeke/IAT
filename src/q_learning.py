import random 
from matplotlib import pyplot as plt
from mdp import *
from collections import defaultdict
from gridworld import *

class q_agent:

    def __init__(self, mdp):# and here...
        self.mdp = mdp

        #self.actions=self.mdp.get_actions()
        
        self.alpha=0.01
        self.q_value =defaultdict(lambda:0.0)


        #self.q_value = [[0.0 for i in range(len(self.actions))] for j in range(len(self.states))]

    def greedy(self, s):
        actions=self.mdp.get_actions(s)
        if random.uniform(0, 1) < 0.9:#epsilon
        # Explore action space
            return random.choice(actions)
        else:
        # Exploit learned values
            max_val=max(self.q_value[(s, a)] for a in actions)
            best_actions = [a for a in actions if self.q_value[(s, a)] == max_val]
            return random.choice(best_actions)
    
    def solve(self):
        #q((0,0),up)value
        q_list=[]
        for episode in range(10000):
            state=self.mdp.get_initial_state()
            while not self.mdp.is_terminal(state):
                action=self.greedy(state)
                next_state,reward=self.mdp.execute(state,action)
                delta=self.get_delta(reward,self.q_value,state,next_state,action)
                self.q_value[(state,action)]+=self.alpha*delta
                state=next_state
            q_list.append(self.q_value[((0,0),UP)])
        plt.plot(q_list)
        plt.show()

    def get_delta(self, reward, q_value, state, next_state,action):
        #Calculate the delta for the update
        next_actions=self.mdp.get_actions(next_state)
        #print(next_actions)
        max_q=max(q_value[(next_state,a)] for a in next_actions)
        #print(max_q)
        #print(self.mdp.get_discount_factor())
        delta=reward+self.mdp.get_discount_factor()*max_q-q_value[(state,action)]
        return delta


    def state_value(self, state):
        #Get the value of a state
        return max(self.q_value[state])

class q_function:
    def __init__(self, agent, states, actions):
        self.agent = agent
        self.states = states
        self.actions = actions

    def get_q_value(self, s, a):
        return self.agent.q_value[(s, a)]
