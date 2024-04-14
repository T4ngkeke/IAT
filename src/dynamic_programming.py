from mdp import *
from collections import defaultdict
from gridworld import *


class dp_agent():
    
    def __init__(self,mdp): #and here...
        self.mdp=mdp
        self.states=self.mdp.get_states()
        self.v=defaultdict(float)
        self.v_bis=defaultdict(float)
        ''' add initialization here! '''
        for s in self.states:
            self.v[s]=0.0
            self.v_bis[s]=0.0

    def get_value(self,s,v):
        #return the value of a specific state s according to value function v
        if s in v:
            return v[s]
        else:
            return 0.0
        
    def get_width(self,v,v_bis):
        #return the absolute norm between two value functions
        return max([abs(v[s]-v_bis[s]) for s in self.states])

    def solve(self):
        #main solving loop
        #initialize value function
        while 1:
            #update value function
            #update width
            self.v_bis=self.v.copy()
            for s in self.states:
                #update value of s
                self.update(s)

            if self.get_width(self.v,self.v_bis)<0.01:
                break

            
    def update(self,s):
        #updates the value of a specific state s
        list_max=[]
        for action in self.mdp.get_actions(s):
            value=0.0
            for s_prime,prob in self.mdp.get_transitions(s,action):
                value+=prob*self.mdp.get_reward(s,action,s_prime)+self.mdp.get_discount_factor()*prob*self.v_bis[s_prime]
            list_max.append(value)
        self.v[s]=max(list_max)  

class value_function():
    def __init__(self,agent,states):
        self.agent=agent
        self.states=states
    def get_value(self,s):
        return self.agent.get_value(s,self.agent.v)

class policy():
    def __init__(self,agent,states):
        self.agent=agent
        self.states=states
    def select_action(self,s):
        max_val=-1000000
        action_taken=None
        next_state_list=[]
        #print(self.agent.mdp.get_actions(s))
        for action in self.agent.mdp.get_actions(s):
            total_val=0.0
            for s_prime,prob in self.agent.mdp.get_transitions(s,action):
                reward=self.agent.mdp.get_reward(s,action,s_prime)
                total_val+=prob*(reward+self.agent.mdp.get_discount_factor()*self.agent.get_value(s_prime,self.agent.v))
            if total_val>max_val:
                max_val=total_val
                action_taken=action
        
        return action_taken            
        
        
