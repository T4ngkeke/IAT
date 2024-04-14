from gridworld import *
from dynamic_programming import *
from q_learning import *


mdp = GridWorld ()
agent=q_agent(mdp)
agent.solve()

print (" states :",mdp.get_states () )
print (" terminal states :",mdp.get_goal_states() )
print (" actions :",mdp.get_actions () )
print(agent.q_value)
#print(agent.v)
#mdp.visualise_value_function(value_function(agent,mdp.get_states()))
#mdp.visualise_policy(policy(agent,mdp.get_states()))
mdp.visualise_q_function(q_function(agent,mdp.get_states(),mdp.get_actions()))
"""
def policy_custom (state) :
	return mdp.UP

while (1) :
	state = mdp.get_initial_state()
	#new_state,_ = mdp.execute(state,policy_custom(state))
	#mdp.initial_state = new_state
	mdp.visualise ()
"""
