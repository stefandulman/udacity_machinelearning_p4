import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import defaultdict
import numpy as np

# global variable for collecting some statistics on the state occurrence
stats = defaultdict(int)
# flag for stats collection
collectstats = False

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        
        # dictionaty for converting state to index
        self.ix = {}
        cnt = 0
        for v1 in ['forward', 'left', 'right']:
          for v2 in ['yes', 'no']:
            for v3 in ['plenty', 'okish', 'gogogo!']:
              self.ix[v1 + '_' + v2 + '_' + v3] = cnt
              cnt = cnt + 1
              
        # dictionary for converting action to index
        self.iy = {
          None:       0,
          'forward':  1,
          'left':     2,
          'right':    3
        }
        
        # and the reverse
        self.valy = {
          0: None,
          1: 'forward',
          2: 'left',
          3: 'right'
        }
        
        # default values for the qlearning constants
        self.default_alpha = 0.5
        self.default_gamma = 0.8
        self.default_epsilon = 0.1
        
        # variable holds the rewards for the runs
        self.totalrewards = list()
        self.lasttotalreward = 0
        
        self._myinit()

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self._myinit()
        
        # store the rewards from last run, if any
        if self.lasttotalreward is not None:
          self.totalrewards.append(self.lasttotalreward)
          self.lasttotalreward = 0

    def overwritect(self, a, g, e):
        '''function overwrites the constants and resets agent'''
        self.default_alpha = a
        self.default_gamma = g
        self.default_epsilon = e
        # reset everything
        self._myinit()


    def _myinit(self):
        '''single place initialization for all local stuff '''

        # Q learning constants
        self.alpha = self.default_alpha
        self.gamma = self.default_gamma
        self.epsilon = self.default_epsilon

        # variable holds initial value for deadline
        self.deadline_max = None       
        
        # variable for storing the Q(s,a) vals for states (rows) and actions (column)
        self.qsa = np.zeros((18, 4))
        
        # variables for storing the info from last step
        self.prevstate = None
        self.prevaction = None
        self.prevreward = None
        

    def update(self, t):

        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        
        # update the deadline_max variable
        if self.deadline_max is None:
          self.deadline_max = self.env.get_deadline(self)

        # TODO: Update state
        (state_traffic, state_deadline) = self.getstate(deadline, inputs)
        # collect some stats
        if collectstats:
          key = state_traffic + '_' + state_deadline
          stats[key] = stats[key] + 1
        
        # TODO: Select action according to your policy
        ### dumb cab - random walk action
        #action = random.choice(self.env.valid_actions)
        #
        ### perfect driver, careless of time
        #action = None
        #if state_traffic == 'forward_yes':
        #  action = 'forward'
        #elif state_traffic == 'left_yes':
        #  action = 'left'
        #elif state_traffic == 'right_yes':
        #  action = 'right'
        #
        ### selection based on learning
        action = self.getaction(state_traffic + '_' + state_deadline)

        # Execute action and get reward
        reward = self.env.act(self, action)
        
        self.lasttotalreward = self.lasttotalreward + reward
        if deadline <= 0:
          self.lasttotalreward = -1000

        # TODO: Learn policy based on state, action, reward
        if self.prevstate != None:
          self.learnpolicy(state_traffic + '_' + state_deadline)
        # update variables
        self.prevstate = state_traffic + '_' + state_deadline
        self.prevaction = action
        self.prevreward = reward
        
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        
      
    
    def learnpolicy(self, newstate):
        ''' update Q(s,a) based onto the values we have so far '''
        
        s = self.ix[newstate]
        termright = self.prevreward + self.gamma * np.amax(self.qsa[s, :])
        
        s = self.ix[self.prevstate]
        a = self.iy[self.prevaction]
        self.qsa[s, a] = (1 - self.alpha) * self.qsa[s, a] + self.alpha * termright
        
    
    
    def getaction(self, state):
        ''' function returns an action based onto what was learnt so far '''
        
        # maximum q - shuffle array otherwise max always outputs first version of max
        s = self.ix[state]
        posmax = np.argwhere(self.qsa[s, :] == np.amax(self.qsa[s, :]))
        choicemax = random.choice(posmax.flatten().tolist())
                
        # random choice
        choicerand = random.randint(0, 3)
        
        if random.random() < self.epsilon:
          return self.valy[choicerand]

        return self.valy[choicemax]
        
      
      
    def getstate(self, deadline, inputs):
        ''' returns the state as two strings - one reflecting the traffic and one reflecting the remaining time '''
        
        # arbitrary thresholds for being late
        thr1 = 3 * self.deadline_max / 5;
        thr2 = 1 * self.deadline_max / 5;
                
        # determine deadline-based state
        if deadline > thr1:
          res_time = 'plenty'
        elif deadline > thr2:
          res_time = 'okish'
        else:
          res_time = 'gogogo!'
        
        # determine the traffic-based state
        # need to move forward
        if self.next_waypoint == 'forward':
          if inputs['light'] == 'green':
            return ('forward_yes', res_time)
          return ('forward_no', res_time)
        
        # need to move left
        if self.next_waypoint == 'left':
          if inputs['light'] == 'red':
            return ('left_no', res_time)          
          if inputs['oncoming'] == 'left' or inputs['oncoming'] == None:
            return ('left_yes', res_time)
          return ('left_no', res_time)
          
        # need to move right
        if self.next_waypoint == 'right':
          if inputs['light'] == 'green':
            return ('right_yes', res_time)
          if inputs['left'] == 'forward':
            return ('right_no', res_time)
          return ('right_yes', res_time)

        # something went wrong
        print 'wrong waypoint specified:', self.next_waypoint
        error(1)

        

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    print "collected statistics", stats
    print "rewards:", a.totalrewards
    print "last ten:", a.totalrewards[-10:]
    print "Q(s,a) matrix", a.qsa



def scanparameters():
  
    bestscore = -1000000
    bestsettings = ""
  
    for alpha in [0.6, 0.7, 0.8, 0.9]:
      for gamma in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        for epsilon in [0.05, 0.1, 0.15, 0.2]:
    
          # create things
          e = Environment()
          a = e.create_agent(LearningAgent)
          e.set_primary_agent(a, enforce_deadline=True)

          # set up the new agent with our constants
          a.overwritect(alpha, gamma, epsilon)

          # Now simulate it
          sim = Simulator(e, update_delay=0.0, display=False) 
          sim.run(n_trials=1000)  # run for a specified number of trials
  
          # compute the score based on last ten runs
          score = sum(a.totalrewards[-10:])
          if score > bestscore:
            bestscore = score
            bestsettings = [alpha, gamma, epsilon, a.totalrewards[-10:]]
            
    print "best run: ", bestscore
    print bestsettings
    
    
if __name__ == '__main__':
    #run()
    scanparameters()
