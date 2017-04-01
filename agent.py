import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import defaultdict
import numpy as np
import sys

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
          for v2 in ['free', 'busy']:
            self.ix[v1 + '_' + v2] = cnt
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
        self.default_alpha = 0.6
        self.default_gamma = 0.2
        self.default_epsilon = 0.05
        
        # hack to identify first run
        self.firstrun = True
        # statistic variable holding the rewards for the runs
        self.totalrewards = list()
        self.lasttotalreward = 0
        # statistic variable holding the deadline variable at finish
        self.totalfinish = list()
        self.lasttotalfinish = 0
        # statistic variable holding the deadline variable at start
        self.totalstart = list()
        
        # statistics on the state occurrence
        self.stats = defaultdict(int)
        
        # variable for storing the Q(s,a) vals for states (rows) and actions (column)
        self.qsa = np.zeros((6, 4))
        
        self._myinit()


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self._myinit()
        
        # store the rewards from last run, if any
        if self.firstrun == False:
          self.totalrewards.append(self.lasttotalreward)
          self.lasttotalreward = 0
          self.totalfinish.append(self.lasttotalfinish)
          self.lasttotalfinish = 1
        else:
          self.firstrun = False


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
          # record statistic
          self.totalstart.append(self.deadline_max)

        # TODO: Update state
        state_traffic = self.getstate(deadline, inputs)
        
        # TODO: Select action according to your policy
        ### dumb cab - random walk action
        #action = random.choice(self.env.valid_actions)
        #
        ### perfect kamikaze driver, careless of time
        #action = None
        #if state_traffic == 'forward_free':
        #  action = 'forward'
        #elif state_traffic == 'left_free':
        #  action = 'left'
        #elif state_traffic == 'right_free':
        #  action = 'right'
        #
        ### selection based on learning
        action = self.getaction(state_traffic)

        # update the state
        self.state = state_traffic

        # Execute action and get reward
        reward = self.env.act(self, action)
        
        # collect statistics
        self.lasttotalreward = self.lasttotalreward + reward
        self.lasttotalfinish = deadline
        self.stats[state_traffic] = self.stats[state_traffic] + 1
        
        # TODO: Learn policy based on state, action, reward
        if self.prevstate != None:
          self.learnpolicy(state_traffic)
        # update variables
        self.prevstate = state_traffic
        self.prevaction = action
        self.prevreward = reward
        
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        
      
    
    def learnpolicy(self, newstate):
        ''' update Q(s,a) based onto the values we have so far '''
        
        s = self.ix[newstate]
        termright = self.prevreward + self.gamma * np.amax(self.qsa[s, :])
        
        # print 'new state:', s, 'old q[s,a]:', self.qsa[s, :]
        
        s = self.ix[self.prevstate]
        a = self.iy[self.prevaction]
        self.qsa[s, a] = (1 - self.alpha) * self.qsa[s, a] + self.alpha * termright
        
        # print '  prev state:', s, 'prev action:', a, 'new qsa[s,a]:', self.qsa[s, a]
    
    
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
        
        # determine the traffic-based state
        # need to move forward
        if self.next_waypoint == 'forward':
          if inputs['light'] == 'red':
            return 'forward_busy'
          return 'forward_free'
        
        # need to move left
        if self.next_waypoint == 'left':
          if inputs['light'] == 'red':
            return 'left_busy'
          if inputs['oncoming'] == 'right' or inputs['oncoming'] == 'forward':
            return 'left_busy'
          return 'left_free'
          
        # need to move right
        if self.next_waypoint == 'right':
          if inputs['light'] == 'green':
            return 'right_free'
          if inputs['left'] == 'forward':
            return 'right_busy'
          return 'right_free'

        # something went wrong
        print 'wrong waypoint specified:', self.next_waypoint
        error(1)


def print_stats(ag):
  ''' function pretty prints the collected statistics of the agent '''
  
  total_runs = len(ag.totalfinish)
  timeok_runs  = len( [x for x in ag.totalfinish if x > 0 ] )  
  print ''
  print "Number of total runs:", total_runs
  print "Successful runs: " + str(100.0 * timeok_runs / total_runs) + "%"
  print ''
  
  # -----
  
  print "First rewards:", ag.totalrewards[:5]
  print "Last rewards:", ag.totalrewards[-5:]
  print ''
  
  # -----
  print ''
  print 'Q matrix'
  print '                 None    Forw.   Left    Right'
  labels = [
    'forward_free', 
    'forward_busy', 
    'left_free', 
    'left_busy', 
    'right_free', 
    'right_busy'
  ]
  for i in range(0,6):
    sys.stdout.write('{:13}'.format(labels[i]))
    for j in range(0,4):
      sys.stdout.write('{:8.2f}'.format(ag.qsa[i, j]))
    print ''
  print ''

  # -----
  print 'Statistics of the states seen (state, percentage, total nr.):'
  print ''
  sum = 0
  for k in labels:
    sum = sum + ag.stats[k]
    
  for k in labels:
    print '{:13}'.format(k), '{:6.2f}'.format(100.0 * ag.stats[k] / sum), '{:6.0f}'.format(ag.stats[k])
  print ''
  

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    print_stats(a)



def scanparameters():
  
    bestscore = -1000000
    bestsettings = ""
  
    for alpha in [0.5, 0.6, 0.7]:
      for gamma in [ 0.2, 0.3, 0.4]:
        for epsilon in [0.03, 0.05, 0.07, 0.1]:
    
          # create things
          e = Environment()
          a = e.create_agent(LearningAgent)
          e.set_primary_agent(a, enforce_deadline=True)

          # set up the new agent with our constants
          a.overwritect(alpha, gamma, epsilon)

          # Now simulate it
          sim = Simulator(e, update_delay=0.0, display=False) 
          sim.run(n_trials=100)  # run for a specified number of trials
  
          # compute the score as maximum reward gained in the last ten rounds
          score = sum(np.array(a.totalrewards)[-10:])
          if score > bestscore:
            bestscore = score
            bestsettings = [alpha, gamma, epsilon, a.totalrewards]

          
          # score - sum of relative times in the last ten rounds
          #deadline = np.array(a.totalfinish)[-10:]
          #deadline_max = np.array(a.totalstart)[-11:-1]
          #score = sum(1.0 * (deadline_max - deadline) / deadline_max)
          #if score < bestscore:
          #  bestscore = score
          #  bestsettings = [alpha, gamma, epsilon, a.totalrewards]
            
    print "best run: ", bestscore
    print bestsettings
    
    
if __name__ == '__main__':
    run()
    #scanparameters()
