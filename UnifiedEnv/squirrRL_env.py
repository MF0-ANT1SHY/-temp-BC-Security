import numpy as np
import copy as copy
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import mdptoolbox
import markov_util
from scipy.stats import truncnorm
import gymnasium as gym
from gymnasium import spaces

def Normalize(v):
    norm = 0
    for i in range(len(v)):
        norm += v[i]
    if norm == 0:
       return v
    for i in range(len(v)):
        v[i] /= norm
    return v

def random_normal_trunc(mean, dev, low, up):
    x = np.random.normal(mean, dev)
    return np.clip(x, low, up)

class alpha_random_process:

    def __init__(self, alpha, dev, interval, name = "iid"):
        self._attacker_start = alpha
        self._attacker = alpha
        self._other = 1 - alpha
        self._other_start = self._other
        self._dev = dev
        self._interval = interval
        if (self._interval[0] == 0):
            self._interval = (1e-6, self._interval[1])
        self._name = name

    def reset(self):
        self._other = self._other_start
        self._attacker = self._attacker_start
        return self.get()

    def next(self):
        if (self._name == "iid"):
            #lower_bound = self._attacker / self._interval[1] - self._attacker
            #upper_bound = self._attacker / self._interval[0] - self._attacker
            lower_bound = self._interval[0]
            upper_bound = self._interval[1]
            #self._other = random_normal_trunc(self._other_start, self._dev, lower_bound, upper_bound)
            self._attacker = random_normal_trunc(self._attacker_start, self._dev, lower_bound, upper_bound)
            self._other = 1 - self._attacker
        elif (self._name == "brown") :
            #lower_bound = (1 - self._interval[1]) * self._attacker / self._interval[1] / self._other
            #upper_bound = (1 - self._interval[0]) * self._attacker / self._interval[0] / self._other
            lower_bound = (1 - self._interval[1]) * self._attacker / self._interval[1]
            upper_bound = (1 - self._interval[0]) * self._attacker / self._interval[0]
            self._other = random_normal_trunc(self._other, self._dev, lower_bound, upper_bound)

        return self.get()

    def get(self):
        return self._attacker / (self._attacker + self._other)

class real_alpha_process:
    def __init__(self, alpha, interval, array):
        self._start_alpha = alpha
        self._alpha = alpha
        self._array = array
        self._interval = interval
        avg = 0.0
        length = 1600
        for i in range(length):
            avg += array[i]
        avg /= length
        self._attacker_hashrate = alpha * avg
        self._pointer = 0
        self._interval = interval

    def reset(self):
        self._alpha = self._start_alpha
        self._pointer = 0

    def get(self):
        return self._alpha

    def next(self):
        self._pointer += 1
        while (self._array[self._pointer] == 0): self._pointer += 1
        self._alpha = np.clip(self._attacker_hashrate / self._array[self._pointer], self._interval[0], self._interval[1])
        return self.get()

    def get_total(self):
        return self._array[self._pointer]

class SM_env:

    # max_hidden_block : limit the max hidden block of attacker
    # attacker_fraction : usually denoted as alpha, the hash power of the attacker against the whole network
    # follower_fraction : usually denoted as gamma, the follower's fraction against the honest miner
    # relative_p : the coefficient of the linear trick from the Optimal Selfish Mining paper.
    #
    # The following params describe the random process of the alpha.
    # If you want a fixed alpha, simply use the default values.
    #
    # dev : the standard deviation of the random process of alpha. If you want a fixed alpha, set dev=0.
    # random_interval : the reasonable range of the alpha. You can use (0, 0.5).
    # frequency: the update frequency of the alpha w.r.t block generation.
    # random_process : "iid" or "brown".
    # array : used for the history data simulation.

    def SM_theoratical_gain(self, a, gamma):
        rate = (a * (1 - a) * (1 - a) * (4.0 * a + gamma * (1 - 2 * a)) - np.power(a, 3)) / (1 - a * (1 + (2 - a) * a))
        #rate = a
        return rate

    def seed(self, sd):
        np.random.seed(sd)

    def __init__(self, max_hidden_block, attacker_fraction, follower_fraction, relative_p = 0, dev = 0, random_interval = (0, 1), frequency = 1, random_process = "iid", array = []):

        self._max_hidden_block = max_hidden_block
        self._state_space = []
        self._state_dict = {}
        self._alpha = attacker_fraction
        self._gamma = follower_fraction
        self._accumulated_steps = 0
        self._attack_block = 0
        self._honest_block = 0
        self._action_space_n = 3
        self._state_vector_n = 3
        self._matrix_init = False
        self._frequency = frequency
        self._dev = dev
        #self._current_alpha = random_normal_trunc(self._alpha, self._dev, 0, 1)
        if (random_process == "real"):
            self._random_process = real_alpha_process(attacker_fraction, random_interval, array)
        else:
            self._random_process = alpha_random_process(attacker_fraction, dev, random_interval, random_process)

        self._current_alpha = self._random_process.get()

        rept = 1000000
        alpha = 0.0
        for i in range(rept):
            a = random_normal_trunc(self._alpha, self._dev, random_interval[0], random_interval[1])
            alpha += a
        alpha /= rept
        self._expected_alpha = alpha

        #self._attacker_block_reward = 1 - attacker_fraction
        #self._honest_block_reward = - attacker_fraction
        '''
        if (relative_reward == True):
            t = self.SM_theoratical_gain(attacker_fraction, follower_fraction)
            print(t)
            self._attacker_block_reward = 1 - t
            self._honest_block_reward = - t
        else:
            self._attacker_block_reward = 1 - 0.2
            self._honest_block_reward = - 0.2
        '''

        #print(relative_p)
        if (relative_p == 0): rp = self.SM_theoratical_gain(attacker_fraction, follower_fraction)
        else : rp = relative_p
        self._attacker_block_reward = 1 - rp
        self._honest_block_reward = - rp

        #a = length of attacker's private fork
        #b = length of honest miner's public fork
        #normal   : no fork
        #catch up : when attacker mines a new block and catch up the public chain (a = b)
        #forking  : the attacker publish a fork, which length equals to the public fork,
        #           causing a fork situation

        #construct state space
        #a < b
        for a in range(0, max_hidden_block + 1):
            for b in range(a + 1, max_hidden_block + 1):
                self._state_space.append((a, b, "normal"))
        #a = b = 0
        self._state_space.append((0, 0, "normal"))

        #a = b
        for a in range(1, max_hidden_block + 1):
            self._state_space.append((a, a, "normal"))
            self._state_space.append((a, a, "catch up"))
            self._state_space.append((a, a, "forking"))

        #a > b
        for a in range(1, max_hidden_block + 1):
            self._state_space.append((a, 0, "normal"))
            for b in range(1, a):
                self._state_space.append((a, b, "normal"))
                self._state_space.append((a, b, "forking"))

        #a = max_hidden_block + 1, b < a
        for b in range(0, max_hidden_block + 1):
            self._state_space.append((max_hidden_block + 1, b, "normal"))
            if (b > 0):
                self._state_space.append((max_hidden_block + 1, b, "forking"))

        #a < b, b = max_hidden_block + 1
        for a in range(0, max_hidden_block + 1):
            self._state_space.append((a, max_hidden_block + 1, "normal"))

        self._state_space_n = len(self._state_space)
        self._state_dict = dict(zip(self._state_space, range(0, self._state_space_n)))
        self._current_state = self._state_dict[(0, 0, "normal")]

    #input a state description, return its index
    def _name_to_index(self, s):
        return self._state_dict[s]

    def _vector_to_index(self, s):
        s = list(s)
        if (s[2] == 0):
            s[2] = "normal"
        if (s[2] == 1):
            s[2] = "catch up"
        if (s[2] == 2):
            s[2] = "forking"
        s = tuple(s)
        return self._state_dict[s]

    #input a state index, return its description
    def _index_to_name(self, idx):
        return self._state_space[idx]

    '''
    def _index_to_vector(self, idx, with_alpha = False):
        a, b, status = self._state_space[idx]
        st = 0
        if (status == "normal"):
            st = 0
        if (status == "forking"):
            st = 1
        if (status == "catch up"):
            st = 2
        if (with_alpha == False):
            return (a, b, st)
        else:
            return (a, b, st, self._current_alpha)
    '''

    def _index_to_vector(self, idx, with_alpha = False):
        a, b, status = self._state_space[idx]
        st = 0
        if (status == "normal"):
            st = 0
        if (status == "catch up"):
            st = 1
        if (status == "forking"):
            st = 2
        if (with_alpha == False):
            return (a, b, st)
        else:
            return (a, b, st, self._current_alpha)

    @property
    def current_alpha(self):
        return self._current_alpha

    #reset the environment to the starting state
    def reset(self):
        self._current_alpha = self._alpha
        self._random_process.reset()
        self._accumulated_steps = 0
        self._current_state = self._name_to_index((0, 0, "normal"))
        self._honest_block = 0
        self._attack_block = 0
        return self._current_state

    #input a state index and an action, return next state index, reward, and flag to trigger reset
    #action-value: meaning
    #0 : release private fork to match the public fork (release b block).
    #   if a < b, it means abandon private fork.
    #1 : override the public fork
    #2 : wait and mine on private fork

    #mapping = True : map illegal move to a legal one

    def unmapped_step(self, idx, action, move = True):

        a, b, status = self._index_to_name(idx)

        reward = -10000000 # if reward remains no change, then it means 'action' is an illegal move

        next_a = a
        next_b = b
        next_status = status

        #alpha = np.random.normal(self._alpha, self._dev)
        alpha = self._current_alpha
        gamma = self._gamma

        # out of bound..force to override
        if (a == self._max_hidden_block + 1 and b < a):
            #override, publish (b + 1) blocks
            if (action == 1):
                reward = (b + 1) * self._attacker_block_reward
                next_a = a - b - 1
                next_b = 0
                next_status = "normal"

        # out of bound... force to give up
        elif (b == self._max_hidden_block + 1 and a < b):
            #match -- abandon, accept b blocks
            if (action == 0):
                reward = b * self._honest_block_reward
                next_a = 0
                next_b = 0
                next_status = "normal"
        elif (a < b):
            # attacker abandons his private fork
            if (action == 0):
                reward = b * self._honest_block_reward
                next_a = 0
                next_b = 0
                next_status = "normal"
            if (action == 2):
                event = np.random.choice(2, p = [alpha, 1 - alpha])
                # attacker mines a block
                if (event == 0):
                    reward = 0
                    next_a = a + 1
                    next_b = b
                    if (next_a == next_b):
                        next_status = "catch up"
                    else:
                        next_status = "normal"
                # honest miner mines a block
                if (event == 1):
                    reward = 0
                    next_a = a
                    next_b = b + 1
                    next_status
if (event == 1):
                    reward = 0
                    next_a = a
                    next_b = b + 1
                    next_status = "normal"

        elif (a == b and a == 0):
            if (action == 2):
                event = np.random.choice(2, p = [alpha, 1 - alpha])
                # attacker mines a block
                if (event == 0):
                    reward = 0
                    next_a = a + 1
                    next_b = b
                    next_status = "normal"
                # honest miner mines a block
                if (event == 1):
                    reward = 0
                    next_a = 0
                    next_b = 1
                    next_status = "normal"

        elif (a == b and status == "normal"):
            # attacker publishes all block and matches
            if (action == 0):
                reward = 0
                next_a = a
                next_b = b
                next_status = "forking"
            # wait
            if (action == 2):
                event = np.random.choice(2, p = [alpha, 1 - alpha])
                # attacker mines a block
                if (event == 0):
                    reward = 0
                    next_a = a + 1
                    next_b = b
                    next_status = "normal"
                # honest miner mines a block
                if (event == 1):
                    reward = 0
                    next_a = a
                    next_b = b + 1
                    next_status = "normal"

        elif (a == b and status == "catch up"):
            # in this situation, the attacker cannot match!
            # wait
            if (action == 2):
                event = np.random.choice(2, p = [alpha, 1 - alpha])
                # attacker mines a block
                if (event == 0):
                    reward = 0
                    next_a = a + 1
                    next_b = b
                    next_status = "normal"
                # honest miner mines a block
                if (event == 1):
                    reward = 0
                    next_a = a
                    next_b = b + 1
                    next_status = "normal"

        elif (a == b and status == "forking"):
            # wait, 3 fork possibilities
            if (action == 2):
                event = np.random.choice(3, p = [alpha, (1 - alpha) * gamma, (1 - alpha) * (1 - gamma)])
                # attacker mines a block
                if (event == 0):
                    reward = 0
                    next_a = a + 1
                    next_b = b
                    next_status = "forking"
                # follower mines a block
                if (event == 1):
                    reward = b * self._attacker_block_reward
                    next_a = a - b
                    next_b = 1
                    next_status = "normal"
                # unfollower mines a block
                if (event == 2):
                    reward = 0
                    next_a = a
                    next_b = b + 1
                    next_status = "normal"

        elif (a > b and b == 0):
            # override, publish a block
            if (action == 1):
                reward = self._attacker_block_reward
                next_a = a - 1
                next_b = 0
                next_status = "normal"
            # wait
            if (action == 2):
                event = np.random.choice(2, p = [alpha, 1 - alpha])
                # attacker mines a block
                if (event == 0):
                    reward = 0
                    next_a = a + 1
                    next_b = b
                    next_status = "normal"
                # honest miner mines a block
                if (event == 1):
                    reward = 0
                    next_a = a
                    next_b = b + 1
                    next_status = "normal"

        elif (a > b and b > 0 and status == "normal"):
            # match, publish b blocks
            if (action == 0):
                reward = 0
                next_a = a
                next_b = b
                next_status = "forking"
            # override, publish (b + 1) blocks
            if (action == 1):
                reward = (b + 1) * self._attacker_block_reward
                next_a = a - b - 1
                next_b = 0
                next_status = "normal"
            # wait
            if (action == 2):
                event = np.random.choice(2, p = [alpha, 1 - alpha])
                # attacker mines a block
                if (event == 0):
                    reward = 0
                    next_a = a + 1
                    next_b = b
                    next_status = "normal"
                # honest miner mines a block
                if (event == 1):
                    reward = 0
                    next_a = a
                    next_b = b + 1
                    next_status = "normal"

        elif (a > b and b > 0 and status == "forking"):
            # don't need match...
            # override, publish (b + 1) blocks
            if (action == 1):
                reward = (b + 1) * self._attacker_block_reward
                next_a = a - b - 1
                next_b = 0
                next_status = "normal"
            # wait, 3 fork possibilities
            if (action == 2):
                event = np.random.choice(3, p = [alpha, (1 - alpha) * gamma, (1 - alpha) * (1 - gamma)])
                # attacker mines a block
                if (event == 0):
                    reward = 0
                    next_a = a + 1
                    next_b = b
                    next_status = "forking"
                    #next_status = "normal"
                # follower mines a block
                if (event == 1):
                    reward = b * self._attacker_block_reward
                    next_a = a - b
                    next_b = 1
                    next_status = "normal"
                # unfollower mines a block
                if (event == 2):
                    reward = 0
                    next_a = a
                    next_b = b + 1
                    #next_status = "forking"
                    next_status = "normal"

        next_state = self._state_dict[(next_a, next_b, next_status)]
        reset_flag = False

        if (move == True and reward > -100):
            self._accumulated_steps += 1
            if (self._accumulated_steps % self._frequency == 0):
                self._current_alpha = self._random_process.next()
                '''
                if (self._random_process == "iid") :
                    self._current_alpha = random_normal_trunc(self._alpha, self._dev, self._random_interval[0], self._random_interval[1])
                elif (self._random_process == "brown") :
                    self._current_alpha = random_normal_trunc(self._current_alpha, self._dev, self._random_interval[0], self._random_interval[1])
                '''

            self._current_state = next_state
            if (abs(reward) > 0.005):
                if (reward < 0):
                    self._honest_block += reward / self._honest_block_reward
                else:
                    self._attack_block += reward / self._attacker_block_reward

        if (self._accumulated_steps > 1000000):
            reset_flag = True

        return next_state, reward, reset_flag

    def is_legal_move(self, s, a):
        s1, r, d = self.unmapped_step(s, a, move = False)
        return r > -100

    def legal_move_list(self, s):
        legal_move = []
        for i in range(self._action_space_n):
            if (self.is_legal_move(s, i) == True):
                legal_move.append(i)
        return legal_move

    # mapped_step :
    # If this action is not a legal move, then find a legal move for it.
    # order from 0 to 2
    # return s, r, d, a'
    # s : next state
    # r : instant reward
    # d : end episode flag
    # a : actual action

    def step(self, idx, action, move = True):

        if (self.is_legal_move(idx, action) == True):
            s, r, d = self.unmapped_step(idx, action, move)
            return s, r, d, action

        for i in range(3):
            if (self.is_legal_move(idx, i) == True):
                s, r, d = self.unmapped_step(idx, i, move)
                return s, r, d, i

    @property
    def observation_space_n(self):
        return self._state_space_n

    @property
    def action_space_n(self):
        return self._action_space_n

    @property
    def state_vector_n(self):
        return self._state_vector_n

    @property
    def reward_fraction(self):
        return (self._attack_block) / (self._honest_block + self._attack_block)

    def name_of_action(self, idx, action):
        state = self._index_to_name(idx)
        action_name = ""
        if (action == 0):
            if (state[0] < state[1]):
                action_name = "abandon"
            else: action_name = "match"
        if (action == 1):
            action_name = "override"
        if (action == 2):
            action_name = "wait"
        return action_name

    def map_to_legal_action(self, idx, action):
        s, r, d, a = self.step(idx, action, move = False)
        return a

    def mapped_name_of_action(self, idx, action):
        s, r, d, a = self.step(idx, action, move = False)
        return self.name_of_action(idx, a)


    # add a transition to MDP matrices
    def add_transition(self, a, s1, s2, p, r):
        s1_idx = self._name_to_index(s1)
        s2_idx = self._name_to_index(s2)
        self.transition_matrix[a, s1_idx, s2_idx] = p
        self.reward_matrix[a, s1_idx, s2_idx] = r

    # initialize necessary matrices for MDP solver
    # A : action space size
    # S : state space size
    # transition_matrix : (A, S, S) , probability
    # reward_matrix : (A, S, S) , reward
    def MDP_matrix_init(self):
        self._matrix_init = True
        alpha = self._alpha
        gamma = self._gamma
        self.transition_matrix = np.zeros((self._action_space_n, self._state_space_n, self._state_space_n))
        self.reward_matrix = np.zeros((self._action_space_n, self._state_space_n, self._state_space_n))
        for action in range(self._action_space_n):
            for s1 in range(0, self._state_space_n):

                a, b, status = self._index_to_name(s1)

                legal = False

                # out of bound..force to override
                if (a == self._max_hidden_block + 1 and b < a):
                    #override, publish (b + 1) blocks
                    if (action == 1):
                        legal = True
                        self.add_transition(action, (a, b, status), (a - b - 1, 0, "normal"), 1, (b + 1) * self._attacker_block_reward)

                # out of bound... force to give up
                elif (b == self._max_hidden_block + 1 and a < b):
                    #match -- abandon, accept b blocks
                    if (action == 0):
                        legal = True
                        self.add_transition(action, (a, b, status), (0, 0, "normal"), 1, b * self._honest_block_reward)
                elif (a < b):
                    # attacker abandons his private fork
                    if (action == 0):
                        legal = True
                        self.add_transition(action, (a, b, status), (0, 0, "normal"), 1, b * self._honest_block_reward)
                    if (action == 2):
                        legal = True
                        if (a + 1 == b):
                            self.add_transition(action, (a, b, status), (a + 1, b, "catch up"), alpha, 0)
                        else:
                            self.add_transition(action, (a, b, status), (a + 1, b, "normal"), alpha, 0)

                        self.add_transition(action, (a, b, status), (a, b + 1, "normal"), 1 - alpha, 0)

                elif (a == b and a == 0):
                    if (action == 2):
                        legal = True
                        self.add_transition(action, (a, b, status), (a + 1, b, "normal"), alpha, 0)
                        self.add_transition(action, (a, b, status), (a, b + 1, "normal"), 1 - alpha, 0)

                elif (a == b and status == "normal"):
                    # attacker publishes all block and matches
                    if (action == 0):
                        legal = True
                        self.add_transition(action, (a, b, status), (a, b, "forking"), 1, 0)
                    # wait
                    if (action == 2):
                        legal = True
                        self.add_transition(action, (a, b, status), (a + 1, b, "normal"), alpha, 0)
                        self.add_transition(action, (a, b, status), (a, b + 1, "normal"), 1 - alpha, 0)

                elif (a == b and status == "catch up"):
                    # in this situation, the attacker cannot match!
                    # wait
                    if (action == 2):
                        legal = True
                        self.add_transition(action, (a, b, status), (a + 1, b, "normal"), alpha, 0)
                        self.add_transition(action, (a, b, status), (a, b + 1, "normal"), 1 - alpha, 0)

                elif (a == b and status == "forking"):
                    # wait, 3 fork possibilities
                    if (action == 2):
                        legal = True
                        self.add_transition(action, (a, b, status), (a + 1, b, "forking"), alpha, 0)
                        self.add_transition(action, (a, b, status), (a - b, 1, "normal"), (1 - alpha) * gamma, b * self._attacker_block_reward)
                        self.add_transition(action, (a, b, status), (a, b + 1, "normal"), (1 - alpha) * (1 - gamma), 0)

                elif (a > b and b == 0):
                    # override, publish a block
                    if (action == 1):
                        legal = True
                        self.add_transition(action, (a, b, status), (a - 1, 0, "normal"), 1, self._attacker_block_reward)
                    # wait
                    if (action == 2):
                        legal = True
                        self.add_transition(action, (a, b, status), (a + 1, b, "normal"), alpha, 0)
                        self.add_transition(action, (a, b, status), (a, b + 1, "normal"), 1 - alpha, 0)

                elif (a > b and b > 0 and status == "normal"):
                    # match, publish b blocks
                    if (action == 0):
                        legal = True
                        self.add_transition(action, (a, b, status), (a, b, "forking"), 1, 0)
                    # override, publish (b + 1) blocks
                    if (action == 1):
                        legal = True
                        self.add_transition(action, (a, b, status), (a - b - 1, 0, "normal"), 1, (b + 1) * self._attacker_block_reward)
                    # wait
                    if (action == 2):
                        legal
elif (a > b and b > 0 and status == "normal"):
                    # match, publish b blocks
                    if (action == 0):
                        legal = True
                        self.add_transition(action, (a, b, status), (a, b, "forking"), 1, 0)
                    # override, publish (b + 1) blocks
                    if (action == 1):
                        legal = True
                        self.add_transition(action, (a, b, status), (a - b - 1, 0, "normal"), 1, (b + 1) * self._attacker_block_reward)
                    # wait
                    if (action == 2):
                        legal = True
                        self.add_transition(action, (a, b, status), (a + 1, b, "normal"), alpha, 0)
                        self.add_transition(action, (a, b, status), (a, b + 1, "normal"), 1 - alpha, 0)

                elif (a > b and b > 0 and status == "forking"):
                    # don't need match...
                    # override, publish (b + 1) blocks
                    if (action == 1):
                        legal = True
                        self.add_transition(action, (a, b, status), (a - b - 1, 0, "normal"), 1, (b + 1) * self._attacker_block_reward)
                    # wait, 3 fork possibilities
                    if (action == 2):
                        legal = True
                        self.add_transition(action, (a, b, status), (a + 1, b, "forking"), alpha, 0)
                        self.add_transition(action, (a, b, status), (a - b, 1, "normal"), (1 - alpha) * gamma, b * self._attacker_block_reward)
                        self.add_transition(action, (a, b, status), (a, b + 1, "normal"), (1 - alpha) * (1 - gamma), 0)

                if (legal == False):
                    self.add_transition(action, (a, b, status), (a, b, status), 1, -1000000)

        mdptoolbox.util.check(self.transition_matrix, self.reward_matrix)

    def get_MDP_matrix(self):
        if (self._matrix_init == False): self.MDP_matrix_init()
        return self.transition_matrix, self.reward_matrix

    def theoretical_attacker_fraction(self, policy):

        trans, reward = self.get_MDP_matrix()
        policy = np.array(policy, dtype = np.int32)
        n = self._state_space_n
        A = np.zeros((n, n))
        R_attacker = np.zeros((n, n))
        R_honest = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                A[i, j] = trans[policy[i], i, j]
                r = reward[policy[i], i, j]
                if (r > 0) :
                    R_attacker[i, j] = r * 1.0 / self._attacker_block_reward
                elif (r < 0):
                    R_honest[i, j] = r * 1.0 / self._honest_block_reward

        #print(R_attacker)

        r_attacker = markov_util.MRP_expected_reward(A, R_attacker)
        r_honest = markov_util.MRP_expected_reward(A, R_honest)

        return r_attacker / (r_attacker + r_honest)

    # use binary search to find the best stategy
    # it will fine-tune the reward function!
    def optimal_mdp_solver(self):
        eps = 1e-4
        low = self._alpha
        high = 1
        ret = np.zeros(self._state_space_n)
        while (high - low > eps):
            mid = (low + high) / 2

            self._attacker_block_reward = 1 - mid
            self._honest_block_reward = - mid
            self.MDP_matrix_init()
            P, R = self.get_MDP_matrix()
            solver = mdptoolbox.mdp.PolicyIteration(P, R, 0.99)
            solver.run()
            if (solver.V[self._vector_to_index((0, 0, 0))] > -eps):
                low = mid
                ret = solver.policy
                self._relative_p = mid
            else:
                high = mid

        print("alpha = ", self._alpha, "OSM = ", low)
        return ret

class eth_env(gym.Env):

    # max_hidden_block : limit the max hidden block of attacker
    # attacker_fraction : usually denoted as alpha, the hash power of the attacker against the whole network
    # follower_fraction : usually denoted as gamma, the follower's fraction against the honest miner

    #state space
    # (a, b, fork, d1 ... d6)
    metadata = {"render_modes": ["human"]}

    def SM_theoratical_gain(self, a, gamma):
        rate = (a * (1 - a) * (1 - a) * (4.0 * a + gamma * (1 - 2 * a)) - np.power(a, 3)) / (1 - a * (1 + (2 - a) * a))
        #rate = a
        return rate

    def __init__(self, max_hidden_block, attacker_fraction, follower_fraction, relative_p = 0, dev = 0, random_interval = (0, 1), frequency = 1, know_alpha = False, random_process = "iid", render_mode = None):

        super(eth_env, self).__init__()
        self.render_mode = render_mode

        self._max_hidden_block = max_hidden_block
        self._alpha = attacker_fraction
        self._gamma = follower_fraction
        self._accumulated_steps = 0
        self._attacker_gain = 0
        self._honest_gain = 0
        self._action_space_n = 3
        # (a, b, status, special_block, uncle_0, uncle_1, uncle_2, uncle_3, uncle_4, uncle_5) + (self._current_alpha)
        self._state_vector_n = 10
        #self._current_state = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        self._know_alpha = know_alpha
        if (relative_p == 0): self._relative_p = max(self._alpha, self.SM_theoratical_gain(self._alpha, self._gamma) * 1.05)
        else : self._relative_p = relative_p
        self._current_alpha = self._alpha
        self._random_process = alpha_random_process(attacker_fraction, dev, random_interval, random_process)

        if (self._know_alpha == True) :
            #self._current_state = self._current_state + (self._current_alpha,)
            self._state_vector_n += 1

        self._dev = dev
        self._random_interval = random_interval
        self._frequency = frequency
        #self._current_alpha = random_normal_trunc(self._alpha, self._dev, 0, 1)

        rept = 1000000
        alpha = 0.0
        for i in range(rept):
            a = random_normal_trunc(self._alpha, self._dev, random_interval[0], random_interval[1])
            alpha += a
        alpha /= rept
        self._expected_alpha = alpha
        
        # Define the action space: 0, 1, or 2.
        self.action_space = spaces.Discrete(3)

        # Define the observation space: a tuple of integers and floats.
        # The observation space is a tuple of:
        # (a, b, status, special_block, uncle_0, uncle_1, uncle_2, uncle_3, uncle_4, uncle_5, alpha)
        # a, b: integers from 0 to max_hidden_block + 1
        # status, special_block: integers from 0 to 2
        # uncle_0, ..., uncle_5: integers from 0 to 2
        # alpha: float from 0 to 1
        # low = np.array([0] * (self._state_vector_n - 1) + [0], dtype=np.float32)
        # high = np.array([self._max_hidden_block + 1, self._max_hidden_block + 1] + [2] * 8 + [1], dtype=np.float32)
        
        low = np.array([0] * 3 + [0] * 7, dtype=np.float32)
        high = np.array([self._max_hidden_block + 1, self._max_hidden_block + 1, 2] + [2] * 7, dtype=np.float32)

        if (self._know_alpha == True) :
            low = np.append(low, [0])
            high = np.append(high, [1])
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
    # no index representation...

    #reset the environment to the starting state
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        self._random_process.reset()
        self._accumulated_steps = 0
        self._current_state = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        self._current_alpha = self._alpha
        if (self._know_alpha == True) :
            self._current_state = self._current_state + (self._current_alpha,)
        self._honest_gain = 0
        self._attacker_gain = 0
        self._special_block = 0

        #history info
        self._attacker_block = 0
        self._honest_block = 0
        self._aa_num = 0
        self._aa_distance = 0
        self._ha_num = 0
        self._ha_distance = 0

        return np.array(self._current_state, dtype=np.float32), {}

    #input a state index and an action, return next state index, reward, and flag to trigger reset
    #action-value: meaning
    #0 : release private fork to match the public fork (release b block).
    #   if a < b, it means abandon private fork.
    #1 : override the public fork
    #2 : wait and mine on private fork

    # normal : 0
    # catch up : 1
    # forking : 2

    def unmapped_step(self, s, action, move = True):

        a, b, status = int(s[0]), int(s[1]), int(s[2])

        next_a = a
        next_b = b
        next_status = status

        alpha = self._current_alpha
        gamma = self._gamma

        legal = False
        max_uncle_block = 2
        attacker_get = 0
        attacker_uncle = 0
        attacker_nephew = 0
        honest_get = 0
        honest_uncle = 0
        honest_nephew = 0

        # out of bound..force to override
        if (a == self._max_hidden_block + 1 and b < a):
            #override, publish (b + 1) blocks
            if (action == 1):
                #reward = (b + 1) * self._attacker_block_reward
                attacker_get = b + 1
                next_a = a - b - 1
                next_b = 0
                next_status = 0 # normal
                legal = True

        # out of bound... force to give up
        elif (b == self._max_hidden_block + 1 and a < b):
            #match -- abandon, accept b blocks
            if (action == 0):
                legal = True
                #reward = b * self._honest_block_reward
                if (b > 100):
                    honest_get = b - 1
                    next_a = 0
                    next_b = 1
                    next_status = 0 # normal
                else :
                    honest_get = b
                    next_a = 0
                    next_b = 0
                    next_status = 0 # normal

        elif (a < b):
            # attacker abandons his private fork
            if (action == 0):
                #reward = b * self._honest_block_reward
                legal = True
                if (b > 100):
                    honest_get = b - 1
                    next_a = 0
                    next_b = 1
                    next_status = 0 # normal
                else :
                    honest_get = b
                    next_a = 0
                    next_b = 0
                    next_status = 0 # normal

            if (action == 2):
                legal = True
                event = self.np_random.choice(2, p = [alpha, 1 - alpha])
                # attacker mines a block
                if (event == 0):
                    #reward = 0
                    next_a = a + 1
                    next_b = b
                    #if (next_a == next_b):
                    #    next_status = 1 #"catch up"
                    #else:
                    next_status = 0 # "normal"
                # honest miner mines a block
                if (event == 1):
                    #reward = 0
                    next_a = a
                    next_b = b + 1
                    next_status = 0 #"normal"

        elif (a == b and a == 0):
            if (action == 2):
                legal = True
                event = self.np_random.choice(2, p = [alpha, 1 - alpha])
                # attacker mines a block
                if (event == 0):
                    #reward = 0
                    next_a = a + 1
                    next_b = b
                    next_status = 0 #"normal"
                # honest miner mines a block
                if (event == 1):
                    #reward = 0
                    next_a = 0
                    next_b = 1
                    next_status = 0 #"normal"

        #elif (a == b and status == "normal"):
        elif (a == b and status == 0):
            # attacker publishes all block and matches
            if (action == 0):
                #print("match!")
                legal = True
                #reward = 0
                next_a = a
                next_b = b
                next_status = 2 #"forking"
            # wait
            if (action == 2):
                legal = True
                event = self.np_random.choice(2, p = [alpha, 1 - alpha])
                # attacker mines a block
                if (event == 0):
                    #reward = 0
                    next_a = a + 1
                    next_b = b
                    next_status = 0 #"normal"
                # honest miner mines a block
                if (event == 1):
                    #reward = 0
                    next_a = a
                    next_b = b + 1
                    next_status = 0 #"normal"

        #elif (a == b and status == "catch up"):
        elif (a == b and status == 1):
            # in this situation, the attacker cannot match!
            # wait
            if (action == 2):
                legal = True
                event = self.np_random.choice(2, p = [alpha, 1 - alpha])
                # attacker mines a block
                if (event == 0):
                    #reward = 0
                    next_a = a + 1
                    next_b = b
                    next_status = 0 #"normal"
                # honest miner mines a block
                if (event == 1):
                    #reward = 0
                    next_a = a
                    next_b = b + 1
                    next_status = 0 #"normal"

        #elif (a == b and status == "forking"):
        elif (a == b and status == 2):
            # wait, 3 fork possibilities
            if (action == 2):
                legal = True
                event = self.np_random.choice(3, p = [alpha, (1 - alpha) * gamma, (1 - alpha) * (1 - gamma)])
                # attacker mines a block
                if (event == 0):
                    #reward = 0
                    next_a = a + 1
                    next_b = b
                    next_status = 2 #"forking"
                # follower mines a block
                if (event == 1):
                    #reward = b * self._attacker_block_reward
                    attacker_get = b
                    next_a = a - b
                    next_b = 1
                    next_status = 0 #"normal"
                # unfollower mines a block
                if (event == 2):
                    #reward = 0
                    next_a = a
                    next_b = b + 1
                    next_status = 0 #"normal"

        elif (a > b and b == 0):
            # override, publish a block
            if (action == 1):
                legal = True
                #reward = self._attacker_block_reward
                attacker_get = 1
                next_a = a - 1
                next_b = 0
                next_status = 0 #"normal"
            # wait
            if (action == 2):
                legal = True
                event = self.np_random.choice(2, p = [alpha, 1 - alpha])
                # attacker mines a block
                if (event == 0):
                    #reward = 0
                    next_a = a + 1
                    next_b = b
                    next_status = 0#"normal"
                # honest miner mines a block
                if (event == 1):
                    #reward = 0
                    next_a = a
                    next_b = b + 1
                    next_status = 0# "normal"

        #elif (a > b and b > 0 and status == "normal"):
        elif (a > b and b > 0 and status == 0):
            # match, publish b blocks
            if (action == 0):
                legal = True
                #reward = 0
                next_a = a
                next_b = b
                next_status = 2 #"forking"
            # override, publish (b + 1) blocks
            if (action == 1):
                legal = True
                #reward = (b + 1) * self._attacker_block_reward
                attacker_get = b + 1
                next_a = a - b - 1
                next_b = 0
                next_status = 0 #"normal"
            # wait
            if (action == 2):
                legal = True
                event = self.np_random.choice(2, p = [alpha, 1 - alpha])
                # attacker mines a block
                if (event == 0):
                    #reward = 0
                    next_a = a + 1
                    next_b = b
                    next_status = 0 #"normal"
                # honest miner mines a block
                if (event == 1):
                    #reward = 0
                    next_a = a
                    next_b = b + 1
                    next_status = 0 #"normal"

        #elif (a > b and b > 0 and status == "forking"):
        elif (a > b and b > 0 and status == 2):
            # don't need match...
            # override, publish (b + 1) blocks
            if (action == 1):
                legal = True
                #reward = (b + 1) * self._attacker_block_reward
                attacker_get = b + 1
                next_a = a - b - 1
                next_b = 0
                next_status = 0 #"normal"
            # wait, 3 fork possibilities
            if (action == 2):
                legal = True
                event = self.np_random.choice(3, p = [alpha, (1 - alpha) * gamma, (1 - alpha) * (1 - gamma)])
                # attacker mines a block
                if (event == 0):
                    #reward = 0
                    next_a = a + 1
                    next_b = b
                    next_status = 2 #"forking"
                    #next_status = "normal"
                # follower mines a block
                if (event == 1):
                    #reward = b * self._attacker_block_reward
                    attacker_get = b
                    next_a = a - b
                    next_b = 1
                    next_status = 0 #"normal"
                # unfollower mines a block
                if (event == 2):
                    #reward = 0
                    next_a = a
                    next_b = b + 1
                    next_status = 0 #"normal"

        if (legal == False):
            return s, -1000000, False, {}

        #special_block = s[3] | (status == 2) # if the honest miner know the special block!
        special_block = self._special_block
        uncle = list(s[4 : 10])
        new_uncle = [0] * 6

        ha_distance = 0
        ha_num = 0
        aa_distance = 0
        aa_num = 0

        # reference strategy : earliest block first and only refer to his own block
        if (attacker_get > 0):
            j = 5
            i = 1
            while (i <= attacker_get and j >= 0):
                for k in range(max_uncle_block):
                    while (j >= 0 and uncle[j] != 1): j -= 1
                    if (j >= 0):
                        attacker_nephew += 1
                        attacker_uncle += (8 - (i + j)) / 8.0
                        #attacker_uncle += 4 / 8.0
                        uncle[j] = 0 # used

                        aa_num += 1
                        aa_distance += (i + j)
                i += 1

            for i in range(0, 6 - attacker_get):
                new_uncle[i + attacker_get] = uncle[i]

            if (b > 0 and attacker_get <= 6): new_uncle[attacker_get - 1] = 2
            special_block = 0
            #print("attacker get :", attacker_get, "nephew:", attacker_nephew, "uncle:", attacker_uncle)

        # reference strategy : earliest block first and refer to all block
        elif (honest_get > 0):
            j = 5
            i = 1
            while (i <= honest_get and j >= 0):
                for k in range(max_uncle_block):
                    while (j >= 0 and uncle[j] == 0): j -= 1
                    if (j < 0) : break
                    honest_nephew += 1
                    if (uncle[j] == 1):
                        attacker_uncle += (8 - (i + j)) / 8.0
                        ha_num += 1
                        ha_distance += (i + j)
                        #print("hanging attacker uncle with distance", (i + j))
                        #attacker_uncle += 4 / 8.0
                    elif (uncle[j] == 2):
                        honest_uncle += (8 - (i + j)) / 8.0
                        #honest_uncle += 4 / 8.0
                    uncle[j] = 0 # used
                if (j < 0): break
                i += 1

            if (i < honest_get and i == 1): i += 1 # try the special block
            i = max(i, special_block + 1) # only after K blocks, the honest miner can refer to the special block
            # special block - the new fork block from attacker ! only in forking status
            if (special_block > 0 and i <= honest_get and i <= 7 and i > 1):
                #print("special block work!")
                attacker_uncle += (8 - (i - 1)) / 8.0
                #attacker_uncle += 4 / 8.0
                honest_nephew += 1
                ha_distance += (i - 1)
                #print("special uncle with distance", (i - 1))
                ha_num += 1
            else:
                if (a > 0 and honest_get <= 6): new_uncle[honest_get - 1] = 1

            for i in range(0, 6 - honest_get):
                new_uncle[i + honest_get] = uncle[i]

            special_block = 0
            #print("honest get :", honest_get, "honest nephew:", honest_nephew, "honest uncle", honest_uncle, "attacker uncle", attacker_uncle)

        else:
            new_uncle = uncle
            # if the special block has not been revealed and the attacker reveals now it by matching:
            if (special_block == 0 and next_status == 2 and a > 0):
                special_block = b

            #if (next_status == 2): special_block = 1 # know the special block !

        attacker_instant_gain = (attacker_get + attacker_uncle + attacker_nephew / 32.0)
        honest_instant_gain = (honest_get + honest_uncle + honest_nephew / 32.0)

        #self._relative_p = max(self._current_alpha, self.SM_theoratical_gain(self._current_alpha, self._gamma))
        #print(self._relative_p)
        #if (self._relative_p > 0.8): self._relative_p *= 0.9

        reward = attacker_instant_gain * (1 - self._relative_p) - honest_instant_gain * self._relative_p

        if (move == True):
            self._special_block = special_block
        #special_block = max(special_block, 1)
        
        next_state = (next_a, next_b, next_status, special_block) + tuple(new_uncle)
        if (self._know_alpha == True) :
            next_state = next_state + (self._current_alpha,)

        if (move == True):
            self._accumulated_steps += 1
            if (self._accumulated_steps % self._frequency == 0):
                self._current_alpha = self._random_process.next()
                '''
                if (self._random_process == "iid") :
                    self._current_alpha = random_normal_trunc(self._alpha, self._dev, self._random_interval[0], self._random_interval[1])
                elif (self._random_process == "brown") :
                    self._current_alpha = random_normal_trunc(self._current_alpha, self._dev, self._random_interval[0], self._random_interval[1])
                '''

        if (move == True) :
            self._attacker_gain += attacker_instant_gain
            self._honest_gain += honest_instant_gain
            self._attacker_block += attacker_get
            self._honest_block += honest_get
            self._aa_num += aa_num
            self._aa_distance += aa_distance
            self._ha_num += ha_num
            self._ha_distance += ha_distance
            self._current_state = next_state

        done = False

        if (self._accumulated_steps > 1000000):
            done = True

        #return self._current_state, reward, reset_flag
        return np.array(next_state, dtype=np.float32), float(reward), done, {}

    def is_legal_move(self, s, a):
        s1, r, d, _ = self.unmapped_step(s, a, move = False)
        return r > -100

    def legal_move_list(self, s):
        legal_move = []
        for i in range(self._action_space_n):
            if (self.is_legal_move(s, i) == True):
                legal_move.append(i)
        return legal_move

    # mapped_step :
    # If this action is not a legal move, then find a legal move for it.
    # order from 0 to 2
    # return s, r, d, a'
    # s : next state
    # r : instant reward
    # d : end episode flag
    # a : actual action

    def step(self, action):
        
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        if (isinstance(self._current_state, tuple)):
            state = self._current_state
        else:
            state = tuple(self._current_state.tolist())

        if (self.is_legal_move(state, action) == True):
            s, r, d, info = self.unmapped_step(state, action, move=True)
            return s, r, d, info

        for i in range(3):
            if (self.is_legal_move(state, i) == True):
                s, r, d, info = self.unmapped_step(state, i, move=True)
                return s, r, d, info
        #print("False")
        #print(s, a)

    @property
    def observation_space_n(self):
        return self._state_space_n

    @property
    def action_space_n(self):
        return self._action_space_n

    @property
    def state_vector_n(self):
        return self._state_vector_n

    @property
    def reward_fraction(self):
        return self._attacker_gain / (self._attacker_gain + self._honest_gain)

    def map_to_legal_action(self, state, action):
        s, r, d, a = self.unmapped_step(state, action, move = False)
        return a

    def uncle_info(self):
        '''
        '''

        aa_ratio = self._aa_num / self._attacker_block
        if (self._aa_num == 0):
            aa_distance = 0
        else: aa_distance = self._aa_distance / self._aa_num
        ha_ratio = self._ha_num / self._honest_block
        if (self._ha_num == 0):
            ha_distance = 0
        else : ha_distance = self._ha_distance / self._ha_num


        print("attacker block", self._attacker_block)
        print("honest block", self._honest_block)
        print("aa_num", self._aa_num)
        print("ha num", self._ha_num)
        print("aa ratio", aa_ratio)
        print("aa distance", aa_distance)
        print("ha ratio", ha_ratio)
        print("ha distance", ha_distance)

        return aa_ratio, aa_distance, ha_ratio, ha_distance

    def render(self, mode='human'):
        # This is a placeholder. You can implement actual rendering logic here.
        if self.render_mode == "human":
            print(f"Current State: {self._current_state}")
            print(f"Attacker Gain: {self._attacker_gain}, Honest Gain: {self._honest_gain}")
        else:
            super().render(mode=mode) # just raise an exception

    def close(self):
        pass  # Add any cleanup code here, if necessary