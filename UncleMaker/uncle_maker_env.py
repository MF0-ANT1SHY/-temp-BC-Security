import numpy as np
import math

from utils import save_chain

HONEST = "honest"
ATTACK = "attack"
MAIN = "main"
LAST = -1
SECOND_LAST = -2
BLOCK_INITIAL_DIFFICULTY = 4096
TIME_STAMP_UPPER_BOUND = 72


class UMBlock:
    """
    Initialize one block

    :param identity: Indicates who mined this block
    :param timestamp: When this block is mined, used for difficulty calculation
    :param difficulty: Indicates the difficulty of this block, affects the block after it
    """

    def __init__(self,
                 timestamp,
                 mining_time,
                 interval,
                 identity=MAIN,
                 difficulty=BLOCK_INITIAL_DIFFICULTY):
        self.identity = identity
        self.timestamp = timestamp      # Indicates the time when the miner starts mining this block
        self.mining_time = mining_time  # Time consumed for the miner to mine this block
        self.difficulty = difficulty    # Used for choosing main chain

        # The following property are for debug purpose
        self.replace = False        # Whether the current block has replaced another block
        self.replaced_block = None  # If self.replace=True, the replaced block
        self.actual_timestamp = timestamp   # The true timestamp of the attacker's block
        self.interval = interval    # time interval between the parent block

    def set_difficulty(self, difficulty):
        self.difficulty = difficulty

    def set_timestamp(self, timestamp):
        self.timestamp = timestamp

    @staticmethod
    def create_genesis_block(timestamp, time_interval, difficulty1, difficulty2):
        block_1 = UMBlock(timestamp=timestamp, mining_time=9, interval=0, difficulty=difficulty1)
        block_2 = UMBlock(timestamp=timestamp + time_interval,
                          mining_time=time_interval, interval=time_interval, difficulty=difficulty2)
        return [block_1, block_2]

    def __repr__(self) -> str:
        if not self.replace:
            return (f"[{'-' * 23}]\n"
                    f"[  Identity:    {self.identity:<8}]\n"
                    f"[  Timestamp:   {self.timestamp:<8}]\n"
                    f"[  MiningTime:  {self.mining_time:<8}]\n"
                    f"[  Difficulty:  {self.difficulty:<8}]\n"
                    f"[{'-' * 23}]\n")
        else:
            return (f"[{'-' * 23}]"
                    f"{' ' * 3}"
                    f"[{'-' * 23}]\n"
                    f"[  Identity:    {self.identity:<8}]"
                    f"{' ' * 3}"
                    f"[  Identity:    {self.replaced_block.identity:<8}]\n"
                    f"[  Timestamp:   {self.timestamp:<8}]"
                    f"{' ' * 3}"
                    f"[  Timestamp:   {self.replaced_block.timestamp:<8}]\n"
                    f"[  MiningTime:  {self.mining_time:<8}]"
                    f"{' ' * 3}"
                    f"[  MiningTime:  {self.replaced_block.mining_time:<8}]\n"
                    f"[  Difficulty:  {self.difficulty:<8}]"
                    f"{' ' * 3}"
                    f"[  Difficulty:  {self.replaced_block.difficulty:<8}]\n"
                    f"[{'-' * 23}]"
                    f"{' ' * 3}"
                    f"[{'-' * 23}]\n")


class UMChain:
    """
    Initialize one blockchain

    :param timestamp: Blockchain creation time
    :param time_interval: The time interval between the first two blocks
    :param target_time: The target block time for each block, used for difficulty adjustment
    :param upper_bound: Limit the size of the state space
    """

    def __init__(self,
                 timestamp,
                 time_interval,
                 target_time=12,
                 difficulty=BLOCK_INITIAL_DIFFICULTY,
                 upper_bound=TIME_STAMP_UPPER_BOUND
                 ) -> None:
        # Genesis information
        self.genesis_timestamp = timestamp
        self.genesis_time_interval = time_interval
        self.genesis_difficulty = difficulty

        # Define the environment at the start of each game
        self.target_time = target_time
        self.upper_bound = upper_bound
        self.difficulty = self.genesis_difficulty
        self.compute_power = self.difficulty / self.target_time

        # Create initial chain
        self.blocks = (
            UMBlock.create_genesis_block(
                timestamp=self.genesis_timestamp, time_interval=self.genesis_time_interval,
                difficulty1=self.difficulty,  # genesis block's difficulty
                difficulty2=self.calculate_difficulty(self.genesis_difficulty, self.genesis_time_interval)
                )
        )

    """
    Reset the chain after each game
    """

    def reset(self):
        self.difficulty = self.genesis_difficulty
        self.compute_power = self.difficulty / self.target_time

        self.blocks = (
            UMBlock.create_genesis_block(
                timestamp=self.genesis_timestamp, time_interval=self.genesis_time_interval,
                difficulty1=self.difficulty,  # genesis block's difficulty
                difficulty2=self.calculate_difficulty(self.genesis_difficulty, self.genesis_time_interval)
            )
        )

    """
    Obtaining the difficulty of a block at a specified place

    :param place: LAST(-1) or SECOND_LAST(-2)
    """

    def get_difficulty(self, place):
        return self.blocks[place].difficulty

    """
    Obtaining the timestamp of a block at a specified place

    :param place: LAST(-1) or SECOND_LAST(-2)
    """

    def get_timestamp(self, place):
        return self.blocks[place].timestamp

    """
    Obtaining the mining time of a block at a specified place

    :param place: LAST(-1) or SECOND_LAST(-2)
    """

    def get_mining_time(self, place):
        return self.blocks[place].mining_time

    """
    Obtaining the identity of a block at a specified place

    :param place: LAST(-1) or SECOND_LAST(-2)
    """

    def get_identity(self, place):
        return self.blocks[place].identity

    @property
    def length(self):
        return len(self.blocks)

    @property
    def total_difficulty(self):
        return sum(block.difficulty for block in self.blocks)

    @property
    def total_time_interval(self):
        return int(self.blocks[0].timestamp) - int(self.blocks[-1].timestamp)

    """
    Calculate the new difficulty based on specific block, 
    based on the following formula:
    d: parent's difficulty
    t: timestamp difference from its parent
    u: 1 indicates the current block specified an uncle, 0 otherwise
    d + max(1+u-⌊t/9⌋, -99) ⋅ ⌊d/2048⌋ 

    :param difficulty: Difficulty of its parent block
    :param time_difference: Timestamp difference from its parent (in seconds)
    :param uncle: Whether this block specified an uncle (0/1)
    :return: New difficulty for the next block
    """

    @staticmethod
    def calculate_difficulty(difficulty, time_difference, uncle=0):
        inner_max = max(1 + uncle - math.floor(time_difference / 9), -99)
        return difficulty + inner_max * math.floor(difficulty / 2048)

    """
    Return the mining time based on the specified block (difficulty) 
    and the current computing power

    :param difficulty: Difficulty of the specified block
    :param fraction: The computing power fraction of the current miner(Honest Miner/Attacker)
    :return: A random mining time that follows an exponential distribution
    """

    def mine_block(self, difficulty, fraction):
        actor_mining_power = self.compute_power * fraction
        mining_time = np.random.exponential(difficulty / actor_mining_power)
        mining_time_rounded = int(round(mining_time))

        # At least it should be larger than one
        return max(1, min(mining_time_rounded, self.upper_bound))

    """
    Update blockchain
    replace=True: Update the block at the end of the chain using the input block
    replace=False: Append the input block at the end of the chain
    """

    def add_block(self, block, replace=False):
        if replace:
            block.replaced_block = self.blocks.pop()
        block.replace = replace
        self.blocks.append(block)


class UMEnv:
    """
    Initialize the gaming environment

    :param attacker_fraction: Specifies the proportion of network computing power owned by the attacker
    :param target_time: Blockchain target block time
    :param timestamp_upper_bound: The upper bound of the block's timestamp, indicates the size of the state space
    :param pre_defined_time_interval: The time interval between the initial two blocks
    """

    def __init__(self,
                 attacker_fraction,
                 target_time,
                 timestamp_upper_bound,
                 pre_defined_time_interval=9):

        # The blockchain this environment depends on
        self.chain = (
            UMChain(timestamp=0,
                    time_interval=pre_defined_time_interval,
                    target_time=target_time,
                    upper_bound=timestamp_upper_bound)
        )

        # The proportion of total computing power occupied by the attacker
        # Assume it will not change during the whole game
        self._alpha = attacker_fraction

        # Determines the size of the state space, a discrete space with interval of 1
        self.timestamp_upper_bound = timestamp_upper_bound

        # Actions an attacker can perform [0, 1]
        # 0: Mining on the last block, 1: Mining on the second last block
        self._action_space_n = 2

        # State explain:
        # 0: Indicates the difference in timestamps between the last and the second last block
        # 1: If the attacker launches an attack, its mining time
        self._current_state = self.chain.get_timestamp(LAST) - self.chain.get_timestamp(SECOND_LAST)
        self._accumulated_steps = 0  # Total game steps so far

    def reset(self):
        self.chain.reset()

        self._accumulated_steps = 0
        self._current_state = self.chain.get_timestamp(LAST) - self.chain.get_timestamp(SECOND_LAST)
        return self._current_state, 0

    @property
    def reward_fraction(self):
        honest_count = len([block for block in self.chain.blocks if block.identity == HONEST])
        attack_count = len([block for block in self.chain.blocks if block.identity == ATTACK])
        return attack_count / (attack_count + honest_count)

    @staticmethod
    def manipulate_timestamp(interval):
        return ((interval - 9) // 9) * 9 + 8

    """
    There seems to be no action that cannot be performed in whatever status
    status explain: 
    A number from 0 to timestamp_upper_bound that describes the time difference between the last two blocks
    Actions explain:
    0: Mining honestly, on the latest block.
    1: Trying to attack, mining on the previous block.
    
    :param action: the action that attacker decided to take
    :param time: The last determined time the attacker mined
    """

    def step(self, state, action, time):
        # s_l_* = second_last_*
        s_l_difficulty, s_l_mining_time = \
            self.chain.get_difficulty(SECOND_LAST), self.chain.get_mining_time(SECOND_LAST)
        s_l_timestamp = self.chain.get_timestamp(SECOND_LAST)
        # l_*=last_*
        l_difficulty, l_mining_time = self.chain.get_difficulty(LAST), self.chain.get_mining_time(LAST)
        l_timestamp = self.chain.get_timestamp(LAST)

        # Honest Miner (always mine on the latest block)
        h_difficulty = self.chain.calculate_difficulty(l_difficulty, l_mining_time)
        h_mining_time = self.chain.mine_block(difficulty=h_difficulty, fraction=1 - self._alpha)

        # Attacker decides to mine on the latest block
        if action == 0:
            a_p_timestamp = l_timestamp
            a_p_mining_time = l_mining_time

            a_difficulty = self.chain.calculate_difficulty(l_difficulty, l_mining_time)
            a_mining_time = self.chain.mine_block(difficulty=a_difficulty, fraction=self._alpha)

        # Attacker decides to attack, mine on the second last block and try to replace the last block
        else:
            a_p_timestamp = s_l_timestamp
            a_p_mining_time = self.manipulate_timestamp(s_l_mining_time) # manipulate its parent's mining time

            a_difficulty = self.chain.calculate_difficulty(s_l_difficulty, self.manipulate_timestamp(s_l_mining_time))
            a_mining_time = self.chain.mine_block(difficulty=a_difficulty, fraction=self._alpha)

        attack_succeed = False
        # Attacker wins the bookkeeping rights
        if a_mining_time < h_mining_time:
            if action == 1:
                attack_succeed = True

            new_identity = ATTACK
            new_timestamp = a_p_timestamp + a_p_mining_time
            new_mining_time = a_mining_time
            new_difficulty = a_difficulty
            new_interval = a_p_mining_time
        # Honest Miner wins the bookkeeping rights
        else:
            new_identity = HONEST
            new_timestamp = l_timestamp + l_mining_time
            new_mining_time = h_mining_time
            new_difficulty = h_difficulty
            new_interval = l_mining_time
            
        # Update environment
        new_block = UMBlock(timestamp=new_timestamp, mining_time=new_mining_time, interval=new_interval,
                            identity=new_identity, difficulty=new_difficulty)
        self.chain.add_block(new_block, replace=attack_succeed)
        
        # Update reward and the status of the next round
        reward = 1 if a_mining_time < h_mining_time else 0
        reward = reward + 1 if attack_succeed else reward

        # The attack can be launched if the following two requirements are
        # satisfied
        # 1. Honest miner mined the last block
        # 2. The attacker has room to tamper with the timestamp so that its difficulty
        #    will be strictly greater than the previous last block (9 <= t)
        if self.chain.get_identity(LAST) == HONEST:
            if 9 <= self.chain.get_mining_time(SECOND_LAST):
                return new_mining_time, reward, True, action, a_mining_time
        return new_mining_time, reward, False, action, a_mining_time
