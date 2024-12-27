import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy as copy
import mdptoolbox
from _logger import setup_logging
from _random_process import (
    alpha_random_process,
    real_alpha_process,
    random_normal_trunc,
)
from _chain import Block, Chain
import os
from state_visualizer import StateVisualizer

logger = setup_logging("unified_env", "critical")


def Normalize(v):
    norm = 0
    for i in range(len(v)):
        norm += v[i]
    if norm == 0:
        return v
    for i in range(len(v)):
        v[i] /= norm
    return v


class eth_env(gym.Env):
    """
    Ethereum mining environment simulation with the following key parameters:
    - max_hidden_block: Maximum number of blocks attacker can keep private
    - attacker_fraction (alpha): Hash power ratio of attacker vs total network
    - follower_fraction (gamma): Ratio of miners that follow attacker's chain

    State space is represented as: (a, b, fork, advantage, d1...d6) where:
    - a: Number of blocks in attacker's fork
    - b: Number of blocks in honest chain
    - fork: Fork status
    - advantage: Advantage of attacker over honest chain
    - d1...d6: Additional state information
    """

    def __init__(
        self,
        max_hidden_block,
        attacker_fraction,
        follower_fraction,
        relative_p=0.0,
        dev=0.0,
        random_interval=(0, 1),
        frequency=1,
        know_alpha=False,
        random_process="iid",
        is_random_process=False,
        visualize=False,
        mask_argument=True,
        discrete=True,
    ):
        logger.info(
            f"Initializing eth_env with max_hidden_block={max_hidden_block}, alpha={attacker_fraction}, gamma={follower_fraction}"
        )
        super().__init__()

        self._mask_argument = mask_argument
        if mask_argument:
            logger.warning("Masking action argument, are you sure?")

        self.CRITERIA = 1
        self.REMAINING_BLOCKS = 1

        # Initialize blockchain components
        self.adj = 1
        self.chain = Chain(0, 13, 1, self.adj, name="mainchain")
        self.attackfork = Chain(0, 13, attacker_fraction, self.adj, name="attackfork")
        self.honestfork = Chain(
            0, 13, 1 - attacker_fraction, self.adj, name="honestfork"
        )
        self.honestfork.adjust(0)
        self.honestfork.set_external_info(self.chain.blocks[-1].difficulty)
        self.attackfork.adjust(0)
        self.attackfork.set_external_info(self.chain.blocks[-1].difficulty)

        # Initialize mining parameters
        self._max_hidden_block = max_hidden_block
        self._alpha = attacker_fraction
        self._gamma = follower_fraction
        self._accumulated_steps = 0
        self._attacker_gain = 0
        self._honest_gain = 0
        self._know_alpha = know_alpha
        self._discrete = discrete

        # Define action space
        if self._discrete:
            self.action_space = spaces.Discrete(
                4, start=0
            )  # 0: adopt, 1: match, 2: mine, 3: override
        else:
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32
            )

        # Define observation space
        if not know_alpha:
            self.observation_space = spaces.Dict(
                {
                    "len_attacker_forking": spaces.Discrete(
                        max_hidden_block + 2
                    ),  # 0 to max_hidden_block + 1
                    "len_honest_forking": spaces.Discrete(
                        max_hidden_block + 2
                    ),  # 0 to max_hidden_block + 1
                    "forking_status": spaces.Discrete(2),  # 0 or 1
                    "advantage": spaces.Discrete(3),  # 0, 1, or 2
                    "special_block": spaces.Discrete(
                        max_hidden_block + 1
                    ),  # 0 to max_hidden_block
                    "uncle": spaces.MultiDiscrete([3] * 6),  # 6 values of 0, 1, or 2
                }
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "len_attacker_forking": spaces.Discrete(max_hidden_block + 2),
                    "len_honest_forking": spaces.Discrete(max_hidden_block + 2),
                    "forking_status": spaces.Discrete(2),
                    "advantage": spaces.Discrete(3),
                    "special_block": spaces.Discrete(max_hidden_block + 1),
                    "uncle": spaces.MultiDiscrete([3] * 6),
                    "alpha": spaces.Box(
                        low=0.0, high=1.0, shape=(1,), dtype=np.float32
                    ),
                }
            )

        # Set relative mining power
        if relative_p == 0:
            self._relative_p = max(
                self._alpha, self.SM_theoratical_gain(self._alpha, self._gamma) * 1.05
            )
        else:
            self._relative_p = relative_p

        self._current_alpha = self._alpha
        self._random_process = alpha_random_process(
            attacker_fraction, dev, random_interval, random_process
        )

        self._dev = dev
        self._random_interval = random_interval
        self._frequency = frequency
        self._is_random_process = is_random_process

        # Initialize state visualizer
        self._visualize = visualize
        if visualize:
            self.visualizer = StateVisualizer()

    def _seed(self, sd):
        if sd == 41:
            logger.warning("Using seed 41, you might get fixed results")
        np.random.seed(sd)

    def SM_theoratical_gain(self, a, gamma):
        # Calculate theoretical gain based on selfish mining formula
        rate = (
            a * (1 - a) * (1 - a) * (4.0 * a + gamma * (1 - 2 * a)) - np.power(a, 3)
        ) / (1 - a * (1 + (2 - a) * a))
        return rate

    @property
    def expected_alpha(self):
        # Calculate expected alpha
        rept = 1000000
        alpha = 0.0
        for i in range(rept):
            a = random_normal_trunc(
                self._alpha,
                self._dev,
                self._random_interval[0],
                self._random_interval[1],
            )
            alpha += a
        alpha /= rept
        return alpha

    def _map_box_to_action(self, box_action):
        """
        Map normalized box action to discrete action and argument
        input: box_action: np.array([action_type, action_arg])
        output: mapped_action, mapped_arg
        """
        if self._discrete:
            return box_action, 0
        action_type = box_action[0]  # [-1, 1]
        action_arg = box_action[1]  # [-1, 1]

        # Map first dimension to action type
        if action_type < -0.5:  # e.g., [-0.6, -0.5]
            mapped_action = 0  # adopt
        elif action_type < 0:  # e.g., [-0.1,0]
            mapped_action = 1  # match
        elif action_type < 0.5:  # e.g., [0.4,0.5]
            mapped_action = 2  # mine
        else:  # e.g., [0.5, 0.6]
            mapped_action = 3  # override

        # Map second dimension to action argument
        if mapped_action == 2:  # mine action, timestamp manipulation strategy
            if action_arg < -0.33:
                mapped_arg = 0  # normal time
            elif action_arg < 0.33:
                mapped_arg = 1  # minimum time
            else:
                mapped_arg = 2  # maximum time
        else:
            # maps a continuous value between -1 and 1 to a discrete integer between 0 and self._max_hidden_block
            mapped_arg = max(
                0,
                min(
                    int((action_arg + 1) * self._max_hidden_block / 2),
                    self._max_hidden_block,
                ),
            )

        if self._mask_argument:
            mapped_arg = 0

        return mapped_action, mapped_arg

    def _explain_action(self, action):
        actionType, actionArg = self._map_box_to_action(action)
        if actionType == 0:
            return f"Adopt {actionArg} blocks"
        elif actionType == 1:
            return f"Match"
        elif actionType == 2:
            return f"Mine {actionArg} blocks"
        elif actionType == 3:
            return f"Override"

    def updateAbsTimestamp(self, time):
        self.attackfork._absolute_timestamp = time
        self.honestfork._absolute_timestamp = time
        self.chain._absolute_timestamp = time

    def reset(self, seed=41):
        logger.info("Resetting environment")
        self._seed(seed)
        if self._accumulated_steps > 0 and self._visualize:  # Save previous episode
            self.visualizer.save_visualization(self._alpha, self._accumulated_steps)
        self.chain.reset()
        self.attackfork.reset()
        self.honestfork.reset()
        self.honestfork.adjust(0)
        self.honestfork.set_external_info(self.chain.blocks[-1].difficulty)
        self.attackfork.adjust(0)
        self.attackfork.set_external_info(self.chain.blocks[-1].difficulty)
        self._random_process.reset()
        self._accumulated_steps = 0
        self._current_state = {
            "len_attacker_forking": 0,
            "len_honest_forking": 0,
            "forking_status": 0,
            "advantage": 0,
            "special_block": 0,
            "uncle": np.array([0, 0, 0, 0, 0, 0]),
        }
        if self._know_alpha:
            self._current_state["alpha"] = self._current_alpha
        self._current_alpha = self._alpha
        self._honest_gain = 1
        self._attacker_gain = 0
        self._special_block = 0

        # Track mining statistics
        self._attacker_block = 0
        self._honest_block = 1  #  we start with one genesis block
        self._aa_num = 0
        self._aa_distance = 0
        self._ha_num = 0
        self._ha_distance = 0

        return self._current_state, {}

    def step(self, action):
        """Execute a step in the environment with the given state and action."""
        # print(
        #     f"Current state: {self._current_state}, action: {self._explain_action(action)}, pure_action: {action}"
        # )
        logger.info(
            f"Current alpha: {self._current_alpha}, Advantage: {self._current_state['advantage']}"
        )

        terminated = False
        truncated = False

        # Store current state before step
        current_state = self._current_state.copy()

        actionType, actionArg = self._map_box_to_action(action)

        lenAttackerForking = self._current_state["len_attacker_forking"]
        lenHonestForking = self._current_state["len_honest_forking"]
        forkingStatus = self._current_state["forking_status"]
        advantage = self._current_state["advantage"]
        special_block = self._current_state["special_block"]
        uncle = self._current_state["uncle"]

        # Log fork status changes
        if forkingStatus == 0:
            logger.debug("Current fork status: Normal")
        elif forkingStatus == 1:
            logger.debug("Current fork status: Match")
        else:
            raise Exception(f"Unknown fork status: {forkingStatus}")

        next_lenAttackerForking = lenAttackerForking
        next_lenHonestForking = lenHonestForking
        next_forkingStatus = forkingStatus

        gamma = self._gamma

        legal = False
        max_uncle_block = 2
        attacker_get = 0
        attacker_uncle = 0
        attacker_nephew = 0
        honest_get = 0
        honest_uncle = 0
        honest_nephew = 0

        # Force override if attacker fork is too long
        if (
            lenAttackerForking == self._max_hidden_block + 1
            and lenHonestForking < lenAttackerForking
        ):
            if actionType == 3:
                legal = True
                attacker_get = lenHonestForking + 1

                next_lenAttackerForking = lenAttackerForking - lenHonestForking - 1
                next_lenHonestForking = 0

                blocks = self.attackfork._extract_blocks(lenHonestForking + 1)
                for block in blocks:
                    self.chain.add_block(block)

                if next_lenAttackerForking == 0:
                    self.attackfork.set_external_info(self.chain.blocks[-1].difficulty)
                if next_lenHonestForking == 0:
                    self.honestfork.set_external_info(self.chain.blocks[-1].difficulty)
                self.attackfork.adjust(next_lenAttackerForking)
                self.honestfork.adjust(next_lenHonestForking)

                next_forkingStatus = 0
                next_advantage = 0

        # Force adopt if honest fork is too long
        elif (
            lenHonestForking == self._max_hidden_block + 1
            and lenAttackerForking < lenHonestForking
        ):
            if actionType == 0:
                legal = True
                continueToAttack = actionArg
                honest_get = lenHonestForking - continueToAttack

                next_lenAttackerForking = 0
                next_lenHonestForking = continueToAttack

                blocks: list[Block] = self.honestfork._extract_blocks(
                    lenHonestForking - continueToAttack
                )
                for block in blocks:
                    self.chain.add_block(block)
                self.honestfork.adjust(continueToAttack)
                self.attackfork.adjust(next_lenAttackerForking)
                if next_lenAttackerForking == 0:
                    self.attackfork.set_external_info(self.chain.blocks[-1].difficulty)
                if next_lenHonestForking == 0:
                    self.honestfork.set_external_info(self.chain.blocks[-1].difficulty)
                next_forkingStatus = forkingStatus
                next_advantage = advantage

        # Normal fork status
        elif forkingStatus == 0:
            if actionType == 0:  # Adopt action
                continueToAttack = actionArg
                if continueToAttack > min(lenHonestForking, self.REMAINING_BLOCKS):
                    Exception(f"Invalid action argument: {continueToAttack}")
                else:
                    legal = True
                    honest_get = lenHonestForking - continueToAttack

                    next_lenAttackerForking = 0
                    next_lenHonestForking = continueToAttack

                    blocks: list[Block] = self.honestfork._extract_blocks(
                        lenHonestForking - continueToAttack
                    )
                    for block in blocks:
                        self.chain.add_block(block)
                    self.honestfork.adjust(continueToAttack)
                    self.attackfork.adjust(next_lenAttackerForking)
                    if next_lenAttackerForking == 0:
                        self.attackfork.set_external_info(
                            self.chain.blocks[-1].difficulty
                        )
                    if next_lenHonestForking == 0:
                        self.honestfork.set_external_info(
                            self.chain.blocks[-1].difficulty
                        )
                    next_forkingStatus = forkingStatus
                    next_advantage = advantage

            elif actionType == 1:  # Match action
                # Can only match when forks are equal length and attacker has advantage
                if lenAttackerForking == lenHonestForking and advantage == 1:
                    legal = True
                    next_lenAttackerForking = lenAttackerForking
                    next_lenHonestForking = lenHonestForking
                    next_forkingStatus = 1
                    next_advantage = 0
                else:
                    pass

            elif actionType == 2:  # Mine action
                logger.debug(f"Mining attempt - ActionArg: {actionArg}")
                legal = True
                attacker_time = self.attackfork.mine_block()
                honest_time = self.honestfork.mine_block()

                # Attacker mines first
                if attacker_time < honest_time:
                    if actionArg == 0:
                        new_block = Block("attacker", attacker_time)
                    elif actionArg == 1:
                        new_block = Block("attacker", 0)
                    elif actionArg == 2:
                        new_block = Block("attacker", 2 * 60 * 59)
                    else:
                        pass
                    logger.debug(
                        f"Adding block to attackfork: {new_block},mainchain: {self.chain.blocks}"
                    )
                    self.attackfork.add_block(new_block, self.chain)

                    next_lenAttackerForking = lenAttackerForking + 1
                    next_lenHonestForking = lenHonestForking
                    next_forkingStatus = forkingStatus

                    # Check if attacker gains advantage
                    if (
                        lenHonestForking - lenAttackerForking <= self.CRITERIA
                        and lenHonestForking - lenAttackerForking > 0
                    ):
                        if self.honestfork.total_diff > self.attackfork.total_diff:
                            next_advantage = advantage
                        elif self.honestfork.total_diff == self.attackfork.total_diff:
                            next_advantage = 1
                        else:  # honestfork.total_diff < attackfork.total_diff
                            next_advantage = 2
                    else:
                        next_advantage = advantage

                # Honest miners mine first
                elif attacker_time >= honest_time:
                    new_block = Block("honest", honest_time)
                    logger.debug(
                        f"Adding block to honestfork: {new_block},mainchain: {len(self.chain.blocks)} {self.chain.blocks[-1].nonce}"
                    )
                    self.honestfork.add_block(new_block, self.chain)

                    next_lenAttackerForking = lenAttackerForking
                    next_lenHonestForking = lenHonestForking + 1
                    next_forkingStatus = forkingStatus

                    # Check if honest miners gain advantage
                    if (
                        lenAttackerForking - lenHonestForking <= self.CRITERIA
                        and lenAttackerForking - lenHonestForking > 0
                    ):
                        if self.honestfork.total_diff > self.attackfork.total_diff:
                            next_advantage = advantage
                        elif self.honestfork.total_diff == self.attackfork.total_diff:
                            next_advantage = 1
                        else:  # honestfork.total_diff < attackfork.total_diff
                            next_advantage = 2
                    else:
                        next_advantage = advantage

            elif actionType == 3:  # Override action
                if lenAttackerForking <= lenHonestForking and advantage == 2:
                    legal = True
                    attacker_get = lenAttackerForking

                    blocks = self.attackfork._extract_blocks(lenAttackerForking)
                    for block in blocks:
                        self.chain.add_block(block)

                    next_lenAttackerForking = 0
                    next_lenHonestForking = 0
                    self.attackfork.adjust(next_lenAttackerForking)
                    self.honestfork.adjust(next_lenHonestForking)
                    if next_lenAttackerForking == 0:
                        self.attackfork.set_external_info(
                            self.chain.blocks[-1].difficulty
                        )
                    if next_lenHonestForking == 0:
                        self.honestfork.set_external_info(
                            self.chain.blocks[-1].difficulty
                        )
                    next_forkingStatus = 0
                    next_advantage = 0
                elif lenAttackerForking > lenHonestForking:
                    legal = True
                    attacker_get = lenHonestForking + 1

                    next_lenAttackerForking = lenAttackerForking - lenHonestForking - 1
                    next_lenHonestForking = 0

                    blocks = self.attackfork._extract_blocks(lenHonestForking + 1)
                    for block in blocks:
                        self.chain.add_block(block)

                    if next_lenAttackerForking == 0:
                        self.attackfork.set_external_info(
                            self.chain.blocks[-1].difficulty
                        )
                    if next_lenHonestForking == 0:
                        self.honestfork.set_external_info(
                            self.chain.blocks[-1].difficulty
                        )
                    self.attackfork.adjust(next_lenAttackerForking)
                    self.honestfork.adjust(next_lenHonestForking)

                    next_forkingStatus = 0
                    next_advantage = 0

        # Match fork status
        elif forkingStatus == 1:
            if actionType == 0:  # Adopt action
                continueToAttack = actionArg
                # if continueToAttack > min(lenHonestForking, self.REMAINING_BLOCKS):
                #     # raise Exception(f"Invalid action argument: {continueToAttack}")
                #     pass
                # else:
                if continueToAttack <= min(lenHonestForking, self.REMAINING_BLOCKS):
                    legal = True
                    honest_get = lenHonestForking - continueToAttack

                    next_lenAttackerForking = 0
                    next_lenHonestForking = continueToAttack

                    blocks = self.honestfork._extract_blocks(
                        lenHonestForking - continueToAttack
                    )
                    for block in blocks:
                        self.chain.add_block(block)
                    if next_lenAttackerForking == 0:
                        self.attackfork.set_external_info(
                            self.chain.blocks[-1].difficulty
                        )
                    if next_lenHonestForking == 0:
                        self.honestfork.set_external_info(
                            self.chain.blocks[-1].difficulty
                        )
                    self.attackfork.adjust(next_lenAttackerForking)
                    self.honestfork.adjust(next_lenHonestForking)

                    next_forkingStatus = 0
                    next_advantage = 0

            elif actionType == 2:  # Mine action
                legal = True
                attacker_time = self.attackfork.mine_block()
                honest_time = self.honestfork.mine_block()

                # Attacker mines first
                if attacker_time < honest_time:
                    if actionArg == 0:
                        new_block = Block("attacker", attacker_time)
                    elif actionArg == 1:
                        new_block = Block("attacker", 0)
                    elif actionArg == 2:
                        new_block = Block("attacker", 2 * 60 * 59)
                    self.attackfork.add_block(new_block, self.chain)

                    next_lenAttackerForking = lenAttackerForking + 1
                    next_lenHonestForking = lenHonestForking
                    next_forkingStatus = forkingStatus
                    next_advantage = advantage

                # Honest miners mine first
                elif attacker_time >= honest_time:
                    new_block = Block("honest", honest_time)
                    event = np.random.choice(2, p=[gamma, 1 - gamma])
                    if event == 0:  # Follower mines on attacker's chain
                        attacker_get = lenHonestForking

                        blocks = self.attackfork._extract_blocks(lenHonestForking)

                        next_lenAttackerForking = lenAttackerForking - lenHonestForking
                        next_lenHonestForking = 1
                        self.attackfork.adjust(next_lenAttackerForking)
                        self.honestfork.adjust(0)
                        self.honestfork.add_block(new_block, self.chain)
                        for block in blocks:
                            self.chain.add_block(block)
                        if next_lenAttackerForking == 0:
                            self.attackfork.set_external_info(
                                self.chain.blocks[-1].difficulty
                            )
                        if next_lenHonestForking == 0:
                            self.honestfork.set_external_info(
                                self.chain.blocks[-1].difficulty
                            )
                        next_forkingStatus = 0
                        next_advantage = advantage
                    else:  # Non-follower mines on honest chain
                        blocks = self.attackfork._extract_blocks(lenHonestForking)

                        next_lenAttackerForking = lenAttackerForking - lenHonestForking
                        next_lenHonestForking = 1
                        self.attackfork.adjust(next_lenAttackerForking)
                        self.honestfork.adjust(0)
                        self.honestfork.add_block(new_block, self.chain)
                        for block in blocks:
                            self.chain.add_block(block)
                        if next_lenAttackerForking == 0:
                            self.attackfork.set_external_info(
                                self.chain.blocks[-1].difficulty
                            )
                        if next_lenHonestForking == 0:
                            self.honestfork.set_external_info(
                                self.chain.blocks[-1].difficulty
                            )
                        next_forkingStatus = 0
                        next_advantage = advantage

            elif actionType == 3:  # Override action
                if lenAttackerForking > lenHonestForking:
                    legal = True
                    attacker_get = lenHonestForking + 1

                    next_lenAttackerForking = lenAttackerForking - lenHonestForking - 1
                    next_lenHonestForking = 0

                    blocks = self.attackfork._extract_blocks(lenHonestForking + 1)
                    for block in blocks:
                        self.chain.add_block(block)

                    self.attackfork.adjust(next_lenAttackerForking)
                    self.honestfork.adjust(next_lenHonestForking)
                    if next_lenAttackerForking == 0:
                        self.attackfork.set_external_info(
                            self.chain.blocks[-1].difficulty
                        )
                    if next_lenHonestForking == 0:
                        self.honestfork.set_external_info(
                            self.chain.blocks[-1].difficulty
                        )

                    next_forkingStatus = 0
                    next_advantage = 0

        if not legal:
            reward = 0
            terminated = False
            truncated = False
        else:
            special_block = self._special_block
            uncle = list(uncle)
            new_uncle = [0] * 6

            ha_distance = 0
            ha_num = 0
            aa_distance = 0
            aa_num = 0

            # Process uncle blocks for attacker
            if attacker_get > 0:
                j = 5
                i = 1
                while i <= attacker_get and j >= 0:
                    for k in range(max_uncle_block):
                        while j >= 0 and uncle[j] != 1:
                            j -= 1
                        if j >= 0:
                            attacker_nephew += 1
                            attacker_uncle += (8 - (i + j)) / 8.0
                            uncle[j] = 0  # Mark as used

                            aa_num += 1
                            aa_distance += i + j
                    i += 1

                for i in range(0, 6 - attacker_get):
                    new_uncle[i + attacker_get] = uncle[i]

                if lenHonestForking > 0 and attacker_get <= 6:
                    new_uncle[attacker_get - 1] = 2
                special_block = 0

            # Process uncle blocks for honest miners
            elif honest_get > 0:
                j = 5
                i = 1
                while i <= honest_get and j >= 0:
                    for k in range(max_uncle_block):
                        while j >= 0 and uncle[j] == 0:
                            j -= 1
                        if j < 0:
                            break
                        honest_nephew += 1
                        if uncle[j] == 1:
                            attacker_uncle += (8 - (i + j)) / 8.0
                            ha_num += 1
                            ha_distance += i + j
                        elif uncle[j] == 2:
                            honest_uncle += (8 - (i + j)) / 8.0
                        uncle[j] = 0  # Mark as used
                    if j < 0:
                        break
                    i += 1

                if i < honest_get and i == 1:
                    i += 1  # Try the special block
                i = max(i, special_block + 1)
                if special_block > 0 and i <= honest_get and i <= 7 and i > 1:
                    attacker_uncle += (8 - (i - 1)) / 8.0
                    honest_nephew += 1
                    ha_distance += i - 1
                    ha_num += 1
                else:
                    if lenAttackerForking > 0 and honest_get <= 6:
                        new_uncle[honest_get - 1] = 1

                for i in range(0, 6 - honest_get):
                    new_uncle[i + honest_get] = uncle[i]

                special_block = 0

            else:
                new_uncle = uncle
                if (
                    special_block == 0
                    and next_forkingStatus == 1
                    and lenAttackerForking > 0
                ):
                    special_block = lenHonestForking

            # Calculate rewards
            attacker_instant_gain = (
                attacker_get + attacker_uncle + attacker_nephew / 32.0
            )
            honest_instant_gain = honest_get + honest_uncle + honest_nephew / 32.0

            reward = (
                attacker_instant_gain * (1 - self._relative_p)
                - honest_instant_gain * self._relative_p
            )

            self._special_block = special_block
            self._accumulated_steps += 1
            if (
                self._accumulated_steps % self._frequency == 0
                and self._is_random_process
            ):
                self._current_alpha = self._random_process.next()

            self._attacker_gain += attacker_instant_gain
            self._honest_gain += honest_instant_gain
            self._attacker_block += attacker_get
            self._honest_block += honest_get
            self._aa_num += aa_num
            self._aa_distance += aa_distance
            self._ha_num += ha_num
            self._ha_distance += ha_distance
            self._current_state = {
                "len_attacker_forking": next_lenAttackerForking,
                "len_honest_forking": next_lenHonestForking,
                "forking_status": next_forkingStatus,
                "advantage": next_advantage,
                "special_block": special_block,
                "uncle": np.array(new_uncle),
            }
            if self._know_alpha:
                self._current_state["alpha"] = self._current_alpha

            if self._accumulated_steps > 100000:
                truncated = True

            # Log chain state changes when blocks are added
            if legal:
                if attacker_get > 0:
                    logger.info(f"Attacker gained {attacker_get} blocks")
                    logger.info(
                        f"Chain state after attacker blocks: AttackerFork={next_lenAttackerForking}, HonestFork={next_lenHonestForking}"
                    )

                if honest_get > 0:
                    logger.info(f"Honest miners gained {honest_get} blocks")
                    logger.info(
                        f"Chain state after honest blocks: AttackerFork={next_lenAttackerForking}, HonestFork={next_lenHonestForking}"
                    )

                # Log block properties
                if len(self.attackfork.blocks) > 0:
                    latest_attack_block = self.attackfork.blocks[-1]
                    logger.info(f"Latest attacker block: {latest_attack_block}")
                    logger.info(
                        f"Attacker fork total difficulty: {self.attackfork.total_diff}"
                    )

                if len(self.honestfork.blocks) > 0:
                    latest_honest_block = self.honestfork.blocks[-1]
                    logger.info(f"Latest honest block: {latest_honest_block}")
                    logger.info(
                        f"Honest fork total difficulty: {self.honestfork.total_diff}"
                    )

                logger.info(
                    f"Rewards - Attacker: {attacker_instant_gain}, Honest: {honest_instant_gain}"
                )
                logger.info(
                    f"Accumulated rewards - Attacker: {self._attacker_gain}, Honest: {self._honest_gain}"
                )
                logger.info(f"Next state: {self._current_state}")
            if self._visualize:
                self.visualizer.add_transition(
                    current_state, (actionType, actionArg), self._current_state
                )

        return self._current_state, reward, terminated, truncated, {}

    @property
    def reward_fraction(self):
        """Fraction of total rewards earned by attacker"""
        return self._attacker_gain / (self._attacker_gain + self._honest_gain)

    @property
    def reward_time_avg(self):
        """Time-averaged reward for attacker"""
        return (
            (self._attacker_gain * (1 - self._relative_p))
            * 600000000
            / (self.honestfork._absolute_timestamp + 10e-6)
        )

    def uncle_info(self):
        """Get statistics about uncle blocks"""
        aa_ratio = self._aa_num / self._attacker_block
        if self._aa_num == 0:
            aa_distance = 0
        else:
            aa_distance = self._aa_distance / self._aa_num
        ha_ratio = self._ha_num / self._honest_block
        if self._ha_num == 0:
            ha_distance = 0
        else:
            ha_distance = self._ha_distance / self._ha_num

        logger.info("\n=== Uncle Block Statistics ===")
        logger.info(f"Attacker-Attacker ratio: {aa_ratio}")
        logger.info(f"Attacker-Attacker distance: {aa_distance}")
        logger.info(f"Honest-Attacker ratio: {ha_ratio}")
        logger.info(f"Honest-Attacker distance: {ha_distance}")

        return aa_ratio, aa_distance, ha_ratio, ha_distance
