import gymnasium as gym
from gymnasium import spaces
import numpy as np
from _logger import setup_logging
from _random_process import (
    alpha_random_process,
    real_alpha_process,
    random_normal_trunc,
)
from _chain import Block, Chain

logger = setup_logging("unified_env", "critical")


class Ethereum(gym.Env):
    """
    Gymnasium environment for Ethereum mining simulation.

    This environment simulates the selfish mining scenario in Ethereum's PoW chain.

    State space: (a, b, fork, advantage, d1...d6) where:
    - a: Number of blocks in attacker's fork
    - b: Number of blocks in honest chain
    - fork: Fork status
    - advantage: Advantage of attacker over honest chain
    - d1...d6: Additional state information
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        max_hidden_block,
        attacker_fraction,
        follower_fraction,
        relative_p=0,
        dev=0,
        random_interval=(0, 1),
        frequency=1,
        know_alpha=False,
        random_process="iid",
        is_random_process=False,
        render_mode=None,
    ):
        super().__init__()

        # Save parameters
        self._max_hidden_block = max_hidden_block
        self._alpha = attacker_fraction
        self._gamma = follower_fraction
        self._know_alpha = know_alpha

        # Define action space
        self.action_space = spaces.Discrete(4)  # Adopt, Match, Mine, Override

        # Define observation space
        self.observation_space = spaces.Tuple(
            (
                spaces.Discrete(
                    max_hidden_block + 2
                ),  # lenAttackerForking (0 to max_hidden_block + 1)
                spaces.Discrete(
                    max_hidden_block + 2
                ),  # lenHonestForking (0 to max_hidden_block + 1)
                spaces.Discrete(2),  # forkingStatus (0 or 1)
                spaces.Discrete(3),  # advantage (0, 1, or 2)
                spaces.Discrete(
                    max_hidden_block + 1
                ),  # special_block (0 to max_hidden_block)
                *[spaces.Discrete(3) for _ in range(6)],  # uncle (0, 1, or 2)
            )
        )

        if self._know_alpha:
            self.observation_space = spaces.Tuple(
                self.observation_space.spaces + (spaces.Box(low=0, high=1, shape=(1,)),)
            )

        # Initialize other parameters
        self.render_mode = render_mode
        self._setup_environment(
            relative_p,
            dev,
            random_interval,
            frequency,
            random_process,
            is_random_process,
        )

    def SM_theoratical_gain(self, a, gamma):
        # Calculate theoretical gain based on selfish mining formula
        rate = (
            a * (1 - a) * (1 - a) * (4.0 * a + gamma * (1 - 2 * a)) - np.power(a, 3)
        ) / (1 - a * (1 + (2 - a) * a))
        return rate

    def _setup_environment(
        self,
        relative_p,
        dev,
        random_interval,
        frequency,
        random_process,
        is_random_process,
    ):
        """Initialize environment parameters and state.

        Args:
            relative_p (float): Relative mining power parameter
            dev (float): Standard deviation for random process
            random_interval (tuple): Range for random values (min, max)
            frequency (int): Update frequency for random process
            random_process (str): Type of random process ("iid" or "brown")
            is_random_process (bool): Whether to use random process
        """
        logger.info("Setting up environment parameters")

        # Initialize blockchain components
        self.CRITERIA = 1
        self.REMAINING_BLOCKS = 1
        self.adj = 1

        # Initialize chains
        self.chain = Chain(0, 13, 1, self.adj, name="mainchain")
        self.attackfork = Chain(0, 13, self._alpha, self.adj, name="attackfork")
        self.honestfork = Chain(0, 13, 1 - self._alpha, self.adj, name="honestfork")

        # Initialize forks with no blocks
        self.honestfork.adjust(0)
        self.honestfork.set_external_info(self.chain.blocks[-1].difficulty)
        self.attackfork.adjust(0)
        self.attackfork.set_external_info(self.chain.blocks[-1].difficulty)

        # Set relative mining power
        if relative_p == 0:
            self._relative_p = max(
                self._alpha, self.SM_theoratical_gain(self._alpha, self._gamma) * 1.05
            )
        else:
            self._relative_p = relative_p

        # Initialize random process
        self._current_alpha = self._alpha
        self._random_process = alpha_random_process(
            self._alpha, dev, random_interval, random_process
        )

        # Set additional parameters
        self._dev = dev
        self._random_interval = random_interval
        self._frequency = frequency
        self._is_random_process = is_random_process

        # Initialize mining statistics
        self._accumulated_steps = 0
        self._attacker_gain = 0
        self._honest_gain = 0
        self._attacker_block = 0
        self._honest_block = 0
        self._aa_num = 0
        self._aa_distance = 0
        self._ha_num = 0
        self._ha_distance = 0
        self._special_block = 0

        # Calculate expected alpha
        self._expected_alpha = self._calculate_expected_alpha(
            self._alpha, dev, random_interval
        )

        logger.info("Environment setup complete")

    def _calculate_expected_alpha(self, alpha, dev, random_interval):
        """Calculate expected alpha value through Monte Carlo sampling."""
        rept = 1000000
        alpha_sum = 0.0
        for _ in range(rept):
            a = random_normal_trunc(alpha, dev, random_interval[0], random_interval[1])
            alpha_sum += a
        return alpha_sum / rept

    def reset(self, seed=None, options=None):
        """Reset environment to initial state.

        Args:
            seed (Optional[int]): Random seed for reproducibility
            options (Optional[dict]): Additional options for reset

        Returns:
            tuple[ObsType, dict[str, Any]]: Initial observation and info dict
        """
        super().reset(seed=seed)
        logger.info("Resetting environment")

        # Reset random process and steps
        self._random_process.reset()
        self._accumulated_steps = 0

        # Reset state
        self._current_state = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        self._current_alpha = self._alpha

        if self._know_alpha:
            self._current_state = self._current_state + (self._current_alpha,)

        # Reset gains
        self._honest_gain = 0
        self._attacker_gain = 0
        self._special_block = 0

        # Reset history info
        self._attacker_block = 0
        self._honest_block = 0
        self._aa_num = 0
        self._aa_distance = 0
        self._ha_num = 0
        self._ha_distance = 0

        # Reset blockchain components
        self.chain.reset()
        self.attackfork.reset()
        self.honestfork.reset()
        self.honestfork.adjust(0)
        self.honestfork.set_external_info(self.chain.blocks[-1].difficulty)
        self.attackfork.adjust(0)
        self.attackfork.set_external_info(self.chain.blocks[-1].difficulty)

        logger.info(f"Environment reset complete. Initial state: {self._current_state}")

        # Return initial observation and empty info dict
        return self._current_state, {}

    def step(self, action):
        """Execute one step in the environment.

        Args:
            action (int): Action to take (0: adopt, 1: override, 2: wait/mine, 3: match)

        Returns:
            tuple[ObsType, float, bool, bool, dict[str, Any]]: Tuple containing:
                - observation: Current state
                - reward: Reward from action
                - terminated: Whether episode ended naturally
                - truncated: Whether episode was artificially terminated
                - info: Additional information
        """
        # Validate action
        if not self.action_space.contains(action):
            logger.warning(f"Invalid action {action}, defaulting to wait/mine (2)")
            action = 2

        # Get current state components
        current_state = self._current_state
        a, b, status = current_state[0:3]
        alpha = self._current_alpha
        gamma = self._gamma

        # Take step in environment
        next_state, reward, reset_flag = self._unmapped_step(
            current_state, action, move=True
        )

        # Prepare info dictionary
        info = {
            "attacker_gain": self._attacker_gain,
            "honest_gain": self._honest_gain,
            "attacker_blocks": self._attacker_block,
            "honest_blocks": self._honest_block,
            "uncle_stats": {
                "aa_ratio": self._aa_num / (self._attacker_block + 1e-10),
                "aa_distance": self._aa_distance / (self._aa_num + 1e-10),
                "ha_ratio": self._ha_num / (self._honest_block + 1e-10),
                "ha_distance": self._ha_distance / (self._ha_num + 1e-10),
            },
            "chain_state": {
                "attacker_fork": len(self.attackfork.blocks),
                "honest_fork": len(self.honestfork.blocks),
                "attacker_difficulty": self.attackfork.total_diff,
                "honest_difficulty": self.honestfork.total_diff,
            },
        }

        # Check if episode should end
        terminated = False
        truncated = reset_flag

        # Log step information
        logger.info(f"Step - Action: {action}, Reward: {reward}")
        logger.info(f"State transition: {current_state} -> {next_state}")

        return next_state, reward, terminated, truncated, info

    def _unmapped_step(self, state, action, move=True):
        """Internal step function that handles state transitions.

        Args:
            state (tuple): Current state
            action (int): Action to take
            move (bool): Whether to actually execute the move

        Returns:
            tuple: (next_state, reward, reset_flag)
        """
        # Initialize variables
        lenAttackerForking, lenHonestForking, forkingStatus = state[0:3]
        advantage = state[3]

        logger.debug("\n=== Step Start ===")
        logger.debug(f"Action: {action}")
        logger.info(f"Current alpha: {self._current_alpha}, Advantage: {advantage}")

        # Initialize next state variables
        next_lenAttackerForking = lenAttackerForking
        next_lenHonestForking = lenHonestForking
        next_forkingStatus = forkingStatus
        next_advantage = advantage

        # Initialize reward components
        legal = False
        attacker_get = honest_get = 0
        attacker_uncle = honest_uncle = 0
        attacker_nephew = honest_nephew = 0

        # Process forced actions first
        if self._should_force_override(lenAttackerForking, lenHonestForking):
            next_state, reward, reset_flag = self._handle_forced_override(
                action, lenAttackerForking, lenHonestForking
            )
            if reward > -100:  # Valid move
                return next_state, reward, reset_flag

        elif self._should_force_adopt(lenAttackerForking, lenHonestForking):
            next_state, reward, reset_flag = self._handle_forced_adopt(
                action, lenAttackerForking, lenHonestForking
            )
            if reward > -100:  # Valid move
                return next_state, reward, reset_flag

        # Process normal actions
        if forkingStatus == 0:  # Normal fork status
            next_state, reward, reset_flag, legal = self._handle_normal_fork(
                action, state, move
            )
        elif forkingStatus == 1:  # Match fork status
            next_state, reward, reset_flag, legal = self._handle_match_fork(
                action, state, move
            )
        else:
            raise ValueError(f"Invalid forking status: {forkingStatus}")

        if not legal:
            return state, -1000000, False

        # Update steps and check for episode end
        if move:
            self._accumulated_steps += 1
            if self._accumulated_steps % self._frequency == 0:
                if self._is_random_process:
                    self._current_alpha = self._random_process.next()

        reset_flag = self._accumulated_steps > 100000

        return next_state, reward, reset_flag

    def _should_force_override(self, lenAttackerForking, lenHonestForking):
        """Check if attacker must override."""
        return (
            lenAttackerForking == self._max_hidden_block + 1
            and lenHonestForking < lenAttackerForking
        )

    def _should_force_adopt(self, lenAttackerForking, lenHonestForking):
        """Check if attacker must adopt."""
        return (
            lenHonestForking == self._max_hidden_block + 1
            and lenAttackerForking < lenHonestForking
        )

    def render(self):
        """Render current state"""
        # Implementation will follow in next step
        pass

    def close(self):
        """Clean up environment resources"""
        pass

    def _handle_normal_fork(self, action, state, move):
        """Handle actions in normal fork status."""
        lenAttackerForking, lenHonestForking, forkingStatus = state[0:3]
        advantage = state[3]

        if action == 0:  # Adopt
            return self._handle_adopt_action(state, move)
        elif action == 1:  # Match
            return self._handle_match_action(state)
        elif action == 2:  # Mine
            return self._handle_mine_action(state, move)
        elif action == 3:  # Override
            return self._handle_override_action(state)
        else:
            raise ValueError(f"Invalid action: {action}")

    def _handle_adopt_action(self, state, move):
        """Handle adopt action."""
        lenAttackerForking, lenHonestForking, forkingStatus = state[0:3]
        advantage = state[3]
        continueToAttack = min(lenHonestForking, self.REMAINING_BLOCKS)

        legal = True
        honest_get = lenHonestForking - continueToAttack

        next_lenAttackerForking = 0
        next_lenHonestForking = continueToAttack

        if move:
            blocks = self.honestfork._extract_blocks(
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

        next_state = self._create_next_state(
            next_lenAttackerForking,
            next_lenHonestForking,
            forkingStatus,
            advantage,
            state,
        )
        reward = self._calculate_reward(0, honest_get)

        return next_state, reward, False, legal

    def _handle_match_action(self, state):
        """Handle match action."""
        lenAttackerForking, lenHonestForking, forkingStatus = state[0:3]
        advantage = state[3]

        if lenAttackerForking == lenHonestForking and advantage == 1:
            legal = True
            next_state = self._create_next_state(
                lenAttackerForking,
                lenHonestForking,
                1,  # Change to match status
                0,  # Reset advantage
                state,
            )
            return next_state, 0, False, legal

        return state, -1000000, False, False

    def _handle_mine_action(self, state, move):
        """Handle mine action."""
        lenAttackerForking, lenHonestForking, forkingStatus = state[0:3]
        advantage = state[3]

        legal = True
        attacker_time = self.attackfork.mine_block()
        honest_time = self.honestfork.mine_block()

        # Attacker mines first
        if attacker_time < honest_time:
            return self._handle_attacker_mines_first(
                state,
                attacker_time,
                lenAttackerForking,
                lenHonestForking,
                forkingStatus,
                advantage,
            )
        # Honest miners mine first
        else:
            return self._handle_honest_mines_first(
                state,
                honest_time,
                lenAttackerForking,
                lenHonestForking,
                forkingStatus,
                advantage,
            )

    def _handle_attacker_mines_first(
        self,
        state,
        attacker_time,
        lenAttackerForking,
        lenHonestForking,
        forkingStatus,
        advantage,
    ):
        """Handle case where attacker mines first."""
        new_block = Block("attacker", attacker_time)
        logger.debug(f"Adding block to attackfork: {new_block}")
        self.attackfork.add_block(new_block, self.chain)

        next_lenAttackerForking = lenAttackerForking + 1
        next_lenHonestForking = lenHonestForking
        next_forkingStatus = forkingStatus

        # Check if attacker gains advantage
        next_advantage = self._calculate_advantage(
            lenHonestForking - lenAttackerForking,
            self.honestfork.total_diff,
            self.attackfork.total_diff,
            advantage,
        )

        next_state = self._create_next_state(
            next_lenAttackerForking,
            next_lenHonestForking,
            next_forkingStatus,
            next_advantage,
            state,
        )

        return next_state, 0, False, True

    def _handle_honest_mines_first(
        self,
        state,
        honest_time,
        lenAttackerForking,
        lenHonestForking,
        forkingStatus,
        advantage,
    ):
        """Handle case where honest miners mine first."""
        new_block = Block("honest", honest_time)
        logger.debug(f"Adding block to honestfork: {new_block}")
        self.honestfork.add_block(new_block, self.chain)

        next_lenAttackerForking = lenAttackerForking
        next_lenHonestForking = lenHonestForking + 1
        next_forkingStatus = forkingStatus

        # Check if honest miners gain advantage
        next_advantage = self._calculate_advantage(
            lenAttackerForking - lenHonestForking,
            self.honestfork.total_diff,
            self.attackfork.total_diff,
            advantage,
        )

        next_state = self._create_next_state(
            next_lenAttackerForking,
            next_lenHonestForking,
            next_forkingStatus,
            next_advantage,
            state,
        )

        return next_state, 0, False, True

    def _calculate_advantage(
        self, length_diff, honest_diff, attacker_diff, current_advantage
    ):
        """Calculate advantage based on fork lengths and difficulties."""
        if length_diff <= self.CRITERIA and length_diff > 0:
            if honest_diff > attacker_diff:
                return current_advantage
            elif honest_diff == attacker_diff:
                return 1
            else:
                return 2
        return current_advantage

    def _handle_override_action(self, state):
        """Handle override action."""
        lenAttackerForking, lenHonestForking, forkingStatus = state[0:3]
        advantage = state[3]

        # Case 1: Invalid override attempt
        if lenAttackerForking < lenHonestForking - self.CRITERIA - 1:
            logger.warning(f"Invalid override: {lenAttackerForking, lenHonestForking}")
            return state, -1000000, False, False

        # Case 2: Override with shorter or equal fork
        if lenAttackerForking <= lenHonestForking:
            if advantage != 2:
                logger.warning(f"Invalid override without advantage: {advantage}")
                return state, -1000000, False, False

            legal = True
            attacker_get = lenAttackerForking
            next_state, reward = self._process_override(lenAttackerForking, 0, 0, state)
            return next_state, reward, False, legal

        # Case 3: Override with longer fork
        legal = True
        attacker_get = lenHonestForking + 1
        next_state, reward = self._process_override(
            lenHonestForking + 1, lenAttackerForking - lenHonestForking - 1, 0, state
        )
        return next_state, reward, False, legal

    def _process_override(
        self, blocks_to_extract, next_attacker_len, next_honest_len, state
    ):
        """Process override action by extracting and adding blocks."""
        blocks = self.attackfork._extract_blocks(blocks_to_extract)
        for block in blocks:
            self.chain.add_block(block)

        self.attackfork.adjust(next_attacker_len)
        self.honestfork.adjust(next_honest_len)

        if next_attacker_len == 0:
            self.attackfork.set_external_info(self.chain.blocks[-1].difficulty)
        if next_honest_len == 0:
            self.honestfork.set_external_info(self.chain.blocks[-1].difficulty)

        next_state = self._create_next_state(
            next_attacker_len,
            next_honest_len,
            0,  # Reset fork status
            0,  # Reset advantage
            state,
        )

        reward = self._calculate_reward(blocks_to_extract, 0)
        return next_state, reward

    def _calculate_reward(self, attacker_get, honest_get, uncle_info=None):
        """Calculate reward based on blocks gained and uncle blocks."""
        if uncle_info is None:
            uncle_info = self._process_uncle_blocks(attacker_get, honest_get)

        attacker_uncle = uncle_info.get("attacker_uncle", 0)
        attacker_nephew = uncle_info.get("attacker_nephew", 0)
        honest_uncle = uncle_info.get("honest_uncle", 0)
        honest_nephew = uncle_info.get("honest_nephew", 0)

        attacker_instant_gain = attacker_get + attacker_uncle + attacker_nephew / 32.0
        honest_instant_gain = honest_get + honest_uncle + honest_nephew / 32.0

        return (
            attacker_instant_gain * (1 - self._relative_p)
            - honest_instant_gain * self._relative_p
        )
