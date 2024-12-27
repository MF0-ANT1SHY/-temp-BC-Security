from _logger import setup_logging
import numpy as np

logger = setup_logging("chain", "critical")


class Block:
    def __init__(self, owner, timestamp, diff=0):
        self.owner = owner  # Can be "honest" or "attacker"
        self.timestamp = timestamp
        self.difficulty = diff
        self.nonce = 0  # Block sequence number

    # Initialize chain with compute power, starting from a difficulty-adjusted interval
    @staticmethod
    def create_genesis_block(timestamp, diff):
        return Block("honest", timestamp, diff)

    def set_difficulty(self, diff):
        self.difficulty = diff

    def __repr__(self):
        return f"Block({self.owner}, {self.timestamp}, {self.difficulty}, {self.nonce})"


class Chain:
    def __init__(
        self,
        timestamp,
        target_time=13,
        compute_power=1,
        adjustment_interval=1,
        name="default",
        diff=2**34,
    ) -> None:
        logger.info(
            f"Initializing Chain {name} with target_time={target_time}, compute_power={compute_power}, diff={diff}"
        )
        # Concrete compute power = difficulty / target time
        self.concreteCP = diff / target_time
        self.lastinterval = 0
        self.otimestamp = timestamp
        self.otarget_time = target_time
        self.ocompute_power = compute_power
        self.oadjustment_interval = adjustment_interval
        self.diff = diff
        self.name = name
        self._newestNonce = 0
        self.reset()

    @property
    def newestNonce(self):
        # Latest nonce (sequence number) of blocks mined on this fork
        if len(self.blocks) > 0:
            self._newestNonce = max(self.blocks[-1].nonce, self._newestNonce)
        return self._newestNonce

    def reset(self):
        self.lastinterval = 0
        self.target_time = self.otarget_time
        self.adjustment_interval = self.oadjustment_interval
        self.compute_power = self.ocompute_power
        self.blocks = [Block.create_genesis_block(self.otimestamp, self.diff)]
        # Allowed timestamp range for blocks
        self.timestamp_interval = [0, 15]
        # Real timestamp, unmodified by attacker
        self._absolute_timestamp = 0
        self._external_diff = self.ocompute_power * self.otarget_time

    def set_external_info(self, diff):
        self._external_diff = diff

    @property
    def current_difficulty(self):
        if len(self.blocks) == 0:
            return self._external_diff
        return self.blocks[-1].difficulty

    @property
    def previous_difficulty(self):
        if len(self.blocks) <= 1:
            return self._external_diff
        return self.blocks[-2].difficulty

    @property
    def length(self):
        return len(self.blocks)

    @property
    def total_diff(self):
        # Calculate total difficulty of all blocks in chain
        if len(self.blocks) == 0:
            return
        diff = 0
        for block in self.blocks:
            diff += block.difficulty
        return diff

    @property
    def total_timestamp(self):
        if len(self.blocks) == 0:
            return
        start_time = int(self.blocks[0].timestamp)
        end_time = int(self.blocks[-1].timestamp)
        time = end_time - start_time
        return time

    def modify_interval_timestamp(self, timestamp_gap):
        # Ethereum doesn't enforce strict timestamp limits for DAA intervals
        return timestamp_gap

    # Difficulty adjustment for Bitcoin
    def set_difficulty(self, mainchain=None, block=None, u=0):
        d = -1
        hasUncle = u  # Whether block has uncle blocks
        t = -1  # Timestamp gap
        diff = -1

        if len(self.blocks) == 0:
            d = self._external_diff
        else:
            d = self.blocks[-1].difficulty

        t = block.timestamp

        # Calculate difficulty adjustment based on timestamp and uncle presence
        inner_max = max(1 + hasUncle - t // 9, -99)
        diff = d + inner_max * (d // 2048)

        return diff

    def mine_block(self, adjust_power=0, uncle=False):
        logger.debug(f"Mining block on {self.name} with adjust_power={adjust_power}")
        cp_factor: int = self.compute_power + adjust_power
        cp = cp_factor * self.concreteCP
        mean_block_time = self.current_difficulty / cp
        time_interval = np.random.exponential(mean_block_time)
        logger.debug(f"Block mined with time_interval={time_interval}")
        return time_interval

    def add_block(self, block, mainchain=None, u=0):
        logger.debug(f"=== Block Addition Start ===")
        logger.debug(f"Chain: {self.name}")
        logger.debug(f"Block to add: {block}")
        logger.debug(f"Has mainchain reference: {mainchain is not None}")

        if mainchain is not None:
            old_diff = block.difficulty
            block.difficulty = self.set_difficulty(mainchain, block, u)
            logger.debug(f"Difficulty changed from {old_diff} to {block.difficulty}")
        else:
            logger.debug("No mainchain reference - keeping original difficulty")
            pass

        # get next nonce, if self.blocks is empty, use mainchain's last nonce, use self.blocks[-1].nonce if self.blocks is not empty
        if len(self.blocks) == 0 and mainchain is not None:
            block.nonce = mainchain.blocks[-1].nonce + 1
        elif len(self.blocks) > 0:
            block.nonce = self.blocks[-1].nonce + 1
        else:
            raise Exception("No blocks exist in either chain")
        logger.debug(f"after nonce: {block.nonce}")
        logger.debug(f"Block nonce set to {block.nonce}")

        self.blocks.append(block)
        logger.debug(
            f"Chain {self.name} length: {len(self.blocks)}, total difficulty: {self.total_diff}"
        )
        logger.debug(f"=== Block Addition Complete ===")
        return True

    def _extract_blocks(self, num):
        if num > len(self.blocks):
            logger.error(
                f"Not enough blocks in chain: {len(self.blocks)} blocks available, requested {num} blocks"
            )
            raise Exception("Not enough blocks in chain")
        return self.blocks[-num:]

    def adjust(self, num=0, blocks=None, mainchain=None):
        if num < 0 or num > self.length:
            return
        else:
            if num > 0:
                # Keep last 'num' blocks
                self.blocks = self.blocks[-num:]
            else:  # num == 0
                self.blocks = []

        if mainchain is not None:
            if blocks is not None and len(blocks) > 0:
                for block in blocks:
                    self.add_block(block, mainchain)
        else:
            if blocks is not None and len(blocks) > 0:
                for block in blocks:
                    self.add_block(block)

    def adjust_end(self, num=0, blocks=None, mainchain=None):
        if num < 0 or num > self.length:
            return
        else:
            if num > 0:
                # Keep first 'num' blocks
                self.blocks = self.blocks[:num]
            else:  # num == 0
                self.blocks = []

        if mainchain is not None:
            if blocks is not None and len(blocks) > 0:
                for block in blocks:
                    self.add_block(block, mainchain)
        else:
            if blocks is not None and len(blocks) > 0:
                for block in blocks:
                    self.add_block(block)

    def _print_block(self):
        for block in self.blocks:
            print(block)
