"""
courier_env.py -- the GAME RULES of Courier (no graphics in this file).

A delivery robot lives on a grid. It must walk to the package, pick it up and
carry it to the goal. This file follows the standard Gymnasium format so a
learning algorithm can plug in without changing the game.

THE LOOP every reinforcement-learning project uses:
    reset()       -> start a new game, get what the robot "sees"
    step(action)  -> robot moves; the game returns (new situation, reward,
                     game over?). Repeat until the game ends.
"""

import gymnasium as gym
from gymnasium import spaces

# The 4 actions, as numbers
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
# How each action changes (row, column). Row 0 is the TOP of the map.
MOVES = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

# REWARDS: the points. This is the only way the robot learns what is good.
STEP_PENALTY = -1.0      # every move costs 1 point -> the robot learns to hurry
BUMP_PENALTY = -4.0      # extra penalty for hitting the map edge or a wall
PICKUP_REWARD = 10.0     # bonus for picking up the package
DELIVERY_REWARD = 50.0   # big bonus for delivering it (the real goal)
DEAD_PENALTY = -20.0     # extra penalty if the battery runs empty (game over)

# ---------------------------------------------------------------------------
# LEVEL SETTINGS. Each level = map size, positions (row, col), walls, time limit.
# To add a new level, add one entry here (and flip "ready" in levels.py).
# ---------------------------------------------------------------------------
LEVEL_CONFIG = {
    # Level 1: small empty map
    1: dict(size=5, package=(0, 3), goal=(4, 1), walls=[], max_steps=50),
    # Level 2 (week 3): bigger 7x7 map with walls the robot must go around
    2: dict(size=7, package=(0, 5), goal=(6, 1), max_steps=80,
            walls=[(1, 3), (2, 3), (3, 3), (4, 3),     # a long wall in the middle
                   (5, 1), (5, 2),                     # a wall protecting the goal
                   (2, 5), (3, 5)]),                   # a wall below the package
    # Level 3 (week 4): battery. Every move uses 1 battery point. Stepping on a
    # charging pad (chargers) refills it. An empty battery ends the game.
    3: dict(size=7, package=(0, 5), goal=(6, 1), max_steps=100,
            walls=[(1, 3), (2, 3), (3, 3), (5, 1), (5, 2)],
            battery=16, chargers=[(4, 5), (2, 1)]),
}


class CourierEnv(gym.Env):
    def __init__(self, level=1, random_start=True):
        """
        level        : which level to play (see LEVEL_CONFIG)
        random_start : True = robot starts on a random square every game
        """
        super().__init__()
        cfg = LEVEL_CONFIG[level]
        self.level = level
        self.size = cfg["size"]
        self.max_steps = cfg["max_steps"]
        self.package_pos = cfg["package"]
        self.goal_pos = cfg["goal"]
        self.walls = set(cfg["walls"])          # squares the robot cannot enter
        self.battery_max = cfg.get("battery")   # None = this level has no battery
        self.chargers = set(cfg.get("chargers", []))   # charging pads
        self.random_start = random_start
        self.default_start = (self.size - 1, self.size - 1)

        self.action_space = spaces.Discrete(4)
        # The robot "sees" ONE number: which square it is on + carrying or not
        # (+ how much battery is left, on levels that have a battery).
        self.n_battery = (self.battery_max + 1) if self.battery_max else 1
        self.observation_space = spaces.Discrete(self.size * self.size * 2 * self.n_battery)

        self.robot_pos = self.default_start
        self.carrying = False
        self.battery = self.battery_max or 0
        self.steps = 0
        self.total_reward = 0.0

    def _get_obs(self):
        row, col = self.robot_pos
        base = (row * self.size + col) * 2 + int(self.carrying)   # square + carrying
        return base * self.n_battery + self.battery                # + battery left

    def reset(self, seed=None, options=None):
        """Start a brand-new game."""
        super().reset(seed=seed)
        if self.random_start:
            # random free square: not a wall, not the package, not the goal
            while True:
                pos = (int(self.np_random.integers(0, self.size)),
                       int(self.np_random.integers(0, self.size)))
                if (pos not in self.walls and pos not in self.chargers
                        and pos not in (self.package_pos, self.goal_pos)):
                    break
            self.robot_pos = pos
        else:
            self.robot_pos = self.default_start
        self.carrying = False
        self.battery = self.battery_max or 0       # always start with a full battery
        self.steps = 0
        self.total_reward = 0.0
        return self._get_obs(), {}

    def step(self, action):
        """The robot makes ONE move. All the rules live here."""
        self.steps += 1
        reward = STEP_PENALTY
        terminated = False
        event = None   # tells the graphics what just happened (for effects)

        # 1) where would the robot end up?
        d_row, d_col = MOVES[int(action)]
        new_pos = (self.robot_pos[0] + d_row, self.robot_pos[1] + d_col)

        # 2) inside the map and not a wall -> move. Otherwise it "bumps":
        #    it stays where it is and loses extra points.
        inside = 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size
        if inside and new_pos not in self.walls:
            self.robot_pos = new_pos
        else:
            reward += BUMP_PENALTY
            event = "bump"

        # 3) battery: every move uses 1 point. A charging pad refills it.
        if self.battery_max:
            self.battery -= 1
            if self.robot_pos in self.chargers:
                if self.battery < self.battery_max - 1:
                    event = "charge"               # tells the graphics to show sparks
                self.battery = self.battery_max

        # 4) package rules (pick-up and delivery are automatic)
        if not self.carrying and self.robot_pos == self.package_pos:
            self.carrying = True
            reward += PICKUP_REWARD
            event = "pickup"
        elif self.carrying and self.robot_pos == self.goal_pos:
            reward += DELIVERY_REWARD
            terminated = True
            event = "deliver"

        # 5) empty battery (and not delivered) -> game over with a penalty
        dead = False
        if self.battery_max and self.battery <= 0 and not terminated:
            dead = True
            terminated = True
            reward += DEAD_PENALTY
            event = "dead"

        # 6) too many moves -> game cut off
        truncated = self.steps >= self.max_steps and not terminated

        self.total_reward += reward
        info = {"steps": self.steps, "total_reward": self.total_reward,
                "delivered": event == "deliver", "dead": dead, "event": event}
        return self._get_obs(), reward, terminated, truncated, info
