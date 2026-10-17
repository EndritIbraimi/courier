"""agents.py -- the "brains" that can play the game.

Week 1: only a random brain exists. From week 2 we add a learning brain here
and get_agent() will return it, so the Watch-the-AI mode upgrades itself.
"""


class RandomAgent:
    name = "Random robot (not trained yet)"

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        """Look at the situation (obs) and return an action. Random ignores it."""
        return self.action_space.sample()


def get_agent(level, env):
    """Return the best available brain for this level."""
    return RandomAgent(env.action_space)
