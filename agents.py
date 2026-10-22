"""
agents.py -- the "brains" that can play the game.

RandomAgent       : moves at random (week 1, our baseline)
QLearningAgent    : learns from points using Q-learning (week 2)

Q-LEARNING IN ONE PICTURE
    The robot keeps a notebook (the "Q-table"): one row for every situation
    it can be in (50 situations), one column for every move (4 moves).
    Each cell says: "how many points do I expect from this move, in this
    situation, from now until the game ends?"
    After every move the robot corrects one cell using what really happened.
    After thousands of games the notebook becomes a good strategy.
"""
import os
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def model_path(level):
    """Where the trained notebook of a level is saved."""
    return os.path.join(MODEL_DIR, f"level{level}_q.npy")


class RandomAgent:
    name = "Random robot (not trained)"

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        return self.action_space.sample()


class QLearningAgent:
    name = "Q-learning robot (trained)"

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95):
        """
        alpha : learning speed. How much one new experience changes the notebook.
        gamma : how much the robot cares about FUTURE points (0 = only now).
        """
        self.q = np.zeros((n_states, n_actions))   # the notebook, starts empty (all 0)
        self.alpha = alpha
        self.gamma = gamma
        self.n_actions = n_actions

    def act(self, obs, epsilon=0.0):
        """Choose a move. With probability epsilon, explore (random move);
        otherwise pick the move with the best score in the notebook."""
        if np.random.random() < epsilon:
            return int(np.random.randint(self.n_actions))
        row = self.q[obs]
        best = np.flatnonzero(row == row.max())     # all moves tied for best...
        return int(np.random.choice(best))          # ...pick one of them at random

    def learn(self, obs, action, reward, next_obs, terminated):
        """THE Q-LEARNING UPDATE: correct one cell of the notebook.
        target   = the points I just got + the best I expect afterwards
        new cell = old cell moved a little (alpha) toward the target"""
        future = 0.0 if terminated else self.gamma * self.q[next_obs].max()
        target = reward + future
        self.q[obs, action] += self.alpha * (target - self.q[obs, action])

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.q)

    @classmethod
    def load(cls, path):
        q = np.load(path)
        agent = cls(q.shape[0], q.shape[1])
        agent.q = q
        return agent


def get_agent(level, env, trained=True):
    """Return the trained robot if a saved notebook exists, else the random one."""
    if trained and os.path.exists(model_path(level)):
        return QLearningAgent.load(model_path(level))
    return RandomAgent(env.action_space)
