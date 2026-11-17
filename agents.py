"""
agents.py -- the "brains" that can play the game.

RandomAgent    : moves at random (our baseline)
QLearningAgent : Q-learning (week 2)
SarsaAgent     : SARSA, the second algorithm we compare against (week 3)

BOTH learning methods keep a "notebook" (the Q-table): one row per situation,
one column per move, each cell = "how many points do I expect from this move
here, until the game ends?". They differ in HOW they correct a cell:

  Q-learning : "...assuming that from the next situation I will play the BEST move."
  SARSA      : "...assuming that from the next situation I will play the move I
                ACTUALLY choose (which can be a random exploring move)."

So SARSA takes its own exploring mistakes into account, Q-learning ignores them.
"""
import os
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
ALGOS = ["qlearning", "sarsa"]


def model_path(level, algo):
    """Where the trained notebook of an algorithm on a level is saved."""
    return os.path.join(MODEL_DIR, f"level{level}_{algo}.npy")


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
        """Choose a move. With probability epsilon explore (random move);
        otherwise pick the move with the best score in the notebook."""
        if np.random.random() < epsilon:
            return int(np.random.randint(self.n_actions))
        row = self.q[obs]
        best = np.flatnonzero(row == row.max())     # all moves tied for best...
        return int(np.random.choice(best))          # ...pick one of them at random

    def learn(self, obs, action, reward, next_obs, terminated):
        """Q-LEARNING UPDATE. target = points now + the BEST I can expect next."""
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


class SarsaAgent(QLearningAgent):
    name = "SARSA robot (trained)"

    def learn(self, obs, action, reward, next_obs, next_action, terminated):
        """SARSA UPDATE. target = points now + what I expect from the move I
        will REALLY make next (next_action), not the best one.
        The name = State, Action, Reward, next State, next Action."""
        future = 0.0 if terminated else self.gamma * self.q[next_obs, next_action]
        target = reward + future
        self.q[obs, action] += self.alpha * (target - self.q[obs, action])


def load_agents(level, env):
    """All brains available for a level, in the order the T key cycles them:
    trained Q-learning, trained SARSA (if their saved notebooks exist), random."""
    agents = {}
    for algo, cls in (("qlearning", QLearningAgent), ("sarsa", SarsaAgent)):
        path = model_path(level, algo)
        if os.path.exists(path):
            agent = cls.load(path)
            agents[algo] = agent
    agents["random"] = RandomAgent(env.action_space)
    return agents
