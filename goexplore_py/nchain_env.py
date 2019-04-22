
import gym

class NChainPos:
	__slots__ = ['state', 'tuple']

	def __init__(self, state):
		self.state = state

		self.set_tuple()

	def set_tuple(self):
		self.tuple = (self.state,)

	def __hash__(self):
		return hash(self.tuple)

	def __eq__(self, other):
		if not isinstance(other, NChainPos):
			return False
		return self.tuple == other.tuple

	def __getstate__(self):
		return self.tuple

	def __setstate__(self, d):
		self.state = d[0]
		self.tuple = d

	def __repr__(self):
		return f'State={self.state}'

class MyNChain:
	def __init__(self, N):
		self.env = gym.make("NChain-v0")
		self.env.env.n = N
		self.env.unwrapped.slip = 0
		self.state = None
		self.pos = None
		self.cur_steps = 0
		self.cur_score = 0
		self.rooms = []
		self.level = 0

	def reset(self):
		self.state = self.env.reset()
		self.cur_steps = 0
		self.cur_score = 0
		self.pos = NChainPos(self.state)
		return self.state

	def step(self, action):
		self.state, reward, done, info = self.env.step(action)
		self.cur_steps += 1
		self.cur_score += reward
		self.pos = NChainPos(self.state)
		return self.state, reward, done, info

	def get_pos(self):
		return self.pos

	def get_restore(self):
		return (self.state, self.pos, self.cur_steps, self.cur_score)

	def restore(self, data):
		self.state, self.pos, self.cur_steps, self.cur_score = data
		self.env.env.state = self.state
		return self.state

	@staticmethod
	def make_pos(pos):
		return NChainPos(pos.state)