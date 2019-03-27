


import tensorflow as tf
import numpy as np

from baselines.ppo2 import ppo2, policies



class PPOExplorer:
	def __init__(self, env,  nexp, lr, lr_decay=1, cl_decay=1, nminibatches=4, n_tr_epochs=4, cliprange=0.1, gamma=0.99, lam=0.95, nenvs=1, policy=policies.CnnPolicy):
		ob_space = env.observation_space
		ac_space = env.action_space

		self.exp = 0
		self.nsteps = nexp
		self.batch = self.nsteps * nenvs
		self.n_mb = nminibatches
		self.nbatch_train = self.batch // self.n_mb
		self.mb_obs, self.mb_rewards, self.mb_actions, self.mb_values, self.mb_dones, self.mb_neglogpacs = [],[],[],[],[],[]
		self.lr = lr
		self.lr_decay = lr_decay
		self.cl_decay = cl_decay
		self.states = None
		self.done = [False for _ in range(nenvs)]
		self.gamma = gamma
		self.lam = lam
		self.nenvs = nenvs


		self.n_train_epoch = n_tr_epochs
		self.cliprange = cliprange
		self.model = ppo2.Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs,
								nbatch_train=self.nbatch_train, nsteps=self.nsteps, ent_coef=0.01, vf_coef=1, max_grad_norm=0.5)

		self.obs = np.zeros((nenvs,) + ob_space.shape, dtype=self.model.train_model.X.dtype.name)


	def init_seed(self):
		# self.exp = 0
		# self.mb_obs, self.mb_rewards, self.mb_actions, self.mb_values, self.mb_dones, self.mb_neglogpacs = [],[],[],[],[],[]
		pass

	def init_trajectory(self, arg=None, arg2=None):
		pass

	def seen_state(self, e):
		self.exp += 1
		self.obs[:] = e.obs
		self.mb_rewards.append(e.reward)
		self.done = e.done

		if self.exp >= self.nsteps:
			self.mb_obs = np.asarray(self.mb_obs, dtype=self.obs.dtype).squeeze()
			self.mb_rewards = np.asarray(self.mb_rewards, dtype=np.float32).squeeze()
			self.mb_actions = np.asarray(self.mb_actions).squeeze()
			self.mb_values = np.asarray(self.mb_values, dtype=np.float32).squeeze()
			self.mb_neglogpacs = np.asarray(self.mb_neglogpacs, dtype=np.float32).squeeze()
			self.mb_dones = np.asarray(self.mb_dones, dtype=np.bool).squeeze()



			# From baselines' ppo2 runner():
			# Calculate returns

			last_values = self.model.value(self.obs)

			mb_returns = np.zeros_like(self.mb_rewards)
			mb_advs = np.zeros_like(self.mb_rewards)
			lastgaelam = 0
			for t in reversed(range(self.nsteps)):
				if t == self.nsteps - 1:
					nextnonterminal = 1.0 - self.done
					nextvalues = last_values
				else:
					nextnonterminal = 1.0 - self.mb_dones[t + 1]
					nextvalues = self.mb_values[t + 1]
				delta = self.mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - self.mb_values[t]
				mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam

			mb_returns[:] = mb_advs + self.mb_values
			# Swap and flatten axis 0 and 1
			if self.nenvs > 1:
				self.mb_obs, mb_returns, self.mb_dones, self.mb_actions, self.mb_values, self.mb_neglogpacs = \
					map(ppo2.sf01, (self.mb_obs, mb_returns, self.mb_dones, self.mb_actions, self.mb_values, self.mb_neglogpacs))


			#train model for multiple epoch in n minibacthes pr epoch
			#From baselines' ppo2 learn()
			inds = np.arange(self.batch)
			for _ in range(self.n_train_epoch):
				np.random.shuffle(inds)
				for start in range(0, self.batch, self.nbatch_train):
					end = start + self.nbatch_train
					mbinds = inds[start:end]
					slices = (arr[mbinds] for arr in (self.mb_obs, mb_returns, self.mb_dones, self.mb_actions, self.mb_values, self.mb_neglogpacs))
					self.model.train(self.lr, self.cliprange, *slices)





			self.lr *= self.lr_decay
			self.cliprange *=  self.cl_decay
			self.exp = 0
			self.mb_obs, self.mb_rewards, self.mb_actions, self.mb_values, self.mb_dones, self.mb_neglogpacs = [], [], [], [], [], []

	def get_action(self, state, env):

		self.obs[:] = state

		actions, values, self.states, neglogpacs = self.model.step(self.obs)

		self.mb_obs.append(self.obs.copy())
		self.mb_actions.append(actions)
		self.mb_values.append(values)
		self.mb_neglogpacs.append(neglogpacs)
		self.mb_dones.append(self.done)

		return actions

class PPOExplorer_v2:
	def __init__(self, env, actors,  nexp, lr, lr_decay=1, cl_decay=1, nminibatches=4, n_tr_epochs=4, cliprange=0.1, gamma=0.99, lam=0.95, policy=policies.CnnPolicy):
		ob_space = env.observation_space
		ac_space = env.action_space

		self.nacts = actors
		self.actor = 0

		self.exp = 0
		self.nsteps = nexp
		self.batch = self.nsteps * actors
		self.n_mb = nminibatches
		self.nbatch_train = self.batch // self.n_mb
		self.mb_obs, self.mb_rewards, self.mb_actions, self.mb_values, self.mb_dones, self.mb_neglogpacs = [[]],[[]],[[]],[[]],[[]],[[]]
		self.lr = lr
		self.lr_decay = lr_decay
		self.cl_decay = cl_decay
		self.states = None
		self.done = [False for _ in range(1)]
		self.gamma = gamma
		self.lam = lam



		self.n_train_epoch = n_tr_epochs
		self.cliprange = cliprange
		self.model = ppo2.Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=1,
								nbatch_train=self.nbatch_train, nsteps=self.nsteps, ent_coef=0.01, vf_coef=1, max_grad_norm=0.5)

		self.obs = np.zeros((1,) + ob_space.shape, dtype=self.model.train_model.X.dtype.name)


	def init_seed(self):
		# self.exp = 0
		# self.mb_obs, self.mb_rewards, self.mb_actions, self.mb_values, self.mb_dones, self.mb_neglogpacs = [],[],[],[],[],[]
		pass

	def init_trajectory(self, arg=None, arg2=None):

		pass

	def seen_state(self, e):
		self.exp += 1
		self.obs[:] = e.obs
		self.mb_rewards[self.actor].append(e.reward)
		self.done = e.done

		if self.exp >= self.nsteps:
			self.actor += 1
			self.exp = 0



			if self.actor != self.nacts:
				self.mb_obs.append([])
				self.mb_actions.append([])
				self.mb_values.append([])
				self.mb_neglogpacs.append([])
				self.mb_dones.append([])
				self.mb_rewards.append([])
			else:
				self.actor = 0

				self.mb_obs = np.asarray(self.mb_obs, dtype=self.obs.dtype,).squeeze()
				self.mb_rewards = np.asarray(self.mb_rewards, dtype=np.float32).squeeze()
				self.mb_actions = np.asarray(self.mb_actions).squeeze()
				self.mb_values = np.asarray(self.mb_values, dtype=np.float32).squeeze()
				self.mb_neglogpacs = np.asarray(self.mb_neglogpacs, dtype=np.float32).squeeze()
				self.mb_dones = np.asarray(self.mb_dones, dtype=np.bool).squeeze()

				self.mb_obs = self.mb_obs.swapaxes(0, 1)
				self.mb_rewards = self.mb_rewards.swapaxes(0, 1)
				self.mb_actions = self. mb_actions.swapaxes(0, 1)
				self.mb_values = self.mb_values.swapaxes(0, 1)
				self.mb_neglogpacs = self.mb_neglogpacs.swapaxes(0, 1)
				self.mb_dones = self.mb_dones.swapaxes(0, 1)

				# From baselines' ppo2 runner():
				# Calculate returns

				last_values = self.model.value(self.obs)

				mb_returns = np.zeros_like(self.mb_rewards)
				mb_advs = np.zeros_like(self.mb_rewards)
				lastgaelam = 0
				for t in reversed(range(self.nsteps)):
					if t == self.nsteps - 1:
						nextnonterminal = 1.0 - self.done
						nextvalues = last_values
					else:
						nextnonterminal = 1.0 - self.mb_dones[t + 1]
						nextvalues = self.mb_values[t + 1]
					delta = self.mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - self.mb_values[t]
					mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam

				mb_returns[:] = mb_advs + self.mb_values
				# Swap and flatten axis 0 and 1
				if self.nacts > 1:
					self.mb_obs, mb_returns, self.mb_dones, self.mb_actions, self.mb_values, self.mb_neglogpacs = \
						map(ppo2.sf01, (self.mb_obs, mb_returns, self.mb_dones, self.mb_actions, self.mb_values, self.mb_neglogpacs))


				#train model for multiple epoch in n minibacthes pr epoch
				#From baselines' ppo2 learn()
				inds = np.arange(self.batch)
				for _ in range(self.n_train_epoch):
					np.random.shuffle(inds)
					for start in range(0, self.batch, self.nbatch_train):
						end = start + self.nbatch_train
						mbinds = inds[start:end]
						slices = (arr[mbinds] for arr in (self.mb_obs, mb_returns, self.mb_dones, self.mb_actions, self.mb_values, self.mb_neglogpacs))
						self.model.train(self.lr, self.cliprange, *slices)





				self.lr *= self.lr_decay
				self.cliprange *=  self.cl_decay

				self.mb_obs, self.mb_rewards, self.mb_actions, self.mb_values, self.mb_dones, self.mb_neglogpacs = [[]], [[]], [[]], [[]], [[]], [[]]

	def get_action(self, state, env):

		self.obs[:] = state

		actions, values, self.states, neglogpacs = self.model.step(self.obs)

		self.mb_obs[self.actor].append(self.obs.copy())
		self.mb_actions[self.actor].append(actions)
		self.mb_values[self.actor].append(values)
		self.mb_neglogpacs[self.actor].append(neglogpacs)
		self.mb_dones[self.actor].append(self.done)

		return actions

	def __repr__(self):
		return 'ppo'



class DQNExplorer:
	def __init__(self, env):
		self.model = tf.keras.Sequential()
		self.model.add(tf.keras.layers.Conv2D(16, 8, input_shape=env.observation_space.shape, strides=4, activation=tf.nn.relu))
		self.model.add(tf.keras.layers.Conv2D(32, 4, strides=2, activation=tf.nn.relu))
		self.model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
		self.model.add(tf.keras.layers.Dense(env.action_space.n, activation=tf.nn.softmax))

		self.model.compile(optimizer=tf.keras.optimizers.Adam)


# class DQNet(tf.keras.Model):
#     def __init__(self, observation_sp, action_sp):
#         super(DQNet, self).__init__()
#         self.conv1 = tf.keras.layers.Conv2D(16, 8, strides=4, activation=tf.nn.relu, input_shape=(observation_sp))
#         self.conv2 = tf.keras.layers.Conv2D(32, 4, strides=2, activation=tf.nn.relu)
#         self.dense = tf.keras.layers.Dense(256, activation=tf.nn.relu)
#         self.output = tf.keras.layers.Dense(action_sp, activation=tf.nn.softmax)

#     def call(self, inputs):
#         x = self.conv1(inputs)
#         x = self.conv2(x)
#         x = self.dense(x)
#         return self.output(x)
