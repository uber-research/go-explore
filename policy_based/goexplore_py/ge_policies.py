# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import numpy as np
import tensorflow as tf
import atari_reset.atari_reset.policies as po
logger = logging.getLogger(__name__)


class GRUPolicyGoalConSimpleFlexEnt(object):
    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, memsize=800, test_mode=False, reuse=False,
                 goal_space=None):
        nh, nw, nc = ob_space.shape
        nbatch = nenv*nsteps
        ob_shape = (nbatch, nh, nw, nc)
        logger.info(f'goal_space.shape: {goal_space.shape}')
        goal_shape = tuple([nbatch] + list(goal_space.shape))
        logger.info(f'goal_shape: {goal_shape}')
        nact = ac_space.n

        # use variables instead of placeholder to keep data on GPU if we're training
        nn_input = tf.placeholder(tf.uint8, ob_shape, 'input')  # obs
        goal = tf.placeholder(tf.float32, goal_shape, 'goal')  # goal
        mask = tf.placeholder(tf.float32, [nbatch], 'done_mask')  # mask (done t-1)
        states = tf.placeholder(tf.float32, [nenv, memsize], 'hidden_state')  # states
        entropy = tf.placeholder(tf.float32, [nbatch], 'entropy_factor')
        fake_actions = tf.placeholder(tf.int64, [nbatch], 'fake_actions')
        logger.info(f'fake_actions.shape: {fake_actions.shape}')
        logger.info(f'fake_actions.dtype: {fake_actions.dtype}')

        with tf.variable_scope("model", reuse=reuse):
            logger.info(f'input.shape {nn_input.shape}')
            h = tf.nn.relu(po.conv(tf.cast(nn_input, tf.float32)/255., 'c1', noutchannels=64, filtsize=8, stride=4))
            logger.info(f'h.shape: {h.shape}')
            h2 = tf.nn.relu(po.conv(h, 'c2', noutchannels=128, filtsize=4, stride=2))
            logger.info(f'h2.shape: {h2.shape}')
            h3 = tf.nn.relu(po.conv(h2, 'c3', noutchannels=128, filtsize=3, stride=1))
            logger.info(f'h3.shape: {h3.shape}')
            h3 = po.to2d(h3)
            logger.info(f'h3.shape: {h3.shape}')
            g1 = tf.cast(goal, tf.float32)
            logger.info(f'g1.shape: {g1.shape}')
            h3 = tf.concat([h3, g1], axis=1)
            logger.info(f'h3.shape: {h3.shape}')
            h4 = tf.contrib.layers.layer_norm(po.fc(h3, 'fc1', nout=memsize), center=False, scale=False,
                                              activation_fn=tf.nn.relu)
            logger.info(f'h4.shape: {h4.shape}')
            h5 = tf.reshape(h4, [nenv, nsteps, memsize])

            m = tf.reshape(mask, [nenv, nsteps, 1])
            cell = po.GRUCell(memsize, 'gru1', nin=memsize)
            h6, snew = tf.nn.dynamic_rnn(cell, (h5, m), dtype=tf.float32, time_major=False,
                                         initial_state=states, swap_memory=True)
            logger.info(f'h6.shape: {h6.shape}')

            h7 = tf.concat([tf.reshape(h6, [nbatch, memsize]), h4], axis=1)
            pi = po.fc(h7, 'pi', nact, init_scale=0.01)
            if test_mode:
                pi *= 2.
            else:
                pi /= tf.reshape(entropy, (nbatch, 1))
            logger.info(f'h7.shape: {h7.shape}')
            vf_before_squeeze = po.fc(h7, 'v', 1, init_scale=0.01)
            logger.info(f'vf_before_squeeze.shape: {vf_before_squeeze.shape}')
            vf = tf.squeeze(vf_before_squeeze, axis=[1])
            logger.info(f'vf.shape: {vf.shape}')

        self.pdtype = po.make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)
        a0 = self.pd.sample()
        logger.info(f'a0.shape: {a0.shape}')
        logger.info(f'a0.dtype: {a0.dtype}')
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, memsize), dtype=np.float32)

        neg_log_fake_a = self.pd.neglogp(fake_actions)

        def step(local_ob, local_goal, local_state, local_mask, local_increase_ent):
            return sess.run([a0, vf, snew, neglogp0],
                            {nn_input: local_ob, states: local_state, mask: local_mask, entropy: local_increase_ent,
                             goal: local_goal})

        def step_fake_action(local_ob, local_goal, local_state, local_mask, local_increase_ent, local_fake_action):
            return sess.run([a0, vf, snew, neglogp0, neg_log_fake_a],
                            {nn_input: local_ob,
                             states: local_state,
                             mask: local_mask,
                             entropy: local_increase_ent,
                             goal: local_goal,
                             fake_actions: local_fake_action})

        def value(local_ob, local_goal, local_state, local_mask):
            return sess.run(vf, {nn_input: local_ob, states: local_state, mask: local_mask, goal: local_goal})

        self.X = nn_input
        self.goal = goal
        self.M = mask
        self.S = states
        self.E = entropy
        self.pi = pi
        self.vf = vf
        self.step = step
        self.step_fake_action = step_fake_action
        self.value = value
