"""
// Modifications Copyright (c) 2020 Uber Technologies Inc.
"""

import numpy as np
import tensorflow as tf
from tensorflow.nn import rnn_cell
from baselines.common.distributions import make_pdtype
import logging
logger = logging.getLogger(__name__)


def to2d(x):
    size = 1
    for shapel in x.get_shape()[1:]:
        size *= shapel.value
    return tf.reshape(x, (-1, size))


def normc_init(std=1.0, axis=0):
    """
    Initialize with normalized columns
    """

    # noinspection PyUnusedLocal
    def _initializer(shape, dtype=None, partition_info=None):  # pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
        return tf.constant(out)
    return _initializer


def ortho_init(scale=1.0):
    # noinspection PyUnusedLocal
    def _ortho_init(shape, dtype, partition_info=None):  # pylint: disable=W0613
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init


def fc(x, scope, nout, init_scale=1.0, init_bias=0.0):
    with tf.variable_scope(scope):  # pylint: disable=E1129
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nout], initializer=normc_init(init_scale))
        b = tf.get_variable("b", [nout], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(x, w) + b


def conv(x, scope, noutchannels, filtsize, stride, pad='VALID', init_scale=1.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[3].value
        w = tf.get_variable("w", [filtsize, filtsize, nin, noutchannels], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [noutchannels], initializer=tf.constant_initializer(0.0))
        z = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=pad)+b
        return z


class GRUCell(rnn_cell.RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""
    def __init__(self, num_units, name, nin, rec_gate_init=0.):
        rnn_cell.RNNCell.__init__(self)
        self._num_units = num_units
        self.rec_gate_init = rec_gate_init
        self.w1 = tf.get_variable(name + "w1", [nin+num_units, 2*num_units], initializer=normc_init(1.))
        self.b1 = tf.get_variable(name + "b1", [2*num_units], initializer=tf.constant_initializer(rec_gate_init))
        self.w2 = tf.get_variable(name + "w2", [nin+num_units, num_units], initializer=normc_init(1.))
        self.b2 = tf.get_variable(name + "b2", [num_units], initializer=tf.constant_initializer(0.))

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    # noinspection PyMethodOverriding
    def call(self, inputs, state):
        """Gated recurrent unit (GRU) with nunits cells."""
        x, new = inputs
        while len(state.get_shape().as_list()) > len(new.get_shape().as_list()):
            new = tf.expand_dims(new, len(new.get_shape().as_list()))
        h = state * (1.0 - new)
        hx = tf.concat([h, x], axis=1)
        mr = tf.sigmoid(tf.matmul(hx, self.w1) + self.b1)
        m, r = tf.split(mr, 2, axis=1)
        rh_x = tf.concat([r * h, x], axis=1)
        htil = tf.tanh(tf.matmul(rh_x, self.w2) + self.b2)
        h = m * h + (1.0 - m) * htil
        return h, h


class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, _nsteps, _test_mode=False, reuse=False):
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        x = tf.placeholder(tf.uint8, ob_shape)
        with tf.variable_scope("model", reuse=reuse):
            h = tf.nn.relu(conv(tf.cast(x, tf.float32)/255., 'c1', noutchannels=64, filtsize=8, stride=4))
            h2 = tf.nn.relu(conv(h, 'c2', noutchannels=128, filtsize=4, stride=2))
            h3 = tf.nn.relu(conv(h2, 'c3', noutchannels=128, filtsize=3, stride=1))
            h3 = to2d(h3)
            h4 = tf.nn.relu(fc(h3, 'fc1', nout=1024))
            pi = fc(h4, 'pi', nact, init_scale=0.01)
            vf = fc(h4, 'v', 1, init_scale=0.01)[:, 0]

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {x: ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {x: ob})

        self.X = x
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class GRUPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, memsize=800, test_mode=False, reuse=False):
        nh, nw, nc = ob_space.shape
        nbatch = nenv*nsteps
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n

        # use variables instead of placeholder to keep data on GPU if we're training
        x = tf.placeholder(tf.uint8, ob_shape)  # obs
        mask = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
        states = tf.placeholder(tf.float32, [nenv, memsize])  # states
        e = tf.placeholder(tf.uint8, [nbatch])

        with tf.variable_scope("model", reuse=reuse):
            h = tf.nn.relu(conv(tf.cast(x, tf.float32)/255., 'c1', noutchannels=64, filtsize=8, stride=4))
            h2 = tf.nn.relu(conv(h, 'c2', noutchannels=128, filtsize=4, stride=2))
            h3 = tf.nn.relu(conv(h2, 'c3', noutchannels=128, filtsize=3, stride=1))
            h3 = to2d(h3)
            h4 = tf.contrib.layers.layer_norm(fc(h3, 'fc1', nout=memsize), center=False, scale=False,
                                              activation_fn=tf.nn.relu)
            h5 = tf.reshape(h4, [nenv, nsteps, memsize])

            m = tf.reshape(mask, [nenv, nsteps, 1])
            cell = GRUCell(memsize, 'gru1', nin=memsize)
            h6, snew = tf.nn.dynamic_rnn(cell, (h5, m), dtype=tf.float32, time_major=False, initial_state=states,
                                         swap_memory=True)

            h7 = tf.concat([tf.reshape(h6, [nbatch, memsize]), h4], axis=1)
            pi = fc(h7, 'pi', nact, init_scale=0.01)
            if test_mode:
                pi *= 2.
            else:
                pi = tf.where(e > 0, pi/2., pi)
            vf = tf.squeeze(fc(h7, 'v', 1, init_scale=0.01))

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, memsize), dtype=np.float32)

        def step(ob, state, mask_, increase_ent):
            return sess.run([a0, vf, snew, neglogp0], {x: ob, states: state, mask: mask_, e: increase_ent})

        def value(ob, state, mask_):
            return sess.run(vf, {x: ob, states: state, mask: mask_})

        self.X = x
        self.M = mask
        self.S = states
        self.E = e
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
