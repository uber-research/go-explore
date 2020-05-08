# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
import tensorflow as tf
from typing import Any
import atari_reset.atari_reset.ppo as ppo


class GoalConditionedModelFlexEnt(ppo.Model):
    def __init__(self):
        super(GoalConditionedModelFlexEnt, self).__init__()

    def init(self, policy, ob_space, ac_space, nenv, nsteps, ent_coef, vf_coef, l2_coef,
             cliprange, adam_epsilon=1e-6, load_path=None, test_mode=False, goal_space=None, disable_hvd=False):
        self.sess = tf.get_default_session()
        self.init_models(policy, ob_space, ac_space, nenv, nsteps, test_mode, goal_space)
        self.init_loss(nenv, nsteps, cliprange, disable_hvd)
        self.loss = self.pg_loss - self.entropy * ent_coef + self.vf_loss * vf_coef + l2_coef * self.l2_loss
        self.finalize(load_path, adam_epsilon)

    def init_models(self, policy, ob_space, ac_space, nenv, nsteps, test_mode, goal_space):
        # At test time, we only need the most recent action in order to take a step.
        self.act_model = policy(self.sess, ob_space, ac_space, nenv, 1, test_mode=test_mode, reuse=False,
                                goal_space=goal_space)
        # At training time, we need to keep track of the last T (nsteps) of actions that we took.
        self.train_model = policy(self.sess, ob_space, ac_space, nenv, nsteps, test_mode=test_mode, reuse=True,
                                  goal_space=goal_space)

    def train_from_runner(self, lr: float, runner: Any):
        return self.train(lr,
                          runner.ar_mb_obs_2.reshape(self.train_model.X.shape),
                          runner.ar_mb_goals,
                          runner.ar_mb_rets,
                          runner.ar_mb_advs,
                          runner.ar_mb_dones,
                          runner.ar_mb_actions,
                          runner.ar_mb_values,
                          runner.ar_mb_neglogpacs,
                          runner.ar_mb_valids,
                          runner.ar_mb_ent,
                          runner.mb_states[0])

    def train(self, lr, obs, goals, returns, advs, masks, actions, values, neglogpacs, valids, increase_ent,
              states=None):
        td_map = {self.LR: lr, self.train_model.X: obs, self.train_model.goal: goals,
                  self.A: actions, self.ADV: advs, self.VALID: valids, self.R: returns,
                  self.OLDNEGLOGPAC: neglogpacs, self.OLDVPRED: values, self.train_model.E: increase_ent}
        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks
        return self.sess.run(self.loss_requested, feed_dict=td_map)[:-1]


class GoalConFlexEntSilModel(GoalConditionedModelFlexEnt):
    def __init__(self):
        super(GoalConFlexEntSilModel, self).__init__()
        self.sil_loss = None
        self.SIL_A = None
        self.SIL_VALID = None
        self.SIL_R = None
        self.sil_pg_loss = None
        self.sil_vf_loss = None
        self.sil_entropy = None
        self.sil_valid_min = None
        self.sil_valid_max = None
        self.sil_valid_mean = None

        self.neglop_sil_min = None
        self.neglop_sil_max = None
        self.neglop_sil_mean = None

        # Debug
        self.mean_val_pred = None
        self.mean_sil_r = None
        self.train_it = 0

    def init(self, policy, ob_space, ac_space, nenv, nsteps, ent_coef, vf_coef, l2_coef,
             cliprange, adam_epsilon=1e-6, load_path=None, test_mode=False, goal_space=None, sil_coef=0.0,
             sil_vf_coef=0.0, sil_ent_coef=0.0, disable_hvd=False):
        self.sess = tf.get_default_session()
        self.init_models(policy, ob_space, ac_space, nenv, nsteps, test_mode, goal_space)
        self.init_loss(nenv, nsteps, cliprange, disable_hvd)
        self.init_sil_loss(nenv, nsteps, sil_vf_coef, sil_ent_coef)
        self.loss = (self.pg_loss
                     - self.entropy * ent_coef
                     + self.vf_loss * vf_coef
                     + l2_coef * self.l2_loss
                     + sil_coef * self.sil_loss)

        self.finalize(load_path, adam_epsilon)
        self.loss_requested_dict = {self.pg_loss: 'policy_loss',
                                    self.vf_loss: 'value_loss',
                                    self.l2_loss: 'l2_loss',
                                    self.entropy: 'policy_entropy',
                                    self.approxkl: 'approxkl',
                                    self.clipfrac: 'clipfrac',
                                    self.sil_pg_loss: 'sil_pg_loss',
                                    self.sil_vf_loss: 'sil_vf_loss',
                                    self.sil_loss: 'sil_loss',
                                    self.sil_entropy: 'sil_entropy',
                                    self.sil_valid_min: 'sil_valid_min',
                                    self.sil_valid_max: 'sil_valid_max',
                                    self.sil_valid_mean: 'sil_valid_mean',
                                    self.neglop_sil_min: 'neglop_sil_min',
                                    self.neglop_sil_max: 'neglop_sil_max',
                                    self.neglop_sil_mean: 'neglop_sil_mean',
                                    self.mean_val_pred: 'mean_val_pred',
                                    self.mean_sil_r: 'mean_sil_r',
                                    self.train_op: ''}
        self.init_requested_loss()

    def init_sil_loss(self, nenv, nsteps, sil_vf_coef, sil_ent_coef):
        self.SIL_A = self.train_model.pdtype.sample_placeholder([nenv*nsteps], name='sil_action')
        self.SIL_VALID = tf.placeholder(tf.float32, [nenv*nsteps], name='sil_valid')
        self.SIL_R = tf.placeholder(tf.float32, [nenv*nsteps], name='sil_return')

        neglogp_sil_ac = self.train_model.pd.neglogp(self.SIL_A)

        self.sil_pg_loss = tf.reduce_mean(neglogp_sil_ac * tf.nn.relu(self.SIL_R - self.OLDVPRED) * self.SIL_VALID)
        self.sil_vf_loss = .5 * tf.reduce_mean(tf.square(tf.nn.relu(self.SIL_R - self.vpred)) * self.SIL_VALID)
        self.sil_entropy = tf.reduce_mean(self.SIL_VALID * self.train_model.pd.entropy())
        self.sil_loss = self.sil_pg_loss + sil_vf_coef * self.sil_vf_loss + sil_ent_coef * self.sil_entropy

        self.sil_valid_min = tf.reduce_min(self.SIL_VALID)
        self.sil_valid_max = tf.reduce_max(self.SIL_VALID)
        self.sil_valid_mean = tf.reduce_mean(self.SIL_VALID)

        self.neglop_sil_min = tf.reduce_min(neglogp_sil_ac)
        self.neglop_sil_max = tf.reduce_max(neglogp_sil_ac)
        self.neglop_sil_mean = tf.reduce_mean(neglogp_sil_ac)

        self.mean_val_pred = tf.reduce_mean(self.OLDVPRED)
        self.mean_sil_r = tf.reduce_mean(self.SIL_R)

    def train_from_runner(self, lr: float, runner: Any):
        obs = runner.ar_mb_obs_2.reshape(self.train_model.X.shape)

        return self.train(lr,
                          obs,
                          runner.ar_mb_goals,
                          runner.ar_mb_rets,
                          runner.ar_mb_advs,
                          runner.ar_mb_dones,
                          runner.ar_mb_actions,
                          runner.ar_mb_values,
                          runner.ar_mb_neglogpacs,
                          runner.ar_mb_valids,
                          runner.ar_mb_ent,
                          runner.ar_mb_sil_actions,
                          runner.ar_mb_sil_rew,
                          runner.ar_mb_sil_valid,
                          runner.mb_states[0])

    def train(self, lr, obs, goals, returns, advs, masks, actions, values, neglogpacs, valids, increase_ent,
              sil_actions=None, sil_rew=None, sil_valid=None, states=None):
        self.train_it += 1
        td_map = {self.LR: lr, self.train_model.X: obs, self.train_model.goal: goals, self.A: actions, self.ADV: advs,
                  self.VALID: valids, self.R: returns,
                  self.OLDNEGLOGPAC: neglogpacs, self.OLDVPRED: values, self.train_model.E: increase_ent,
                  self.SIL_A: sil_actions, self.SIL_R: sil_rew, self.SIL_VALID: sil_valid}
        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks
        return self.filter_requested_losses(self.sess.run(self.loss_requested, feed_dict=td_map))
