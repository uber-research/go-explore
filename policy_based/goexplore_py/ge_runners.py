# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import atari_reset.atari_reset.ppo as ppo


class RunnerFlexEntSilProper(ppo.Runner):
    def __init__(self, env, model, nsteps, gamma, lam, norm_adv, subtract_rew_avg):
        super(RunnerFlexEntSilProper, self).__init__(env, model, nsteps, gamma, lam, norm_adv, subtract_rew_avg)
        self.mb_sil_actions = self.reg_shift_list()
        self.mb_sil_rew = self.reg_shift_list()
        self.mb_sil_valid = self.reg_shift_list()
        self.ar_mb_sil_valid = None
        self.ar_mb_sil_actions = None
        self.ar_mb_sil_rew = None
        self.trunc_lst_mb_sil_valid = None

    def append_mb_data(self, actions, values, states, neglogpacs, obs_and_goals, rewards, dones, infos):
        super(RunnerFlexEntSilProper, self).append_mb_data(actions,
                                                           values,
                                                           states,
                                                           neglogpacs,
                                                           obs_and_goals,
                                                           rewards,
                                                           dones,
                                                           infos)

        def get_sil_valid(info):
            is_valid = float(info.get('sil_action') is not None)
            return is_valid

        self.mb_sil_valid.append([get_sil_valid(info) for info in infos])
        sil_actions = np.zeros_like(actions)
        for cur_info_id, info in enumerate(infos):
            cur_action = info.get('sil_action')
            if cur_action is not None:
                sil_actions[cur_info_id] = cur_action

        self.mb_sil_actions.append(sil_actions)
        self.mb_sil_rew.append([info.get('sil_value', 0) for info in infos])

    def gather_return_info(self, end):
        super(RunnerFlexEntSilProper, self).gather_return_info(end)
        self.ar_mb_sil_valid = ppo.sf01(np.asarray(self.mb_sil_valid[:end], dtype=np.float32), 'sil_valids')
        self.ar_mb_sil_actions = ppo.sf01(np.asarray(self.mb_sil_actions[:end]), 'sil_actions')
        self.ar_mb_sil_rew = ppo.sf01(np.asarray(self.mb_sil_rew[:end], dtype=np.float32), 'sil_rewards')
        self.trunc_lst_mb_sil_valid = ppo.sf01(np.asarray(self.mb_sil_valid[-len(self.mb_cells):len(self.mb_sil_valid)],
                                                          dtype=np.float32), 'trunc_sil_valids')
