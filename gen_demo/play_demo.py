# Copyright (c) 2018-2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import gym
import sys
import pickle
import time

human_agent_action = 0
human_wants_restart = False
human_sets_pause = False

KEYNAMES = {
    32: 'FIRE',
65361: 'LEFT',
65363: 'RIGHT',
65364: 'DOWN',
65362: 'UP'
}

COMBINATIONS = [{'UP', 'DOWN'}, {'LEFT', 'RIGHT'}, {'FIRE',}]

ACTIONS = None

CUR_KEYS = set()

def update_action():
    global human_agent_action
    action_name = ''
    for c in COMBINATIONS:
        for possible in c:
            if possible in CUR_KEYS:
                action_name = action_name + possible
                break

    if action_name == '':
        action_name = 'NOOP'

    human_agent_action =  ACTIONS.index(action_name)

def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key in KEYNAMES:
        CUR_KEYS.add(KEYNAMES[key])
        update_action()
    # a = int( key - ord('0') )
    # if a <= 0 or a >= ACTIONS: return
    # human_agent_action = a

def key_release(key, mod):
    global human_agent_action
    if key in KEYNAMES:
        CUR_KEYS.remove(KEYNAMES[key])
        update_action()

if __name__ == '__main__':
    demo = pickle.load(open(sys.argv[1], 'rb'))
    if len(sys.argv) > 2:
        start = sys.argv[2]
    else:
        print(f'{len(demo["actions"]):,} steps, where to start?')
        start = input()

    start = int(start)
    env = gym.make('MontezumaRevengeNoFrameskip-v4')
    ACTIONS = env.unwrapped.get_action_meanings()
    env.reset()
    for a in demo['actions'][:int(start)]:
        env.step(a)
    print(ACTIONS[demo['actions'][start]])

    env.render()
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release

    done = False
    while not done:
        env.step(human_agent_action)
        env.render()
        time.sleep(0.01)
