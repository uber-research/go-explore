# Copyright (c) 2018-2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import imageio
from PIL import Image, ImageFont, ImageDraw

import gen_demo.atari_demo as atari_demo
import gen_demo.atari_demo.wrappers

from goexplore_py.goexplore import *
# from atari_reset.atari_reset.wrappers import Image, MyResizeFrame, WarpFrame

sys.modules['env'] = sys.modules['goexplore_py.montezuma_env']

import fire

FOLDER = None
DESTINATION = None
FRAME_SKIP = 4

NUM_PIXELS = 8.0
# NUM_PIXELS = 16.0

class RenamingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ['basic',
                      'explorers',
                      'goexplore',
                      'import_ai',
                      'montezuma_env',
                      'pitfall_env',
                      'randselectors',
                      'utils']:
            module = 'goexplore_py.' + module
        return super().find_class(module, name)


def my_resize_frame(obs, res):
    obs = np.array(Image.fromarray(obs).resize((res[0], res[1]), resample=Image.BILINEAR), dtype=np.uint8)
    return obs.reshape(res)


def convert_state(state, shape):
    import cv2
    frame = (((cv2.resize(cv2.cvtColor(state, cv2.COLOR_RGB2GRAY), (11, 8), interpolation=cv2.INTER_AREA) / 255.0) * NUM_PIXELS).astype(np.uint8) * (255.0 / NUM_PIXELS)).astype(np.uint8)
    # print("shape:", shape, frame.shape)
    frame = np.transpose(frame)
    frame = cv2.resize(frame, dsize=shape, interpolation=cv2.INTER_NEAREST)
    # print("shape:", frame.shape)
    frame = np.transpose(frame)
    return frame


def with_domain_knowledge(key):
    return not isinstance(key, tuple)


class ScoreTrajectories:
    def __init__(self, chosen_demos, data, max_level=float('inf'), max_trajectory=float('inf'), max_score=float('inf')):
        self.chosen_demos = chosen_demos
        self.data = data
        self.select_longest_trajectory = False
        self.max_level = max_level
        self.max_trajectory = max_trajectory
        self.max_score = max_score

    def compute_similarity_weight(self, cell):
        weight = 1.0
        for k in self.chosen_demos:
            cell2 = self.data[k]
            total = 0
            different = 0
            if len(cell.trajectory) == 0 or len(cell2.trajectory) == 0:
                continue
            for a1, a2 in zip(cell.trajectory, cell2.trajectory):
                if not isinstance(a1, tuple) and a1.from_.exact.level >= max_level and a2.from_.exact.level >= max_level:
                    break
                total += 1
                different += a1.action != a2.action
            weight = min(weight, different / total)

        return weight

    def __call__(self, key):
        cell = self.data[key]

        weight = self.compute_similarity_weight(cell)

        if with_domain_knowledge(key):
            if self.select_longest_trajectory:
                return weight * key.level, weight * cell.score, cell.trajectory_len * weight
            else:
                return weight * key.level, weight * cell.score, -cell.trajectory_len * weight
        else:
            if self.select_longest_trajectory:
                return cell.real_cell.level * weight, cell.score * weight, cell.trajectory_len * weight
            else:
                return cell.real_cell.level * weight, cell.score * weight, -cell.trajectory_len * weight


def run(folder, destination, max_level=None, max_trajectory=None, max_score=None, game="montezuma", stop_on_score=False, n_demos=1):
    global FOLDER, DESTINATION
    FOLDER = folder
    DESTINATION = destination
    if game == "montezuma":
        gym_game = 'MontezumaRevengeNoFrameskip-v4'
    elif game == "pitfall":
        gym_game = 'PitfallNoFrameskip-v4'
    else:
        raise NotImplementedError("Unknown game: " + game)

    file = max(e for e in glob.glob(FOLDER + '/*.7z') if '_set' not in e)
    print(file)
    print('size =', len(lzma.open(file).read()))
    data = RenamingUnpickler(lzma.open(file)).load()

    os.makedirs(destination, exist_ok=True)

    chosen_demos = []

    if max_level is None:
        max_level = float('inf')
    if max_trajectory is None:
        max_trajectory = float('inf')
    if max_score is None:
        max_score = float('inf')

    # Experimental: truncate all trajectories to fit the desired criteria
    print("Cell information:")
    for key in data.keys():
        cell = data[key]
        cum_reward = 0
        trajectory_length = 0
        highest_reward = 0
        highest_reward_trajectory_length = 0
        for e in cell.trajectory:
            cum_reward += e.reward
            trajectory_length += 1
            if cum_reward > highest_reward:
                highest_reward = cum_reward
                highest_reward_trajectory_length = trajectory_length
            if trajectory_length >= max_trajectory:
                break
            if cum_reward >= max_score:
                break
        cell.score = highest_reward
        cell.trajectory_len = highest_reward_trajectory_length
        cell.trajectory = cell.trajectory[0:highest_reward_trajectory_length]

    for idx in range(n_demos):
        key = max(data.keys(), key=ScoreTrajectories(chosen_demos, data, max_level, max_trajectory, max_score))
        if with_domain_knowledge(key):
            if key.level < max_level and game != "pitfall":
                print(f'WARNING: Level {max_level} not solved (max={key.level})')
            list_of_actions = [e.action for e in data[key].trajectory if e.to.exact.level < max_level]

        else:
            list_of_actions = [e.action for e in data[key].trajectory]

        if hasattr(data[key].real_cell, 'level'):
            print('Chosen - score:', data[key].score, "length:", data[key].trajectory_len, 'level:', data[key].real_cell.level)
        else:
            print('Chosen - score:', data[key].score, "length:", data[key].trajectory_len)
        chosen_demos.append(key)

        env = gym.make(gym_game)
        env = atari_demo.wrappers.AtariDemo(env)
        env.reset()
        frames = [env.render(mode='rgb_array').repeat(2, axis=1)]
        total = 0

        for a in [0] * 3 + list_of_actions:
            # done = False
            for _ in range(4):
                _, reward, done, _ = env.step(a)
                frame = env.render(mode='rgb_array').repeat(2, axis=1)
                frames.append(np.array(frame))

                total += reward

        frames += [frames[-1]] * 200
        print("Created demo:", len(frames), total, data[key].score)

        env.save_to_file(DESTINATION + f'/{idx}.demo')

        imageio.mimsave(DESTINATION + f'/{idx}.mp4', frames[::7], fps=24)


if __name__ == '__main__':
    fire.Fire(run)
