# Copyright (c) 2018-2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import imageio
import gym
import json
import argparse
from tqdm import tqdm


def render_video(env, actions, out, no_ops=0, large_frame=False, speedup=4):
    frame = env.reset()
    writer = imageio.get_writer(out, fps=60) 
    if large_frame:
        frame = frame.repeat(2, axis=1)
        frame = frame.repeat(10, axis=1)
        frame = frame.repeat(10, axis=0)
    writer.append_data(frame)
    for i in range(no_ops):
        frame, reward, done, _ = env.step(0)
        if large_frame:
            frame = frame.repeat(2, axis=1)
            frame = frame.repeat(10, axis=1)
            frame = frame.repeat(10, axis=0)
        if i % speedup == 0:
            writer.append_data(frame)
    for i, e in enumerate(tqdm(actions)):
        frame, reward, done, _ = env.step(e)
        if large_frame:
            frame = frame.repeat(2, axis=1)
            frame = frame.repeat(10, axis=1)
            frame = frame.repeat(10, axis=0)
        if i % speedup == 0:
            writer.append_data(frame)
    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', metavar='F', type=str, nargs=1,
                        help='the json file from which to render')
    parser.add_argument('--game', '-g', type=str, default='Montezuma', 
                        help='determines which game to render')
    parser.add_argument('--out', '-o', type=str, default='out.mp4', 
                        help='name of the output file')
    parser.add_argument('--large_frame', action='store_true', default=False,
                        help='render the video with large frames')
    parser.add_argument('--seed', type=int, default=0,
                        help='which seed (i.e. which of the rollouts) to render')
    parser.add_argument('--speed', type=int, default=4,
                        help='what speedup to apply')
    args = parser.parse_args()

    filename = args.input[0]
    large_frame = args.large_frame
    seed = args.seed
    speedup = args.seed
    output_file = args.out
    game = args.game

    data = json.load(open(filename))
    env = gym.make(game + 'NoFrameskip-v4')
    no_ops = int(filename.split('/')[-1].split('.')[0])
    render_video(env, data[seed][0]['actions'], output_file, no_ops=no_ops, large_frame=large_frame, speedup=speedup)


if __name__ == '__main__':
    main()
