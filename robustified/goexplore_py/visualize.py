
# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
import pickle
from collections import defaultdict

from goexplore_py.montezuma_env import PYRAMID


def render_with_known(data, filename):
    height, width = data[1][1].shape[:2]

    final_image = np.zeros((height * 4, width * 9, 3), dtype=np.uint8) + 255

    positions = PYRAMID

    def room_pos(room):
        for height, l in enumerate(positions):
            for width, r in enumerate(l):
                if r == room:
                    return (height, width)
        return None

    points = defaultdict(int)

    # print(final_image)

    for room in range(24):
        if room in data:
            img = data[room][1]
        else:
            img = np.zeros((height, width, 3)) + 127
        y_room, x_room = room_pos(room)
        y_room *= height
        x_room *= width
        final_image[y_room:y_room + height, x_room:x_room + width, :] = img

    plt.figure(figsize=(final_image.shape[1] // 30, final_image.shape[0] // 30))

    plt.imshow(final_image)

    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def main():
    with open("all_rooms.pickle", "rb") as file:
        data = pickle.load(file)
    render_with_known(data, "test.png")


if __name__ == "__main__":
    main()
