# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
import linecache


def display_top(snapshot, key_type='traceback', limit=20):
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        filename = frame.filename
        print("#%s: %s:%s: %.1f MB"
              % (index, filename, frame.lineno, stat.size / (1024*1024)))
        line = linecache.getline(frame.filename, frame.lineno).strip()

        for frame in stat.traceback.format():
            print(frame)

        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f MB" % (len(other), size / (1024*1024)))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f MB" % (total / (1024*1024)))
