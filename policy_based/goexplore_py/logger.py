# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
class SimpleLogger:
    def __init__(self, file_name):
        self.file_handle = open(file_name, 'w')
        self.column_names = []
        self.values = []
        self.first_line = True

    def write(self, name, value):
        if self.first_line:
            self.column_names.append(name)
        self.values.append(value)

    def flush(self):
        if self.first_line:
            self.first_line = False
            for i, column_name in enumerate(self.column_names):
                self.file_handle.write(column_name)
                if i < len(self.column_names) - 1:
                    self.file_handle.write(', ')
            self.file_handle.write('\n')
        for i, value in enumerate(self.values):
            self.file_handle.write(str(value))
            if i < len(self.column_names) - 1:
                self.file_handle.write(', ')
        self.file_handle.write('\n')
        self.file_handle.flush()
        self.values = []

    def close(self):
        self.file_handle.close()
