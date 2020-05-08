# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
COMM_WORLD = None
COMM_TYPE_SHARED = None


def init_mpi():
    global COMM_WORLD
    global COMM_TYPE_SHARED
    import mpi4py.rc
    mpi4py.rc.initialize = False
    from mpi4py import MPI
    COMM_WORLD = MPI.COMM_WORLD
    COMM_TYPE_SHARED = MPI.COMM_TYPE_SHARED


def get_comm_world():
    return COMM_WORLD


def get_comm_type_shared():
    return COMM_TYPE_SHARED
