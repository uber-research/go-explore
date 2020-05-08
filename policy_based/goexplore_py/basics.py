# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import collections
import functools
import warnings as _warnings

try:
    from dataclasses import dataclass, field as datafield

    def copyfield(data):
        return datafield(default_factory=lambda: copy.deepcopy(data))
except ImportError:
    _warnings.warn('dataclasses not found. To get it, use Python 3.7 or pip install dataclasses')
    raise

infinity = float('inf')


def notebook_max_width():
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:100% !important; }</style>"))


class Memoized(object):
    """
    Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    """

    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self):
        """Return the function's docstring."""
        return self.func.__doc__

    def __get__(self, obj, objtype):
        """Support instance methods."""
        return functools.partial(self.__call__, obj)
