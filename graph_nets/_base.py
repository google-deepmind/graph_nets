# Lint as: python2, python3
# Copyright 2018 The GraphNets Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Base classes for modules, defined depending on the Sonnet version.

Strategy to be compatible with both Sonnet 1 and Sonnet 2 works as follows:
 - Dynamically decide which version we are using.
 - Create an adapter base class, with a unified API, that would allow child
   classes to implement interfaces similar to Sonnet 1.
 - All GraphNet modules Networks inherit for that same base class, and work with
   either Sonnet 1 or Sonnet 2, depending on how the library is configured.

We do not recommmend users to inherit from this main class, as we only adapt the
functionality for the GraphNets use cases.

We also define a `WrappedModelFnModule`. This is similar to `sonnet.v1.Module`,
except that is receives a callable that returns the build method, rather than
receiving the build method directly. We need this because:
 - There is no analogous to `sonnet.v1.Module` in Sonnet 2.
 - `sonnet.v1.Module` relies on `get_variable` to return always the same
   variables in subsequent calls to the Sonnet module. This means that passing
   a single build method that builds submodules inside of it, yields the right
   variable sharing when called multiple times, thanks to custom variable
   getters. This mechanism does not work in Sonnet 2, and it would lead to
   separate varaibles/submodules being isntantiated every time the module is
   connected. This is why our `WrappedModelFnModule` instead, takes a callable
   that can be called in the `__init__` similarly to how `*_model_fn` arguments
   work in `blocks.py` and `modules.py`.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import contextlib

import six
import sonnet as snt

_sonnet_version = snt.__version__
_sonnet_major_version = int(_sonnet_version.split(".")[0])

if _sonnet_major_version == 1:

  AbstractModule = snt.AbstractModule

elif _sonnet_major_version == 2:

  @six.add_metaclass(abc.ABCMeta)
  class AbstractModule(snt.Module):
    """Makes Sonnet1-style childs from this look like a Sonnet2 module."""

    def __init__(self, *args, **kwargs):
      super(AbstractModule, self).__init__(*args, **kwargs)
      self.__call__.__func__.__doc__ = self._build.__doc__

    # In snt2 calls to `_enter_variable_scope` are ignored.
    @contextlib.contextmanager
    def _enter_variable_scope(self, *args, **kwargs):
      yield None

    def __call__(self, *args, **kwargs):
      return self._build(*args, **kwargs)

    @abc.abstractmethod
    def _build(self, *args, **kwargs):
      """Similar to Sonnet 1 ._build method."""


else:
  raise RuntimeError(
      "Unexpected sonnet major version %d" % (_sonnet_major_version))


class WrappedModelFnModule(AbstractModule):
  """Wraps a model_fn as a Sonnet module with a name.

  Following `blocks.py` convention, a `model_fn` is a callable that, when called
  with no arguments, returns a callable similar to a Sonnet module instance.

  """

  def __init__(self, model_fn, name):
    """Inits the module.

    Args:
      model_fn: callable that, when called with no arguments, returns a callable
          similar to a Sonnet module instance.
      name: Name for the wrapper module.

    """
    super(WrappedModelFnModule, self).__init__(name=name)
    with self._enter_variable_scope():
      self._model = model_fn()

  def _build(self, *args, **kwargs):
    return self._model(*args, **kwargs)
