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

"""Utilities for `utils_np_test` and `utils_tf_test`.

This provides a base class for tests involving `graphs.GraphsTuple`
containing either numpy or tensorflow data. This base class is populated with
test data and also provides a convenience method for asserting graph equality.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import itertools

from graph_nets import graphs
from graph_nets import utils_np
import numpy as np
import tensorflow as tf


@contextlib.contextmanager
def assert_new_op_prefixes(test, expected_prefix, assert_some_new_ops=True):
  """Asserts the namescope of tf ops created within the context manager."""
  ops_before = [n.name for n in tf.get_default_graph().as_graph_def().node]
  yield
  ops_after = [n.name for n in tf.get_default_graph().as_graph_def().node]
  new_ops = set(ops_after) - set(ops_before)
  prefix_length = len(expected_prefix)
  if assert_some_new_ops:
    test.assertNotEqual(0, len(new_ops))
  for op_name in new_ops:
    test.assertEqual(expected_prefix, op_name[:prefix_length])


def mask_leading_dimension(tensor):
  return tf.placeholder_with_default(tensor,
                                     [None] + tensor.get_shape().as_list()[1:])


class GraphsTest(tf.test.TestCase):
  """A base class for tests that operate on GraphsNP or GraphsTF."""

  def populate_test_data(self, max_size):
    """Populates the class fields with data used for the tests.

    This creates a batch of graphs with number of nodes from 0 to `num`,
    number of edges ranging from 1 to `num`, plus an empty graph with no nodes
    and no edges (so that the total number of graphs is 1 + (num ** (num + 1)).

    The nodes states, edges states and global states of the graphs are
    created to have different types and shapes.

    Those graphs are stored both as dictionaries (in `self.graphs_dicts_in`,
    without `n_node` and `n_edge` information, and in `self.graphs_dicts_out`
    with these two fields filled), and a corresponding numpy
    `graphs.GraphsTuple` is stored in `self.reference_graph`.

    Args:
      max_size: The maximum number of nodes and edges (inclusive).
    """
    filt = lambda x: (x[0] > 0) or (x[1] == 0)
    n_node, n_edge = zip(*list(
        filter(filt, itertools.product(
            range(max_size + 1), range(max_size + 1)))))

    graphs_dicts = []
    nodes = []
    edges = []
    receivers = []
    senders = []
    globals_ = []

    def _make_default_state(shape, dtype):
      return np.arange(np.prod(shape)).reshape(shape).astype(dtype)

    for i, (n_node_, n_edge_) in enumerate(zip(n_node, n_edge)):
      n = _make_default_state([n_node_, 7, 11], "f4") + i * 100.
      e = _make_default_state([n_edge_, 13, 14], np.float64) + i * 100. + 1000.
      r = _make_default_state([n_edge_], np.int32) % n_node[i]
      s = (_make_default_state([n_edge_], np.int32) + 1) % n_node[i]
      g = _make_default_state([5, 3], "f4") - i * 100. - 1000.

      nodes.append(n)
      edges.append(e)
      receivers.append(r)
      senders.append(s)
      globals_.append(g)
      graphs_dict = dict(nodes=n, edges=e, receivers=r, senders=s, globals=g)
      graphs_dicts.append(graphs_dict)

    # Graphs dicts without n_node / n_edge (to be used as inputs).
    self.graphs_dicts_in = graphs_dicts
    # Graphs dicts with n_node / n_node (to be checked against outputs).
    self.graphs_dicts_out = []
    for dict_ in self.graphs_dicts_in:
      completed_dict = dict_.copy()
      completed_dict["n_node"] = completed_dict["nodes"].shape[0]
      completed_dict["n_edge"] = completed_dict["edges"].shape[0]
      self.graphs_dicts_out.append(completed_dict)

    # pylint: disable=protected-access
    offset = utils_np._compute_stacked_offsets(n_node, n_edge)
    # pylint: enable=protected-access
    self.reference_graph = graphs.GraphsTuple(**dict(
        nodes=np.concatenate(nodes, axis=0),
        edges=np.concatenate(edges, axis=0),
        receivers=np.concatenate(receivers, axis=0) + offset,
        senders=np.concatenate(senders, axis=0) + offset,
        globals=np.stack(globals_),
        n_node=np.array(n_node),
        n_edge=np.array(n_edge)))

  def _assert_graph_equals_np(self, graph0, graph, force_edges_ordering=False):
    """Asserts that all the graph fields of graph0 and graph match."""
    if graph0.nodes is None:
      self.assertEqual(None, graph.nodes)
    else:
      self.assertAllClose(graph0.nodes, graph.nodes)
    if graph0.globals is None:
      self.assertEqual(None, graph.globals)
    else:
      self.assertAllClose(graph0.globals, graph.globals)
    self.assertAllClose(graph0.n_node, graph.n_node.tolist())
    if graph0.receivers is None:
      self.assertEqual(None, graph.receivers)
      self.assertEqual(None, graph.senders)
      self.assertEqual(None, graph.edges)
      self.assertAllEqual(graph0.n_edge, graph.n_edge)
      return
    self.assertAllClose(graph0.n_edge, graph.n_edge.tolist())

    if not force_edges_ordering:
      self.assertAllClose(graph0.receivers, graph.receivers)
      self.assertAllClose(graph0.senders, graph.senders)
      if graph0.edges is not None:
        self.assertAllClose(graph0.edges, graph.edges)
      else:
        self.assertEqual(None, graph.edges)
      return
    # To compare edges content, we need to make sure they appear in the same
    # order
    if graph0.edges is not None:
      sorted_receivers0, sorted_senders0, sorted_content0 = zip(
          *sorted(zip(graph0.receivers, graph0.senders, graph0.edges.tolist())))
      sorted_receivers, sorted_senders, sorted_content = zip(
          *sorted(zip(graph.receivers, graph.senders, graph.edges.tolist())))
      self.assertAllClose(sorted_content0, sorted_content)
    elif graph.receivers is not None:
      sorted_receivers0, sorted_senders0 = zip(
          *sorted(zip(graph0.receivers, graph0.senders)))
      sorted_receivers, sorted_senders = zip(
          *sorted(zip(graph.receivers, graph.senders)))
    else:
      return
    self.assertAllClose(sorted_receivers0, sorted_receivers)
    self.assertAllClose(sorted_senders0, sorted_senders)

  def setUp(self):
    self.populate_test_data(max_size=2)
    tf.reset_default_graph()
