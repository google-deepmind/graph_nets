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

"""Tests for blocks.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl.testing import parameterized
from graph_nets import blocks
from graph_nets import graphs
from graph_nets import utils_np
from graph_nets import utils_tf
import numpy as np
import sonnet as snt
import tensorflow as tf


SMALL_GRAPH_1 = {
    "globals": [1.1, 1.2, 1.3, 1.4],
    "nodes": [[10.1, 10.2], [20.1, 20.2], [30.1, 30.2]],
    "edges": [[101., 102., 103., 104.], [201., 202., 203., 204.]],
    "senders": [0, 1],
    "receivers": [1, 2],
}

SMALL_GRAPH_2 = {
    "globals": [-1.1, -1.2, -1.3, -1.4],
    "nodes": [[-10.1, -10.2], [-20.1, -20.2], [-30.1, -30.2]],
    "edges": [[-101., -102., -103., -104.]],
    "senders": [1,],
    "receivers": [2,],
}

SMALL_GRAPH_3 = {
    "globals": [1.1, 1.2, 1.3, 1.4],
    "nodes": [[10.1, 10.2], [20.1, 20.2], [30.1, 30.2]],
    "edges": [[101., 102., 103., 104.], [201., 202., 203., 204.]],
    "senders": [1, 1],
    "receivers": [0, 2],
}

SMALL_GRAPH_4 = {
    "globals": [1.1, 1.2, 1.3, 1.4],
    "nodes": [[10.1, 10.2], [20.1, 20.2], [30.1, 30.2]],
    "edges": [[101., 102., 103., 104.], [201., 202., 203., 204.]],
    "senders": [0, 2],
    "receivers": [1, 1],
}


class GraphModuleTest(tf.test.TestCase, parameterized.TestCase):
  """Base class for all the tests in this file."""

  def setUp(self):
    super(GraphModuleTest, self).setUp()
    tf.set_random_seed(0)

  def _get_input_graph(self, none_fields=None):
    if none_fields is None:
      none_fields = []
    input_graph = utils_tf.data_dicts_to_graphs_tuple(
        [SMALL_GRAPH_1, SMALL_GRAPH_2, SMALL_GRAPH_3, SMALL_GRAPH_4])
    input_graph = input_graph.map(lambda _: None, none_fields)
    return input_graph

  def _get_shaped_input_graph(self):
    return graphs.GraphsTuple(
        nodes=tf.zeros([3, 4, 5, 11], dtype=tf.float32),
        edges=tf.zeros([5, 4, 5, 12], dtype=tf.float32),
        globals=tf.zeros([2, 4, 5, 13], dtype=tf.float32),
        receivers=tf.range(5, dtype=tf.int32) // 3,
        senders=tf.range(5, dtype=tf.int32) % 3,
        n_node=tf.constant([2, 1], dtype=tf.int32),
        n_edge=tf.constant([3, 2], dtype=tf.int32),
    )

  def _assert_build_and_run(self, network, input_graph):
    # No error at construction time.
    output = network(input_graph)
    # No error at runtime.
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(output)


BROADCAST_GLOBAL_TO_EDGES = [
    [1.1, 1.2, 1.3, 1.4],
    [1.1, 1.2, 1.3, 1.4],
    [-1.1, -1.2, -1.3, -1.4],
]

BROADCAST_GLOBAL_TO_NODES = [
    [1.1, 1.2, 1.3, 1.4],
    [1.1, 1.2, 1.3, 1.4],
    [1.1, 1.2, 1.3, 1.4],
    [-1.1, -1.2, -1.3, -1.4],
    [-1.1, -1.2, -1.3, -1.4],
    [-1.1, -1.2, -1.3, -1.4],
]

SENDER_NODES_TO_EDGES = [
    [10.1, 10.2],
    [20.1, 20.2],
    [-20.1, -20.2],
]

RECEIVER_NODES_TO_EDGES = [
    [20.1, 20.2],
    [30.1, 30.2],
    [-30.1, -30.2],
]


class BroadcastersTest(GraphModuleTest):
  """Tests for the broadcasters."""

  @parameterized.named_parameters(
      ("globals_to_edges",
       blocks.broadcast_globals_to_edges, BROADCAST_GLOBAL_TO_EDGES),
      ("globals_to_nodes",
       blocks.broadcast_globals_to_nodes, BROADCAST_GLOBAL_TO_NODES),
      ("sender_nodes_to_edges",
       blocks.broadcast_sender_nodes_to_edges, SENDER_NODES_TO_EDGES),
      ("receiver_nodes_to_edges",
       blocks.broadcast_receiver_nodes_to_edges, RECEIVER_NODES_TO_EDGES),
  )
  def test_output_values(self, broadcaster, expected):
    """Test the broadcasted output value."""
    input_graph = utils_tf.data_dicts_to_graphs_tuple(
        [SMALL_GRAPH_1, SMALL_GRAPH_2])
    broadcasted = broadcaster(input_graph)
    with self.test_session() as sess:
      broadcasted_out = sess.run(broadcasted)
    self.assertNDArrayNear(
        np.array(expected, dtype=np.float32), broadcasted_out, err=1e-4)

  @parameterized.named_parameters(
      ("globals_to_edges",
       blocks.broadcast_globals_to_edges, BROADCAST_GLOBAL_TO_EDGES),
      ("globals_to_nodes",
       blocks.broadcast_globals_to_nodes, BROADCAST_GLOBAL_TO_NODES),
      ("sender_nodes_to_edges",
       blocks.broadcast_sender_nodes_to_edges, SENDER_NODES_TO_EDGES),
      ("receiver_nodes_to_edges",
       blocks.broadcast_receiver_nodes_to_edges, RECEIVER_NODES_TO_EDGES),
  )
  def test_output_values_larger_rank(self, broadcaster, expected):
    """Test the broadcasted output value."""
    input_graph = utils_tf.data_dicts_to_graphs_tuple(
        [SMALL_GRAPH_1, SMALL_GRAPH_2])
    input_graph = input_graph.map(
        lambda v: tf.reshape(v, [v.get_shape().as_list()[0]] + [2, -1]))
    broadcasted = broadcaster(input_graph)
    with self.test_session() as sess:
      broadcasted_out = sess.run(broadcasted)
    self.assertNDArrayNear(
        np.reshape(np.array(expected, dtype=np.float32),
                   [len(expected)] + [2, -1]),
        broadcasted_out,
        err=1e-4)

  @parameterized.named_parameters(
      ("globals_to_edges_no_globals",
       blocks.broadcast_globals_to_edges, ("globals",)),
      ("globals_to_nodes_no_globals",
       blocks.broadcast_globals_to_nodes, ("globals",)),
      ("sender_nodes_to_edges_none_nodes",
       blocks.broadcast_sender_nodes_to_edges, ("nodes",)),
      ("sender_nodes_to_edges_none_senders",
       blocks.broadcast_sender_nodes_to_edges,
       ("edges", "senders", "receivers")),
      ("receiver_nodes_to_edges_none_nodes",
       blocks.broadcast_receiver_nodes_to_edges, ("nodes",)),
  )
  def test_missing_field_raises_exception(self, broadcaster, none_fields):
    """Test that an error is raised if a required field is `None`."""
    input_graph = self._get_input_graph(none_fields)
    with self.assertRaisesRegexp(
        ValueError, "field cannot be None when broadcasting"):
      broadcaster(input_graph)


class ReducersTest(GraphModuleTest):
  """Tests for the reducers."""

  @parameterized.parameters(
      (blocks.unsorted_segment_min_or_zero,
       [[0., 0.],
        [0.1, -0.1],
        [0.2, -0.3],
        [0.4, -0.6],
        [0.7, -1.],
        [0.9, -0.9],
        [0., 0.]]),
      (blocks.unsorted_segment_max_or_zero,
       [[0., 0.],
        [0.1, -0.1],
        [0.3, -0.2],
        [0.6, -0.4],
        [1., -0.7],
        [0.9, -0.9],
        [0., 0.]]),
  )
  def test_output_values(self, reducer, expected_values):
    input_values_np = np.array([[0.1, -0.1],
                                [0.2, -0.2],
                                [0.3, -0.3],
                                [0.4, -0.4],
                                [0.5, -0.5],
                                [0.6, -0.6],
                                [0.7, -0.7],
                                [0.8, -0.8],
                                [0.9, -0.9],
                                [1., -1.]], dtype=np.float32)
    input_indices_np = np.array([1, 2, 2, 3, 3, 3, 4, 4, 5, 4], dtype=np.int32)
    num_groups_np = np.array(7, dtype=np.int32)

    input_indices = tf.constant(input_indices_np, dtype=tf.int32)
    input_values = tf.constant(input_values_np, dtype=tf.float32)
    num_groups = tf.constant(num_groups_np, dtype=tf.int32)

    reduced = reducer(input_values, input_indices, num_groups)

    with self.test_session() as sess:
      reduced_out = sess.run(reduced)

    self.assertNDArrayNear(
        np.array(expected_values, dtype=np.float32), reduced_out, err=1e-4)


SEGMENT_SUM_EDGES_TO_GLOBALS = [
    [302., 304., 306., 308.],
    [-101., -102., -103., -104.],
    [302., 304., 306., 308.],
    [302., 304., 306., 308.],
]

SEGMENT_SUM_NODES_TO_GLOBALS = [
    [60.3, 60.6],
    [-60.3, -60.6],
    [60.3, 60.6],
    [60.3, 60.6],
]

SEGMENT_SUM_SENT_EDGES_TO_NODES = [
    [101., 102., 103., 104.],
    [201., 202., 203., 204.],
    [0., 0., 0., 0.],
    [0., 0., 0., 0.],
    [-101., -102., -103., -104.],
    [0., 0., 0., 0.],
    [0., 0., 0., 0.],
    [302., 304., 306., 308.],
    [0., 0., 0., 0.,],
    [101., 102., 103., 104.],
    [0., 0., 0., 0.],
    [201., 202., 203., 204.],
]

SEGMENT_SUM_RECEIVED_EDGES_TO_NODES = [
    [0., 0., 0., 0.],
    [101., 102., 103., 104.],
    [201., 202., 203., 204.],
    [0., 0., 0., 0.],
    [0., 0., 0., 0.],
    [-101., -102., -103., -104.],
    [101., 102., 103., 104.],
    [0., 0., 0., 0.],
    [201., 202., 203., 204.],
    [0., 0., 0., 0.],
    [302., 304., 306., 308,],
    [0., 0., 0., 0.],
]


class FieldAggregatorsTest(GraphModuleTest):

  @parameterized.named_parameters(
      ("edges_to_globals",
       blocks.EdgesToGlobalsAggregator(tf.unsorted_segment_sum),
       SEGMENT_SUM_EDGES_TO_GLOBALS,),
      ("nodes_to_globals",
       blocks.NodesToGlobalsAggregator(tf.unsorted_segment_sum),
       SEGMENT_SUM_NODES_TO_GLOBALS,),
      ("sent_edges_to_nodes",
       blocks.SentEdgesToNodesAggregator(tf.unsorted_segment_sum),
       SEGMENT_SUM_SENT_EDGES_TO_NODES,),
      ("received_edges_to_nodes",
       blocks.ReceivedEdgesToNodesAggregator(tf.unsorted_segment_sum),
       SEGMENT_SUM_RECEIVED_EDGES_TO_NODES),
  )
  def test_output_values(self, aggregator, expected):
    input_graph = self._get_input_graph()
    aggregated = aggregator(input_graph)
    with self.test_session() as sess:
      aggregated_out = sess.run(aggregated)
    self.assertNDArrayNear(
        np.array(expected, dtype=np.float32), aggregated_out, err=1e-4)

  @parameterized.named_parameters(
      ("edges_to_globals",
       blocks.EdgesToGlobalsAggregator(tf.unsorted_segment_sum),
       SEGMENT_SUM_EDGES_TO_GLOBALS,),
      ("nodes_to_globals",
       blocks.NodesToGlobalsAggregator(tf.unsorted_segment_sum),
       SEGMENT_SUM_NODES_TO_GLOBALS,),
      ("sent_edges_to_nodes",
       blocks.SentEdgesToNodesAggregator(tf.unsorted_segment_sum),
       SEGMENT_SUM_SENT_EDGES_TO_NODES,),
      ("received_edges_to_nodes",
       blocks.ReceivedEdgesToNodesAggregator(tf.unsorted_segment_sum),
       SEGMENT_SUM_RECEIVED_EDGES_TO_NODES),
  )
  def test_output_values_larger_rank(self, aggregator, expected):
    input_graph = self._get_input_graph()
    input_graph = input_graph.map(
        lambda v: tf.reshape(v, [v.get_shape().as_list()[0]] + [2, -1]))
    aggregated = aggregator(input_graph)
    with self.test_session() as sess:
      aggregated_out = sess.run(aggregated)
    self.assertNDArrayNear(
        np.reshape(np.array(expected, dtype=np.float32),
                   [len(expected)] + [2, -1]),
        aggregated_out,
        err=1e-4)

  @parameterized.named_parameters(
      ("received edges to nodes missing edges",
       blocks.ReceivedEdgesToNodesAggregator, "edges"),
      ("sent edges to nodes missing edges",
       blocks.SentEdgesToNodesAggregator, "edges"),
      ("nodes to globals missing nodes",
       blocks.NodesToGlobalsAggregator, "nodes"),
      ("edges to globals missing nodes",
       blocks.EdgesToGlobalsAggregator, "edges"),)
  def test_missing_field_raises_exception(self, constructor, none_field):
    """Tests that aggregator fail if a required field is missing."""
    input_graph = self._get_input_graph([none_field])
    with self.assertRaisesRegexp(ValueError, none_field):
      constructor(tf.unsorted_segment_sum)(input_graph)

  @parameterized.named_parameters(
      ("received edges to nodes missing nodes and globals",
       blocks.ReceivedEdgesToNodesAggregator, ["nodes", "globals"]),
      ("sent edges to nodes missing nodes and globals",
       blocks.SentEdgesToNodesAggregator, ["nodes", "globals"]),
      ("nodes to globals missing edges and globals",
       blocks.NodesToGlobalsAggregator,
       ["edges", "receivers", "senders", "globals"]),
      ("edges to globals missing globals",
       blocks.EdgesToGlobalsAggregator, ["globals"]),
  )
  def test_unused_field_can_be_none(self, constructor, none_fields):
    """Tests that aggregator fail if a required field is missing."""
    input_graph = self._get_input_graph(none_fields)
    constructor(tf.unsorted_segment_sum)(input_graph)


class EdgeBlockTest(GraphModuleTest):

  def setUp(self):
    super(EdgeBlockTest, self).setUp()
    self._scale = 10.
    self._edge_model_fn = lambda: lambda features: features * self._scale

  @parameterized.named_parameters(
      ("all inputs", True, True, True, True),
      ("edges nodes only", True, False, False, False),
      ("receiver nodes only", False, True, False, False),
      ("sender nodes only", False, False, True, False),
      ("globals only", False, False, False, True),
      ("edges and sender nodes", True, False, True, False),
      ("receiver nodes and globals", False, True, False, True),
  )
  def test_output_values(
      self, use_edges, use_receiver_nodes, use_sender_nodes, use_globals):
    """Compares the output of an EdgeBlock to an explicit computation."""
    input_graph = self._get_input_graph()
    edge_block = blocks.EdgeBlock(
        edge_model_fn=self._edge_model_fn,
        use_edges=use_edges,
        use_receiver_nodes=use_receiver_nodes,
        use_sender_nodes=use_sender_nodes,
        use_globals=use_globals)
    output_graph = edge_block(input_graph)

    model_inputs = []
    if use_edges:
      model_inputs.append(input_graph.edges)
    if use_receiver_nodes:
      model_inputs.append(blocks.broadcast_receiver_nodes_to_edges(input_graph))
    if use_sender_nodes:
      model_inputs.append(blocks.broadcast_sender_nodes_to_edges(input_graph))
    if use_globals:
      model_inputs.append(blocks.broadcast_globals_to_edges(input_graph))

    model_inputs = tf.concat(model_inputs, axis=-1)
    self.assertEqual(input_graph.nodes, output_graph.nodes)
    self.assertEqual(input_graph.globals, output_graph.globals)

    with self.test_session() as sess:
      output_graph_out, model_inputs_out = sess.run(
          (output_graph, model_inputs))

    expected_output_edges = model_inputs_out * self._scale
    self.assertNDArrayNear(
        expected_output_edges, output_graph_out.edges, err=1e-4)

  @parameterized.named_parameters(
      ("all inputs", True, True, True, True, 12),
      ("edges only", True, False, False, False, 4),
      ("receivers only", False, True, False, False, 2),
      ("senders only", False, False, True, False, 2),
      ("globals only", False, False, False, True, 4),
  )
  def test_created_variables(self,
                             use_edges, use_receiver_nodes, use_sender_nodes,
                             use_globals, expected_first_dim_w):
    """Verifies the variable names and shapes created by an EdgeBlock."""
    output_size = 10
    expected_var_shapes_dict = {
        "edge_block/mlp/linear_0/b:0": [output_size],
        "edge_block/mlp/linear_0/w:0": [expected_first_dim_w, output_size]}

    input_graph = self._get_input_graph()

    edge_block = blocks.EdgeBlock(
        edge_model_fn=functools.partial(snt.nets.MLP,
                                        output_sizes=[output_size]),
        use_edges=use_edges,
        use_receiver_nodes=use_receiver_nodes,
        use_sender_nodes=use_sender_nodes,
        use_globals=use_globals)
    edge_block(input_graph)

    variables = edge_block.get_variables()
    var_shapes_dict = {var.name: var.get_shape().as_list() for var in variables}
    self.assertDictEqual(expected_var_shapes_dict, var_shapes_dict)

  @parameterized.named_parameters(
      ("missing node (receivers only)", False, True, False, False, ("nodes",)),
      ("missing node (senders only)", False, False, True, False, ("nodes",)),
      ("missing edge data", True, False, False, False, ("edges",)),
      ("missing edges (but no edge consumption)", False, True, True, False,
       ("edges", "senders", "receivers")),
      ("missing globals", False, False, False, True, ("globals",)),
  )
  def test_missing_field_raises_exception(
      self, use_edges, use_receiver_nodes, use_sender_nodes, use_globals,
      none_fields):
    """Checks that missing a required field raises an exception."""
    input_graph = self._get_input_graph(none_fields)
    edge_block = blocks.EdgeBlock(
        edge_model_fn=self._edge_model_fn,
        use_edges=use_edges,
        use_receiver_nodes=use_receiver_nodes,
        use_sender_nodes=use_sender_nodes,
        use_globals=use_globals)
    with self.assertRaisesRegexp(ValueError, "field cannot be None"):
      edge_block(input_graph)

  def test_compatible_higher_rank_no_raise(self):
    """No exception should occur with higher ranks tensors."""
    input_graph = self._get_shaped_input_graph()
    input_graph = input_graph.map(lambda v: tf.transpose(v, [0, 2, 1, 3]))
    network = blocks.EdgeBlock(
        functools.partial(snt.Conv2D, output_channels=10, kernel_shape=[3, 3]))
    self._assert_build_and_run(network, input_graph)

  @parameterized.named_parameters(
      ("mismatched edges and r. nodes", True, True, False, False, "nodes"),
      ("mismatched edges and s. nodes", True, False, True, False, "nodes"),
      ("mismatched edges and globals", True, False, False, True, "edges"),
      ("mismatched nodes and globals", False, True, True, True, "globals"),
  )
  def test_incompatible_higher_rank_inputs_raises(self,
                                                  use_edges,
                                                  use_receiver_nodes,
                                                  use_sender_nodes,
                                                  use_globals,
                                                  field):
    """A exception should be raised if the inputs have incompatible shapes."""
    input_graph = self._get_shaped_input_graph()
    input_graph = input_graph.replace(
        **{field: tf.transpose(getattr(input_graph, field), [0, 2, 1, 3])})
    network = blocks.EdgeBlock(
        functools.partial(snt.Conv2D, output_channels=10, kernel_shape=[3, 3]),
        use_edges=use_edges,
        use_receiver_nodes=use_receiver_nodes,
        use_sender_nodes=use_sender_nodes,
        use_globals=use_globals
    )
    with self.assertRaisesRegexp(ValueError, "in both shapes must be equal"):
      network(input_graph)

  @parameterized.named_parameters(
      ("mismatched nodes", True, False, False, True, "nodes"),
      ("mismatched edges", False, True, True, True, "edges"),
      ("mismatched globals", True, True, True, False, "globals"),
  )
  def test_incompatible_higher_rank_inputs_no_raise(self,
                                                    use_edges,
                                                    use_receiver_nodes,
                                                    use_sender_nodes,
                                                    use_globals,
                                                    field):
    """No exception should occur if a differently shapped field is not used."""
    input_graph = self._get_shaped_input_graph()
    input_graph = input_graph.replace(
        **{field: tf.transpose(getattr(input_graph, field), [0, 2, 1, 3])})
    network = blocks.EdgeBlock(
        functools.partial(snt.Conv2D, output_channels=10, kernel_shape=[3, 3]),
        use_edges=use_edges,
        use_receiver_nodes=use_receiver_nodes,
        use_sender_nodes=use_sender_nodes,
        use_globals=use_globals
    )
    self._assert_build_and_run(network, input_graph)

  @parameterized.named_parameters(
      ("no edges", False, True, True, "edges"),
      ("no nodes", True, False, True, "nodes"),
      ("no globals", True, True, False, "globals"),
  )
  def test_unused_field_can_be_none(
      self, use_edges, use_nodes, use_globals, none_field):
    """Checks that computation can handle non-necessary fields left None."""
    input_graph = self._get_input_graph([none_field])
    edge_block = blocks.EdgeBlock(
        edge_model_fn=self._edge_model_fn,
        use_edges=use_edges,
        use_receiver_nodes=use_nodes,
        use_sender_nodes=use_nodes,
        use_globals=use_globals)
    output_graph = edge_block(input_graph)

    model_inputs = []
    if use_edges:
      model_inputs.append(input_graph.edges)
    if use_nodes:
      model_inputs.append(blocks.broadcast_receiver_nodes_to_edges(input_graph))
      model_inputs.append(blocks.broadcast_sender_nodes_to_edges(input_graph))
    if use_globals:
      model_inputs.append(blocks.broadcast_globals_to_edges(input_graph))

    model_inputs = tf.concat(model_inputs, axis=-1)
    self.assertEqual(input_graph.nodes, output_graph.nodes)
    self.assertEqual(input_graph.globals, output_graph.globals)

    with self.test_session() as sess:
      actual_edges, model_inputs_out = sess.run(
          (output_graph.edges, model_inputs))

    expected_output_edges = model_inputs_out * self._scale
    self.assertNDArrayNear(expected_output_edges, actual_edges, err=1e-4)

  def test_no_input_raises_exception(self):
    """Checks that receiving no input raises an exception."""
    with self.assertRaisesRegexp(ValueError, "At least one of "):
      blocks.EdgeBlock(
          edge_model_fn=self._edge_model_fn,
          use_edges=False,
          use_receiver_nodes=False,
          use_sender_nodes=False,
          use_globals=False)


class NodeBlockTest(GraphModuleTest):

  def setUp(self):
    super(NodeBlockTest, self).setUp()
    self._scale = 10.
    self._node_model_fn = lambda: lambda features: features * self._scale

  @parameterized.named_parameters(
      ("all inputs, custom reductions", True, True, True, True,
       tf.unsorted_segment_sum, tf.unsorted_segment_mean),
      ("received edges only, blocks reducer",
       True, False, False, False, blocks.unsorted_segment_max_or_zero, None),
      ("sent edges only, custom reduction",
       False, True, False, False, None, tf.unsorted_segment_prod),
      ("nodes only",
       False, False, True, False, None, None),
      ("globals only",
       False, False, False, True, None, None),
      ("received edges and nodes, custom reductions",
       True, False, True, False,
       blocks.unsorted_segment_min_or_zero, tf.unsorted_segment_prod),
      ("sent edges and globals, custom reduction",
       False, True, False, True, None, blocks.unsorted_segment_min_or_zero),
  )
  def test_output_values(
      self, use_received_edges, use_sent_edges, use_nodes,
      use_globals, received_edges_reducer, sent_edges_reducer):
    """Compares the output of a NodeBlock to an explicit computation."""
    input_graph = self._get_input_graph()
    node_block = blocks.NodeBlock(
        node_model_fn=self._node_model_fn,
        use_received_edges=use_received_edges,
        use_sent_edges=use_sent_edges,
        use_nodes=use_nodes,
        use_globals=use_globals,
        received_edges_reducer=received_edges_reducer,
        sent_edges_reducer=sent_edges_reducer)
    output_graph = node_block(input_graph)

    model_inputs = []
    if use_received_edges:
      model_inputs.append(
          blocks.ReceivedEdgesToNodesAggregator(
              received_edges_reducer)(input_graph))
    if use_sent_edges:
      model_inputs.append(
          blocks.SentEdgesToNodesAggregator(sent_edges_reducer)(input_graph))
    if use_nodes:
      model_inputs.append(input_graph.nodes)
    if use_globals:
      model_inputs.append(blocks.broadcast_globals_to_nodes(input_graph))

    model_inputs = tf.concat(model_inputs, axis=-1)
    self.assertEqual(input_graph.edges, output_graph.edges)
    self.assertEqual(input_graph.globals, output_graph.globals)

    with self.test_session() as sess:
      output_graph_out, model_inputs_out = sess.run(
          (output_graph, model_inputs))

    expected_output_nodes = model_inputs_out * self._scale
    self.assertNDArrayNear(
        expected_output_nodes, output_graph_out.nodes, err=1e-4)

  @parameterized.named_parameters(
      ("all inputs", True, True, True, True, 14),
      ("received edges only", True, False, False, False, 4),
      ("sent edges only", False, True, False, False, 4),
      ("nodes only", False, False, True, False, 2),
      ("globals only", False, False, False, True, 4),
  )
  def test_created_variables(self,
                             use_received_edges, use_sent_edges, use_nodes,
                             use_globals, expected_first_dim_w):
    """Verifies the variable names and shapes created by a NodeBlock."""
    output_size = 10
    expected_var_shapes_dict = {
        "node_block/mlp/linear_0/b:0": [output_size],
        "node_block/mlp/linear_0/w:0": [expected_first_dim_w, output_size]}

    input_graph = self._get_input_graph()

    node_block = blocks.NodeBlock(
        node_model_fn=functools.partial(snt.nets.MLP,
                                        output_sizes=[output_size]),
        use_received_edges=use_received_edges,
        use_sent_edges=use_sent_edges,
        use_nodes=use_nodes,
        use_globals=use_globals)

    node_block(input_graph)

    variables = node_block.get_variables()
    var_shapes_dict = {var.name: var.get_shape().as_list() for var in variables}
    self.assertDictEqual(expected_var_shapes_dict, var_shapes_dict)

  @parameterized.named_parameters(
      ("missing nodes", False, False, True, False, ("nodes",)),
      ("missing edge data (receivers only)",
       True, False, False, False, ("edges",)),
      ("missing edge data (senders only)",
       False, True, False, False, ("edges",)),
      ("missing globals", False, False, False, True, ("globals",)),
  )
  def test_missing_field_raises_exception(
      self, use_received_edges, use_sent_edges, use_nodes, use_globals,
      none_fields):
    """Checks that missing a required field raises an exception."""
    input_graph = self._get_input_graph(none_fields)
    node_block = blocks.NodeBlock(
        node_model_fn=self._node_model_fn,
        use_received_edges=use_received_edges,
        use_sent_edges=use_sent_edges,
        use_nodes=use_nodes,
        use_globals=use_globals)
    with self.assertRaisesRegexp(ValueError, "field cannot be None"):
      node_block(input_graph)

  @parameterized.named_parameters(
      ("no received edges reducer", True, False, None, tf.unsorted_segment_sum),
      ("no sent edges reducer", False, True, tf.unsorted_segment_sum, None),
  )
  def test_missing_aggregation_raises_exception(
      self, use_received_edges, use_sent_edges,
      received_edges_reducer, sent_edges_reducer):
    """Checks that missing a required aggregation argument raises an error."""
    with self.assertRaisesRegexp(ValueError, "should not be None"):
      blocks.NodeBlock(
          node_model_fn=self._node_model_fn,
          use_received_edges=use_received_edges,
          use_sent_edges=use_sent_edges,
          use_nodes=False,
          use_globals=False,
          received_edges_reducer=received_edges_reducer,
          sent_edges_reducer=sent_edges_reducer)

  def test_compatible_higher_rank_no_raise(self):
    """No exception should occur with higher ranks tensors."""
    input_graph = self._get_shaped_input_graph()
    input_graph = input_graph.map(lambda v: tf.transpose(v, [0, 2, 1, 3]))
    network = blocks.NodeBlock(
        functools.partial(snt.Conv2D, output_channels=10, kernel_shape=[3, 3]))
    self._assert_build_and_run(network, input_graph)

  @parameterized.named_parameters(
      ("mismatched nodes and r. edges", True, False, True, False, "edges"),
      ("mismatched nodes and s. edges", True, False, True, False, "edges"),
      ("mismatched edges and globals", True, False, False, True, "globals"),
      ("mismatched nodes and globals", False, True, True, True, "globals"),
  )
  def test_incompatible_higher_rank_inputs_raises(self,
                                                  use_received_edges,
                                                  use_sent_edges,
                                                  use_nodes,
                                                  use_globals,
                                                  field):
    """A exception should be raised if the inputs have incompatible shapes."""
    input_graph = self._get_shaped_input_graph()
    input_graph = input_graph.replace(
        **{field: tf.transpose(getattr(input_graph, field), [0, 2, 1, 3])})
    network = blocks.NodeBlock(
        functools.partial(snt.Conv2D, output_channels=10, kernel_shape=[3, 3]),
        use_received_edges=use_received_edges,
        use_sent_edges=use_sent_edges,
        use_nodes=use_nodes,
        use_globals=use_globals
    )
    with self.assertRaisesRegexp(ValueError, "in both shapes must be equal"):
      network(input_graph)

  @parameterized.named_parameters(
      ("mismatched nodes", True, True, False, True, "nodes"),
      ("mismatched edges", False, False, True, True, "edges"),
      ("mismatched globals", True, True, True, False, "globals"),
  )
  def test_incompatible_higher_rank_inputs_no_raise(self,
                                                    use_received_edges,
                                                    use_sent_edges,
                                                    use_nodes,
                                                    use_globals,
                                                    field):
    """No exception should occur if a differently shapped field is not used."""
    input_graph = self._get_shaped_input_graph()
    input_graph = input_graph.replace(
        **{field: tf.transpose(getattr(input_graph, field), [0, 2, 1, 3])})
    network = blocks.NodeBlock(
        functools.partial(snt.Conv2D, output_channels=10, kernel_shape=[3, 3]),
        use_received_edges=use_received_edges,
        use_sent_edges=use_sent_edges,
        use_nodes=use_nodes,
        use_globals=use_globals
    )
    self._assert_build_and_run(network, input_graph)

  @parameterized.named_parameters(
      ("no edges", False, True, True, "edges"),
      ("no nodes", True, False, True, "nodes"),
      ("no globals", True, True, False, "globals"),
  )
  def test_unused_field_can_be_none(
      self, use_edges, use_nodes, use_globals, none_field):
    """Checks that computation can handle non-necessary fields left None."""
    input_graph = self._get_input_graph([none_field])
    node_block = blocks.NodeBlock(
        node_model_fn=self._node_model_fn,
        use_received_edges=use_edges,
        use_sent_edges=use_edges,
        use_nodes=use_nodes,
        use_globals=use_globals)
    output_graph = node_block(input_graph)

    model_inputs = []
    if use_edges:
      model_inputs.append(
          blocks.ReceivedEdgesToNodesAggregator(
              tf.unsorted_segment_sum)(input_graph))
      model_inputs.append(
          blocks.SentEdgesToNodesAggregator(
              tf.unsorted_segment_sum)(input_graph))
    if use_nodes:
      model_inputs.append(input_graph.nodes)
    if use_globals:
      model_inputs.append(blocks.broadcast_globals_to_nodes(input_graph))

    model_inputs = tf.concat(model_inputs, axis=-1)
    self.assertEqual(input_graph.edges, output_graph.edges)
    self.assertEqual(input_graph.globals, output_graph.globals)

    with self.test_session() as sess:
      actual_nodes, model_inputs_out = sess.run(
          (output_graph.nodes, model_inputs))

    expected_output_nodes = model_inputs_out * self._scale
    self.assertNDArrayNear(expected_output_nodes, actual_nodes, err=1e-4)

  def test_no_input_raises_exception(self):
    """Checks that receiving no input raises an exception."""
    with self.assertRaisesRegexp(ValueError, "At least one of "):
      blocks.NodeBlock(
          node_model_fn=self._node_model_fn,
          use_received_edges=False,
          use_sent_edges=False,
          use_nodes=False,
          use_globals=False)


class GlobalBlockTest(GraphModuleTest):
  """Tests for the GlobalBlock."""

  def setUp(self):
    super(GlobalBlockTest, self).setUp()
    self._scale = 10.
    self._global_model_fn = lambda: lambda features: features * self._scale

  @parameterized.named_parameters(
      ("all_inputs, custom reductions",
       True, True, True, tf.unsorted_segment_sum, tf.unsorted_segment_mean),
      ("edges only, blocks reducer",
       True, False, False, blocks.unsorted_segment_max_or_zero, None),
      ("nodes only, custom reduction",
       False, True, False, None, tf.unsorted_segment_prod),
      ("globals only",
       False, False, True, None, None),
      ("edges and nodes, blocks reducer",
       True, True, False, blocks.unsorted_segment_min_or_zero,
       tf.unsorted_segment_prod),
      ("nodes and globals, blocks reducer",
       False, True, True, None, blocks.unsorted_segment_min_or_zero),
  )
  def test_output_values(
      self, use_edges, use_nodes, use_globals, edges_reducer, nodes_reducer):
    """Compares the output of a GlobalBlock to an explicit computation."""
    input_graph = self._get_input_graph()
    global_block = blocks.GlobalBlock(
        global_model_fn=self._global_model_fn,
        use_edges=use_edges,
        use_nodes=use_nodes,
        use_globals=use_globals,
        edges_reducer=edges_reducer,
        nodes_reducer=nodes_reducer)
    output_graph = global_block(input_graph)

    model_inputs = []
    if use_edges:
      model_inputs.append(
          blocks.EdgesToGlobalsAggregator(edges_reducer)(input_graph))
    if use_nodes:
      model_inputs.append(
          blocks.NodesToGlobalsAggregator(nodes_reducer)(input_graph))
    if use_globals:
      model_inputs.append(input_graph.globals)

    model_inputs = tf.concat(model_inputs, axis=-1)
    self.assertEqual(input_graph.edges, output_graph.edges)
    self.assertEqual(input_graph.nodes, output_graph.nodes)

    with self.test_session() as sess:
      output_graph_out, model_inputs_out = sess.run(
          (output_graph, model_inputs))

    expected_output_globals = model_inputs_out * self._scale
    self.assertNDArrayNear(
        expected_output_globals, output_graph_out.globals, err=1e-4)

  @parameterized.named_parameters(
      ("default", True, True, True, 10),
      ("use edges only", True, False, False, 4),
      ("use nodes only", False, True, False, 2),
      ("use globals only", False, False, True, 4),
  )
  def test_created_variables(self, use_edges, use_nodes,
                             use_globals, expected_first_dim_w):
    """Verifies the variable names and shapes created by a GlobalBlock."""
    output_size = 10
    expected_var_shapes_dict = {
        "global_block/mlp/linear_0/b:0": [output_size],
        "global_block/mlp/linear_0/w:0": [expected_first_dim_w, output_size]}

    input_graph = self._get_input_graph()

    global_block = blocks.GlobalBlock(
        global_model_fn=functools.partial(snt.nets.MLP,
                                          output_sizes=[output_size]),
        use_edges=use_edges,
        use_nodes=use_nodes,
        use_globals=use_globals)

    global_block(input_graph)

    variables = global_block.get_variables()
    var_shapes_dict = {var.name: var.get_shape().as_list() for var in variables}
    self.assertDictEqual(expected_var_shapes_dict, var_shapes_dict)

  @parameterized.named_parameters(
      ("missing edges", True, False, False, "edges"),
      ("missing nodes", False, True, False, "nodes"),
      ("missing globals", False, False, True, "globals"),
  )
  def test_missing_field_raises_exception(
      self, use_edges, use_nodes, use_globals, none_field):
    """Checks that missing a required field raises an exception."""
    input_graph = self._get_input_graph([none_field])
    global_block = blocks.GlobalBlock(
        global_model_fn=self._global_model_fn,
        use_edges=use_edges,
        use_nodes=use_nodes,
        use_globals=use_globals)
    with self.assertRaisesRegexp(ValueError, "field cannot be None"):
      global_block(input_graph)

  @parameterized.named_parameters(
      ("no edges", False, True, True, "edges"),
      ("no nodes", True, False, True, "nodes"),
      ("no globals", True, True, False, "globals"),
  )
  def test_unused_field_can_be_none(
      self, use_edges, use_nodes, use_globals, none_field):
    """Checks that computation can handle non-necessary fields left None."""
    input_graph = self._get_input_graph([none_field])
    global_block = blocks.GlobalBlock(
        global_model_fn=self._global_model_fn,
        use_edges=use_edges,
        use_nodes=use_nodes,
        use_globals=use_globals)
    output_graph = global_block(input_graph)

    model_inputs = []
    if use_edges:
      model_inputs.append(
          blocks.EdgesToGlobalsAggregator(tf.unsorted_segment_sum)(input_graph))
    if use_nodes:
      model_inputs.append(
          blocks.NodesToGlobalsAggregator(tf.unsorted_segment_sum)(input_graph))
    if use_globals:
      model_inputs.append(input_graph.globals)

    model_inputs = tf.concat(model_inputs, axis=-1)
    self.assertEqual(input_graph.edges, output_graph.edges)
    self.assertEqual(input_graph.nodes, output_graph.nodes)

    with self.test_session() as sess:
      actual_globals, model_inputs_out = sess.run(
          (output_graph.globals, model_inputs))

    expected_output_globals = model_inputs_out * self._scale
    self.assertNDArrayNear(expected_output_globals, actual_globals, err=1e-4)

  def test_compatible_higher_rank_no_raise(self):
    """No exception should occur with higher ranks tensors."""
    input_graph = self._get_shaped_input_graph()
    input_graph = input_graph.map(lambda v: tf.transpose(v, [0, 2, 1, 3]))
    network = blocks.GlobalBlock(
        functools.partial(snt.Conv2D, output_channels=10, kernel_shape=[3, 3]))
    self._assert_build_and_run(network, input_graph)

  @parameterized.named_parameters(
      ("mismatched nodes and edges", True, True, False, "edges"),
      ("mismatched edges and globals", True, False, True, "globals"),
      ("mismatched nodes and globals", False, True, True, "globals"),
  )
  def test_incompatible_higher_rank_inputs_raises(self,
                                                  use_edges,
                                                  use_nodes,
                                                  use_globals,
                                                  field):
    """A exception should be raised if the inputs have incompatible shapes."""
    input_graph = self._get_shaped_input_graph()
    input_graph = input_graph.replace(
        **{field: tf.transpose(getattr(input_graph, field), [0, 2, 1, 3])})
    network = blocks.GlobalBlock(
        functools.partial(snt.Conv2D, output_channels=10, kernel_shape=[3, 3]),
        use_edges=use_edges,
        use_nodes=use_nodes,
        use_globals=use_globals
    )
    with self.assertRaisesRegexp(ValueError, "in both shapes must be equal"):
      network(input_graph)

  @parameterized.named_parameters(
      ("mismatched nodes", True, False, True, "nodes"),
      ("mismatched edges", False, True, True, "edges"),
      ("mismatched globals", True, True, False, "globals"),
  )
  def test_incompatible_higher_rank_inputs_no_raise(self,
                                                    use_edges,
                                                    use_nodes,
                                                    use_globals,
                                                    field):
    """No exception should occur if a differently shapped field is not used."""
    input_graph = self._get_shaped_input_graph()
    input_graph = input_graph.replace(
        **{field: tf.transpose(getattr(input_graph, field), [0, 2, 1, 3])})
    network = blocks.GlobalBlock(
        functools.partial(snt.Conv2D, output_channels=10, kernel_shape=[3, 3]),
        use_edges=use_edges,
        use_nodes=use_nodes,
        use_globals=use_globals
    )
    self._assert_build_and_run(network, input_graph)

  def test_no_input_raises_exception(self):
    """Checks that receiving no input raises an exception."""
    with self.assertRaisesRegexp(ValueError, "At least one of "):
      blocks.GlobalBlock(
          global_model_fn=self._global_model_fn,
          use_edges=False,
          use_nodes=False,
          use_globals=False)

  @parameterized.named_parameters(
      ("missing edges reducer", True, False, None, tf.unsorted_segment_sum),
      ("missing nodes reducer", False, True, tf.unsorted_segment_sum, None),
  )
  def test_missing_aggregation_raises_exception(
      self, use_edges, use_nodes, edges_reducer,
      nodes_reducer):
    """Checks that missing a required aggregation argument raises an error."""
    with self.assertRaisesRegexp(ValueError, "should not be None"):
      blocks.GlobalBlock(
          global_model_fn=self._global_model_fn,
          use_edges=use_edges,
          use_nodes=use_nodes,
          use_globals=False,
          edges_reducer=edges_reducer,
          nodes_reducer=nodes_reducer)


def _mask_leading_dimension(tensor):
  return tf.placeholder_with_default(tensor,
                                     [None] + tensor.get_shape().as_list()[1:])


class CommonBlockTests(GraphModuleTest):
  """Tests that are common to the EdgeBlock, NodeBlock and GlobalBlock."""

  @parameterized.named_parameters(
      ("edge block", blocks.EdgeBlock),
      ("node block", blocks.NodeBlock),
      ("global block", blocks.GlobalBlock),
  )
  def test_dynamic_batch_sizes(self, block_constructor):
    """Checks that all batch sizes are as expected through a GraphNetwork."""
    input_graph = self._get_input_graph()
    placeholders = input_graph.map(_mask_leading_dimension, graphs.ALL_FIELDS)
    model = block_constructor(
        functools.partial(snt.nets.MLP, output_sizes=[10]))
    output = model(placeholders)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      other_input_graph = utils_np.data_dicts_to_graphs_tuple(
          [SMALL_GRAPH_1, SMALL_GRAPH_2])
      actual = sess.run(output, {placeholders: other_input_graph})
    for k, v in other_input_graph._asdict().items():
      self.assertEqual(v.shape[0], getattr(actual, k).shape[0])

  @parameterized.named_parameters(
      ("float64 data, edge block", tf.float64, tf.int32, blocks.EdgeBlock),
      ("int64 indices, edge block", tf.float32, tf.int64, blocks.EdgeBlock),
      ("float64 data, node block", tf.float64, tf.int32, blocks.NodeBlock),
      ("int64 indices, node block", tf.float32, tf.int64, blocks.NodeBlock),
      ("float64 data, global block", tf.float64, tf.int32, blocks.GlobalBlock),
      ("int64 indices, global block", tf.float32, tf.int64, blocks.GlobalBlock),
  )
  def test_dtypes(self, data_dtype, indices_dtype, block_constructor):
    """Checks that all the output types are as expected for blocks."""
    input_graph = self._get_input_graph()
    input_graph = input_graph.map(lambda v: tf.cast(v, data_dtype),
                                  ["nodes", "edges", "globals"])
    input_graph = input_graph.map(lambda v: tf.cast(v, indices_dtype),
                                  ["receivers", "senders"])
    model = block_constructor(
        functools.partial(snt.nets.MLP, output_sizes=[10]))
    output = model(input_graph)
    for field in ["nodes", "globals", "edges"]:
      self.assertEqual(data_dtype, getattr(output, field).dtype)
    for field in ["receivers", "senders"]:
      self.assertEqual(indices_dtype, getattr(output, field).dtype)

if __name__ == "__main__":
  tf.test.main()
