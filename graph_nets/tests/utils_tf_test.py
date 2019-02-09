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

"""Tests for utils_tf.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from graph_nets import graphs
from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets.tests import test_utils
import networkx as nx
import numpy as np
from six.moves import range
import tensorflow as tf


class RepeatTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for `_axis_to_inside`, `_inside_to_axis` and `repeat`."""

  def test_axis_to_inside(self):
    t = np.arange(24).reshape(3, 4, 2)
    tensor = tf.constant(t)
    axis = 1
    expected = np.transpose(t, (axis, 0, 2))
    op = utils_tf._axis_to_inside(tensor, axis)
    with self.test_session() as sess:
      actual = sess.run(op)
    self.assertAllEqual(expected, actual)

  def test_inside_to_axis(self):
    t = np.arange(24).reshape(3, 4, 2)
    tensor = tf.constant(t)
    axis = 1
    expected = np.transpose(t, (1, 0, 2))
    op = utils_tf._inside_to_axis(tensor, axis)
    with self.test_session() as sess:
      actual = sess.run(op)
    self.assertAllEqual(expected, actual)

  def test_repeat(self):
    t = np.arange(24).reshape(3, 2, 4)
    tensor = tf.constant(t)
    repeats = [2, 3]
    axis = 1
    expected = np.repeat(t, repeats, axis=axis)
    op = utils_tf.repeat(tensor, repeats, axis=axis)
    with self.test_session() as sess:
      actual = sess.run(op)
    self.assertAllEqual(expected, actual)

  @parameterized.named_parameters(("default", "custom_name", None),
                                  ("custom", None, "repeat"))
  def test_name_scope(self, name, expected_name):
    kwargs = {"name": name} if name else {}
    expected_name = expected_name if expected_name else name

    t = tf.zeros([3, 2, 4])
    indices = tf.constant([2, 3])
    with test_utils.assert_new_op_prefixes(self, expected_name + "/"):
      utils_tf.repeat(t, indices, axis=1, **kwargs)


def _generate_graph(batch_index, n_nodes=4, add_edges=True):
  graph = nx.DiGraph()
  for node in range(n_nodes):
    node_data = {"features": np.array([node, batch_index], dtype=np.float32)}
    graph.add_node(node, **node_data)
  if add_edges:
    for edge, (receiver, sender) in enumerate(zip([0, 0, 1], [1, 2, 3])):
      if sender < n_nodes and receiver < n_nodes:
        edge_data = np.array([edge, edge + 1, batch_index], dtype=np.float64)
        graph.add_edge(sender, receiver, features=edge_data, index=edge)
  graph.graph["features"] = np.array([batch_index], dtype=np.float32)
  return graph


class ConcatTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for `concat`, along various axis."""

  @parameterized.named_parameters(
      ("no nones", []), ("stateless graph", ["nodes", "edges", "globals"]),
      ("no edges", ["edges", "receivers", "senders"]))
  def test_concat_first_axis(self, none_fields):
    graph_0 = utils_np.networkxs_to_graphs_tuple(
        [_generate_graph(0, 3), _generate_graph(1, 2)])
    graph_1 = utils_np.networkxs_to_graphs_tuple([_generate_graph(2, 2)])
    graph_2 = utils_np.networkxs_to_graphs_tuple([_generate_graph(3, 3)])
    graphs_ = [
        gr.map(tf.convert_to_tensor, graphs.ALL_FIELDS)
        for gr in [graph_0, graph_1, graph_2]
    ]
    graphs_ = [gr.map(lambda _: None, none_fields) for gr in graphs_]
    concat_graph = utils_tf.concat(graphs_, axis=0)
    for none_field in none_fields:
      self.assertEqual(None, getattr(concat_graph, none_field))
    concat_graph = concat_graph.map(tf.no_op, none_fields)
    with self.test_session() as sess:
      concat_graph = sess.run(concat_graph)
    if "nodes" not in none_fields:
      self.assertAllEqual(
          np.array([0, 1, 2, 0, 1, 0, 1, 0, 1, 2]),
          [x[0] for x in concat_graph.nodes])
      self.assertAllEqual(
          np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 3]),
          [x[1] for x in concat_graph.nodes])
    if "edges" not in none_fields:
      self.assertAllEqual(
          np.array([0, 1, 0, 0, 0, 1]), [x[0] for x in concat_graph.edges])
      self.assertAllEqual(
          np.array([0, 0, 1, 2, 3, 3]), [x[2] for x in concat_graph.edges])
    self.assertAllEqual(np.array([3, 2, 2, 3]), concat_graph.n_node)
    self.assertAllEqual(np.array([2, 1, 1, 2]), concat_graph.n_edge)
    if "senders" not in none_fields:
      # [1, 2], [1], [1], [1, 2] and 3, 2, 2, 3 nodes
      # So we are summing [1, 2, 1, 1, 2] with [0, 0, 3, 5, 7, 7]
      self.assertAllEqual(np.array([1, 2, 4, 6, 8, 9]), concat_graph.senders)
    if "receivers" not in none_fields:
      # [0, 0], [0], [0], [0, 0] and 3, 2, 2, 3 nodes
      # So we are summing [0, 0, 0, 0, 0, 0] with [0, 0, 3, 5, 7, 7]
      self.assertAllEqual(np.array([0, 0, 3, 5, 7, 7]), concat_graph.receivers)
    if "globals" not in none_fields:
      self.assertAllEqual(np.array([[0], [1], [2], [3]]), concat_graph.globals)

  def test_concat_last_axis(self):
    graph0 = utils_np.networkxs_to_graphs_tuple(
        [_generate_graph(0, 3), _generate_graph(1, 2)])
    graph1 = utils_np.networkxs_to_graphs_tuple(
        [_generate_graph(2, 3), _generate_graph(3, 2)])
    graph0 = graph0.map(tf.convert_to_tensor, graphs.ALL_FIELDS)
    graph1 = graph1.map(tf.convert_to_tensor, graphs.ALL_FIELDS)
    with self.test_session() as sess:
      concat_graph = sess.run(utils_tf.concat([graph0, graph1], axis=-1))
    self.assertAllEqual(
        np.array([[0, 0, 0, 2], [1, 0, 1, 2], [2, 0, 2, 2], [0, 1, 0, 3],
                  [1, 1, 1, 3]]), concat_graph.nodes)
    self.assertAllEqual(
        np.array([[0, 1, 0, 0, 1, 2], [1, 2, 0, 1, 2, 2], [0, 1, 1, 0, 1, 3]]),
        concat_graph.edges)
    self.assertAllEqual(np.array([3, 2]), concat_graph.n_node)
    self.assertAllEqual(np.array([2, 1]), concat_graph.n_edge)
    self.assertAllEqual(np.array([1, 2, 4]), concat_graph.senders)
    self.assertAllEqual(np.array([0, 0, 3]), concat_graph.receivers)
    self.assertAllEqual(np.array([[0, 2], [1, 3]]), concat_graph.globals)


class BuildPlaceholdersTest(test_utils.GraphsTest, parameterized.TestCase):

  def _assert_expected_shapes(self, placeholders, but_for=None,
                              num_graphs=None):
    if but_for is None:
      but_for = []
    if "nodes" not in but_for:
      self.assertAllEqual([None, 2], placeholders.nodes.shape.as_list())
    if "edges" not in but_for:
      self.assertAllEqual([None, 3], placeholders.edges.shape.as_list())
    if "globals" not in but_for:
      self.assertAllEqual([num_graphs, 1], placeholders.globals.shape.as_list())
    for key in ["receivers", "senders"]:
      if key not in but_for:
        self.assertAllEqual([None], getattr(placeholders, key).shape.as_list())
    for key in ["n_node", "n_edge"]:
      if key not in but_for:
        self.assertAllEqual([num_graphs],
                            getattr(placeholders, key).shape.as_list())

  @parameterized.named_parameters(
      ("all_field_defined", [], False),
      ("no features", ["nodes", "edges", "globals"], False),
      ("no edges", ["edges", "receivers", "senders"], False),
      ("dynamic", [], True))
  def test_build_placeholders_from_specs(self,
                                         none_fields,
                                         force_dynamic_num_graphs=False):
    num_graphs = 3
    shapes = graphs.GraphsTuple(
        nodes=[3, 4],
        edges=[2],
        globals=[num_graphs, 4, 6],
        receivers=[None],
        senders=[18],
        n_node=[num_graphs],
        n_edge=[num_graphs],
    )
    dtypes = graphs.GraphsTuple(
        nodes=tf.float64,
        edges=tf.int32,
        globals=tf.float32,
        receivers=tf.int64,
        senders=tf.int64,
        n_node=tf.int32,
        n_edge=tf.int64)
    dtypes = dtypes.map(lambda _: None, none_fields)
    shapes = shapes.map(lambda _: None, none_fields)
    placeholders = utils_tf._build_placeholders_from_specs(
        dtypes, shapes, force_dynamic_num_graphs=force_dynamic_num_graphs)
    for k in graphs.ALL_FIELDS:
      placeholder = getattr(placeholders, k)
      if k in none_fields:
        self.assertEqual(None, placeholder)
      else:
        self.assertEqual(getattr(dtypes, k), placeholder.dtype)
        if k not in ["n_node", "n_edge", "globals"] or force_dynamic_num_graphs:
          self.assertAllEqual([None] + getattr(shapes, k)[1:],
                              placeholder.shape.as_list())
        else:
          self.assertAllEqual([num_graphs] + getattr(shapes, k)[1:],
                              placeholder.shape.as_list())

  @parameterized.named_parameters(("static_num_graphs", False),
                                  ("dynamic_num_graphs", True))
  def test_placeholders_from_data_dicts(self, force_dynamic_num_graphs):
    num_graphs = len(self.graphs_dicts_in)
    placeholders = utils_tf.placeholders_from_data_dicts(
        self.graphs_dicts_in, force_dynamic_num_graphs=force_dynamic_num_graphs)
    self.assertAllEqual([None, 7, 11], placeholders.nodes.shape.as_list())
    self.assertAllEqual([None, 13, 14], placeholders.edges.shape.as_list())

    global_shape = placeholders.globals.shape.as_list()
    if force_dynamic_num_graphs:
      self.assertAllEqual([None, 5, 3], global_shape)
    else:
      self.assertAllEqual([num_graphs, 5, 3], global_shape)
    for key in ["receivers", "senders"]:
      self.assertAllEqual([None], getattr(placeholders, key).shape.as_list())
    for key in ["n_node", "n_edge"]:
      shape = getattr(placeholders, key).shape.as_list()
      if force_dynamic_num_graphs:
        self.assertAllEqual([None], shape)
      else:
        self.assertAllEqual([num_graphs], shape)

  def test_placeholders_from_networkxs(self):
    num_graphs = 16
    networkxs = [
        _generate_graph(batch_index) for batch_index in range(num_graphs)
    ]
    placeholders = utils_tf.placeholders_from_networkxs(
        networkxs, force_dynamic_num_graphs=False)
    self._assert_expected_shapes(placeholders, num_graphs=num_graphs)
    self.assertEqual(tf.float32, placeholders.nodes.dtype)
    self.assertEqual(tf.float64, placeholders.edges.dtype)

  def test_placeholders_from_networkxs_missing_nodes(self):
    num_graphs = 16
    networkxs = [
        _generate_graph(batch_index, n_nodes=0, add_edges=False)
        for batch_index in range(num_graphs)
    ]
    placeholders = utils_tf.placeholders_from_networkxs(
        networkxs, force_dynamic_num_graphs=False)
    self.assertEqual(None, placeholders.nodes)
    self.assertEqual(None, placeholders.edges)
    self._assert_expected_shapes(
        placeholders, but_for=["nodes", "edges"], num_graphs=num_graphs)

  def test_placeholders_from_networkxs_hints(self):
    num_graphs = 16
    networkxs = [
        _generate_graph(batch_index, n_nodes=0, add_edges=False)
        for batch_index in range(num_graphs)
    ]
    placeholders = utils_tf.placeholders_from_networkxs(
        networkxs,
        node_shape_hint=[14],
        edge_shape_hint=[17],
        data_type_hint=tf.float64,
        force_dynamic_num_graphs=False)
    self.assertAllEqual([None, 14], placeholders.nodes.shape.as_list())
    self.assertAllEqual([None, 17], placeholders.edges.shape.as_list())
    self._assert_expected_shapes(
        placeholders, but_for=["nodes", "edges"], num_graphs=num_graphs)
    self.assertEqual(tf.float64, placeholders.nodes.dtype)
    self.assertEqual(tf.float64, placeholders.edges.dtype)

  def test_placeholders_from_networkxs_missing_edges(self):
    num_graphs = 16
    networkxs = [
        _generate_graph(batch_index, add_edges=False)
        for batch_index in range(num_graphs)
    ]
    placeholders = utils_tf.placeholders_from_networkxs(
        networkxs, force_dynamic_num_graphs=False)
    self.assertEqual(None, placeholders.edges)
    self._assert_expected_shapes(
        placeholders, but_for=["edges"], num_graphs=num_graphs)

  def test_feed_data(self):
    networkx = [_generate_graph(batch_index) for batch_index in range(16)]
    placeholders = utils_tf.placeholders_from_networkxs(
        networkx, force_dynamic_num_graphs=True)
    # Does not need to be the same size
    networkxs = [_generate_graph(batch_index) for batch_index in range(2)]
    with self.test_session() as sess:
      output = sess.run(
          placeholders,
          utils_tf.get_feed_dict(placeholders,
                                 utils_np.networkxs_to_graphs_tuple(networkxs)))
    self.assertAllEqual(
        np.array([[0, 0], [1, 0], [2, 0], [3, 0], [0, 1], [1, 1], [2, 1],
                  [3, 1]]), output.nodes)
    self.assertEqual(np.float32, output.nodes.dtype)
    self.assertAllEqual(np.array([[0], [1]]), output.globals)
    self.assertEqual(np.float32, output.globals.dtype)
    sorted_edges_content = sorted(
        [(x, y, z)
         for x, y, z in zip(output.receivers, output.senders, output.edges)])
    self.assertAllEqual([0, 0, 1, 4, 4, 5],
                        [x[0] for x in sorted_edges_content])
    self.assertAllEqual([1, 2, 3, 5, 6, 7],
                        [x[1] for x in sorted_edges_content])
    self.assertEqual(np.float64, output.edges.dtype)
    self.assertAllEqual(
        np.array([[0, 1, 0], [1, 2, 0], [2, 3, 0], [0, 1, 1], [1, 2, 1],
                  [2, 3, 1]]), [x[2] for x in sorted_edges_content])

  @parameterized.named_parameters((
      "no features",
      ["nodes", "edges", "globals"],
  ), (
      "no edges",
      ["edges", "receivers", "senders"],
  ))
  def test_get_feed_dict_raises(self, none_fields):
    networkxs = [_generate_graph(batch_index) for batch_index in range(16)]
    placeholders = utils_tf.placeholders_from_networkxs(networkxs)
    feed_values = utils_np.networkxs_to_graphs_tuple(networkxs)
    with self.assertRaisesRegexp(ValueError, ""):
      utils_tf.get_feed_dict(
          placeholders.map(lambda _: None, none_fields), feed_values)
    with self.assertRaisesRegexp(ValueError, ""):
      utils_tf.get_feed_dict(placeholders,
                             feed_values.map(lambda _: None, none_fields))

  def test_feed_data_no_nodes(self):
    networkx = [
        _generate_graph(batch_index, n_nodes=0, add_edges=False)
        for batch_index in range(16)
    ]
    placeholders = utils_tf.placeholders_from_networkxs(
        networkx, force_dynamic_num_graphs=True)
    # Does not need to be the same size
    networkxs = [
        _generate_graph(batch_index, n_nodes=0, add_edges=False)
        for batch_index in range(2)
    ]
    self.assertEqual(None, placeholders.nodes)
    self.assertEqual(None, placeholders.edges)
    with self.test_session() as sess:
      output = sess.run(
          placeholders.replace(nodes=tf.no_op(), edges=tf.no_op()),
          utils_tf.get_feed_dict(placeholders,
                                 utils_np.networkxs_to_graphs_tuple(networkxs)))
    self.assertAllEqual(np.array([[0], [1]]), output.globals)
    self.assertEqual(np.float32, output.globals.dtype)

  def test_feed_data_no_edges(self):
    networkx = [
        _generate_graph(batch_index, add_edges=False)
        for batch_index in range(16)
    ]
    placeholders = utils_tf.placeholders_from_networkxs(
        networkx, force_dynamic_num_graphs=True)
    # Does not need to be the same size
    networkxs = [
        _generate_graph(batch_index, add_edges=False)
        for batch_index in range(2)
    ]
    self.assertEqual(None, placeholders.edges)
    with self.test_session() as sess:
      output = sess.run(
          placeholders.replace(edges=tf.no_op()),
          utils_tf.get_feed_dict(placeholders,
                                 utils_np.networkxs_to_graphs_tuple(networkxs)))
    self.assertAllEqual(
        np.array([[0, 0], [1, 0], [2, 0], [3, 0], [0, 1], [1, 1], [2, 1],
                  [3, 1]]), output.nodes)
    self.assertAllEqual(np.array([[0], [1]]), output.globals)
    self.assertEqual(np.float32, output.globals.dtype)


class StopGradientsGraphTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(StopGradientsGraphTest, self).setUp()
    self._graph = utils_tf.data_dicts_to_graphs_tuple([{
        "senders": tf.zeros([10], dtype=tf.int32),
        "receivers": tf.zeros([10], dtype=tf.int32),
        "nodes": tf.ones([5, 7]),
        "edges": tf.zeros([10, 6]),
        "globals": tf.zeros([1, 8])
    }])

  def _check_if_gradients_exist(self, stopped_gradients_graph):
    gradients = []
    for field in ["globals", "nodes", "edges"]:
      xs = getattr(self._graph, field)
      ys = getattr(stopped_gradients_graph, field)
      gradient = tf.gradients(ys, xs)[0] if ys is not None else ys
      gradients.append(gradient)
    return [True if grad is not None else False for grad in gradients]

  @parameterized.named_parameters(
      ("stop_all_fields", True, True, True),
      ("stop_globals", True, False, False), ("stop_nodes", False, True, False),
      ("stop_edges", False, False, True), ("stop_none", False, False, False))
  def test_stop_gradients_outputs(self, stop_globals, stop_nodes, stop_edges):
    stopped_gradients_graph = utils_tf.stop_gradient(
        self._graph,
        stop_globals=stop_globals,
        stop_nodes=stop_nodes,
        stop_edges=stop_edges)

    gradients_exist = self._check_if_gradients_exist(stopped_gradients_graph)
    expected_gradients_exist = [
        not stop_globals, not stop_nodes, not stop_edges
    ]
    self.assertAllEqual(expected_gradients_exist, gradients_exist)

  @parameterized.named_parameters(("no_nodes", "nodes"), ("no_edges", "edges"),
                                  ("no_globals", "globals"))
  def test_stop_gradients_with_missing_field_raises(self, none_field):
    self._graph = self._graph.map(lambda _: None, [none_field])
    with self.assertRaisesRegexp(ValueError, none_field):
      utils_tf.stop_gradient(self._graph)

  def test_stop_gradients_default_params(self):
    """Tests for the default params of `utils_tf.stop_gradient`."""
    stopped_gradients_graph = utils_tf.stop_gradient(self._graph)
    gradients_exist = self._check_if_gradients_exist(stopped_gradients_graph)
    expected_gradients_exist = [False, False, False]
    self.assertAllEqual(expected_gradients_exist, gradients_exist)


class IdentityTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for the `identity` method."""

  def setUp(self):
    super(IdentityTest, self).setUp()
    self._graph = utils_tf.data_dicts_to_graphs_tuple([{
        "senders": tf.random_uniform([10], maxval=10, dtype=tf.int32),
        "receivers": tf.random_uniform([10], maxval=10, dtype=tf.int32),
        "nodes": tf.random_uniform([5, 7]),
        "edges": tf.random_uniform([10, 6]),
        "globals": tf.random_uniform([1, 8])
    }])

  def test_name_scope(self):
    """Tests that the name scope are correctly pushed through this function."""
    graph = self._graph
    with tf.name_scope("test"):
      graph_id = utils_tf.identity(graph)
    for field in [
        "nodes", "edges", "globals", "receivers", "senders", "n_node", "n_edge"
    ]:
      self.assertEqual("test", getattr(graph_id, field).name.split("/")[0])

  @parameterized.named_parameters(
      ("all fields defined", []), ("no node features", ["nodes"]),
      ("no edge features", ["edges"]), ("no global features", ["globals"]),
      ("no edges", ["edges", "receivers", "senders"]))
  def test_output(self, none_fields):
    """Tests that this function produces the identity."""
    graph = self._graph.map(lambda _: None, none_fields)
    with tf.name_scope("test"):
      graph_id = utils_tf.identity(graph)
    graph = utils_tf.make_runnable_in_session(graph)
    graph_id = utils_tf.make_runnable_in_session(graph_id)
    with self.test_session() as sess:
      expected_out, actual_out = sess.run([graph, graph_id])
    for field in [
        "nodes", "edges", "globals", "receivers", "senders", "n_node", "n_edge"
    ]:
      if field in none_fields:
        self.assertEqual(None, getattr(actual_out, field))
      else:
        self.assertNDArrayNear(
            getattr(expected_out, field), getattr(actual_out, field), err=1e-4)


class RunGraphWithNoneInSessionTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(RunGraphWithNoneInSessionTest, self).setUp()
    self._graph = utils_tf.data_dicts_to_graphs_tuple([{
        "senders": tf.random_uniform([10], maxval=10, dtype=tf.int32),
        "receivers": tf.random_uniform([10], maxval=10, dtype=tf.int32),
        "nodes": tf.random_uniform([5, 7]),
        "edges": tf.random_uniform([10, 6]),
        "globals": tf.random_uniform([1, 8])
    }])

  @parameterized.named_parameters(
      ("all fields defined", []), ("no node features", ["nodes"]),
      ("no edge features", ["edges"]), ("no global features", ["globals"]),
      ("no edges", ["edges", "receivers", "senders"]))
  def test_output(self, none_fields):
    """Tests that this function produces the identity."""
    graph = self._graph.map(lambda _: None, none_fields)
    with tf.name_scope("test"):
      graph_id = utils_tf.make_runnable_in_session(graph)
    graph = graph.map(tf.no_op, none_fields)
    with self.test_session() as sess:
      expected_out, actual_out = sess.run([graph, graph_id])
    for field in [
        "nodes", "edges", "globals", "receivers", "senders", "n_node", "n_edge"
    ]:
      if field in none_fields:
        self.assertEqual(None, getattr(actual_out, field))
      else:
        self.assertNDArrayNear(
            getattr(expected_out, field), getattr(actual_out, field), err=1e-4)


class ComputeOffsetTest(tf.test.TestCase):
  """Tests for the `compute_stacked_offsets` method."""

  def setUp(self):
    tf.reset_default_graph()
    self.sizes = [5, 4, 3, 1, 2, 0, 3, 0, 4, 7]
    self.repeats = [2, 2, 0, 2, 1, 3, 2, 0, 3, 2]
    self.offset = [
        0, 0, 5, 5, 12, 12, 13, 15, 15, 15, 15, 15, 18, 18, 18, 22, 22
    ]

  def test_compute_stacked_offsets(self):
    offset0 = utils_tf._compute_stacked_offsets(self.sizes, self.repeats)
    offset1 = utils_tf._compute_stacked_offsets(
        np.array(self.sizes), np.array(self.repeats))
    offset2 = utils_tf._compute_stacked_offsets(
        tf.constant(self.sizes, dtype=tf.int32),
        tf.constant(self.repeats, dtype=tf.int32))

    with self.test_session() as sess:
      o0, o1, o2 = sess.run([offset0, offset1, offset2])

    self.assertAllEqual(self.offset, o0.tolist())
    self.assertAllEqual(self.offset, o1.tolist())
    self.assertAllEqual(self.offset, o2.tolist())


class DataDictsCompletionTests(test_utils.GraphsTest, parameterized.TestCase):
  """Tests for the methods creating complete graphs from partial graphs."""

  def _assert_indices_sizes(self, dict_, n_relation):
    for key in ["receivers", "senders"]:
      self.assertAllEqual((n_relation,), dict_[key].get_shape().as_list())

  @parameterized.named_parameters(
      ("static", utils_tf._create_complete_edges_from_nodes_static),
      ("dynamic", utils_tf._create_complete_edges_from_nodes_dynamic),
  )
  def test_create_complete_edges_from_nodes_include_self_edges(self, method):
    for graph_dict in self.graphs_dicts_in:
      n_node = graph_dict["nodes"].shape[0]
      edges_dict = method(n_node, exclude_self_edges=False)
      self._assert_indices_sizes(edges_dict, n_node**2)

  @parameterized.named_parameters(
      ("static", utils_tf._create_complete_edges_from_nodes_static),
      ("dynamic", utils_tf._create_complete_edges_from_nodes_dynamic),
  )
  def test_create_complete_edges_from_nodes_exclude_self_edges(self, method):
    for graph_dict in self.graphs_dicts_in:
      n_node = graph_dict["nodes"].shape[0]
      edges_dict = method(n_node, exclude_self_edges=True)
      self._assert_indices_sizes(edges_dict, n_node * (n_node - 1))

  def test_create_complete_edges_from_nodes_dynamic_number_of_nodes(self):
    for graph_dict in self.graphs_dicts_in:
      n_node = tf.shape(tf.constant(graph_dict["nodes"]))[0]
      edges_dict = utils_tf._create_complete_edges_from_nodes_dynamic(
          n_node, exclude_self_edges=False)
      n_relation_op = n_node**2
      with self.test_session() as sess:
        n_relation, receivers, senders, n_edge = sess.run([
            n_relation_op, edges_dict["receivers"], edges_dict["senders"],
            edges_dict["n_edge"]
        ])
      self.assertAllEqual((n_relation,), receivers.shape)
      self.assertAllEqual((n_relation,), senders.shape)
      self.assertEqual(n_relation, n_edge)


class GraphsCompletionTests(test_utils.GraphsTest, parameterized.TestCase):
  """Tests for completing partial GraphsTuple."""

  def _assert_indices_sizes(self, graph, n_relation):
    for key in ["receivers", "senders"]:
      self.assertAllEqual((n_relation,),
                          getattr(graph, key).get_shape().as_list())

  @parameterized.named_parameters(("edge size 0", 0), ("edge size 1", 1))
  def test_fill_edge_state(self, edge_size):
    """Tests for filling the edge state with a constant content."""
    for g in self.graphs_dicts_in:
      g.pop("edges")
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    n_edges = np.sum(self.reference_graph.n_edge)
    graphs_tuple = utils_tf.set_zero_edge_features(graphs_tuple, edge_size)
    self.assertAllEqual((n_edges, edge_size),
                        graphs_tuple.edges.get_shape().as_list())

  @parameterized.named_parameters(("edge size 0", 0), ("edge size 1", 1))
  def test_fill_edge_state_dynamic(self, edge_size):
    """Tests for filling the edge state with a constant content."""
    for g in self.graphs_dicts_in:
      g.pop("edges")
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    graphs_tuple = graphs_tuple._replace(
        n_edge=tf.placeholder_with_default(
            graphs_tuple.n_edge, shape=graphs_tuple.n_edge.get_shape()))
    n_edges = np.sum(self.reference_graph.n_edge)
    graphs_tuple = utils_tf.set_zero_edge_features(graphs_tuple, edge_size)
    with self.test_session() as sess:
      actual_edges = sess.run(graphs_tuple.edges)
    self.assertNDArrayNear(
        np.zeros((n_edges, edge_size)), actual_edges, err=1e-4)

  @parameterized.named_parameters(("global size 0", 0), ("global size 1", 1))
  def test_fill_global_state(self, global_size):
    """Tests for filling the global state with a constant content."""
    for g in self.graphs_dicts_in:
      g.pop("globals")
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    n_graphs = self.reference_graph.n_edge.shape[0]
    graphs_tuple = utils_tf.set_zero_global_features(graphs_tuple, global_size)
    self.assertAllEqual((n_graphs, global_size),
                        graphs_tuple.globals.get_shape().as_list())

  @parameterized.named_parameters(("global size 0", 0), ("global size 1", 1))
  def test_fill_global_state_dynamic(self, global_size):
    """Tests for filling the global state with a constant content."""
    for g in self.graphs_dicts_in:
      g.pop("globals")
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    # Hide global shape information
    graphs_tuple = graphs_tuple._replace(
        n_node=tf.placeholder_with_default(graphs_tuple.n_node, shape=[None]))
    n_graphs = self.reference_graph.n_edge.shape[0]
    graphs_tuple = utils_tf.set_zero_global_features(graphs_tuple, global_size)
    with self.test_session() as sess:
      actual_globals = sess.run(graphs_tuple.globals)
    self.assertNDArrayNear(
        np.zeros((n_graphs, global_size)), actual_globals, err=1e-4)

  @parameterized.named_parameters(("node size 0", 0), ("node size 1", 1))
  def test_fill_node_state(self, node_size):
    """Tests for filling the node state with a constant content."""
    for g in self.graphs_dicts_in:
      g["n_node"] = g["nodes"].shape[0]
      g.pop("nodes")
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    n_nodes = np.sum(self.reference_graph.n_node)
    graphs_tuple = utils_tf.set_zero_node_features(graphs_tuple, node_size)
    self.assertAllEqual((n_nodes, node_size),
                        graphs_tuple.nodes.get_shape().as_list())

  @parameterized.named_parameters(("node size 0", 0), ("node size 1", 1))
  def test_fill_node_state_dynamic(self, node_size):
    """Tests for filling the node state with a constant content."""
    for g in self.graphs_dicts_in:
      g["n_node"] = g["nodes"].shape[0]
      g.pop("nodes")
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    graphs_tuple = graphs_tuple._replace(
        n_node=tf.placeholder_with_default(
            graphs_tuple.n_node, shape=graphs_tuple.n_node.get_shape()))
    n_nodes = np.sum(self.reference_graph.n_node)
    graphs_tuple = utils_tf.set_zero_node_features(graphs_tuple, node_size)
    with self.test_session() as sess:
      actual_nodes = sess.run(graphs_tuple.nodes)
    self.assertNDArrayNear(
        np.zeros((n_nodes, node_size)), actual_nodes, err=1e-4)

  def test_fill_edge_state_with_missing_fields_raises(self):
    """Edge field cannot be filled if receivers or senders are missing."""
    for g in self.graphs_dicts_in:
      g.pop("receivers")
      g.pop("senders")
      g.pop("edges")
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    with self.assertRaisesRegexp(ValueError, "receivers"):
      graphs_tuple = utils_tf.set_zero_edge_features(graphs_tuple, edge_size=1)

  def test_fill_state_default_types(self):
    """Tests that the features are created with the correct default type."""
    for g in self.graphs_dicts_in:
      g.pop("nodes")
      g.pop("globals")
      g.pop("edges")
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    graphs_tuple = utils_tf.set_zero_edge_features(graphs_tuple, edge_size=1)
    graphs_tuple = utils_tf.set_zero_node_features(graphs_tuple, node_size=1)
    graphs_tuple = utils_tf.set_zero_global_features(
        graphs_tuple, global_size=1)
    self.assertEqual(tf.float32, graphs_tuple.edges.dtype)
    self.assertEqual(tf.float32, graphs_tuple.nodes.dtype)
    self.assertEqual(tf.float32, graphs_tuple.globals.dtype)

  @parameterized.parameters(
      (tf.float64,),
      (tf.int32,),
  )
  def test_fill_state_user_specified_types(self, dtype):
    """Tests that the features are created with the correct default type."""
    for g in self.graphs_dicts_in:
      g.pop("nodes")
      g.pop("globals")
      g.pop("edges")
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    graphs_tuple = utils_tf.set_zero_edge_features(graphs_tuple, 1, dtype)
    graphs_tuple = utils_tf.set_zero_node_features(graphs_tuple, 1, dtype)
    graphs_tuple = utils_tf.set_zero_global_features(graphs_tuple, 1, dtype)
    self.assertEqual(dtype, graphs_tuple.edges.dtype)
    self.assertEqual(dtype, graphs_tuple.nodes.dtype)
    self.assertEqual(dtype, graphs_tuple.globals.dtype)

  @parameterized.named_parameters(
      ("no self edges", False),
      ("self edges", True),
  )
  def test_fully_connect_graph_dynamic(self, exclude_self_edges):
    for g in self.graphs_dicts_in:
      g.pop("edges")
      g.pop("receivers")
      g.pop("senders")
    n_relation = 0
    for g in self.graphs_dicts_in:
      n_node = g["nodes"].shape[0]
      if exclude_self_edges:
        n_relation += n_node * (n_node - 1)
      else:
        n_relation += n_node * n_node

    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    graphs_tuple = utils_tf.fully_connect_graph_dynamic(graphs_tuple,
                                                        exclude_self_edges)
    with self.test_session() as sess:
      actual_receivers, actual_senders = sess.run(
          [graphs_tuple.receivers, graphs_tuple.senders])
    self.assertAllEqual((n_relation,), actual_receivers.shape)
    self.assertAllEqual((n_relation,), actual_senders.shape)
    self.assertAllEqual((len(self.graphs_dicts_in),),
                        graphs_tuple.n_edge.get_shape().as_list())

  @parameterized.named_parameters(
      ("no self edges", False),
      ("self edges", True),
  )
  def test_fully_connect_graph_dynamic_with_dynamic_sizes(
      self, exclude_self_edges):
    for g in self.graphs_dicts_in:
      g.pop("edges")
      g.pop("receivers")
      g.pop("senders")
    n_relation = 0
    for g in self.graphs_dicts_in:
      n_node = g["nodes"].shape[0]
      if exclude_self_edges:
        n_relation += n_node * (n_node - 1)
      else:
        n_relation += n_node * n_node

    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    graphs_tuple = graphs_tuple.map(test_utils.mask_leading_dimension,
                                    ["nodes", "globals", "n_node", "n_edge"])
    graphs_tuple = utils_tf.fully_connect_graph_dynamic(graphs_tuple,
                                                        exclude_self_edges)
    with self.test_session() as sess:
      actual_receivers, actual_senders, actual_n_edge = sess.run(
          [graphs_tuple.receivers, graphs_tuple.senders, graphs_tuple.n_edge])
    self.assertAllEqual((n_relation,), actual_receivers.shape)
    self.assertAllEqual((n_relation,), actual_senders.shape)
    self.assertAllEqual((len(self.graphs_dicts_in),), actual_n_edge.shape)
    expected_edges = []
    offset = 0
    for graph in self.graphs_dicts_in:
      n_node = graph["nodes"].shape[0]
      for e1 in range(n_node):
        for e2 in range(n_node):
          if not exclude_self_edges or e1 != e2:
            expected_edges.append((e1 + offset, e2 + offset))
      offset += n_node
    actual_edges = zip(actual_receivers, actual_senders)
    self.assertSetEqual(set(actual_edges), set(expected_edges))

  @parameterized.named_parameters(
      ("no self edges", False),
      ("self edges", True),
  )
  def test_fully_connect_graph_static(self, exclude_self_edges):
    for g in self.graphs_dicts_in:
      g.pop("edges")
      g.pop("receivers")
      g.pop("senders")
    num_graphs = 2
    num_nodes = 3
    if exclude_self_edges:
      n_edges = num_nodes * (num_nodes - 1)
    else:
      n_edges = num_nodes * num_nodes
    n_relation = num_graphs * n_edges
    graphs_dicts = [{
        "nodes": tf.zeros([num_nodes, 1])
    } for _ in range(num_graphs)]
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(graphs_dicts)
    graphs_tuple = utils_tf.fully_connect_graph_static(graphs_tuple,
                                                       exclude_self_edges)
    self.assertAllEqual((n_relation,),
                        graphs_tuple.receivers.get_shape().as_list())
    self.assertAllEqual((n_relation,),
                        graphs_tuple.senders.get_shape().as_list())
    self.assertAllEqual((num_graphs,),
                        graphs_tuple.n_edge.get_shape().as_list())
    with self.test_session() as sess:
      actual_receivers, actual_senders, actual_n_edge = sess.run(
          [graphs_tuple.receivers, graphs_tuple.senders, graphs_tuple.n_edge])
    expected_edges = []
    offset = 0
    for _ in range(num_graphs):
      for v1 in range(num_nodes):
        for v2 in range(num_nodes):
          if not exclude_self_edges or v1 != v2:
            expected_edges.append((v1 + offset, v2 + offset))
      offset += num_nodes
    actual_edges = zip(actual_receivers, actual_senders)
    self.assertNDArrayNear(
        np.array([n_edges] * num_graphs), actual_n_edge, 1e-4)
    self.assertSetEqual(set(actual_edges), set(expected_edges))

  def test_fully_connect_graph_static_with_dynamic_sizes_raises(self):
    for g in self.graphs_dicts_in:
      g.pop("edges")
      g.pop("receivers")
      g.pop("senders")
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    graphs_tuple_1 = graphs_tuple.map(test_utils.mask_leading_dimension,
                                      ["n_node"])
    with self.assertRaisesRegexp(ValueError, "known at construction time"):
      utils_tf.fully_connect_graph_static(graphs_tuple_1)
    graphs_tuple_2 = graphs_tuple.map(test_utils.mask_leading_dimension,
                                      ["nodes"])
    with self.assertRaisesRegexp(ValueError, "known at construction time"):
      utils_tf.fully_connect_graph_static(graphs_tuple_2)
    with self.assertRaisesRegexp(ValueError, "the same in all graphs"):
      utils_tf.fully_connect_graph_static(graphs_tuple)


class GraphsTupleConversionTests(test_utils.GraphsTest, parameterized.TestCase):
  """Tests for the method converting between data dicts and GraphsTuple."""

  @parameterized.named_parameters(("all fields defined", []), (
      "no edge features",
      ["edges"],
  ), (
      "no node features",
      ["nodes"],
  ), (
      "no globals",
      ["globals"],
  ), (
      "no edges",
      ["edges", "receivers", "senders"],
  ))
  def test_data_dicts_to_graphs_tuple(self, none_fields):
    """Fields in `none_fields` will be cleared out."""
    for field in none_fields:
      for graph_dict in self.graphs_dicts_in:
        if field in graph_dict:
          if field == "nodes":
            graph_dict["n_node"] = graph_dict["nodes"].shape[0]
          graph_dict[field] = None
        self.reference_graph = self.reference_graph._replace(**{field: None})
      if field == "senders":
        self.reference_graph = self.reference_graph._replace(
            n_edge=np.zeros_like(self.reference_graph.n_edge))
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    for field in none_fields:
      self.assertEqual(None, getattr(graphs_tuple, field))
    graphs_tuple = graphs_tuple.map(tf.no_op, none_fields)
    with self.test_session() as sess:
      self._assert_graph_equals_np(self.reference_graph, sess.run(graphs_tuple))

  @parameterized.parameters(("receivers",), ("senders",))
  def test_data_dicts_to_graphs_tuple_raises(self, none_field):
    """Fields that cannot be missing."""
    for graph_dict in self.graphs_dicts_in:
      graph_dict[none_field] = None
    with self.assertRaisesRegexp(ValueError, none_field):
      utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)

  def test_data_dicts_to_graphs_tuple_no_raise(self):
    """Not having nodes is fine, if the number of nodes is provided."""
    for graph_dict in self.graphs_dicts_in:
      graph_dict["n_node"] = graph_dict["nodes"].shape[0]
      graph_dict["nodes"] = None
    utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)

  def test_data_dicts_to_graphs_tuple_cast_types(self):
    """Index and number fields should be cast to tensors of the right type."""
    for graph_dict in self.graphs_dicts_in:
      graph_dict["n_node"] = np.array(
          graph_dict["nodes"].shape[0], dtype=np.int64)
      graph_dict["receivers"] = graph_dict["receivers"].astype(np.int16)
      graph_dict["senders"] = graph_dict["senders"].astype(np.float64)
      graph_dict["nodes"] = graph_dict["nodes"].astype(np.float64)
      graph_dict["edges"] = tf.constant(graph_dict["edges"], dtype=tf.float64)
    out = utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    for key in ["n_node", "n_edge", "receivers", "senders"]:
      self.assertEqual(tf.int32, getattr(out, key).dtype)
      self.assertEqual(type(tf.int32), type(getattr(out, key).dtype))
    for key in ["nodes", "edges"]:
      self.assertEqual(type(tf.float64), type(getattr(out, key).dtype))
      self.assertEqual(tf.float64, getattr(out, key).dtype)


class GraphsIndexingTests(test_utils.GraphsTest, parameterized.TestCase):
  """Tests for the `get_graph` method."""

  @parameterized.named_parameters(("int_index", False),
                                  ("tensor_index", True))
  def test_getitem_one(self, use_tensor_index):
    index = 2
    expected = self.graphs_dicts_out[index]

    if use_tensor_index:
      index = tf.constant(index)

    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    graph_op = utils_tf.get_graph(graphs_tuple, index)
    graph_op = utils_tf.make_runnable_in_session(graph_op)

    with self.test_session() as sess:
      graph = sess.run(graph_op)
    actual, = utils_np.graphs_tuple_to_data_dicts(graph)

    for k, v in expected.items():
      self.assertAllClose(v, actual[k])
    self.assertEqual(expected["nodes"].shape[0], actual["n_node"])
    self.assertEqual(expected["edges"].shape[0], actual["n_edge"])

  @parameterized.named_parameters(("int_slice", False),
                                  ("tensor_slice", True))
  def test_getitem(self, use_tensor_slice):
    index = slice(1, 3)
    expected = self.graphs_dicts_out[index]

    if use_tensor_slice:
      index = slice(tf.constant(index.start), tf.constant(index.stop))

    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    graphs2_op = utils_tf.get_graph(graphs_tuple, index)
    graphs2_op = utils_tf.make_runnable_in_session(graphs2_op)

    with self.test_session() as sess:
      graphs2 = sess.run(graphs2_op)
    actual = utils_np.graphs_tuple_to_data_dicts(graphs2)

    for ex, ac in zip(expected, actual):
      for k, v in ex.items():
        self.assertAllClose(v, ac[k])
      self.assertEqual(ex["nodes"].shape[0], ac["n_node"])
      self.assertEqual(ex["edges"].shape[0], ac["n_edge"])

  @parameterized.named_parameters(
      ("index_bad_type", 1.,
       TypeError, "Index must be a valid scalar integer"),
      ("index_bad_shape", tf.constant([0, 1]),
       TypeError, "Valid tensor indices must be scalars"),
      ("index_bad_dtype", tf.constant(1.),
       TypeError, "Valid tensor indices must have types"),
      ("slice_bad_type_stop", slice(1.),
       TypeError, "Valid tensor indices must be integers"),
      ("slice_bad_shape_stop", slice(tf.constant([0, 1])),
       TypeError, "Valid tensor indices must be scalars"),
      ("slice_bad_dtype_stop", slice(tf.constant(1.)),
       TypeError, "Valid tensor indices must have types"),
      ("slice_bad_type_start", slice(0., 1),
       TypeError, "Valid tensor indices must be integers"),
      ("slice_bad_shape_start", slice(tf.constant([0, 1]), 1),
       TypeError, "Valid tensor indices must be scalars"),
      ("slice_bad_dtype_start", slice(tf.constant(0.), 1),
       TypeError, "Valid tensor indices must have types"),
      ("slice_with_step", slice(0, 1, 1),
       ValueError, "slices with step/stride are not supported"),
  )
  def test_raises(self, index, error_type, message):
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    with self.assertRaisesRegexp(error_type, message):
      utils_tf.get_graph(graphs_tuple, index)


class TestNumGraphs(test_utils.GraphsTest):
  """Tests for the `get_num_graphs` function."""

  def setUp(self):
    super(TestNumGraphs, self).setUp()
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    self.empty_graph = graphs_tuple.map(lambda _: None,
                                        graphs.GRAPH_DATA_FIELDS)

  def test_num_graphs_static(self):
    graph = self.empty_graph.replace(n_node=tf.zeros([3], dtype=tf.int32))
    self.assertEqual(3, utils_tf.get_num_graphs(graph))

  def test_num_graphs_dynamic(self):
    n_node_placeholder = tf.placeholder(tf.int32, [None])
    graph = self.empty_graph.replace(n_node=n_node_placeholder)
    num_graphs = utils_tf.get_num_graphs(graph)
    with self.test_session() as sess:
      actual_num_graphs = sess.run(
          num_graphs, {n_node_placeholder: np.zeros([3], dtype=np.int32)})
    self.assertEqual(3, actual_num_graphs)


if __name__ == "__main__":
  tf.test.main()
