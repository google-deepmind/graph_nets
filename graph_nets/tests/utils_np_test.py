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

"""Test for utils_np."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from absl.testing import parameterized
from graph_nets import utils_np
from graph_nets.tests import test_utils
import networkx as nx
import numpy as np
from six.moves import range
import tensorflow as tf


class ConcatenationTest(test_utils.GraphsTest, parameterized.TestCase):

  def test_compute_stacked_offsets(self):
    sizes = np.array([5, 4, 3, 1, 2, 0, 3, 0, 4, 7])
    repeats = [2, 2, 0, 2, 1, 3, 2, 0, 3, 2]
    offsets0 = utils_np._compute_stacked_offsets(sizes, repeats)
    offsets1 = utils_np._compute_stacked_offsets(sizes, np.array(repeats))
    expected_offsets = [
        0, 0, 5, 5, 12, 12, 13, 15, 15, 15, 15, 15, 18, 18, 18, 22, 22
    ]
    self.assertAllEqual(expected_offsets, offsets0.tolist())
    self.assertAllEqual(expected_offsets, offsets1.tolist())

  def test_concatenate_data_dicts(self):
    cat = utils_np._concatenate_data_dicts(self.graphs_dicts_in)
    for k, v in cat.items():
      self.assertAllEqual(getattr(self.reference_graph, k), v)


class DataDictsConversionTest(test_utils.GraphsTest, parameterized.TestCase):

  @parameterized.parameters(([],),
                            (["edges"],),
                            (["globals"],),
                            (["edges", "receivers", "senders"],))
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
    graphs = utils_np.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    for field in none_fields:
      self.assertEqual(None, getattr(graphs, field))
    self._assert_graph_equals_np(self.reference_graph, graphs)

  @parameterized.parameters(("receivers",), ("senders",))
  def test_data_dicts_to_graphs_tuple_missing_field_raises(self, none_field):
    """Fields that cannot be missing."""
    for graph_dict in self.graphs_dicts_in:
      graph_dict[none_field] = None
    with self.assertRaisesRegexp(ValueError, none_field):
      utils_np.data_dicts_to_graphs_tuple(self.graphs_dicts_in)

  def test_data_dicts_to_graphs_tuple_infer_n_node(self):
    """Not having nodes is fine if providing the number of nodes."""
    for graph_dict in self.graphs_dicts_in:
      graph_dict["n_node"] = graph_dict["nodes"].shape[0]
      graph_dict["nodes"] = None
    out = utils_np.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    self.assertAllEqual([0, 1, 1, 1, 2, 2, 2], out.n_node)

  def test_data_dicts_to_graphs_tuple_cast_types(self):
    """Index and number fields should be cast to numpy arrays."""
    for graph_dict in self.graphs_dicts_in:
      graph_dict["n_node"] = np.array(
          graph_dict["nodes"].shape[0], dtype=np.int64)
      graph_dict["receivers"] = graph_dict["receivers"].astype(np.int16)
      graph_dict["senders"] = graph_dict["senders"].astype(np.float64)
      graph_dict["nodes"] = graph_dict["nodes"].astype(np.float64)
      graph_dict["edges"] = graph_dict["edges"].astype(np.float64)
    out = utils_np.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    for key in ["n_node", "n_edge", "receivers", "senders"]:
      self.assertEqual(np.int32, getattr(out, key).dtype)
    for key in ["nodes", "edges"]:
      self.assertEqual(tf.float64, getattr(out, key).dtype)

  def test_data_dicts_to_graphs_tuple_from_lists(self):
    """Tests creatings a GraphsTuple from python lists."""
    for graph_dict in self.graphs_dicts_in:
      graph_dict["receivers"] = graph_dict["receivers"].tolist()
      graph_dict["senders"] = graph_dict["senders"].tolist()
    graphs = utils_np.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    self._assert_graph_equals_np(self.reference_graph, graphs)

  @parameterized.named_parameters(
      ("all_fields", []),
      ("no_data", ["nodes", "edges", "globals"]),
      ("no_edges", ["edges", "receivers", "senders"]))
  def test_graphs_tuple_to_data_dicts(self, none_fields):
    graphs_tuple = utils_np.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    graphs_tuple = graphs_tuple.map(lambda _: None, none_fields)
    data_dicts = utils_np.graphs_tuple_to_data_dicts(graphs_tuple)
    for none_field, data_dict in itertools.product(none_fields, data_dicts):
      self.assertEqual(None, data_dict[none_field])
    for expected_data_dict, data_dict in zip(self.graphs_dicts_out, data_dicts):
      for k, v in expected_data_dict.items():
        if k not in none_fields:
          self.assertAllClose(v, data_dict[k])


def _single_data_dict_to_networkx(data_dict):
  graph_nx = nx.OrderedMultiDiGraph()
  if data_dict["nodes"].size > 0:
    for i, x in enumerate(data_dict["nodes"]):
      graph_nx.add_node(i, features=x)

  if data_dict["edges"].size > 0:
    edge_data = zip(data_dict["senders"], data_dict["receivers"], [{
        "features": x
    } for x in data_dict["edges"]])
    graph_nx.add_edges_from(edge_data)
  graph_nx.graph["features"] = data_dict["globals"]

  return graph_nx


class NetworkxConversionTest(test_utils.GraphsTest, parameterized.TestCase):

  def test_order_preserving(self):
    """Tests that edges order can be preserved when importing from networks."""
    graph = nx.DiGraph()
    for node_index in range(4):
      graph.add_node(node_index, features=np.array([node_index]))
    receivers = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    senders = [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]
    for edge_index, (receiver, sender) in enumerate(zip(receivers, senders)):
      # Removing the "index" key makes this test fail 100%.
      edge_data = {"features": np.array([edge_index]), "index": edge_index}
      graph.add_edge(sender, receiver, **edge_data)
    graph.graph["features"] = np.array([0.])
    graphs_graph = utils_np.networkx_to_data_dict(graph)
    self.assertAllEqual(receivers, graphs_graph["receivers"])
    self.assertAllEqual(senders, graphs_graph["senders"])
    self.assertAllEqual([[x] for x in range(4)], graphs_graph["nodes"])
    self.assertAllEqual([[x] for x in range(12)], graphs_graph["edges"])

  def test_networkxs_to_graphs_tuple_with_none_fields(self):
    graph_nx = nx.OrderedMultiDiGraph()
    data_dict = utils_np.networkx_to_data_dict(
        graph_nx,
        node_shape_hint=None,
        edge_shape_hint=None)
    self.assertEqual(None, data_dict["edges"])
    self.assertEqual(None, data_dict["globals"])
    self.assertEqual(None, data_dict["nodes"])
    graph_nx.add_node(0, features=None)
    data_dict = utils_np.networkx_to_data_dict(
        graph_nx,
        node_shape_hint=1,
        edge_shape_hint=None)
    self.assertEqual(None, data_dict["nodes"])
    graph_nx.add_edge(0, 0, features=None)
    data_dict = utils_np.networkx_to_data_dict(
        graph_nx,
        node_shape_hint=[1],
        edge_shape_hint=[1])
    self.assertEqual(None, data_dict["edges"])
    graph_nx.graph["features"] = None
    utils_np.networkx_to_data_dict(graph_nx)
    self.assertEqual(None, data_dict["globals"])

  def test_networkxs_to_graphs_tuple(self):
    graph0 = utils_np.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    graph_nxs = []
    for data_dict in self.graphs_dicts_in:
      graph_nxs.append(_single_data_dict_to_networkx(data_dict))
      hints = {
          "edge_shape_hint": data_dict["edges"].shape[1:],
          "node_shape_hint": data_dict["nodes"].shape[1:],
          "data_type_hint": data_dict["nodes"].dtype,
      }
    graph = utils_np.networkxs_to_graphs_tuple(graph_nxs, **hints)
    self._assert_graph_equals_np(graph0, graph, force_edges_ordering=True)

  def test_networkxs_to_graphs_tuple_raises_key_error(self):
    """If the "features" field is not present in the nodes or edges."""
    graph_nx = _single_data_dict_to_networkx(self.graphs_dicts_in[-1])
    first_node = list(graph_nx.nodes(data=True))[0]
    del first_node[1]["features"]
    with self.assertRaisesRegexp(
        KeyError, "This could be due to the node having been silently added"):
      utils_np.networkxs_to_graphs_tuple([graph_nx])
    graph_nx = _single_data_dict_to_networkx(self.graphs_dicts_in[-1])
    first_edge = list(graph_nx.edges(data=True))[0]
    del first_edge[2]["features"]
    with self.assertRaises(KeyError):
      utils_np.networkxs_to_graphs_tuple([graph_nx])

  def test_networkxs_to_graphs_tuple_raises_assertion_error(self):
    """Either all nodes (resp. edges) should have features, or none of them."""
    graph_nx = _single_data_dict_to_networkx(self.graphs_dicts_in[-1])
    first_node = list(graph_nx.nodes(data=True))[0]
    first_node[1]["features"] = None
    with self.assertRaisesRegexp(
        ValueError, "Either all the nodes should have features"):
      utils_np.networkxs_to_graphs_tuple([graph_nx])
    graph_nx = _single_data_dict_to_networkx(self.graphs_dicts_in[-1])
    first_edge = list(graph_nx.edges(data=True))[0]
    first_edge[2]["features"] = None
    with self.assertRaisesRegexp(
        ValueError, "Either all the edges should have features"):
      utils_np.networkxs_to_graphs_tuple([graph_nx])

  @parameterized.named_parameters(
      ("all fields defined", []),
      ("stateless", ["nodes", "edges", "globals"]))
  def test_graphs_tuple_to_networkxs(self, none_fields):
    if "nodes" in none_fields:
      for graph in self.graphs_dicts_in:
        graph["n_node"] = graph["nodes"].shape[0]
    graphs = utils_np.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    graphs = graphs.map(lambda _: None, none_fields)
    graph_nxs = utils_np.graphs_tuple_to_networkxs(graphs)
    for data_dict, graph_nx in zip(self.graphs_dicts_out, graph_nxs):
      if "globals" in none_fields:
        self.assertEqual(None, graph_nx.graph["features"])
      else:
        self.assertAllClose(data_dict["globals"], graph_nx.graph["features"])
      nodes_data = graph_nx.nodes(data=True)
      for i, (v, (j, n)) in enumerate(zip(data_dict["nodes"], nodes_data)):
        self.assertEqual(i, j)
        if "nodes" in none_fields:
          self.assertEqual(None, n["features"])
        else:
          self.assertAllClose(v, n["features"])
      edges_data = sorted(
          graph_nx.edges(data=True), key=lambda x: x[2]["index"])
      for v, (_, _, e) in zip(data_dict["edges"], edges_data):
        if "edges" in none_fields:
          self.assertEqual(None, e["features"])
        else:
          self.assertAllClose(v, e["features"])
      for r, s, (i, j, _) in zip(
          data_dict["receivers"], data_dict["senders"], edges_data):
        self.assertEqual(s, i)
        self.assertEqual(r, j)


class GetItemTest(test_utils.GraphsTest, parameterized.TestCase):

  def test_get_single_item(self):
    index = 2
    expected = self.graphs_dicts_out[index]

    graphs = utils_np.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    graph = utils_np.get_graph(graphs, index)
    actual, = utils_np.graphs_tuple_to_data_dicts(graph)

    for k, v in expected.items():
      self.assertAllClose(v, actual[k])

  def test_get_many_items(self):
    index = slice(1, 3)
    expected = self.graphs_dicts_out[index]

    graphs = utils_np.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    graphs2 = utils_np.get_graph(graphs, index)
    actual = utils_np.graphs_tuple_to_data_dicts(graphs2)

    for ex, ac in zip(expected, actual):
      for k, v in ex.items():
        self.assertAllClose(v, ac[k])

if __name__ == "__main__":
  tf.test.main()
