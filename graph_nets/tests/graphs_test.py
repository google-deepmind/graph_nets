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

"""Tests for `graphs.py`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl.testing import parameterized
from graph_nets import graphs
import tensorflow as tf


class GraphsTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(GraphsTest, self).setUp()
    all_fields = graphs.GRAPH_DATA_FIELDS + graphs.GRAPH_NUMBER_FIELDS
    self.graph = {k: k for k in all_fields}

  @parameterized.named_parameters(
      ("no n_node", ["n_node"],),
      ("no n_edge", ["n_edge"],),
      ("receivers but no senders", ["edges", "senders"],),
      ("senders but no receivers", ["edges", "receivers"],),
      ("edges but no senders/receivers", ["receivers", "senders"],),
  )
  def test_inconsistent_none_fields_raise_error_on_creation(self, none_fields):
    for none_field in none_fields:
      self.graph[none_field] = None
    with self.assertRaisesRegexp(ValueError, none_fields[-1]):
      graphs.GraphsTuple(**self.graph)

  @parameterized.named_parameters(
      ("no n_node", ["n_node"],),
      ("no n_edge", ["n_edge"],),
      ("receivers but no senders", ["edges", "senders"],),
      ("senders but no receivers", ["edges", "receivers"],),
      ("edges but no senders/receivers", ["receivers", "senders"],),
  )
  def test_inconsistent_none_fields_raise_error_on_replace(self, none_fields):
    graph = graphs.GraphsTuple(**self.graph)
    with self.assertRaisesRegexp(ValueError, none_fields[-1]):
      graph.replace(**{none_field: None for none_field in none_fields})

  @parameterized.named_parameters(
      ("all fields defined", [],),
      ("no node state", ["nodes"],),
      ("no edge state", ["edges"],),
      ("no global state", ["globals"],),
      ("no state", ["nodes", "edges", "globals"],),
      ("no graph", ["nodes", "edges", "globals", "receivers", "senders"],),
      ("no edges", ["edges", "receivers", "senders"],),
  )
  def test_creation_with_valid_none_fields(self, none_fields):
    for none_field in none_fields:
      self.graph[none_field] = None
    graph = graphs.GraphsTuple(**self.graph)
    for k, v in self.graph.items():
      self.assertEqual(v, getattr(graph, k))

  @parameterized.named_parameters(
      ("all fields defined", [],),
      ("no node state", ["nodes"],),
      ("no edge state", ["edges"],),
      ("no global state", ["globals"],),
      ("no state", ["nodes", "edges", "globals"],),
      ("no graph", ["nodes", "edges", "globals", "receivers", "senders"],),
      ("no edges", ["edges", "receivers", "senders"],),
  )
  def test_replace_with_valid_none_fields(self, none_fields):
    # Create a graph with different values.
    graph = graphs.GraphsTuple(**{k: v + v for k, v in self.graph.items()})
    # Update with a graph containing the initial values, or Nones.
    for none_field in none_fields:
      self.graph[none_field] = None
    graph = graph.replace(**self.graph)
    for k, v in self.graph.items():
      self.assertEqual(v, getattr(graph, k))

  @parameterized.parameters(
      ([],),
      (["nodes"],),
      (["edges"],),
      (["globals"],),
      (["receivers"],),
      (["senders"],),
      (["n_node"],),
      (["n_edge"],),
      (["receivers", "senders"],),
      (["nodes", "edges", "globals"],),
      (["nodes", "edges", "globals", "receivers", "senders",
        "n_node", "n_edge"],),
  )
  def test_map_fields_as_expected(self, fields_to_map):
    """Tests that the fields are mapped are as expected."""
    graph = graphs.GraphsTuple(**self.graph)
    graph = graph.map(lambda v: v + v, fields_to_map)
    for field in graphs.ALL_FIELDS:
      if field in fields_to_map:
        self.assertEqual(field + field, getattr(graph, field))
      else:
        self.assertEqual(field, getattr(graph, field))

  def test_map_field_called_only_once(self):
    """Tests that the mapping function is called exactly once per field."""
    graph = graphs.GraphsTuple(**self.graph)
    mapped_fields = []
    def map_fn(v):
      mapped_fields.append(v)
      return v
    graph = graph.map(map_fn, graphs.ALL_FIELDS)
    self.assertListEqual(sorted(mapped_fields), sorted(graphs.ALL_FIELDS))

  def test_map_field_default_value(self):
    """Tests the default value for the `fields` argument."""
    graph = graphs.GraphsTuple(**self.graph)
    mapped_fields = []
    graph = graph.map(mapped_fields.append)
    self.assertListEqual(sorted(mapped_fields),
                         sorted([graphs.EDGES, graphs.GLOBALS, graphs.NODES]))

  def test_map_field_is_parallel(self):
    """Tests that fields are mapped parallelaly, not sequentially."""
    graph = graphs.GraphsTuple(**self.graph)
    graph = graph.map(lambda v: None, ["edges", "receivers", "senders"])


if __name__ == "__main__":
  tf.test.main()
