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

"""A class that defines graph-structured data.

The main purpose of the `GraphsTuple` is to represent multiple graphs with
different shapes and sizes in a way that supports batched processing.

This module first defines the string constants which are used to represent
graph(s) as tuples or dictionaries: `N_NODE, N_EDGE, NODES, EDGES, RECEIVERS,
SENDERS, GLOBALS`.

This representation could typically take the following form, for a batch of
`n_graphs` graphs stored in a `GraphsTuple` called graph:

  - N_NODE: The number of nodes per graph. It is a vector of integers with shape
    `[n_graphs]`, such that `graph.N_NODE[i]` is the number of nodes in the i-th
    graph.

  - N_EDGE: The number of edges per graph. It is a vector of integers with shape
    `[n_graphs]`, such that `graph.N_NODE[i]` is the number of edges in the i-th
    graph.

  - NODES: The nodes features. It is either `None` (the graph has no node
    features), or a vector of shape `[n_nodes] + node_shape`, where
    `n_nodes = sum(graph.N_NODE)` is the total number of nodes in the batch of
    graphs, and `node_shape` represents the shape of the features of each node.
    The relative index of a node from the batched version can be recovered from
    the `graph.N_NODE` property. For instance, the second node of the third
    graph will have its features in the
    `1 + graph.N_NODE[0] + graph.N_NODE[1]`-th slot of graph.NODES.
    Observe that having a `None` value for this field does not mean that the
    graphs have no nodes, only that they do not have features.

  - EDGES: The edges features. It is either `None` (the graph has no edge
    features), or a vector of shape `[n_edges] + edge_shape`, where
    `n_edges = sum(graph.N_EDGE)` is the total number of edges in the batch of
    graphs, and `edge_shape` represents the shape of the features of each ede.
    The relative index of an edge from the batched version can be recovered from
    the `graph.N_EDGE` property. For instance, the third edge of the third
    graph will have its features in the `2 + graph.N_EDGE[0] + graph.N_EDGE[1]`-
    th slot of graph.EDGES.
    Observe that having a `None` value for this field does not necessarily mean
    that the graph has no edges, only that they do not have features.

  - RECEIVERS: The indices of the receiver nodes, for each edge. It is either
    `None` (if the graph has no edges), or a vector of integers of shape
    `[n_edges]`, such that `graph.RECEIVERS[i]` is the index of the node
    receiving from the i-th edge.
    Observe that the index is absolute (in other words, cumulative), i.e.
    `graphs.RECEIVERS` take value in `[0, n_nodes]`. For instance, an edge
    connecting the vertices with relative indices 2 and 3 in the second graph of
    the batch would have a `RECEIVERS` value of `3 + graph.N_NODE[0]`.
    If `graphs.RECEIVERS` is `None`, then `graphs.EDGES` and `graphs.SENDERS`
    should also be `None`.

  - SENDERS: The indices of the sender nodes, for each edge. It is either
    `None` (if the graph has no edges), or a vector of integers of shape
    `[n_edges]`, such that `graph.SENDERS[i]` is the index of the node
    sending from the i-th edge.
    Observe that the index is absolute, i.e. `graphs.RECEIVERS` take value in
    `[0, n_nodes]`. For instance, an edge connecting the vertices with relative
    indices 1 and 3 in the third graph of the batch would have a `SENDERS` value
    of `1 + graph.N_NODE[0] + graph.N_NODE[1]`.
    If `graphs.SENDERS` is `None`, then `graphs.EDGES` and `graphs.RECEIVERS`
    should also be `None`.

  - GLOBALS: The global features of the graph. It is either `None` (the graph
    has no global features), or a vector of shape `[n_graphs] + global_shape`
    representing graph level features.

The `utils_np` and `utils_tf` modules provide convenience methods to work with
graph that contain numpy and tensorflow data, respectively: conversion,
batching, unbatching, indexing, among others.

The `GraphsTuple` class, however, is not restricted to storing vectors, and can
be used to store attributes of graphs as well (for instance, types or shapes).

The only assertions it makes are that the `None` fields are compatible with the
definition of a graph given above, namely:

  - the N_NODE and N_EDGE fields cannot be `None`;

  - if RECEIVERS is None, then SENDERS must be `None` (and vice-versa);

  - if RECEIVERS and SENDERS are `None`, then `EDGES` must be `None`.

Those assumptions are checked both upon initialization and when replacing a
field by calling the `replace` or `map` method.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


NODES = "nodes"
EDGES = "edges"
RECEIVERS = "receivers"
SENDERS = "senders"
GLOBALS = "globals"
N_NODE = "n_node"
N_EDGE = "n_edge"

GRAPH_FEATURE_FIELDS = (NODES, EDGES, GLOBALS)
GRAPH_INDEX_FIELDS = (RECEIVERS, SENDERS)
GRAPH_DATA_FIELDS = (NODES, EDGES, RECEIVERS, SENDERS, GLOBALS)
GRAPH_NUMBER_FIELDS = (N_NODE, N_EDGE)
ALL_FIELDS = (NODES, EDGES, RECEIVERS, SENDERS, GLOBALS, N_NODE, N_EDGE)


class GraphsTuple(
    collections.namedtuple("GraphsTuple",
                           GRAPH_DATA_FIELDS + GRAPH_NUMBER_FIELDS)):
  """Default namedtuple describing `Graphs`s.

  A children of `collections.namedtuple`s, which allows it to be directly input
  and output from `tensorflow.Session.run()` calls
  """

  def _validate_none_fields(self):
    """Asserts that the set of `None` fields in the instance is valid."""
    if self.n_node is None:
      raise ValueError("Field `n_node` cannot be None")
    if self.n_edge is None:
      raise ValueError("Field `n_edge` cannot be None")
    if self.receivers is None and self.senders is not None:
      raise ValueError(
          "Field `senders` must be None as field `receivers` is None")
    if self.senders is None and self.receivers is not None:
      raise ValueError(
          "Field `receivers` must be None as field `senders` is None")
    if self.receivers is None and self.edges is not None:
      raise ValueError(
          "Field `edges` must be None as field `receivers` and `senders` are "
          "None")

  def __init__(self, *args, **kwargs):
    del args, kwargs
    # The fields of a `namedtuple` are filled in the `__new__` method.
    # `__init__` does not accept parameters.
    super(GraphsTuple, self).__init__()
    self._validate_none_fields()

  def replace(self, **kwargs):
    output = self._replace(**kwargs)
    output._validate_none_fields()  # pylint: disable=protected-access
    return output

  def map(self, field_fn, fields=GRAPH_FEATURE_FIELDS):
    """Applies `field_fn` to the fields `fields` of the instance.

    `field_fn` is applied exactly once per field in `fields`. The result must
    satisfy the `GraphsTuple` requirement w.r.t. `None` fields, i.e. the
    `SENDERS` cannot be `None` if the `EDGES` or `RECEIVERS` are not `None`,
    etc.

    Args:
      field_fn: A callable that take a single argument.
      fields: (iterable of `str`). An iterable of the fields to apply
        `field_fn` to.

    Returns:
      A copy of the instance, with the fields in `fields` replaced by the result
      of applying `field_fn` to them.
    """
    return self.replace(**{k: field_fn(getattr(self, k)) for k in fields})
