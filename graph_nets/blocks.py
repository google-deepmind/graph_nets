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

"""Building blocks for Graph Networks.

This module contains elementary building blocks of graph networks:

  - `broadcast_{field_1}_to_{field_2}` propagates the features from `field_1`
    onto the relevant elements of `field_2`;

  - `{field_1}To{field_2}Aggregator` propagates and then reduces the features
    from `field_1` onto the relevant elements of `field_2`;

  - the `EdgeBlock`, `NodeBlock` and `GlobalBlock` are elementary graph networks
    that only update the edges (resp. the nodes, the globals) of their input
    graph (as described in https://arxiv.org/abs/1806.01261).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from graph_nets import graphs
from graph_nets import utils_tf
import sonnet as snt
import tensorflow as tf


NODES = graphs.NODES
EDGES = graphs.EDGES
GLOBALS = graphs.GLOBALS
RECEIVERS = graphs.RECEIVERS
SENDERS = graphs.SENDERS
GLOBALS = graphs.GLOBALS
N_NODE = graphs.N_NODE
N_EDGE = graphs.N_EDGE


def _validate_graph(graph, mandatory_fields, additional_message=None):
  for field in mandatory_fields:
    if getattr(graph, field) is None:
      message = "`{}` field cannot be None".format(field)
      if additional_message:
        message += " " + format(additional_message)
      message += "."
      raise ValueError(message)


def _validate_broadcasted_graph(graph, from_field, to_field):
  additional_message = "when broadcasting {} to {}".format(from_field, to_field)
  _validate_graph(graph, [from_field, to_field], additional_message)


def broadcast_globals_to_edges(graph, name="broadcast_globals_to_edges"):
  """Broadcasts the global features to the edges of a graph.

  Args:
    graph: A `graphs.GraphsTuple` containing `Tensor`s, with globals features of
      shape `[n_graphs] + global_shape`, and `N_EDGE` field of shape
      `[n_graphs]`.
    name: (string, optional) A name for the operation.

  Returns:
    A tensor of shape `[n_edges] + global_shape`, where
    `n_edges = sum(graph.n_edge)`. The i-th element of this tensor is given by
    `globals[j]`, where j is the index of the graph the i-th edge belongs to
    (i.e. is such that
    `sum_{k < j} graphs.n_edge[k] <= i < sum_{k <= j} graphs.n_edge[k]`).

  Raises:
    ValueError: If either `graph.globals` or `graph.n_edge` is `None`.
  """
  _validate_broadcasted_graph(graph, GLOBALS, N_EDGE)
  with tf.name_scope(name):
    return utils_tf.repeat(graph.globals, graph.n_edge, axis=0)


def broadcast_globals_to_nodes(graph, name="broadcast_globals_to_nodes"):
  """Broadcasts the global features to the nodes of a graph.

  Args:
    graph: A `graphs.GraphsTuple` containing `Tensor`s, with globals features of
      shape `[n_graphs] + global_shape`, and `N_NODE` field of shape
      `[n_graphs]`.
    name: (string, optional) A name for the operation.

  Returns:
    A tensor of shape `[n_nodes] + global_shape`, where
    `n_nodes = sum(graph.n_node)`. The i-th element of this tensor is given by
    `globals[j]`, where j is the index of the graph the i-th node belongs to
    (i.e. is such that
    `sum_{k < j} graphs.n_node[k] <= i < sum_{k <= j} graphs.n_node[k]`).

  Raises:
    ValueError: If either `graph.globals` or `graph.n_node` is `None`.
  """
  _validate_broadcasted_graph(graph, GLOBALS, N_NODE)
  with tf.name_scope(name):
    return utils_tf.repeat(graph.globals, graph.n_node, axis=0)


def broadcast_sender_nodes_to_edges(
    graph, name="broadcast_sender_nodes_to_edges"):
  """Broadcasts the node features to the edges they are sending into.

  Args:
    graph: A `graphs.GraphsTuple` containing `Tensor`s, with nodes features of
      shape `[n_nodes] + node_shape`, and `senders` field of shape
      `[n_edges]`.
    name: (string, optional) A name for the operation.

  Returns:
    A tensor of shape `[n_edges] + node_shape`, where
    `n_edges = sum(graph.n_edge)`. The i-th element is given by
    `graph.nodes[graph.senders[i]]`.

  Raises:
    ValueError: If either `graph.nodes` or `graph.senders` is `None`.
  """
  _validate_broadcasted_graph(graph, NODES, SENDERS)
  with tf.name_scope(name):
    return tf.gather(graph.nodes, graph.senders)


def broadcast_receiver_nodes_to_edges(
    graph, name="broadcast_receiver_nodes_to_edges"):
  """Broadcasts the node features to the edges they are receiving from.

  Args:
    graph: A `graphs.GraphsTuple` containing `Tensor`s, with nodes features of
      shape `[n_nodes] + node_shape`, and receivers of shape `[n_edges]`.
    name: (string, optional) A name for the operation.

  Returns:
    A tensor of shape `[n_edges] + node_shape`, where
    `n_edges = sum(graph.n_edge)`. The i-th element is given by
    `graph.nodes[graph.receivers[i]]`.

  Raises:
    ValueError: If either `graph.nodes` or `graph.receivers` is `None`.
  """
  _validate_broadcasted_graph(graph, NODES, RECEIVERS)
  with tf.name_scope(name):
    return tf.gather(graph.nodes, graph.receivers)


class EdgesToGlobalsAggregator(snt.AbstractModule):
  """Aggregates all edges into globals."""

  def __init__(self, reducer, name="edges_to_globals_aggregator"):
    """Initializes the EdgesToGlobalsAggregator module.

    The reducer is used for combining per-edge features (one set of edge
    feature vectors per graph) to give per-graph features (one feature
    vector per graph). The reducer should take a `Tensor` of edge features, a
    `Tensor` of segment indices, and a number of graphs. It should be invariant
    under permutation of edge features within each graph.

    Examples of compatible reducers are:
    * tf.unsorted_segment_sum
    * tf.unsorted_segment_mean
    * tf.unsorted_segment_prod
    * unsorted_segment_min_or_zero
    * unsorted_segment_max_or_zero

    Args:
      reducer: A function for reducing sets of per-edge features to individual
        per-graph features.
      name: The module name.
    """
    super(EdgesToGlobalsAggregator, self).__init__(name=name)
    self._reducer = reducer

  def _build(self, graph):
    _validate_graph(graph, (EDGES,),
                    additional_message="when aggregating from edges.")
    num_graphs = utils_tf.get_num_graphs(graph)
    graph_index = tf.range(num_graphs)
    indices = utils_tf.repeat(graph_index, graph.n_edge, axis=0)
    return self._reducer(graph.edges, indices, num_graphs)


class NodesToGlobalsAggregator(snt.AbstractModule):
  """Aggregates all nodes into globals."""

  def __init__(self, reducer, name="nodes_to_globals_aggregator"):
    """Initializes the NodesToGlobalsAggregator module.

    The reducer is used for combining per-node features (one set of node
    feature vectors per graph) to give per-graph features (one feature
    vector per graph). The reducer should take a `Tensor` of node features, a
    `Tensor` of segment indices, and a number of graphs. It should be invariant
    under permutation of node features within each graph.

    Examples of compatible reducers are:
    * tf.unsorted_segment_sum
    * tf.unsorted_segment_mean
    * tf.unsorted_segment_prod
    * unsorted_segment_min_or_zero
    * unsorted_segment_max_or_zero

    Args:
      reducer: A function for reducing sets of per-node features to individual
        per-graph features.
      name: The module name.
    """
    super(NodesToGlobalsAggregator, self).__init__(name=name)
    self._reducer = reducer

  def _build(self, graph):
    _validate_graph(graph, (NODES,),
                    additional_message="when aggregating from nodes.")
    num_graphs = utils_tf.get_num_graphs(graph)
    graph_index = tf.range(num_graphs)
    indices = utils_tf.repeat(graph_index, graph.n_node, axis=0)
    return self._reducer(graph.nodes, indices, num_graphs)


class _EdgesToNodesAggregator(snt.AbstractModule):
  """Agregates sent or received edges into the corresponding nodes."""

  def __init__(self, reducer, use_sent_edges=False,
               name="edges_to_nodes_aggregator"):
    super(_EdgesToNodesAggregator, self).__init__(name=name)
    self._reducer = reducer
    self._use_sent_edges = use_sent_edges

  def _build(self, graph):
    _validate_graph(graph, (EDGES, SENDERS, RECEIVERS,),
                    additional_message="when aggregating from edges.")
    num_nodes = tf.reduce_sum(graph.n_node)
    indices = graph.senders if self._use_sent_edges else graph.receivers
    return self._reducer(graph.edges, indices, num_nodes)


class SentEdgesToNodesAggregator(_EdgesToNodesAggregator):
  """Agregates sent edges into the corresponding sender nodes."""

  def __init__(self, reducer, name="sent_edges_to_nodes_aggregator"):
    """Constructor.

    The reducer is used for combining per-edge features (one set of edge
    feature vectors per node) to give per-node features (one feature
    vector per node). The reducer should take a `Tensor` of edge features, a
    `Tensor` of segment indices, and a number of nodes. It should be invariant
    under permutation of edge features within each segment.

    Examples of compatible reducers are:
    * tf.unsorted_segment_sum
    * tf.unsorted_segment_mean
    * tf.unsorted_segment_prod
    * unsorted_segment_min_or_zero
    * unsorted_segment_max_or_zero

    Args:
      reducer: A function for reducing sets of per-edge features to individual
        per-node features.
      name: The module name.
    """
    super(SentEdgesToNodesAggregator, self).__init__(
        use_sent_edges=True,
        reducer=reducer,
        name=name)


class ReceivedEdgesToNodesAggregator(_EdgesToNodesAggregator):
  """Agregates received edges into the corresponding receiver nodes."""

  def __init__(self, reducer, name="received_edges_to_nodes_aggregator"):
    """Constructor.

    The reducer is used for combining per-edge features (one set of edge
    feature vectors per node) to give per-node features (one feature
    vector per node). The reducer should take a `Tensor` of edge features, a
    `Tensor` of segment indices, and a number of nodes. It should be invariant
    under permutation of edge features within each segment.

    Examples of compatible reducers are:
    * tf.unsorted_segment_sum
    * tf.unsorted_segment_mean
    * tf.unsorted_segment_prod
    * unsorted_segment_min_or_zero
    * unsorted_segment_max_or_zero

    Args:
      reducer: A function for reducing sets of per-edge features to individual
        per-node features.
      name: The module name.
    """
    super(ReceivedEdgesToNodesAggregator, self).__init__(
        use_sent_edges=False, reducer=reducer, name=name)


def _unsorted_segment_reduction_or_zero(reducer, values, indices, num_groups):
  """Common code for unsorted_segment_{min,max}_or_zero (below)."""
  reduced = reducer(values, indices, num_groups)
  present_indices = tf.unsorted_segment_max(
      tf.ones_like(indices, dtype=reduced.dtype), indices, num_groups)
  present_indices = tf.clip_by_value(present_indices, 0, 1)
  present_indices = tf.reshape(
      present_indices, [num_groups] + [1] * (reduced.shape.ndims - 1))
  reduced *= present_indices
  return reduced


def unsorted_segment_min_or_zero(values, indices, num_groups,
                                 name="unsorted_segment_min_or_zero"):
  """Aggregates information using elementwise min.

  Segments with no elements are given a "min" of zero instead of the most
  positive finite value possible (which is what `tf.unsorted_segment_min`
  would do).

  Args:
    values: A `Tensor` of per-element features.
    indices: A 1-D `Tensor` whose length is equal to `values`' first dimension.
    num_groups: A `Tensor`.
    name: (string, optional) A name for the operation.

  Returns:
    A `Tensor` of the same type as `values`.
  """
  with tf.name_scope(name):
    return _unsorted_segment_reduction_or_zero(
        tf.unsorted_segment_min, values, indices, num_groups)


def unsorted_segment_max_or_zero(values, indices, num_groups,
                                 name="unsorted_segment_max_or_zero"):
  """Aggregates information using elementwise max.

  Segments with no elements are given a "max" of zero instead of the most
  negative finite value possible (which is what `tf.unsorted_segment_max` would
  do).

  Args:
    values: A `Tensor` of per-element features.
    indices: A 1-D `Tensor` whose length is equal to `values`' first dimension.
    num_groups: A `Tensor`.
    name: (string, optional) A name for the operation.

  Returns:
    A `Tensor` of the same type as `values`.
  """
  with tf.name_scope(name):
    return _unsorted_segment_reduction_or_zero(
        tf.unsorted_segment_max, values, indices, num_groups)


class EdgeBlock(snt.AbstractModule):
  """Edge block.

  A block that updates the features of each edge in a batch of graphs based on
  (a subset of) the previous edge features, the features of the adjacent nodes,
  and the global features of the corresponding graph.

  See https://arxiv.org/abs/1806.01261 for more details.
  """

  def __init__(self,
               edge_model_fn,
               use_edges=True,
               use_receiver_nodes=True,
               use_sender_nodes=True,
               use_globals=True,
               name="edge_block"):
    """Initializes the EdgeBlock module.

    Args:
      edge_model_fn: A callable that will be called in the variable scope of
        this EdgeBlock and should return a Sonnet module (or equivalent
        callable) to be used as the edge model. The returned module should take
        a `Tensor` (of concatenated input features for each edge) and return a
        `Tensor` (of output features for each edge). Typically, this module
        would input and output `Tensor`s of rank 2, but it may also be input or
        output larger ranks. See the `_build` method documentation for more
        details on the acceptable inputs to this module in that case.
      use_edges: (bool, default=True). Whether to condition on edge attributes.
      use_receiver_nodes: (bool, default=True). Whether to condition on receiver
        node attributes.
      use_sender_nodes: (bool, default=True). Whether to condition on sender
        node attributes.
      use_globals: (bool, default=True). Whether to condition on global
        attributes.
      name: The module name.

    Raises:
      ValueError: When fields that are required are missing.
    """
    super(EdgeBlock, self).__init__(name=name)

    if not (use_edges or use_sender_nodes or use_receiver_nodes or use_globals):
      raise ValueError("At least one of use_edges, use_sender_nodes, "
                       "use_receiver_nodes or use_globals must be True.")

    self._use_edges = use_edges
    self._use_receiver_nodes = use_receiver_nodes
    self._use_sender_nodes = use_sender_nodes
    self._use_globals = use_globals

    with self._enter_variable_scope():
      self._edge_model = edge_model_fn()

  def _build(self, graph):
    """Connects the edge block.

    Args:
      graph: A `graphs.GraphsTuple` containing `Tensor`s, whose individual edges
        features (if `use_edges` is `True`), individual nodes features (if
        `use_receiver_nodes` or `use_sender_nodes` is `True`) and per graph
        globals (if `use_globals` is `True`) should be concatenable on the last
        axis.

    Returns:
      An output `graphs.GraphsTuple` with updated edges.

    Raises:
      ValueError: If `graph` does not have non-`None` receivers and senders, or
        if `graph` has `None` fields incompatible with the selected `use_edges`,
        `use_receiver_nodes`, `use_sender_nodes`, or `use_globals` options.
    """
    _validate_graph(
        graph, (SENDERS, RECEIVERS, N_EDGE), " when using an EdgeBlock")

    edges_to_collect = []

    if self._use_edges:
      _validate_graph(graph, (EDGES,), "when use_edges == True")
      edges_to_collect.append(graph.edges)

    if self._use_receiver_nodes:
      edges_to_collect.append(broadcast_receiver_nodes_to_edges(graph))

    if self._use_sender_nodes:
      edges_to_collect.append(broadcast_sender_nodes_to_edges(graph))

    if self._use_globals:
      edges_to_collect.append(broadcast_globals_to_edges(graph))

    collected_edges = tf.concat(edges_to_collect, axis=-1)
    updated_edges = self._edge_model(collected_edges)
    return graph.replace(edges=updated_edges)


class NodeBlock(snt.AbstractModule):
  """Node block.

  A block that updates the features of each node in batch of graphs based on
  (a subset of) the previous node features, the aggregated features of the
  adjacent edges, and the global features of the corresponding graph.

  See https://arxiv.org/abs/1806.01261 for more details.
  """

  def __init__(self,
               node_model_fn,
               use_received_edges=True,
               use_sent_edges=False,
               use_nodes=True,
               use_globals=True,
               received_edges_reducer=tf.unsorted_segment_sum,
               sent_edges_reducer=tf.unsorted_segment_sum,
               name="node_block"):
    """Initializes the NodeBlock module.

    Args:
      node_model_fn: A callable that will be called in the variable scope of
        this NodeBlock and should return a Sonnet module (or equivalent
        callable) to be used as the node model. The returned module should take
        a `Tensor` (of concatenated input features for each node) and return a
        `Tensor` (of output features for each node). Typically, this module
        would input and output `Tensor`s of rank 2, but it may also be input or
        output larger ranks. See the `_build` method documentation for more
        details on the acceptable inputs to this module in that case.
      use_received_edges: (bool, default=True) Whether to condition on
        aggregated edges received by each node.
      use_sent_edges: (bool, default=False) Whether to condition on aggregated
        edges sent by each node.
      use_nodes: (bool, default=True) Whether to condition on node attributes.
      use_globals: (bool, default=True) Whether to condition on global
        attributes.
      received_edges_reducer: Reduction to be used when aggregating received
        edges. This should be a callable whose signature matches
        `tf.unsorted_segment_sum`.
      sent_edges_reducer: Reduction to be used when aggregating sent edges.
        This should be a callable whose signature matches
        `tf.unsorted_segment_sum`.
      name: The module name.

    Raises:
      ValueError: When fields that are required are missing.
    """

    super(NodeBlock, self).__init__(name=name)

    if not (use_nodes or use_sent_edges or use_received_edges or use_globals):
      raise ValueError("At least one of use_received_edges, use_sent_edges, "
                       "use_nodes or use_globals must be True.")

    self._use_received_edges = use_received_edges
    self._use_sent_edges = use_sent_edges
    self._use_nodes = use_nodes
    self._use_globals = use_globals

    with self._enter_variable_scope():
      self._node_model = node_model_fn()
      if self._use_received_edges:
        if received_edges_reducer is None:
          raise ValueError(
              "If `use_received_edges==True`, `received_edges_reducer` "
              "should not be None.")
        self._received_edges_aggregator = ReceivedEdgesToNodesAggregator(
            received_edges_reducer)
      if self._use_sent_edges:
        if sent_edges_reducer is None:
          raise ValueError(
              "If `use_sent_edges==True`, `sent_edges_reducer` "
              "should not be None.")
        self._sent_edges_aggregator = SentEdgesToNodesAggregator(
            sent_edges_reducer)

  def _build(self, graph):
    """Connects the node block.

    Args:
      graph: A `graphs.GraphsTuple` containing `Tensor`s, whose individual edges
        features (if `use_received_edges` or `use_sent_edges` is `True`),
        individual nodes features (if `use_nodes` is True) and per graph globals
        (if `use_globals` is `True`) should be concatenable on the last axis.

    Returns:
      An output `graphs.GraphsTuple` with updated nodes.
    """

    nodes_to_collect = []

    if self._use_received_edges:
      nodes_to_collect.append(self._received_edges_aggregator(graph))

    if self._use_sent_edges:
      nodes_to_collect.append(self._sent_edges_aggregator(graph))

    if self._use_nodes:
      _validate_graph(graph, (NODES,), "when use_nodes == True")
      nodes_to_collect.append(graph.nodes)

    if self._use_globals:
      nodes_to_collect.append(broadcast_globals_to_nodes(graph))

    collected_nodes = tf.concat(nodes_to_collect, axis=-1)
    updated_nodes = self._node_model(collected_nodes)
    return graph.replace(nodes=updated_nodes)


class GlobalBlock(snt.AbstractModule):
  """Global block.

  A block that updates the global features of each graph in a batch based on
  (a subset of) the previous global features, the aggregated features of the
  edges of the graph, and the aggregated features of the nodes of the graph.

  See https://arxiv.org/abs/1806.01261 for more details.
  """

  def __init__(self,
               global_model_fn,
               use_edges=True,
               use_nodes=True,
               use_globals=True,
               nodes_reducer=tf.unsorted_segment_sum,
               edges_reducer=tf.unsorted_segment_sum,
               name="global_block"):
    """Initializes the GlobalBlock module.

    Args:
      global_model_fn: A callable that will be called in the variable scope of
        this GlobalBlock and should return a Sonnet module (or equivalent
        callable) to be used as the global model. The returned module should
        take a `Tensor` (of concatenated input features) and return a `Tensor`
        (the global output features). Typically, this module would input and
        output `Tensor`s of rank 2, but it may also input or output larger
        ranks. See the `_build` method documentation for more details on the
        acceptable inputs to this module in that case.
      use_edges: (bool, default=True) Whether to condition on aggregated edges.
      use_nodes: (bool, default=True) Whether to condition on node attributes.
      use_globals: (bool, default=True) Whether to condition on global
        attributes.
      nodes_reducer: Reduction to be used when aggregating nodes. This should
        be a callable whose signature matches tf.unsorted_segment_sum.
      edges_reducer: Reduction to be used when aggregating edges. This should
        be a callable whose signature matches tf.unsorted_segment_sum.
      name: The module name.

    Raises:
      ValueError: When fields that are required are missing.
    """

    super(GlobalBlock, self).__init__(name=name)

    if not (use_nodes or use_edges or use_globals):
      raise ValueError("At least one of use_edges, "
                       "use_nodes or use_globals must be True.")

    self._use_edges = use_edges
    self._use_nodes = use_nodes
    self._use_globals = use_globals

    with self._enter_variable_scope():
      self._global_model = global_model_fn()
      if self._use_edges:
        if edges_reducer is None:
          raise ValueError(
              "If `use_edges==True`, `edges_reducer` should not be None.")
        self._edges_aggregator = EdgesToGlobalsAggregator(
            edges_reducer)
      if self._use_nodes:
        if nodes_reducer is None:
          raise ValueError(
              "If `use_nodes==True`, `nodes_reducer` should not be None.")
        self._nodes_aggregator = NodesToGlobalsAggregator(
            nodes_reducer)

  def _build(self, graph):
    """Connects the global block.

    Args:
      graph: A `graphs.GraphsTuple` containing `Tensor`s, whose individual edges
        (if `use_edges` is `True`), individual nodes (if `use_nodes` is True)
        and per graph globals (if `use_globals` is `True`) should be
        concatenable on the last axis.

    Returns:
      An output `graphs.GraphsTuple` with updated globals.
    """
    globals_to_collect = []

    if self._use_edges:
      _validate_graph(graph, (EDGES,), "when use_edges == True")
      globals_to_collect.append(self._edges_aggregator(graph))

    if self._use_nodes:
      _validate_graph(graph, (NODES,), "when use_nodes == True")
      globals_to_collect.append(self._nodes_aggregator(graph))

    if self._use_globals:
      _validate_graph(graph, (GLOBALS,), "when use_globals == True")
      globals_to_collect.append(graph.globals)

    collected_globals = tf.concat(globals_to_collect, axis=-1)
    updated_globals = self._global_model(collected_globals)
    return graph.replace(globals=updated_globals)
