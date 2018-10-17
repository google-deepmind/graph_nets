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
"""Common Graph Network architectures.

The modules in this files are Sonnet modules that:

  - take a `graphs.GraphsTuple` containing `Tensor`s as input, with possibly
    `None` fields (depending on the module);

  - return a `graphs.GraphsTuple` with updated values for some fields
    (depending on the module).


The provided modules are:

  - `GraphNetwork`: a general purpose Graph Network composed of configurable
    `EdgeBlock`, `NodeBlock` and `GlobalBlock` from `blocks.py`;

  - `GraphIndependent`: a Graph Network producing updated edges (resp. nodes,
    globals) based on the input's edges (resp. nodes, globals) only;

  - `InteractionNetwork` (from https://arxiv.org/abs/1612.00222): a
    network propagating information on the edges and nodes of a graph;

  - RelationNetwork (from https://arxiv.org/abs/1706.01427): a network
    updating the global property based on the relation between the input's
    nodes properties;

  - DeepSets (from https://arxiv.org/abs/1703.06114): a network that operates on
    sets (graphs without edges);

  - CommNet (from https://arxiv.org/abs/1605.07736 and
    https://arxiv.org/abs/1706.06122): a network updating nodes based on their
    previous features and the features of the adjacent nodes.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from graph_nets import blocks
import sonnet as snt
import tensorflow as tf

_DEFAULT_EDGE_BLOCK_OPT = {
    "use_edges": True,
    "use_receiver_nodes": True,
    "use_sender_nodes": True,
    "use_globals": True,
}

_DEFAULT_NODE_BLOCK_OPT = {
    "use_received_edges": True,
    "use_sent_edges": False,
    "use_nodes": True,
    "use_globals": True,
}

_DEFAULT_GLOBAL_BLOCK_OPT = {
    "use_edges": True,
    "use_nodes": True,
    "use_globals": True,
}


class InteractionNetwork(snt.AbstractModule):
  """Implementation of an Interaction Network.

  An interaction networks computes interactions on the edges based on the
  previous edges features, and on the features of the nodes sending into those
  edges. It then updates the nodes based on the incomming updated edges.
  See https://arxiv.org/abs/1612.00222 for more details.

  This model does not update the graph globals, and they are allowed to be
  `None`.
  """

  def __init__(self,
               edge_model_fn,
               node_model_fn,
               reducer=tf.unsorted_segment_sum,
               name="interaction_network"):
    """Initializes the InteractionNetwork module.

    Args:
      edge_model_fn: A callable that will be passed to `EdgeBlock` to perform
        per-edge computations. The callable must return a Sonnet module (or
        equivalent; see `blocks.EdgeBlock` for details), and the shape of the
        output of this module must match the one of the input nodes, but for the
        first and last axis.
      node_model_fn: A callable that will be passed to `NodeBlock` to perform
        per-node computations. The callable must return a Sonnet module (or
        equivalent; see `blocks.NodeBlock` for details).
      reducer: Reducer to be used by NodeBlock to aggregate edges. Defaults to
        tf.unsorted_segment_sum.
      name: The module name.
    """
    super(InteractionNetwork, self).__init__(name=name)

    with self._enter_variable_scope():
      self._edge_block = blocks.EdgeBlock(
          edge_model_fn=edge_model_fn, use_globals=False)
      self._node_block = blocks.NodeBlock(
          node_model_fn=node_model_fn,
          use_sent_edges=False,
          use_globals=False,
          received_edges_reducer=reducer)

  def _build(self, graph):
    """Connects the InterationNetwork.

    Args:
      graph: A `graphs.GraphsTuple` containing `Tensor`s. `graph.globals` can be
        `None`. The features of each node and edge of `graph` must be
        concatenable on the last axis (i.e., the shapes of `graph.nodes` and
        `graph.edges` must match but for their first and last axis).

    Returns:
      An output `graphs.GraphsTuple` with updated edges and nodes.

    Raises:
      ValueError: If any of `graph.nodes`, `graph.edges`, `graph.receivers` or
        `graph.senders` is `None`.
    """
    return self._node_block(self._edge_block(graph))


class RelationNetwork(snt.AbstractModule):
  """Implementation of a Relation Network.

  See https://arxiv.org/abs/1706.01427 for more details.

  The global and edges features of the input graph are not used, and are
  allowed to be `None` (the receivers and senders properties must be present).
  The output graph has updated, non-`None`, globals.
  """

  def __init__(self,
               edge_model_fn,
               global_model_fn,
               reducer=tf.unsorted_segment_sum,
               name="relation_network"):
    """Initializes the RelationNetwork module.

    Args:
      edge_model_fn: A callable that will be passed to EdgeBlock to perform
        per-edge computations. The callable must return a Sonnet module (or
        equivalent; see EdgeBlock for details).
      global_model_fn: A callable that will be passed to GlobalBlock to perform
        per-global computations. The callable must return a Sonnet module (or
        equivalent; see GlobalBlock for details).
      reducer: Reducer to be used by GlobalBlock to aggregate edges. Defaults to
        tf.unsorted_segment_sum.
      name: The module name.
    """
    super(RelationNetwork, self).__init__(name=name)

    with self._enter_variable_scope():
      self._edge_block = blocks.EdgeBlock(
          edge_model_fn=edge_model_fn,
          use_edges=False,
          use_receiver_nodes=True,
          use_sender_nodes=True,
          use_globals=False)

      self._global_block = blocks.GlobalBlock(
          global_model_fn=global_model_fn,
          use_edges=True,
          use_nodes=False,
          use_globals=False,
          edges_reducer=reducer)

  def _build(self, graph):
    """Connects the RelationNetwork.

    Args:
      graph: A `graphs.GraphsTuple` containing `Tensor`s, except for the edges
        and global properties which may be `None`.

    Returns:
      A `graphs.GraphsTuple` with updated globals.

    Raises:
      ValueError: If any of `graph.nodes`, `graph.receivers` or `graph.senders`
        is `None`.
    """
    output_graph = self._global_block(self._edge_block(graph))
    return graph.replace(globals=output_graph.globals)


def _make_default_edge_block_opt(edge_block_opt):
  """Default options to be used in the EdgeBlock of a generic GraphNetwork."""
  edge_block_opt = edge_block_opt or {}
  for k, v in _DEFAULT_EDGE_BLOCK_OPT.items():
    edge_block_opt[k] = edge_block_opt.get(k, v)
  return edge_block_opt


def _make_default_node_block_opt(node_block_opt, default_reducer):
  """Default options to be used in the NodeBlock of a generic GraphNetwork."""
  node_block_opt = node_block_opt or {}
  for k, v in _DEFAULT_NODE_BLOCK_OPT.items():
    node_block_opt[k] = node_block_opt.get(k, v)
  for key in ["received_edges_reducer", "sent_edges_reducer"]:
    node_block_opt[key] = node_block_opt.get(key, default_reducer)
  return node_block_opt


def _make_default_global_block_opt(global_block_opt, default_reducer):
  """Default options to be used in the GlobalBlock of a generic GraphNetwork."""
  global_block_opt = global_block_opt or {}
  for k, v in _DEFAULT_GLOBAL_BLOCK_OPT.items():
    global_block_opt[k] = global_block_opt.get(k, v)
  for key in ["edges_reducer", "nodes_reducer"]:
    global_block_opt[key] = global_block_opt.get(key, default_reducer)
  return global_block_opt


class GraphNetwork(snt.AbstractModule):
  """Implementation of a Graph Network.

  See https://arxiv.org/abs/1806.01261 for more details.
  """

  def __init__(self,
               edge_model_fn,
               node_model_fn,
               global_model_fn,
               reducer=tf.unsorted_segment_sum,
               edge_block_opt=None,
               node_block_opt=None,
               global_block_opt=None,
               name="graph_network"):
    """Initializes the GraphNetwork module.

    Args:
      edge_model_fn: A callable that will be passed to EdgeBlock to perform
        per-edge computations. The callable must return a Sonnet module (or
        equivalent; see EdgeBlock for details).
      node_model_fn: A callable that will be passed to NodeBlock to perform
        per-node computations. The callable must return a Sonnet module (or
        equivalent; see NodeBlock for details).
      global_model_fn: A callable that will be passed to GlobalBlock to perform
        per-global computations. The callable must return a Sonnet module (or
        equivalent; see GlobalBlock for details).
      reducer: Reducer to be used by NodeBlock and GlobalBlock to to aggregate
        nodes and edges. Defaults to tf.unsorted_segment_sum. This will be
        overridden by the reducers specified in `node_block_opt` and
        `global_block_opt`, if any.
      edge_block_opt: Additional options to be passed to the EdgeBlock. Can
        contain keys `use_edges`, `use_receiver_nodes`, `use_sender_nodes`,
        `use_globals`. By default, these are all True.
      node_block_opt: Additional options to be passed to the NodeBlock. Can
        contain the keys `use_received_edges`, `use_sent_edges`, `use_nodes`,
        `use_globals` (all set to True by default), and
        `received_edges_reducer`, `sent_edges_reducer` (default to `reducer`).
      global_block_opt: Additional options to be passed to the GlobalBlock. Can
        contain the keys `use_edges`, `use_nodes`, `use_globals` (all set to
        True by default), and `edges_reducer`, `nodes_reducer` (defaults to
        `reducer`).
      name: The module name.
    """
    super(GraphNetwork, self).__init__(name=name)
    edge_block_opt = _make_default_edge_block_opt(edge_block_opt)
    node_block_opt = _make_default_node_block_opt(node_block_opt, reducer)
    global_block_opt = _make_default_global_block_opt(global_block_opt, reducer)

    with self._enter_variable_scope():
      self._edge_block = blocks.EdgeBlock(
          edge_model_fn=edge_model_fn, **edge_block_opt)
      self._node_block = blocks.NodeBlock(
          node_model_fn=node_model_fn, **node_block_opt)
      self._global_block = blocks.GlobalBlock(
          global_model_fn=global_model_fn, **global_block_opt)

  def _build(self, graph):
    """Connects the GraphNetwork.

    Args:
      graph: A `graphs.GraphsTuple` containing `Tensor`s. Depending on the block
        options, `graph` may contain `None` fields; but with the default
        configuration, no `None` field is allowed. Moreover, when using the
        default configuration, the features of each nodes, edges and globals of
        `graph` should be concatenable on the last dimension.

    Returns:
      An output `graphs.GraphsTuple` with updated edges, nodes and globals.
    """
    return self._global_block(self._node_block(self._edge_block(graph)))


class GraphIndependent(snt.AbstractModule):
  """A graph block that applies models to the graph elements independently.

  The inputs and outputs are graphs. The corresponding models are applied to
  each element of the graph (edges, nodes and globals) in parallel and
  independently of the other elements. It can be used to encode or
  decode the elements of a graph.
  """

  def __init__(self,
               edge_model_fn=None,
               node_model_fn=None,
               global_model_fn=None,
               name="graph_independent"):
    """Initializes the GraphIndependent module.

    Args:
      edge_model_fn: A callable that returns an edge model function. The
        callable must return a Sonnet module (or equivalent). If passed `None`,
        will pass through inputs (the default).
      node_model_fn: A callable that returns a node model function. The callable
        must return a Sonnet module (or equivalent). If passed `None`, will pass
        through inputs (the default).
      global_model_fn: A callable that returns a global model function. The
        callable must return a Sonnet module (or equivalent). If passed `None`,
        will pass through inputs (the default).
      name: The module name.
    """
    super(GraphIndependent, self).__init__(name=name)

    with self._enter_variable_scope():
      # The use of snt.Module below is to ensure the ops and variables that
      # result from the edge/node/global_model_fns are scoped analogous to how
      # the Edge/Node/GlobalBlock classes do.
      if edge_model_fn is None:
        self._edge_model = lambda x: x
      else:
        self._edge_model = snt.Module(
            lambda x: edge_model_fn()(x), name="edge_model")  # pylint: disable=unnecessary-lambda
      if node_model_fn is None:
        self._node_model = lambda x: x
      else:
        self._node_model = snt.Module(
            lambda x: node_model_fn()(x), name="node_model")  # pylint: disable=unnecessary-lambda
      if global_model_fn is None:
        self._global_model = lambda x: x
      else:
        self._global_model = snt.Module(
            lambda x: global_model_fn()(x), name="global_model")  # pylint: disable=unnecessary-lambda

  def _build(self, graph):
    """Connects the GraphIndependent.

    Args:
      graph: A `graphs.GraphsTuple` containing non-`None` edges, nodes and
        globals.

    Returns:
      An output `graphs.GraphsTuple` with updated edges, nodes and globals.

    """
    return graph.replace(
        edges=self._edge_model(graph.edges),
        nodes=self._node_model(graph.nodes),
        globals=self._global_model(graph.globals))


class DeepSets(snt.AbstractModule):
  """DeepSets module.

  Implementation for the model described in https://arxiv.org/abs/1703.06114
  (M. Zaheer, S. Kottur, S. Ravanbakhsh, B. Poczos, R. Salakhutdinov, A. Smola).
  See also PointNet (https://arxiv.org/abs/1612.00593, C. Qi, H. Su, K. Mo,
  L. J. Guibas) for a related model.

  This module operates on sets, which can be thought of as graphs without
  edges. The nodes features are first updated based on their value and the
  globals features, and new globals features are then computed based on the
  updated nodes features.

  Note that in the original model, only the globals are updated in the returned
  graph, while this implementation also returns updated nodes.
  The original model can be reproduced by writing:
  ```
  deep_sets = DeepSets()
  output = deep_sets(input)
  output = input.replace(globals=output.globals)
  ```

  This module does not use the edges data or the information contained in the
  receivers or senders; the output graph has the same value in those fields as
  the input graph. Those fields can also have `None` values in the input
  `graphs.GraphsTuple`.
  """

  def __init__(self,
               node_model_fn,
               global_model_fn,
               reducer=tf.unsorted_segment_sum,
               name="deep_sets"):
    """Initializes the DeepSets module.

    Args:
      node_model_fn: A callable to be passed to NodeBlock. The callable must
        return a Sonnet module (or equivalent; see NodeBlock for details). The
        shape of this module's output must equal the shape of the input graph's
        global features, but for the first and last axis.
      global_model_fn: A callable to be passed to GlobalBlock. The callable must
        return a Sonnet module (or equivalent; see GlobalBlock for details).
      reducer: Reduction to be used when aggregating the nodes in the globals.
        This should be a callable whose signature matches
        tf.unsorted_segment_sum.
      name: The module name.
    """
    super(DeepSets, self).__init__(name=name)

    with self._enter_variable_scope():
      self._node_block = blocks.NodeBlock(
          node_model_fn=node_model_fn,
          use_received_edges=False,
          use_sent_edges=False,
          use_nodes=True,
          use_globals=True)
      self._global_block = blocks.GlobalBlock(
          global_model_fn=global_model_fn,
          use_edges=False,
          use_nodes=True,
          use_globals=False,
          nodes_reducer=reducer)

  def _build(self, graph):
    """Connects the DeepSets network.

    Args:
      graph: A `graphs.GraphsTuple` containing `Tensor`s, whose edges, senders
        or receivers properties may be `None`. The features of every node and
        global of `graph` should be concatenable on the last axis (i.e. the
        shapes of `graph.nodes` and `graph.globals` must match but for their
        first and last axis).

    Returns:
      An output `graphs.GraphsTuple` with updated globals.
    """
    return self._global_block(self._node_block(graph))


class CommNet(snt.AbstractModule):
  """CommNet module.

  Implementation for the model originally described in
  https://arxiv.org/abs/1605.07736 (S. Sukhbaatar, A. Szlam, R. Fergus), in the
  version presented in https://arxiv.org/abs/1706.06122 (Y. Hoshen).

  This module internally creates edge features based on the features from the
  nodes sending to that edge, and independently learns an embedding for each
  node. It then uses these edges and nodes features to compute updated node
  features.

  This module does not use the global nor the edges features of the input, but
  uses its receivers and senders information. The output graph has the same
  value in edge and global fields as the input graph. The edge and global
  features fields may have a `None` value in the input `gn_graphs.GraphsTuple`.
  """

  def __init__(self,
               edge_model_fn,
               node_encoder_model_fn,
               node_model_fn,
               reducer=tf.unsorted_segment_sum,
               name="comm_net"):
    """Initializes the CommNet module.

    Args:
      edge_model_fn: A callable to be passed to EdgeBlock. The callable must
        return a Sonnet module (or equivalent; see EdgeBlock for details).
      node_encoder_model_fn: A callable to be passed to the NodeBlock
        responsible for the first encoding of the nodes. The callable must
        return a Sonnet module (or equivalent; see NodeBlock for details). The
        shape of this module's output should match the shape of the module built
        by `edge_model_fn`, but for the first and last dimension.
      node_model_fn: A callable to be passed to NodeBlock. The callable must
        return a Sonnet module (or equivalent; see NodeBlock for details).
      reducer: Reduction to be used when aggregating the edges in the nodes.
        This should be a callable whose signature matches
        tf.unsorted_segment_sum.
      name: The module name.
    """
    super(CommNet, self).__init__(name=name)

    with self._enter_variable_scope():
      # Computes $\Psi_{com}(x_j)$ in Eq. (2) of 1706.06122
      self._edge_block = blocks.EdgeBlock(
          edge_model_fn=edge_model_fn,
          use_edges=False,
          use_receiver_nodes=False,
          use_sender_nodes=True,
          use_globals=False)
      # Computes $\Phi(x_i)$ in Eq. (2) of 1706.06122
      self._node_encoder_block = blocks.NodeBlock(
          node_model_fn=node_encoder_model_fn,
          use_received_edges=False,
          use_sent_edges=False,
          use_nodes=True,
          use_globals=False,
          received_edges_reducer=reducer,
          name="node_encoder_block")
      # Computes $\Theta(..)$ in Eq.(2) of 1706.06122
      self._node_block = blocks.NodeBlock(
          node_model_fn=node_model_fn,
          use_received_edges=True,
          use_sent_edges=False,
          use_nodes=True,
          use_globals=False,
          received_edges_reducer=reducer)

  def _build(self, graph):
    """Connects the CommNet network.

    Args:
      graph: A `graphs.GraphsTuple` containing `Tensor`s, with non-`None` nodes,
        receivers and senders.

    Returns:
      An output `graphs.GraphsTuple` with updated nodes.

    Raises:
      ValueError: if any of `graph.nodes`, `graph.receivers` or `graph.senders`
      is `None`.
    """
    node_input = self._node_encoder_block(self._edge_block(graph))
    return graph.replace(nodes=self._node_block(node_input).nodes)


def _unsorted_segment_softmax(data,
                              segment_ids,
                              num_segments,
                              name="unsorted_segment_softmax"):
  """Performs an elementwise softmax operation along segments of a tensor.

  The input parameters are analogous to `tf.unsorted_segment_sum`. It produces
  an output of the same shape as the input data, after performing an
  elementwise sofmax operation between all of the rows with common segment id.

  Args:
    data: A tensor with at least one dimension.
    segment_ids: A tensor of indices segmenting `data` across the first
      dimension.
    num_segments: A scalar tensor indicating the number of segments. It should
      be at least `max(segment_ids) + 1`.
    name: A name for the operation (optional).

  Returns:
    A tensor with the same shape as `data` after applying the softmax operation.

  """
  with tf.name_scope(name):
    segment_maxes = tf.unsorted_segment_max(data, segment_ids, num_segments)
    maxes = tf.gather(segment_maxes, segment_ids)
    # Possibly refactor to `tf.stop_gradient(maxes)` for better performance.
    data -= maxes
    exp_data = tf.exp(data)
    segment_sum_exp_data = tf.unsorted_segment_sum(exp_data, segment_ids,
                                                   num_segments)
    sum_exp_data = tf.gather(segment_sum_exp_data, segment_ids)
    return exp_data / sum_exp_data


def _received_edges_normalizer(graph,
                               normalizer,
                               name="received_edges_normalizer"):
  """Performs elementwise normalization for all received edges by a given node.

  Args:
    graph: A graph containing edge information.
    normalizer: A normalizer function following the signature of
      `modules._unsorted_segment_softmax`.
    name: A name for the operation (optional).

  Returns:
    A tensor with the resulting normalized edges.

  """
  with tf.name_scope(name):
    return normalizer(
        data=graph.edges,
        segment_ids=graph.receivers,
        num_segments=tf.reduce_sum(graph.n_node))


class SelfAttention(snt.AbstractModule):
  """Multi-head self-attention module.

  The module is based on the following three papers:
   * A simple neural network module for relational reasoning (RNs):
       https://arxiv.org/abs/1706.01427
   * Non-local Neural Networks: https://arxiv.org/abs/1711.07971.
   * Attention Is All You Need (AIAYN): https://arxiv.org/abs/1706.03762.

  The input to the modules consists of a graph containing values for each node
  and connectivity between them, a tensor containing keys for each node
  and a tensor containing queries for each node.

  The self-attention step consist of updating the node values, with each new
  node value computed in a two step process:
  - Computing the attention weights between each node and all of its senders
   nodes, by calculating sum(sender_key*receiver_query) and using the softmax
   operation on all attention weights for each node.
  - For each receiver node, compute the new node value as the weighted average
   of the values of the sender nodes, according to the attention weights.
  - Nodes with no received edges, get an updated value of 0.

  Values, keys and queries contain a "head" axis to compute independent
  self-attention for each of the heads.

  """

  def __init__(self, name="self_attention"):
    """Inits the module.

    Args:
      name: The module name.
    """
    super(SelfAttention, self).__init__(name=name)
    self._normalizer = _unsorted_segment_softmax

  def _build(self, node_values, node_keys, node_queries, attention_graph):
    """Connects the multi-head self-attention module.

    The self-attention is only computed according to the connectivity of the
    input graphs, with receiver nodes attending to sender nodes.

    Args:
      node_values: Tensor containing the values associated to each of the nodes.
        The expected shape is [total_num_nodes, num_heads, key_size].
      node_keys: Tensor containing the key associated to each of the nodes. The
        expected shape is [total_num_nodes, num_heads, key_size].
      node_queries: Tensor containing the query associated to each of the nodes.
        The expected shape is [total_num_nodes, num_heads, query_size]. The
        query size must be equal to the key size.
      attention_graph: Graph containing connectivity information between nodes
        via the senders and receivers fields. Node A will only attempt to attend
        to Node B if `attention_graph` contains an edge sent by Node A and
        received by Node B.

    Returns:
      An output `graphs.GraphsTuple` with updated nodes containing the
      aggregated attended value for each of the nodes with shape
      [total_num_nodes, num_heads, value_size].

    Raises:
      ValueError: if the input graph does not have edges.
    """

    # Sender nodes put their keys and values in the edges.
    # [total_num_edges, num_heads, query_size]
    sender_keys = blocks.broadcast_sender_nodes_to_edges(
        attention_graph.replace(nodes=node_keys))
    # [total_num_edges, num_heads, value_size]
    sender_values = blocks.broadcast_sender_nodes_to_edges(
        attention_graph.replace(nodes=node_values))

    # Receiver nodes put their queries in the edges.
    # [total_num_edges, num_heads, key_size]
    receiver_queries = blocks.broadcast_receiver_nodes_to_edges(
        attention_graph.replace(nodes=node_queries))

    # Attention weight for each edge.
    # [total_num_edges, num_heads]
    attention_weights_logits = tf.reduce_sum(
        sender_keys * receiver_queries, axis=-1)
    normalized_attention_weights = _received_edges_normalizer(
        attention_graph.replace(edges=attention_weights_logits),
        normalizer=self._normalizer)

    # Attending to sender values according to the weights.
    # [total_num_edges, num_heads, embedding_size]
    attented_edges = sender_values * normalized_attention_weights[..., None]

    # Summing all of the attended values from each node.
    # [total_num_nodes, num_heads, embedding_size]
    received_edges_aggregator = blocks.ReceivedEdgesToNodesAggregator(
        reducer=tf.unsorted_segment_sum)
    aggregated_attended_values = received_edges_aggregator(
        attention_graph.replace(edges=attented_edges))

    return attention_graph.replace(nodes=aggregated_attended_values)
