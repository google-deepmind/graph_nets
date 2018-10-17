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

"""Tensorflow ops and helpers useful to manipulate graphs.

This module contains utility functions to operate with `Tensor`s representations
of graphs, in particular:

  - `build_placeholders_from_data_dicts` and `build_placeholders_from_networkx`
     create placeholder structures to represent graphs;

  - `get_feed_dict` allow to create a `feed_dict` from a `graphs.GraphsTuple`
    containing numpy arrays and potentially, `None` values;

  - `data_dicts_to_graphs_tuple` converts between data dictionaries and
    `graphs.GraphsTuple`;

  - `fully_connect_graph_static` (resp. `fully_connect_graph_dynamic`) adds
    edges to a `graphs.GraphsTuple` in a fully-connected manner, in the case
    where the number of nodes per graph is known at graph construction time and
    is the same for all graphs (resp. only known at runtime and may depend on
    the graph);

  - `set_zero_node_features`, `set_zero_edge_features` and
    `set_zero_global_features` complete a `graphs.GraphsTuple` with a `Tensor`
    of zeros for the nodes, edges and globals;

  - `concat` batches `graphs.GraphsTuple` together (when using `axis=0`), or
    concatenates them along their data dimension;

  - `repeat` is a utility convenient to broadcast globals to edges or nodes of
    a graph;

  - `get_graph` indexes or slices a `graphs.GraphsTuple` to extract a subgraph
    or a subbatch of graphs;

  - `stop_gradients` stops the gradients flowing through a graph;

  - `identity` applies a `tf.identity` to every field of a graph;

  - `make_runnable_in_session` allows to run a graph containing `None` fields
    through a Tensorflow session.

The functions in these modules are able to deal with graphs containing `None`
fields (e.g. featureless nodes, featureless edges, or no edges).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from graph_nets import graphs
from graph_nets import utils_np
import six
from six.moves import range
import tensorflow as tf


NODES = graphs.NODES
EDGES = graphs.EDGES
GLOBALS = graphs.GLOBALS
RECEIVERS = graphs.RECEIVERS
SENDERS = graphs.SENDERS
GLOBALS = graphs.GLOBALS
N_NODE = graphs.N_NODE
N_EDGE = graphs.N_EDGE

GRAPH_DATA_FIELDS = graphs.GRAPH_DATA_FIELDS
GRAPH_NUMBER_FIELDS = graphs.GRAPH_NUMBER_FIELDS
ALL_FIELDS = graphs.ALL_FIELDS


def _get_shape(tensor):
  """Returns the tensor's shape.

   Each shape element is either:
   - an `int`, when static shape values are available, or
   - a `tf.Tensor`, when the shape is dynamic.

  Args:
    tensor: A `tf.Tensor` to get the shape of.

  Returns:
    The `list` which contains the tensor's shape.
  """

  shape_list = tensor.shape.as_list()
  if all(s is not None for s in shape_list):
    return shape_list
  shape_tensor = tf.shape(tensor)
  return [shape_tensor[i] if s is None else s for i, s in enumerate(shape_list)]


def _axis_to_inside(tensor, axis):
  """Shifts a given axis of a tensor to be the innermost axis.

  Args:
    tensor: A `tf.Tensor` to shift.
    axis: An `int` or `tf.Tensor` that indicates which axis to shift.

  Returns:
    The shifted tensor.
  """

  axis = tf.convert_to_tensor(axis)
  rank = tf.rank(tensor)

  range0 = tf.range(0, limit=axis)
  range1 = tf.range(tf.add(axis, 1), limit=rank)
  perm = tf.concat([[axis], range0, range1], 0)

  return tf.transpose(tensor, perm=perm)


def _inside_to_axis(tensor, axis):
  """Shifts the innermost axis of a tensor to some other axis.

  Args:
    tensor: A `tf.Tensor` to shift.
    axis: An `int` or `tf.Tensor` that indicates which axis to shift.

  Returns:
    The shifted tensor.
  """

  axis = tf.convert_to_tensor(axis)
  rank = tf.rank(tensor)

  range0 = tf.range(1, limit=axis + 1)
  range1 = tf.range(tf.add(axis, 1), limit=rank)
  perm = tf.concat([range0, [0], range1], 0)

  return tf.transpose(tensor, perm=perm)


def _build_placeholders_from_specs(dtypes,
                                   shapes,
                                   force_dynamic_num_graphs=True):
  """Creates a `graphs.GraphsTuple` of placeholders with `dtypes` and `shapes`.

  The dtypes and shapes arguments are instances of `graphs.GraphsTuple` that
  contain dtypes and shapes, or `None` values for the fields for which no
  placeholder should be created. The leading dimension the nodes and edges are
  dynamic because the numbers of nodes and edges can vary.
  If `force_dynamic_num_graphs` is True, then the number of graphs is assumed to
  be dynamic and all fields leading dimensions are set to `None`.
  If `force_dynamic_num_graphs` is False, then the `GRAPH_NUMBER_FIELDS` leading
  dimensions are statically defined.

  Args:
    dtypes: A `graphs.GraphsTuple` that contains `tf.dtype`s or `None`s.
    shapes: A `graphs.GraphsTuple` that contains `list`s of integers,
      `tf.TensorShape`s, or `None`s.
    force_dynamic_num_graphs: A `bool` that forces the batch dimension to be
      dynamic. Defaults to `True`.

  Returns:
    A `graphs.GraphsTuple` containing placeholders.

  Raises:
    ValueError: The `None` fields in `dtypes` and `shapes` do not match.
  """
  dct = {}
  for field in ALL_FIELDS:
    dtype = getattr(dtypes, field)
    shape = getattr(shapes, field)
    if dtype is None or shape is None:
      if not (shape is None and dtype is None):
        raise ValueError(
            "only one of dtype and shape are None for field {}".format(field))
      dct[field] = None
    elif not shape:
      raise ValueError("Shapes must have at least rank 1")
    else:
      shape = list(shape)
      if field in GRAPH_DATA_FIELDS or force_dynamic_num_graphs:
        shape[0] = None
      dct[field] = tf.placeholder(dtype, shape=shape, name=field)

  return graphs.GraphsTuple(**dct)


def _placeholders_from_graphs_tuple(graph, force_dynamic_num_graphs=False):
  """Creates a `graphs.GraphsTuple` of placeholders that matches a numpy graph.

  Args:
    graph: A `graphs.GraphsTuple` that contains numpy data.
    force_dynamic_num_graphs: A `bool` that forces the batch dimension to be
      dynamic. Defaults to `True`.

  Returns:
    A `graphs.GraphsTuple` containing placeholders.
  """
  graph_dtypes = graph.map(
      lambda v: tf.as_dtype(v.dtype) if v is not None else None, ALL_FIELDS)
  graph_shapes = graph.map(lambda v: list(v.shape) if v is not None else None,
                           ALL_FIELDS)
  return _build_placeholders_from_specs(
      graph_dtypes,
      graph_shapes,
      force_dynamic_num_graphs=force_dynamic_num_graphs)


def get_feed_dict(placeholders, graph):
  """Feeds a `graphs.GraphsTuple` of numpy arrays or `None` into `placeholders`.

  When feeding a fully defined graph (no `None` field) into a session, this
  method is not necessary as one can directly do:

  ```
  _ = sess.run(_, {placeholders: graph})
  ```

  However, if the placeholders contain `None`, the above construction would
  fail. This method allows to replace the above call by

  ```
  _ = sess.run(_, get_feed_dict(placeholders: graph))
  ```

  restoring the correct behavior.

  Args:
    placeholders: A `graphs.GraphsTuple` containing placeholders.
    graph: A `graphs.GraphsTuple` containing placeholder compatibale values,
      or `None`s.

  Returns:
    A dictionary with key placeholders and values the fed in values.

  Raises:
    ValueError: If the `None` fields in placeholders and `graph` do not exactly
      match.
  """
  feed_dict = {}
  for field in ALL_FIELDS:
    placeholder = getattr(placeholders, field)
    feed_value = getattr(graph, field)
    if placeholder is None or feed_value is None:
      if not (placeholder is None and feed_value is None):
        raise ValueError("Field {} should be `None` in either none or both of "
                         "the placeholders and feed values.")
    else:
      feed_dict[placeholder] = feed_value
  return feed_dict


def placeholders_from_data_dicts(data_dicts,
                                 force_dynamic_num_graphs=False,
                                 name="placeholders_from_data_dicts"):
  """Constructs placeholders compatible with a list of data dicts.

  Args:
    data_dicts: An iterable of data dicts containing numpy arrays.
    force_dynamic_num_graphs: A `bool` that forces the batch dimension to be
      dynamic. Defaults to `True`.
    name: (string, optional) A name for the operation.

  Returns:
    An instance of `graphs.GraphTuple` placeholders compatible with the
      dimensions of the dictionaries in `data_dicts`.
  """
  with tf.name_scope(name):
    graph = data_dicts_to_graphs_tuple(data_dicts)
    return _placeholders_from_graphs_tuple(
        graph, force_dynamic_num_graphs=force_dynamic_num_graphs)


def placeholders_from_networkxs(graph_nxs,
                                node_shape_hint=None,
                                edge_shape_hint=None,
                                data_type_hint=tf.float32,
                                force_dynamic_num_graphs=True,
                                name="placeholders_from_networkxs"):
  """Constructs placeholders compatible with a list of networkx instances.

  Given a list of networkxs instances, constructs placeholders compatible with
  the shape of those graphs.

  The networkx graph should be set up such that, for fixed shapes `node_shape`,
   `edge_shape` and `global_shape`:
    - `graph_nx.nodes(data=True)[i][-1]["features"]` is, for any node index i, a
      tensor of shape `node_shape`, or `None`;
    - `graph_nx.edges(data=True)[i][-1]["features"]` is, for any edge index i, a
      tensor of shape `edge_shape`, or `None`;
    - `graph_nx.edges(data=True)[i][-1]["index"]`, if present, defines the order
      in which the edges will be sorted in the resulting `data_dict`;
    - `graph_nx.graph["features"] is a tensor of shape `global_shape` or `None`.

  Args:
    graph_nxs: A container of `networkx.MultiDiGraph`s.
    node_shape_hint: (iterable of `int` or `None`, default=`None`) If the graph
      does not contain nodes, the trailing shape for the created `NODES` field.
      If `None` (the default), this field is left `None`. This is not used if
      `graph_nx` contains at least one node.
    edge_shape_hint: (iterable of `int` or `None`, default=`None`) If the graph
      does not contain edges, the trailing shape for the created `EDGES` field.
      If `None` (the default), this field is left `None`. This is not used if
      `graph_nx` contains at least one edge.
    data_type_hint: (numpy dtype, default=`np.float32`) If the `NODES` or
      `EDGES` fields are autocompleted, their type.
    force_dynamic_num_graphs: A `bool` that forces the batch dimension to be
      dynamic. Defaults to `True`.
    name: (string, optional) A name for the operation.

  Returns:
    An instance of `graphs.GraphTuple` placeholders compatible with the
      dimensions of the graph_nxs.
  """
  with tf.name_scope(name):
    graph = utils_np.networkxs_to_graphs_tuple(graph_nxs, node_shape_hint,
                                               edge_shape_hint,
                                               data_type_hint.as_numpy_dtype())
    return _placeholders_from_graphs_tuple(
        graph, force_dynamic_num_graphs=force_dynamic_num_graphs)


def _compute_stacked_offsets(sizes, repeats):
  """Computes offsets to add to indices of stacked tensors (Tensorflow).

  When a set of tensors are stacked, the indices of those from the second on
  must be offset in order to be able to index into the stacked tensor. This
  computes those offsets.

  Args:
    sizes: A 1D `Tensor` of the sizes per graph.
    repeats: A 1D `Tensor` of the number of repeats per graph.

  Returns:
    A 1D `Tensor` containing the index offset per graph.
  """
  sizes = tf.cast(tf.convert_to_tensor(sizes[:-1]), tf.int32)
  offset_values = tf.cumsum(tf.concat([[0], sizes], 0))
  return repeat(offset_values, repeats)


def concat(input_graphs, axis, name="graph_concat"):
  """Returns an op that concatenates graphs along a given axis.

  In all cases, the NODES, EDGES and GLOBALS dimension are concatenated
  along `axis` (if a fields is `None`, the concatenation is just a `None`).
  If `axis` == 0, then the graphs are concatenated along the (underlying) batch
  dimension, i.e. the RECEIVERS, SENDERS, N_NODE and N_EDGE fields of the tuples
  are also concatenated together.
  If `axis` != 0, then there is an underlying asumption that the receivers,
  SENDERS, N_NODE and N_EDGE fields of the graphs in `values` should all match,
  but this is not checked by this op.
  The graphs in `input_graphs` should have the same set of keys for which the
  corresponding fields is not `None`.

  Args:
    input_graphs: A list of `graphs.GraphsTuple` objects containing `Tensor`s
      and satisfying the constraints outlined above.
    axis: An axis to concatenate on.
    name: (string, optional) A name for the operation.

  Returns: An op that returns the concatenated graphs.

  Raises:
    ValueError: If `values` is an empty list, or if the fields which are `None`
      in `input_graphs` are not the same for all the graphs.
  """
  if not input_graphs:
    raise ValueError("List argument `input_graphs` is empty")
  utils_np._check_valid_sets_of_keys([gr._asdict() for gr in input_graphs])  # pylint: disable=protected-access
  if len(input_graphs) == 1:
    return input_graphs[0]
  nodes = [gr.nodes for gr in input_graphs if gr.nodes is not None]
  edges = [gr.edges for gr in input_graphs if gr.edges is not None]
  globals_ = [gr.globals for gr in input_graphs if gr.globals is not None]

  with tf.name_scope(name):
    nodes = tf.concat(nodes, axis, name="concat_nodes") if nodes else None
    edges = tf.concat(edges, axis, name="concat_edges") if edges else None
    if globals_:
      globals_ = tf.concat(globals_, axis, name="concat_globals")
    else:
      globals_ = None
    output = input_graphs[0].replace(nodes=nodes, edges=edges, globals=globals_)
    if axis != 0:
      return output
    n_node_per_tuple = tf.stack(
        [tf.reduce_sum(gr.n_node) for gr in input_graphs])
    n_edge_per_tuple = tf.stack(
        [tf.reduce_sum(gr.n_edge) for gr in input_graphs])
    offsets = _compute_stacked_offsets(n_node_per_tuple, n_edge_per_tuple)
    n_node = tf.concat(
        [gr.n_node for gr in input_graphs], axis=0, name="concat_n_node")
    n_edge = tf.concat(
        [gr.n_edge for gr in input_graphs], axis=0, name="concat_n_edge")
    receivers = [
        gr.receivers for gr in input_graphs if gr.receivers is not None
    ]
    receivers = receivers or None
    if receivers:
      receivers = tf.concat(receivers, axis, name="concat_receivers") + offsets
    senders = [gr.senders for gr in input_graphs if gr.senders is not None]
    senders = senders or None
    if senders:
      senders = tf.concat(senders, axis, name="concat_senders") + offsets
    return output.replace(
        receivers=receivers, senders=senders, n_node=n_node, n_edge=n_edge)


def stop_gradient(graph,
                  stop_edges=True,
                  stop_nodes=True,
                  stop_globals=True,
                  name="graph_stop_gradient"):
  """Stops the gradient flow through a graph.

  Args:
    graph: An instance of `graphs.GraphsTuple` containing `Tensor`s.
    stop_edges: (bool, default=True) indicates whether to stop gradients for
      the edges.
    stop_nodes: (bool, default=True) indicates whether to stop gradients for
      the nodes.
    stop_globals: (bool, default=True) indicates whether to stop gradients for
      the globals.
    name: (string, optional) A name for the operation.

  Returns:
    GraphsTuple after stopping the gradients according to the provided
    parameters.

  Raises:
    ValueError: If attempting to stop gradients through a field which has a
      `None` value in `graph`.
  """

  base_err_msg = "Cannot stop gradient through {0} if {0} are None"
  fields_to_stop = []
  if stop_globals:
    if graph.globals is None:
      raise ValueError(base_err_msg.format(GLOBALS))
    fields_to_stop.append(GLOBALS)
  if stop_nodes:
    if graph.nodes is None:
      raise ValueError(base_err_msg.format(NODES))
    fields_to_stop.append(NODES)
  if stop_edges:
    if graph.edges is None:
      raise ValueError(base_err_msg.format(EDGES))
    fields_to_stop.append(EDGES)

  with tf.name_scope(name):
    return graph.map(tf.stop_gradient, fields_to_stop)


def identity(graph, name="graph_identity"):
  """Pass each element of a graph through a `tf.identity`.

  This allows, for instance, to push a name scope on the graph by writing:
  ```
  with tf.name_scope("encoder"):
    graph = utils_tf.identity(graph)
  ```

  Args:
    graph: A `graphs.GraphsTuple` containing `Tensor`s. `None` values are passed
      through.
    name: (string, optional) A name for the operation.

  Returns:
    A `graphs.GraphsTuple` `graphs_output` such that for any field `x` in NODES,
    EDGES, GLOBALS, RECEIVERS, SENDERS, N_NODE, N_EDGE, if `graph.x` was
    `None`, `graph_output.x` is `None`, and otherwise
    `graph_output.x = tf.identity(graph.x)`
  """
  non_none_fields = [k for k in ALL_FIELDS if getattr(graph, k) is not None]
  with tf.name_scope(name):
    return graph.map(tf.identity, non_none_fields)


def make_runnable_in_session(graph, name="make_graph_runnable_in_session"):
  """Allows a graph containing `None` fields to be run in a `tf.Session`.

  The `None` values of `graph` are replaced by `tf.no_op()`. This function is
  meant to be called just before a call to `sess.run` on a Tensorflow session
  `sess`, as `None` values currently cannot be run through a session.

  Args:
    graph: A `graphs.GraphsTuple` containing `Tensor`s or `None` values.
    name: (string, optional) A name for the operation.

  Returns:
    A `graphs.GraphsTuple` `graph_output` such that, for any field `x` in NODES,
    EDGES, GLOBALS, RECEIVERS, SENDERS, N_NODE, N_EDGE, and a Tensorflow session
    `sess`, if `graph.x` was `None`, `sess.run(graph_output)` is `None`, and
    otherwise
  """
  none_fields = [k for k in ALL_FIELDS if getattr(graph, k) is None]
  with tf.name_scope(name):
    return graph.map(lambda _: tf.no_op(), none_fields)


def repeat(tensor, repeats, axis=0, name="repeat"):
  """Repeats a `tf.Tensor`'s elements along an axis by custom amounts.

  Equivalent to Numpy's `np.repeat`.
  `tensor and `repeats` must have the same numbers of elements along `axis`.

  Args:
    tensor: A `tf.Tensor` to repeat.
    repeats: A 1D sequence of the number of repeats per element.
    axis: An axis to repeat along. Defaults to 0.
    name: (string, optional) A name for the operation.

  Returns:
    The `tf.Tensor` with repeated values.
  """
  with tf.name_scope(name):
    cumsum = tf.cumsum(repeats)
    range_ = tf.range(cumsum[-1])

    indicator_matrix = tf.cast(tf.expand_dims(range_, 1) >= cumsum, tf.int32)
    indices = tf.reduce_sum(indicator_matrix, reduction_indices=1)

    shifted_tensor = _axis_to_inside(tensor, axis)
    repeated_shifted_tensor = tf.gather(shifted_tensor, indices)
    repeated_tensor = _inside_to_axis(repeated_shifted_tensor, axis)

    shape = tensor.shape.as_list()
    shape[axis] = None
    repeated_tensor.set_shape(shape)

    return repeated_tensor


def _populate_number_fields(data_dict):
  """Returns a dict with the number fields N_NODE, N_EDGE filled in.

  The N_NODE field is filled if the graph contains a non-`None` NODES field;
  otherwise, it is set to 0.
  The N_EDGE field is filled if the graph contains a non-`None` RECEIVERS field;
  otherwise, it is set to 0.

  Args:
    data_dict: An input `dict`.

  Returns:
    The data `dict` with number fields.
  """
  dct = data_dict.copy()
  for number_field, data_field in [[N_NODE, NODES], [N_EDGE, RECEIVERS]]:
    if dct.get(number_field) is None:
      if dct[data_field] is not None:
        dct[number_field] = tf.shape(dct[data_field])[0]
      else:
        dct[number_field] = tf.constant(0, dtype=tf.int32)
  return dct


def _to_compatible_data_dicts(data_dicts):
  """Convert the content of `data_dicts` to tensors of the right type.

  All fields are converted to `Tensor`s. The index fields (`SENDERS` and
  `RECEIVERS`) and number fields (`N_NODE`, `N_EDGE`) are cast to `tf.int32`.

  Args:
    data_dicts: An iterable of dictionaries with keys `ALL_KEYS` and
      values either `None`s, or quantities that can be converted to `Tensor`s.

  Returns:
    A list of dictionaries containing `Tensor`s or `None`s.
  """
  results = []
  for data_dict in data_dicts:
    result = {}
    for k, v in data_dict.items():
      if v is None:
        result[k] = None
      else:
        dtype = tf.int32 if k in [SENDERS, RECEIVERS, N_NODE, N_EDGE] else None
        result[k] = tf.convert_to_tensor(v, dtype)
    results.append(result)
  return results


def _concatenate_data_dicts(data_dicts):
  """Concatenate a list of data dicts to create the equivalent batched graph.

  Args:
    data_dicts: An iterable of data dictionaries with keys a subset of
      `GRAPH_DATA_FIELDS`, plus, potentially, a subset of `GRAPH_NUMBER_FIELDS`.
      Every element of `data_dicts` has to contain the same set of keys.
      Moreover, the key `NODES` or `N_NODE` must be present in every element of
      `data_dicts`.

  Returns:
    A data dictionary with the keys `GRAPH_DATA_FIELDS + GRAPH_NUMBER_FIELDS`,
    representing the concatenated graphs.

  Raises:
    ValueError: If two dictionaries in `data_dicts` have a different set of
      keys.
  """
  # Go from a list of dict to a dict of lists
  dct = collections.defaultdict(lambda: [])
  for data_dict in data_dicts:
    data_dict = _populate_number_fields(data_dict)
    for k, v in data_dict.items():
      if v is not None:
        dct[k].append(v)
      elif k not in dct:
        dct[k] = None
  dct = dict(dct)

  # Concatenate the graphs.
  for field, tensors in dct.items():
    if tensors is None:
      dct[field] = None
    elif field in list(GRAPH_NUMBER_FIELDS) + [GLOBALS]:
      dct[field] = tf.stack(tensors)
    else:
      dct[field] = tf.concat(tensors, axis=0)

  # Add offsets to the receiver and sender indices.
  if dct[RECEIVERS] is not None:
    offset = _compute_stacked_offsets(dct[N_NODE], dct[N_EDGE])
    dct[RECEIVERS] += offset
    dct[SENDERS] += offset

  return dct


def _create_complete_edges_from_nodes_static(n_node, exclude_self_edges):
  """Creates complete edges for a graph with `n_node`.

  Args:
    n_node: (python integer) The number of nodes.
    exclude_self_edges: (bool) Excludes self-connected edges.

  Returns:
    A dict of RECEIVERS, SENDERS and N_EDGE data (`Tensor`s of rank 1).
  """
  receivers = []
  senders = []
  n_edges = 0
  for node_1 in range(n_node):
    for node_2 in range(n_node):
      if not exclude_self_edges or node_1 != node_2:
        receivers.append(node_1)
        senders.append(node_2)
        n_edges += 1

  return {
      RECEIVERS: tf.constant(receivers, dtype=tf.int32),
      SENDERS: tf.constant(senders, dtype=tf.int32),
      N_EDGE: tf.constant([n_edges], dtype=tf.int32)
  }


def _create_complete_edges_from_nodes_dynamic(n_node, exclude_self_edges):
  """Creates complete edges for a graph with `n_node`.

  Args:
    n_node: (integer scalar `Tensor`) The number of nodes.
    exclude_self_edges: (bool) Excludes self-connected edges.

  Returns:
    A dict of RECEIVERS, SENDERS and N_EDGE data (`Tensor`s of rank 1).
  """
  rng = tf.range(n_node)
  receivers, senders = tf.meshgrid(rng, rng)
  n_edge = n_node * n_node

  if exclude_self_edges:
    ind = tf.cast(1 - tf.eye(n_node), bool)
    receivers = tf.boolean_mask(receivers, ind)
    senders = tf.boolean_mask(senders, ind)
    n_edge -= n_node

  receivers = tf.reshape(tf.cast(receivers, tf.int32), [n_edge])
  senders = tf.reshape(tf.cast(senders, tf.int32), [n_edge])
  n_edge = tf.reshape(n_edge, [1])

  return {RECEIVERS: receivers, SENDERS: senders, N_EDGE: n_edge}


def _validate_edge_fields_are_all_none(graph):
  if not all(getattr(graph, x) is None for x in [EDGES, RECEIVERS, SENDERS]):
    raise ValueError("Can only add fully connected a graph with `None`"
                     "edges, receivers and senders")


def fully_connect_graph_static(graph,
                               exclude_self_edges=False,
                               name="fully_connect_graph_static"):
  """Adds edges to a graph by fully-connecting the nodes.

  This method can be used if the number of nodes for each graph in `graph` is
  constant and known at graph building time: it will be inferred by dividing
  the number of nodes in the batch(the length of `graph.nodes`) by the number of
  graphs in the batch (the length of `graph.n_node`). It is an error to call
  this method with batches of graphs with dynamic or uneven sizes; in the latter
  case, the method may silently yield an incorrect result.

  Args:
    graph: A `graphs.GraphsTuple` with `None` values for the edges, senders and
      receivers.
    exclude_self_edges (default=False): Excludes self-connected edges.
    name: (string, optional) A name for the operation.

  Returns:
    A `graphs.GraphsTuple` containing `Tensor`s with fully-connected edges.

  Raises:
    ValueError: If any of the `EDGES`, `RECEIVERS` or `SENDERS` field is not
      `None` in `graph`.
    ValueError: If the number of graphs (extracted from `graph.n_node` leading
      dimension) or number of nodes (extracted from `graph.nodes` leading
      dimension) is not known at construction time, or if the latter does not
      divide the former (observe that this is only a necessary condition for
      the constantness of the number of nodes per graph).
  """
  _validate_edge_fields_are_all_none(graph)

  num_graphs = graph.n_node.shape.as_list()[0]
  if num_graphs is None:
    raise ValueError("Number of graphs must be known at construction time when "
                     "using `fully_connect_graph_static`. Did you mean to use "
                     "`fully_connect_graph_dynamic`?")
  num_nodes = graph.nodes.shape.as_list()[0]
  if num_nodes is None:
    raise ValueError("Number of nodes must be known at construction time when "
                     "using `fully_connect_graph_static`. Did you mean to use "
                     "`fully_connect_graph_dynamic`?")
  if num_nodes % num_graphs != 0:
    raise ValueError("Number of nodes must be the same in all graphs when "
                     "using `fully_connect_graph_static`. Did you mean to use "
                     "`fully_connect_graph_dynamic`?")
  num_nodes_per_graph = num_nodes // num_graphs

  with tf.name_scope(name):
    one_graph_edges = _create_complete_edges_from_nodes_static(
        num_nodes_per_graph, exclude_self_edges)
    n_edges = num_nodes_per_graph * (num_nodes_per_graph - 1)
    if not exclude_self_edges:
      n_edges += num_nodes_per_graph

    all_graph_edges = {
        k: tf.tile(v, [num_graphs]) for k, v in six.iteritems(one_graph_edges)
    }
    offsets = [
        num_nodes_per_graph * i
        for i in range(num_graphs)
        for _ in range(n_edges)
    ]
    all_graph_edges[RECEIVERS] += offsets
    all_graph_edges[SENDERS] += offsets
    return graph.replace(**all_graph_edges)


def fully_connect_graph_dynamic(graph,
                                exclude_self_edges=False,
                                name="fully_connect_graph_dynamic"):
  """Adds edges to a graph by fully-connecting the nodes.

  This method does not require the number of nodes per graph to be constant,
  or to be known at graph building time.

  Args:
    graph: A `graphs.GraphsTuple` with `None` values for the edges, senders and
      receivers.
    exclude_self_edges (default=False): Excludes self-connected edges.
    name: (string, optional) A name for the operation.

  Returns:
    A `graphs.GraphsTuple` containing `Tensor`s with fully-connected edges.

  Raises:
    ValueError: if any of the `EDGES`, `RECEIVERS` or `SENDERS` field is not
      `None` in `graph`.
  """
  _validate_edge_fields_are_all_none(graph)

  with tf.name_scope(name):

    def body(i, senders, receivers, n_edge):
      edges = _create_complete_edges_from_nodes_dynamic(graph.n_node[i],
                                                        exclude_self_edges)
      return (i + 1, senders.write(i, edges[SENDERS]),
              receivers.write(i, edges[RECEIVERS]),
              n_edge.write(i, edges[N_EDGE]))

    num_graphs = get_num_graphs(graph)
    loop_condition = lambda i, *_: tf.less(i, num_graphs)
    initial_loop_vars = [0] + [
        tf.TensorArray(dtype=tf.int32, size=num_graphs, infer_shape=False)
        for _ in range(3)  # senders, receivers, n_edge
    ]
    _, senders_array, receivers_array, n_edge_array = tf.while_loop(
        loop_condition, body, initial_loop_vars, back_prop=False)

    n_edge = n_edge_array.concat()
    offsets = _compute_stacked_offsets(graph.n_node, n_edge)
    senders = senders_array.concat() + offsets
    receivers = receivers_array.concat() + offsets
    senders.set_shape(offsets.shape)
    receivers.set_shape(offsets.shape)

    receivers.set_shape([None])
    senders.set_shape([None])

    num_graphs = graph.n_node.get_shape().as_list()[0]
    n_edge.set_shape([num_graphs])

    return graph._replace(senders=senders, receivers=receivers, n_edge=n_edge)


def set_zero_node_features(graph,
                           node_size,
                           dtype=tf.float32,
                           name="set_zero_node_features"):
  """Completes the node state of a graph.

  Args:
    graph: A `graphs.GraphsTuple` with a `None` edge state.
    node_size: (int) the dimension for the created node features.
    dtype: (tensorflow type) the type for the created nodes features.
    name: (string, optional) A name for the operation.

  Returns:
    The same graph but for the node field, which is a `Tensor` of shape
    `[number_of_nodes, node_size]`  where `number_of_nodes = sum(graph.n_node)`,
    with type `dtype`, filled with zeros.

  Raises:
    ValueError: If the `NODES` field is not None in `graph`.
    ValueError: If `node_size` is None.
  """
  if graph.nodes is not None:
    raise ValueError(
        "Cannot complete node state if the graph already has node features.")
  if node_size is None:
    raise ValueError("Cannot complete nodes with None node_size")
  with tf.name_scope(name):
    n_nodes = tf.reduce_sum(graph.n_node)
    return graph._replace(
        nodes=tf.zeros(shape=[n_nodes, node_size], dtype=dtype))


def set_zero_edge_features(graph,
                           edge_size,
                           dtype=tf.float32,
                           name="set_zero_edge_features"):
  """Completes the edge state of a graph.

  Args:
    graph: A `graphs.GraphsTuple` with a `None` edge state.
    edge_size: (int) the dimension for the created edge features.
    dtype: (tensorflow type) the type for the created edge features.
    name: (string, optional) A name for the operation.

  Returns:
    The same graph but for the edge field, which is a `Tensor` of shape
    `[number_of_edges, edge_size]`, where `number_of_edges = sum(graph.n_edge)`,
    with type `dtype` and filled with zeros.

  Raises:
    ValueError: If the `EDGES` field is not None in `graph`.
    ValueError: If the `RECEIVERS` or `SENDERS` field are None in `graph`.
    ValueError: If `edge_size` is None.
  """
  if graph.edges is not None:
    raise ValueError(
        "Cannot complete edge state if the graph already has edge features.")
  if graph.receivers is None or graph.senders is None:
    raise ValueError(
        "Cannot complete edge state if the receivers or senders are None.")
  if edge_size is None:
    raise ValueError("Cannot complete edges with None edge_size")
  with tf.name_scope(name):
    n_edges = tf.reduce_sum(graph.n_edge)
    return graph._replace(
        edges=tf.zeros(shape=[n_edges, edge_size], dtype=dtype))


def set_zero_global_features(graph,
                             global_size,
                             dtype=tf.float32,
                             name="set_zero_global_features"):
  """Completes the global state of a graph.

  Args:
    graph: A `graphs.GraphsTuple` with a `None` global state.
    global_size: (int) the dimension for the created global features.
    dtype: (tensorflow type) the type for the created global features.
    name: (string, optional) A name for the operation.

  Returns:
    The same graph but for the global field, which is a `Tensor` of shape
    `[num_graphs, global_size]`, type `dtype` and filled with zeros.

  Raises:
    ValueError: If the `GLOBALS` field of `graph` is not `None`.
    ValueError: If `global_size` is not `None`.
  """
  if graph.globals is not None:
    raise ValueError(
        "Cannot complete global state if graph already has global features.")
  if global_size is None:
    raise ValueError("Cannot complete globals with None global_size")
  with tf.name_scope(name):
    n_graphs = get_num_graphs(graph)
    return graph._replace(
        globals=tf.zeros(shape=[n_graphs, global_size], dtype=dtype))


def data_dicts_to_graphs_tuple(data_dicts, name="data_dicts_to_graphs_tuple"):
  """Creates a `graphs.GraphsTuple` containing tensors from data dicts.

   All dictionaries must have exactly the same set of keys with non-`None`
   values associated to them. Moreover, this set of this key must define a valid
   graph (i.e. if the `EDGES` are `None`, the `SENDERS` and `RECEIVERS` must be
   `None`, and `SENDERS` and `RECEIVERS` can only be `None` both at the same
   time). The values associated with a key must be convertible to `Tensor`s,
   for instance python lists, numpy arrays, or Tensorflow `Tensor`s.

   This method may perform a memory copy.

   The `RECEIVERS`, `SENDERS`, `N_NODE` and `N_EDGE` fields are cast to
   `np.int32` type.

  Args:
    data_dicts: An iterable of data dictionaries with keys in `ALL_FIELDS`.
    name: (string, optional) A name for the operation.

  Returns:
    A `graphs.GraphTuple` representing the graphs in `data_dicts`.
  """
  for key in ALL_FIELDS:
    for data_dict in data_dicts:
      data_dict.setdefault(key, None)
  utils_np._check_valid_sets_of_keys(data_dicts)  # pylint: disable=protected-access
  with tf.name_scope(name):
    data_dicts = _to_compatible_data_dicts(data_dicts)
    return graphs.GraphsTuple(**_concatenate_data_dicts(data_dicts))


def get_graph(input_graphs, index, name="get_graph"):
  """Indexes into a graph.

  Given a `graphs.graphsTuple` containing `Tensor`s and an index (either
  an `int` or a `slice`), index into the nodes, edges and globals to extract the
  graphs specified by the slice, and returns them into an another instance of a
  `graphs.graphsTuple` containing `Tensor`s.

  Args:
    input_graphs: A `graphs.GraphsTuple` containing `Tensor`s.
    index: An `int` or a `slice`, to index into `graph`. `index` should be
      compatible with the number of graphs in `graphs`.
    name: (string, optional) A name for the operation.

  Returns:
    A `graphs.GraphsTuple` containing `Tensor`s, made of the extracted
      graph(s).

  Raises:
    TypeError: if `index` is not an `int` or a `slice`.
  """

  def safe_slice_none(value, slice_):
    if value is None:
      return value
    return value[slice_]

  if isinstance(index, int):
    graph_slice = slice(index, index + 1)
  elif isinstance(index, slice):
    graph_slice = index
  else:
    raise TypeError("unsupported type: %s" % type(index))

  start_slice = slice(0, graph_slice.start)

  with tf.name_scope(name):
    start_node_index = tf.reduce_sum(
        input_graphs.n_node[start_slice], name="start_node_index")
    start_edge_index = tf.reduce_sum(
        input_graphs.n_edge[start_slice], name="start_edge_index")
    end_node_index = start_node_index + tf.reduce_sum(
        input_graphs.n_node[graph_slice], name="end_node_index")
    end_edge_index = start_edge_index + tf.reduce_sum(
        input_graphs.n_edge[graph_slice], name="end_edge_index")
    nodes_slice = slice(start_node_index, end_node_index)
    edges_slice = slice(start_edge_index, end_edge_index)

    sliced_graphs_dict = {}

    for field in set(GRAPH_NUMBER_FIELDS) | {"globals"}:
      sliced_graphs_dict[field] = safe_slice_none(
          getattr(input_graphs, field), graph_slice)

    field = "nodes"
    sliced_graphs_dict[field] = safe_slice_none(
        getattr(input_graphs, field), nodes_slice)

    for field in {"edges", "senders", "receivers"}:
      sliced_graphs_dict[field] = safe_slice_none(
          getattr(input_graphs, field), edges_slice)
      if (field in {"senders", "receivers"} and
          sliced_graphs_dict[field] is not None):
        sliced_graphs_dict[field] = sliced_graphs_dict[field] - start_node_index

    return graphs.GraphsTuple(**sliced_graphs_dict)


def get_num_graphs(input_graphs, name="get_num_graphs"):
  """Returns the number of graphs (i.e. the batch size) in `input_graphs`.

  Args:
    input_graphs: A `graphs.GraphsTuple` containing tensors.
    name: (string, optional) A name for the operation.

  Returns:
    An `int` (if a static number of graphs is defined) or a `tf.Tensor` (if the
      number of graphs is dynamic).
  """
  with tf.name_scope(name):
    return _get_shape(input_graphs.n_node)[0]
