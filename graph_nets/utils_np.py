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
"""Auxiliary methods that operate on graph structured data.

This modules contains functions to convert between python data structures
representing graphs and `graphs.GraphsTuple` containing numpy arrays.
In particular:

  - `networkx_to_data_dict` and `data_dict_to_networkx` convert from/to an
    instance of `networkx.OrderedMultiDiGraph` from/to a data dictionary;

  - `networkxs_to_graphs_tuple` and `graphs_tuple_to_networkxs` convert
    from instances of `networkx.OrderedMultiDiGraph` to `graphs.GraphsTuple`;

  - `data_dicts_to_graphs_tuple` and `graphs_tuple_to_data_dicts` convert to and
    from lists of data dictionaries and `graphs.GraphsTuple`;

  - `get_graph` allows to index or slice a `graphs.GraphsTuple` to extract a
    subgraph or a subbatch of graphs.

The functions in these modules are able to deal with graphs containing `None`
fields (e.g. featureless nodes, featureless edges, or no edges).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from graph_nets import graphs
import networkx as nx
import numpy as np
from six.moves import range
from six.moves import zip  # pylint: disable=redefined-builtin

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

GRAPH_NX_FEATURES_KEY = "features"


def _check_valid_keys(keys):
  if any([x in keys for x in [EDGES, RECEIVERS, SENDERS]]):
    if not (RECEIVERS in keys and SENDERS in keys):
      raise ValueError("If edges are present, senders and receivers should "
                       "both be defined.")


def _defined_keys(dict_):
  return {k for k, v in dict_.items() if v is not None}


def _check_valid_sets_of_keys(dicts):
  """Checks that all dictionaries have exactly the same valid key sets."""
  prev_keys = None
  for dict_ in dicts:
    current_keys = _defined_keys(dict_)
    _check_valid_keys(current_keys)
    if prev_keys and current_keys != prev_keys:
      raise ValueError(
          "Different set of keys found when iterating over data dictionaries "
          "({} vs {})".format(prev_keys, current_keys))
    prev_keys = current_keys


def _compute_stacked_offsets(sizes, repeats):
  """Computes offsets to add to indices of stacked np arrays.

  When a set of np arrays are stacked, the indices of those from the second on
  must be offset in order to be able to index into the stacked np array. This
  computes those offsets.

  Args:
    sizes: A 1D sequence of np arrays of the sizes per graph.
    repeats: A 1D sequence of np arrays of the number of repeats per graph.

  Returns:
    The index offset per graph.
  """
  return np.repeat(np.cumsum(np.hstack([0, sizes[:-1]])), repeats)


def networkx_to_data_dict(graph_nx,
                          node_shape_hint=None,
                          edge_shape_hint=None,
                          data_type_hint=np.float32):
  """Returns a data dict of Numpy data from a networkx graph.

  The networkx graph should be set up such that, for fixed shapes `node_shape`,
   `edge_shape` and `global_shape`:
    - `graph_nx.nodes(data=True)[i][-1]["features"]` is, for any node index i, a
      tensor of shape `node_shape`, or `None`;
    - `graph_nx.edges(data=True)[i][-1]["features"]` is, for any edge index i, a
      tensor of shape `edge_shape`, or `None`;
    - `graph_nx.edges(data=True)[i][-1]["index"]`, if present, defines the order
      in which the edges will be sorted in the resulting `data_dict`;
    - `graph_nx.graph["features"] is a tensor of shape `global_shape`, or
      `None`.

  The dictionary `type_hints` can provide hints of the "float" and "int" types
  for missing values.

  The output data is a sequence of data dicts with fields:
    NODES, EDGES, RECEIVERS, SENDERS, GLOBALS, N_NODE, N_EDGE.

  Args:
    graph_nx: A `networkx.OrderedMultiDiGraph`.
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

  Returns:
    The data `dict` of Numpy data.

  Raises:
    TypeError: If `graph_nx` is not an instance of networkx.
    KeyError: If `graph_nx` contains at least one node without the "features"
      key in its attribute dictionary, or at least one edge without the
      "features" key in its attribute dictionary.
    ValueError: If `graph_nx` contains at least one node with a `None`
      "features" attribute and one least one node with a non-`None` "features"
      attribute; or if `graph_nx` contains at least one edge with a `None`
      "features" attribute and one least one edge with a non-`None` "features"
      attribute.
  """
  nodes = None
  try:
    number_of_nodes = graph_nx.number_of_nodes()
  except ValueError:
    raise TypeError("Argument `graph_nx` of wrong type {}".format(
        type(graph_nx)))
  if number_of_nodes == 0:
    if node_shape_hint is not None:
      nodes = np.zeros([0] + list(node_shape_hint), dtype=data_type_hint)
  else:
    try:
      nodes_data = [
          x[1][GRAPH_NX_FEATURES_KEY]
          for x in graph_nx.nodes(data=True)
          if x[1][GRAPH_NX_FEATURES_KEY] is not None
      ]
      if nodes_data:
        if len(nodes_data) != number_of_nodes:
          raise ValueError(
              "Either all the nodes should have features, or none of them")
        nodes = np.array(nodes_data)
    except KeyError:
      raise KeyError("Missing 'node' field from the graph nodes. "
                     "This could be due to the node having been silently added "
                     "as a consequence of an edge addition when creating the "
                     "networkx instance")

  edges = None
  number_of_edges = graph_nx.number_of_edges()
  if number_of_edges == 0:
    senders = np.zeros(0, dtype=np.int32)
    receivers = np.zeros(0, dtype=np.int32)
    if edge_shape_hint is not None:
      edges = np.zeros([0] + list(edge_shape_hint), dtype=data_type_hint)
  else:
    if "index" in list(graph_nx.edges(data=True))[0][2]:
      senders, receivers, edge_attr_dicts = zip(
          *sorted(graph_nx.edges(data=True), key=lambda x: x[2]["index"]))
    else:
      senders, receivers, edge_attr_dicts = zip(*graph_nx.edges(data=True))
    senders = np.array(senders, dtype=np.int32)
    receivers = np.array(receivers, dtype=np.int32)
    edges_data = [
        x[GRAPH_NX_FEATURES_KEY]
        for x in edge_attr_dicts
        if x[GRAPH_NX_FEATURES_KEY] is not None
    ]
    if edges_data:
      if len(edges_data) != number_of_edges:
        raise ValueError(
            "Either all the edges should have features, or none of them")
      edges = np.array(edges_data)

  globals_ = None
  if GRAPH_NX_FEATURES_KEY in graph_nx.graph:
    globals_ = graph_nx.graph[GRAPH_NX_FEATURES_KEY]

  return {
      NODES: nodes,
      EDGES: edges,
      RECEIVERS: receivers,
      SENDERS: senders,
      GLOBALS: globals_,
      N_NODE: number_of_nodes,
      N_EDGE: number_of_edges,
  }


def _unstack(array):
  """Similar to `tf.unstack`."""
  num_splits = int(array.shape[0])
  return [np.squeeze(x, 0) for x in np.split(array, num_splits, axis=0)]


def data_dict_to_networkx(data_dict):
  """Returns a networkx graph that contains the stored data.

  Depending on the content of `data_dict`, the returned `networkx` instance has
  the following properties:

  - The nodes feature are placed in the nodes attribute dictionary under the
    "features" key. If the `NODES` fields is `None`, a `None` value is placed
    here;

  - If the `RECEIVERS` field is `None`, no edges are added to the graph.
    Otherwise, edges are added with the order in which they appeared in
    `data_dict` stored in the "index" field of their attributes dictionary;

  - The edges features are placed in the edges attribute dictionary under the
    "features" key. If the `EDGES` field is `None`, a `None` value is placed;

  - The global feature are placed under the key "features" of the graph
    property of the returned instance. If the `GLOBALS` field is `None`, a
    `None` global property is created.

  Args:
    data_dict: A graph `dict` of Numpy data.

  Returns:
    The `networkx.OrderedMultiDiGraph`.

  Raises:
    ValueError: If the `NODES` field of `data_dict` contains `None`, and
      `data_dict` does not have a `N_NODE` field.
  """
  graph_nx = nx.OrderedMultiDiGraph()
  data_dict = _populate_number_fields(data_dict)
  graph_nx.graph[GRAPH_NX_FEATURES_KEY] = data_dict[GLOBALS]

  if data_dict[NODES] is not None:
    if data_dict[NODES].shape[0] > 0:
      nodes_list = _unstack(data_dict[NODES])
      for i, x in enumerate(nodes_list):
        graph_nx.add_node(i, **{GRAPH_NX_FEATURES_KEY: x})
  elif data_dict[N_NODE] is not None:
    for i in range(data_dict[N_NODE]):
      graph_nx.add_node(i, **{GRAPH_NX_FEATURES_KEY: None})
  else:
    raise ValueError("Cannot create a graph with unspecified number of nodes")

  if data_dict[EDGES] is not None and data_dict[EDGES].shape[0] > 0:
    edges_features = [{  # pylint: disable=g-complex-comprehension
        "index": i,
        GRAPH_NX_FEATURES_KEY: x
    } for i, x in enumerate(_unstack(data_dict[EDGES]))]
    edges_data = zip(data_dict[SENDERS], data_dict[RECEIVERS], edges_features)
    graph_nx.add_edges_from(edges_data)
  elif data_dict[RECEIVERS] is not None and data_dict[RECEIVERS].shape[0] > 0:
    edges_features = [{  # pylint: disable=g-complex-comprehension
        "index": i,
        GRAPH_NX_FEATURES_KEY: None
    } for i in range(data_dict[RECEIVERS].shape[0])]
    edges_data = zip(data_dict[SENDERS], data_dict[RECEIVERS], edges_features)
    graph_nx.add_edges_from(edges_data)

  return graph_nx


def networkxs_to_graphs_tuple(graph_nxs,
                              node_shape_hint=None,
                              edge_shape_hint=None,
                              data_type_hint=np.float32):
  """Constructs an instance from an iterable of networkx graphs.

   The networkx graph should be set up such that, for fixed shapes `node_shape`,
   `edge_shape` and `global_shape`:
    - `graph_nx.nodes(data=True)[i][-1]["features"]` is, for any node index i, a
      tensor of shape `node_shape`, or `None`;
    - `graph_nx.edges(data=True)[i][-1]["features"]` is, for any edge index i, a
      tensor of shape `edge_shape`, or `None`;
    - `graph_nx.edges(data=True)[i][-1]["index"]`, if present, defines the order
      in which the edges will be sorted in the resulting `data_dict`;
    - `graph_nx.graph["features"] is a tensor of shape `global_shape`, or
      `None`.

  The output data is a sequence of data dicts with fields:
    NODES, EDGES, RECEIVERS, SENDERS, GLOBALS, N_NODE, N_EDGE.

  Args:
    graph_nxs: A container of `networkx.OrderedMultiDiGraph`s.
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

  Returns:
    The instance.

  Raises:
    ValueError: If `graph_nxs` is not an iterable of networkx instances.
  """
  data_dicts = []
  try:
    for graph_nx in graph_nxs:
      data_dict = networkx_to_data_dict(graph_nx, node_shape_hint,
                                        edge_shape_hint, data_type_hint)
      data_dicts.append(data_dict)
  except TypeError:
    raise ValueError("Could not convert some elements of `graph_nxs`. "
                     "Did you pass an iterable of networkx instances?")

  return data_dicts_to_graphs_tuple(data_dicts)


def graphs_tuple_to_networkxs(graphs_tuple):
  """Converts a `graphs.GraphsTuple` to a sequence of networkx graphs.

  Args:
    graphs_tuple: A `graphs.GraphsTuple` instance containing numpy arrays.

  Returns:
    The list of `networkx.OrderedMultiDiGraph`s.
  """
  return [
      data_dict_to_networkx(x) for x in graphs_tuple_to_data_dicts(graphs_tuple)
  ]


def data_dicts_to_graphs_tuple(data_dicts):
  """Constructs a `graphs.GraphsTuple` from an iterable of data dicts.

  The graphs represented by the `data_dicts` argument are batched to form a
  single instance of `graphs.GraphsTuple` containing numpy arrays.

  Args:
    data_dicts: An iterable of dictionaries with keys `GRAPH_DATA_FIELDS`, plus,
      potentially, a subset of `GRAPH_NUMBER_FIELDS`. The NODES and EDGES fields
      should be numpy arrays of rank at least 2, while the RECEIVERS, SENDERS
      are numpy arrays of rank 1 and same dimension as the EDGES field first
      dimension. The GLOBALS field is a numpy array of rank at least 1.

  Returns:
    An instance of `graphs.GraphsTuple` containing numpy arrays. The
    `RECEIVERS`, `SENDERS`, `N_NODE` and `N_EDGE` fields are cast to `np.int32`
    type.
  """
  data_dicts = [dict(d) for d in data_dicts]
  for key in graphs.GRAPH_DATA_FIELDS:
    for data_dict in data_dicts:
      data_dict.setdefault(key, None)
  _check_valid_sets_of_keys(data_dicts)
  data_dicts = _to_compatible_data_dicts(data_dicts)
  return graphs.GraphsTuple(**_concatenate_data_dicts(data_dicts))


def graphs_tuple_to_data_dicts(graph):
  """Splits the stored data into a list of individual data dicts.

  Each list is a dictionary with fields NODES, EDGES, GLOBALS, RECEIVERS,
  SENDERS.

  Args:
    graph: A `graphs.GraphsTuple` instance containing numpy arrays.

  Returns:
    A list of the graph data dictionaries. The GLOBALS field is a tensor of
      rank at least 1, as the RECEIVERS and SENDERS field (which have integer
      values). The NODES and EDGES fields have rank at least 2.
  """
  offset = _compute_stacked_offsets(graph.n_node, graph.n_edge)

  nodes_splits = np.cumsum(graph.n_node[:-1])
  edges_splits = np.cumsum(graph.n_edge[:-1])
  graph_of_lists = collections.defaultdict(lambda: [])
  if graph.nodes is not None:
    graph_of_lists[NODES] = np.split(graph.nodes, nodes_splits)
  if graph.edges is not None:
    graph_of_lists[EDGES] = np.split(graph.edges, edges_splits)
  if graph.receivers is not None:
    graph_of_lists[RECEIVERS] = np.split(graph.receivers - offset, edges_splits)
    graph_of_lists[SENDERS] = np.split(graph.senders - offset, edges_splits)
  if graph.globals is not None:
    graph_of_lists[GLOBALS] = _unstack(graph.globals)

  n_graphs = graph.n_node.shape[0]
  # Make all fields the same length.
  for k in GRAPH_DATA_FIELDS:
    graph_of_lists[k] += [None] * (n_graphs - len(graph_of_lists[k]))
  graph_of_lists[N_NODE] = graph.n_node
  graph_of_lists[N_EDGE] = graph.n_edge

  result = []
  for index in range(n_graphs):
    result.append({field: graph_of_lists[field][index] for field in ALL_FIELDS})
  return result


def _to_compatible_data_dicts(data_dicts):
  """Converts the content of `data_dicts` to arrays of the right type.

  All fields are converted to numpy arrays. The index fields (`SENDERS` and
  `RECEIVERS`) and number fields (`N_NODE`, `N_EDGE`) are cast to `np.int32`.

  Args:
    data_dicts: An iterable of dictionaries with keys `ALL_KEYS` and values
      either `None`s, or quantities that can be converted to numpy arrays.

  Returns:
    A list of dictionaries containing numpy arrays or `None`s.
  """
  results = []
  for data_dict in data_dicts:
    result = {}
    for k, v in data_dict.items():
      if v is None:
        result[k] = None
      else:
        dtype = np.int32 if k in [SENDERS, RECEIVERS, N_NODE, N_EDGE] else None
        result[k] = np.asarray(v, dtype)
    results.append(result)
  return results


def _populate_number_fields(data_dict):
  """Returns a dict with the number fields N_NODE, N_EDGE filled in.

  The N_NODE field is filled if the graph contains a non-None NODES field;
  otherwise, it is set to 0.
  The N_EDGE field is filled if the graph contains a non-None RECEIVERS field;
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
        dct[number_field] = np.array(
            np.shape(dct[data_field])[0], dtype=np.int32)
      else:
        dct[number_field] = np.array(0, dtype=np.int32)
  return dct


def _concatenate_data_dicts(data_dicts):
  """Concatenate a list of data dicts to create the equivalent batched graph.

  Args:
    data_dicts: An iterable of data dictionaries with keys `GRAPH_DATA_FIELDS`,
      plus, potentially, a subset of `GRAPH_NUMBER_FIELDS`. Each dictionary is
      representing a single graph.

  Returns:
    A data dictionary with the keys `GRAPH_DATA_FIELDS + GRAPH_NUMBER_FIELDS`,
    representing the concatenated graphs.
  """
  # Create a single dict with fields that contain sequences of graph tensors.
  concatenated_dicts = collections.defaultdict(lambda: [])
  for data_dict in data_dicts:
    data_dict = _populate_number_fields(data_dict)
    for k, v in data_dict.items():
      if v is not None:
        concatenated_dicts[k].append(v)
      else:
        concatenated_dicts[k] = None

  concatenated_dicts = dict(concatenated_dicts)

  for field, arrays in concatenated_dicts.items():
    if arrays is None:
      concatenated_dicts[field] = None
    elif field in list(GRAPH_NUMBER_FIELDS) + [GLOBALS]:
      concatenated_dicts[field] = np.stack(arrays)
    else:
      concatenated_dicts[field] = np.concatenate(arrays, axis=0)

  if concatenated_dicts[RECEIVERS] is not None:
    offset = _compute_stacked_offsets(concatenated_dicts[N_NODE],
                                      concatenated_dicts[N_EDGE])
    for field in (RECEIVERS, SENDERS):
      concatenated_dicts[field] += offset

  return concatenated_dicts


def get_graph(input_graphs, index):
  """Indexes into a graph.

  Given a `graphs.GraphsTuple` containing arrays and an index (either
  an `int` or a `slice`), index into the nodes, edges and globals to extract the
  graphs specified by the slice, and returns them into an another instance of a
  `graphs.GraphsTuple` containing `Tensor`s.

  Args:
    input_graphs: A `graphs.GraphsTuple` containing numpy arrays.
    index: An `int` or a `slice`, to index into `graph`. `index` should be
      compatible with the number of graphs in `graphs`.

  Returns:
    A `graphs.GraphsTuple` containing numpy arrays, made of the extracted
      graph(s).

  Raises:
    TypeError: if `index` is not an `int` or a `slice`.
  """
  if isinstance(index, int):
    graph_slice = slice(index, index + 1)
  elif isinstance(index, slice):
    graph_slice = index
  else:
    raise TypeError("unsupported type: %s" % type(index))
  data_dicts = graphs_tuple_to_data_dicts(input_graphs)[graph_slice]
  return graphs.GraphsTuple(**_concatenate_data_dicts(data_dicts))
