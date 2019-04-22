# -*- coding: utf-8 -*-
"""Graph dataset generator and loader

The `GraphDataset` stores the information about how to load and generate
datasets. To create a dataset, you need to specify the features that the
dataset will load. Here 'features' means an instance of a subclass of the
`MyFeature` class. As an example:

```
class ShortestPathGraphDataset(data_util.GraphDataset):
  def __init__(self, data_dir, min_nodes, max_nodes):
    feature_list = [
      data_util.GraphFeature(
          key='input_graph',
          node_feature_size=5,
          edge_feature_size=1,
          global_feature_size=1,
          dtype='float32',
          description='Graph to input to network'),
      data_util.GraphFeature(
          key='target_graph',
          node_feature_size=2,
          edge_feature_size=2,
          global_feature_size=1,
          dtype='float32',
          description='Graph to output from network'),
      # Example of a non-graph feature
      data_util.TensorFeature(
          key='adj_mat_dense',
          shape=[max_nodes, max_nodes],
          dtype='float32',
          description='Sparse adjacency matrix of input graph'),
    ]
    super(ShortestPathGraphDataset, self).__init__(data_dir, feature_list)
    self.min_nodes = min_nodes
    self.max_nodes = max_nodes
  def gen_sample(self, name):
    # ... Compute graphs and labels on graphs
    return {
        'input_graph': input_graph_dict,
        'target_graph': target_graph_dict,
        'adj_mat_dense': adj_mat_dense,
    }
```

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import datetime
import abc

import tqdm
import numpy as np
import tensorflow as tf

from graph_nets import graphs
from graph_nets import utils_tf

# Debug printing
import pprint
pp_xfawedfssa = pprint.PrettyPrinter(indent=2)
def myprint(x):
  if type(x) == str:
    print(x)
  else:
    pp_xfawedfssa.pprint(x)

def np_dense_to_sparse(arr):
  """Takes in a np.ndarray and returns a sparisification of it"""
  idx = np.where(arr != 0.0)
  return idx, arr[idx]


class MyFeature(object):
  """Class for decoding a serialized values in tfrecords or npz files.

  Base class that all other features should subclass. Handles writing out to
  tfrecords, as well as reading from them, and handling of placeholders and
  feed_dicts as well. Could be a single simple thing (e.g. a fixed size Tensor)
  or something more complicated (e.g. a GraphTuple)
  """

  def __init__(self, key, description, shape=None, dtype='float32'):
    """Initialization of MyFeature, giving specification of feature.

    Args:
      key: string acting as name and identifier for this feature
      description: string describing what this feature (for documentation)
      shape: list/tuple of int values describing shape of this feature, if
        applicable. Default= None
      dtype: string for tf.dtype of this feature
    """
    super(MyFeature, self).__init__()
    self.key = key
    self.description = description
    self.shape = shape if shape is not None else []
    self.dtype = dtype

  def get_feature_write(self, value):
    """Returns dictionary of things to write to a tfrecord.

    This creates a dictionary of items to write to the tfrecord. Typically to
    write to a tfrecord it needs to be broken down into basic Tensors (either
    fixed or variable length), and thus if the feature is more complicated this
    function breaks it down into its basic writable components, e.g.  indices
    and values of a sparse tensor.

    Args:
      value: value (usually np.ndarray, but could be anything) to write out to
        the tfrecord

    Returns:
      feat_write: Dictionary of key strings to np.ndarrays/lists to write to
        tfrecord
    """
    return {self.key: value}

  def get_feature_read(self):
    """Returns dictionary of things to read from a tfrecord.

    This creates a dictionary of items to read for the tfrecord parser. The
    keys of the dictionary should be used in `tensors_to_item` to combine into
    the usable form of the feature, e.g. indices and values of a sparse
    tensor.

    Returns:
      feat_read: Dictionary of key strings to values to read from a tfrecord
        (e.g. dictionary of tf.FixedLenFeature, tf.VarLenFeature, etc.)
    """
    return {self.key: tf.FixedLenFeature([], self.dtype)}

  def tensors_to_item(self, keys_to_tensors):
    """Collects relevant items together to create feature.

    This is for the case where there needs to be some post-processing of the
    features or combining of several sub-features to make the feature readable,
    e.g. getting indices and values for a sparse matrix feature. Final
    processing will be done in stack for the batching operation.

    Args:
      keys_to_tensors: dictionary of values loaded from the

    Returns:
      item: Combined values to create the final feature
    """
    item = keys_to_tensors[self.key]
    return item

  def stack(self, arr):
    """Stacks a list of parsed features for batching.

    This is called after loading features and calling tensors_to_item. It takes
    the values and concatenates them in an appropriate way so that the network
    can work with minibatches.

    Args:
      arr: list of parsed features to stack together

    Returns:
      item: item of values stacked together appropriately
    """
    return tf.stack(arr)

  # Placeholder related stuff
  def get_placeholder_and_feature(self, batch=True):
    """Gets dictionary of placeholders and computed value for this feature.

    In the case you are not using tfrecords, this can be used to create the
    appropriate placeholders for this feature. If this feature combines several
    basic components (e.g. a sparse tensor with indices and values) then it
    combines them together into a single value for this feature. Also handles
    batching of values within the placeholder.

    Args:
      batch: (bool, default=True) Whether to batch the output

    Returns:
      placeholder: Dictionary of key strings to placeholders for this feature
    """
    if batch:
      placeholder = tf.placeholder(self.dtype, shape=[None] + self.shape)
    else:
      placeholder = tf.placeholder(self.dtype, shape=self.shape)
    return {self.key: placeholder}, placeholder

  def get_feed_dict(self, placeholders, values, batch=True):
    """Get the `feed_dict` for this feature, mapping placeholders to values.

    This creates the `feed_dict` by mapping the appropriate placeholders to the
    values provided. Also handles batching of values within the placeholder.

    Args:
      placeholders: Dictionary of key strings to placeholders
      values: Values (typically np.ndarrays or lists) needed to build this
        feature
      batch: (bool, default=True) Whether to batch the output

    Returns:
      feed_dict: Dictionary of placeholders to values for this feature
    """
    if batch:
      fdict = np.expand_dims(values, 0)  # Add batch dimension
    else:
      fdict = values
    return {placeholders[self.key]: fdict}

  def npz_value(self, value):
    return {self.key: value}


class IntFeature(MyFeature):
  """Class used for decoding a single serialized int64 value.

  This class is to store a single integer value e.g. the lengths of an array.
  """

  def __init__(self, key, description, dtype='int64'):
    """Initialization of IntFeature, giving specification of feature.

    Args:
      key: string acting as name and identifier for this feature
      description: string describing what this feature (for documentation)
      dtype: string for tf.dtype of this feature (either 'int64' or 'int32')
        Default= 'int64'
    """
    super(IntFeature, self).__init__(key,
                                       description,
                                       shape=[],
                                       dtype='int64')
    assert(dtype in ['int64', 'int32'])
    self.convert_to = dtype

  def get_feature_write(self, value):
    feat = tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    return {self.key: feat}

  def tensors_to_item(self, keys_to_tensors):
    tensor = keys_to_tensors[self.key]
    if self.convert_to != 'int64':
      return tf.cast(tensor, dtype=self.convert_to)
    else:
      return tf.cast(tensor, dtype=tf.int64)

  def get_placeholder_and_feature(self, batch=True):
    if batch:
      placeholder = tf.placeholder(tf.int64, shape=[None])
    else:
      placeholder = tf.placeholder(tf.int64)
    if self.convert_to != 'int64':
      sample = tf.cast(placeholder, dtype=self.convert_to)
    else:
      sample = placeholder
    return {self.key: placeholder}, sample


class TensorFeature(MyFeature):
  """Class used for decoding tensors of fixed size."""

  def __init__(self, key, shape, dtype, description):
    """Initialization of TensorFeature, giving specification of feature.

    Args:
      key: string acting as name and identifier for this feature
      shape: list/tuple of int values describing shape of this feature.
      dtype: string for tf.dtype of this feature
      description: string describing what this feature (for documentation)
    """
    super(TensorFeature, self).__init__(key,
                                        description,
                                        shape=shape,
                                        dtype=dtype)

  def get_feature_write(self, value):
    v = value.astype(self.dtype).tobytes()
    feat = tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))
    return {self.key: feat}

  def get_feature_read(self):
    return {self.key: tf.FixedLenFeature([], tf.string)}

  def tensors_to_item(self, keys_to_tensors):
    tensor = keys_to_tensors[self.key]
    tensor = tf.decode_raw(tensor, out_type=self.dtype)
    tensor = tf.reshape(tensor, self.shape)
    sess = tf.InteractiveSession()
    return tensor


class VarLenIntListFeature(MyFeature):
  """Class used for decoding variable length int64 lists."""

  def __init__(self, key, dtype, description):
    """Initialization of VarLenIntListFeature, giving specification of feature.

    Args:
      key: string acting as name and identifier for this feature
      dtype: string for tf.dtype of this feature
      description: string describing what this feature (for documentation)
    """
    super(VarLenIntListFeature, self).__init__(key,
                                               description,
                                               shape=[None],
                                               dtype=dtype)

  def get_feature_write(self, value):
    """Input `value` should be a list of integers."""
    feat = tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    return {self.key: feat}

  def get_feature_read(self):
    return {self.key: tf.VarLenFeature(tf.int64)}

  def tensors_to_item(self, keys_to_tensors):
    tensor = keys_to_tensors[self.key]
    tensor = tf.sparse_tensor_to_dense(tensor)
    return tf.cast(tensor, self.dtype)

  def get_placeholder_and_feature(self, batch=True):
    placeholder = tf.placeholder(self.dtype, shape=self.shape)
    return {self.key: placeholder}, placeholder

class VarLenFloatFeature(MyFeature):
  """Class used for decoding variable shaped float tensors."""

  def __init__(self, key, shape, description):
    """Initialization of VarLenIntListFeature, giving specification of feature.

    Args:
      key: string acting as name and identifier for this feature
      shape: list/tuple of int values describing shape of this feature.
      description: string describing what this feature (for documentation)
    """
    super(VarLenFloatFeature, self).__init__(key,
                                             description,
                                             shape=shape,
                                             dtype='float32')

  def get_feature_write(self, value):
    """Input `value` has to be compatible with this instance's shape."""
    if isinstance(value, np.ndarray):
      err_msg = "VarLenFloatFeature shape incompatible with input shape"
      if len(value.shape) == len(self.shape):
        for i, sz in enumerate(value.shape):
          if self.shape[i] is not None:
            assert sz == self.shape[i], err_msg
      elif len(value.shape) == 1 and \
            len(self.shape) == 2 and \
            self.shape[0] is None:
        assert value.shape[0] == self.shape[1], err_msg
      else:
        assert False, err_msg
      flist = tf.train.FloatList(value=value.reshape(-1))
    else:
      flist = tf.train.FloatList(value=value)
    return {self.key: tf.train.Feature(float_list=flist)}

  def get_feature_read(self):
    return {self.key: tf.VarLenFeature(tf.float32)}

  def tensors_to_item(self, keys_to_tensors):
    tensor = keys_to_tensors[self.key]
    tensor = tf.sparse_tensor_to_dense(tensor)
    shape = [s if s is not None else -1 for s in self.shape]
    tensor = tf.reshape(tensor, shape)
    return tensor

  def get_placeholder_and_feature(self, batch=True):
    placeholder = tf.placeholder(self.dtype, shape=self.shape)
    return {self.key: placeholder}, placeholder


class SparseTensorFeature(MyFeature):
  """Class used for decoding serialized sparse float tensors."""

  def __init__(self, key, shape, description):
    """Initialization of SparseTensorFeature, giving specification of feature.

    Args:
      key: string acting as name and identifier for this feature
      shape: list/tuple of int values describing shape of this feature.
      description: string describing what this feature (for documentation)
    """
    super(SparseTensorFeature, self).__init__(key,
                                              description,
                                              shape=shape,
                                              dtype='float32')

  # TODO: Make these change into concatenating for 1 index tensor
  def get_feature_write(self, value):
    """Value should be a tuple `(idx, vals)`.

    Value should be a tuple `(idx, vals)` with `idx` being a tuple of lists of
    `int` values of the same length and `vals` is a list of `self.dtype` values
    the same length as `idx[0]`.
    """
    idx, vals = value[0], value[1]
    sptensor_feature = {
        '{}_{:02d}'.format(self.key, i):
        tf.train.Feature(int64_list=tf.train.Int64List(value=idx[i]))
        for i in range(len(self.shape))
    }
    sptensor_feature['{}_value'.format(self.key)] = \
      tf.train.Feature(float_list=tf.train.FloatList(value=vals))
    return sptensor_feature

  def get_feature_read(self):
    feat_read = {
        '{}_{:02d}'.format(self.key, i): tf.VarLenFeature(tf.int64)
        for i in range(len(self.shape))
    }
    feat_read['{}_value'.format(self.key)] = tf.VarLenFeature(self.dtype)
    return feat_read

  def tensors_to_item(self, keys_to_tensors):
    indices_sp = [
        keys_to_tensors['{}_{:02d}'.format(self.key, i)]
        for i in range(len(self.shape))
    ]
    indices_list = [tf.sparse_tensor_to_dense(inds) for inds in indices_sp]
    indices = tf.stack(indices_list, -1)
    values_sp = keys_to_tensors['{}_value'.format(self.key)]
    values = tf.sparse_tensor_to_dense(values_sp)
    tensor = tf.SparseTensor(indices, values, self.shape)
    return tensor

  def stack(self, arr):
    concat_arr = [tf.sparse_reshape(x, [1] + self.shape) for x in arr]
    return tf.sparse_concat(0, concat_arr)

  # Placeholder related
  def get_placeholder_and_feature(self, batch):
    placeholder = tf.sparse_placeholder(self.dtype)
    return {self.key: placeholder}, placeholder

  def get_feed_dict(self, placeholders, values, batch):
    idxs, vals = values[self.key + '_idx'], values[self.key + '_val']
    if batch:
      idxs = np.concatenate((np.zeros((len(idxs), 1)), idxs), -1)
    val = tf.SparseTensorValue(idxs, vals, [1] + self.shape)
    return {placeholders[self.key]: val}

  def npz_value(self, value):
    idx_, val = value[0], value[1]
    idx = np.stack(idx_, -1)
    return {self.key + '_idx': idx, self.key + '_val': val}


class GraphFeature(MyFeature):
  """Custom class used for decoding serialized GraphsTuples."""

  def __init__(self, key, node_feature_size, edge_feature_size,
               global_feature_size, dtype, description):
    super(GraphFeature, self).__init__(key,
                                       description,
                                       shape=[],
                                       dtype=dtype)
    features_list = [
        IntFeature(
            key='n_node',
            dtype='int32',
            description='Number of nodes we are using'),
        VarLenFloatFeature(
            key='nodes',
            shape=[None, node_feature_size],
            description='Initial node embeddings for optimization'),
        IntFeature(
            key='n_edge',
            dtype='int32',
            description='Number of edges in this graph'),
        VarLenFloatFeature(
            key='globals',
            shape=[None, global_feature_size],
            description='Edge features'),
        VarLenFloatFeature(
            key='edges',
            shape=[None, edge_feature_size],
            description='Edge features'),
        VarLenIntListFeature(
            key='receivers',
            dtype='int32',
            description='Recieving nodes for edges'),
        VarLenIntListFeature(
            key='senders',
            dtype='int32',
            description='Sending nodes for edges'),
    ]
    self.features = {}
    for feat in features_list:
      key = feat.key
      self.features[key] = feat
      self.features[key].key = '{}_{}'.format(self.key, key)
    self.node_feature_size = node_feature_size
    self.edge_feature_size = edge_feature_size
    self.global_feature_size = edge_feature_size

  def get_feature_write(self, value):
    """Input `value` should be a dictionary for a `graph_net.GraphsTuple`

    Input `value` should be a dictionary generated by one of the methods in
    `graph_net.util_tf`, a data dictionary for the graph
    """
    feat_write = {}
    for key, feat in self.features.items():
      feat_write.update(feat.get_feature_write(value[key]))
    return feat_write

  def get_feature_read(self):
    feat_read = {}
    for _, feat in self.features.items():
      feat_read.update(feat.get_feature_read())
    return feat_read

  def tensors_to_item(self, keys_to_tensors):
    graph_dict = {}
    for key, feat in self.features.items():
      graph_dict[key] = feat.tensors_to_item(keys_to_tensors)
    # return graphs.GraphsTuple(**graph_dict)
    return graph_dict

  def stack(self, arr):
    return utils_tf.data_dicts_to_graphs_tuple(arr)

  # Placeholder related
  def get_placeholder_and_feature(self, batch):
    placeholders = {}
    sample_dict = {}
    for key, feat in self.features.items():
      ph, val = feat.get_placeholder_and_feature(batch=batch)
      placeholders.update(ph)
      sample_dict[key] = val
    print("-----------------------------------------------------------")
    sample = graphs.GraphsTuple(**sample_dict)
    myprint(sample_dict)
    print(sample)
    print("-----------------------------------------------------------")
    return placeholders, sample

  def get_feed_dict(self, placeholders, values, batch):
    fdict = {}
    for _, feat in self.features.items():
      # Due to how graphs are concatenated we never need a batch dimension
      fdict.update(feat.get_feed_dict(placeholders, values, False))
    return fdict

  def npz_value(self, values):
    graph_dict = {}
    for key, feat in self.features.items():
      graph_dict.update(feat.npz_value(values[key]))
    return graph_dict


# TODO: Make one for storing images in compressed jpg or png format


GRAPH_KEYS = [
    'n_node', 'nodes', 'n_edge', 'edges', 'receivers', 'senders', 'globals'
]


class GraphDataset(abc.ABC):
  """Graph dataset generator and loader

  The `GraphDataset` stores the information about how to load and generate
  datasets.
  """
  MAX_IDX = 700

  def __init__(self, data_dir, feature_list):
    """Initializes GraphDataset

    Args:
      data_dir: string giving the path to where the data is/will be stored
      feature_list: list of objects that are instances of subclesses of
        MyFeature, defining the objects in each element of the dataset
    """
    self.data_dir = data_dir
    self.sizes = {}
    self.features = {v.key: v for v in feature_list}

  def get_placeholders(self, batch=True):
    """Gets appropriate dictionary of placeholders all features of this dataset.

    In the case you are not using tfrecords, this function builds all the
    appropriate placeholders and returns them in a dictionary. It also handles
    batching of values within the placeholder.

    Args:
      batch: (bool, default=True) Whether to batch the output

    Returns:
      placeholders: Dictionary of key strings to placeholders for this dataset
    """
    # Build placeholders
    placeholders = {}
    sample = {}
    for key, feat in self.features.items():
      ph, val = feat.get_placeholder_and_feature(batch=batch)
      placeholders.update(ph)
      sample[key] = val
    # Other placeholders
    return sample, placeholders

  def get_feed_dict(self, placeholders, value_dict):
    """Get the `feed_dict` for this dataset, mapping placeholders to values.

    This creates the `feed_dict` by mapping the appropriate placeholders to the
    values provided in value_dict. Also handles batching of values within the
    placeholders.

    Args:
      placeholders: Dictionary of key strings to placeholders
      value_dict: Dictionary of key strings values (typically np.ndarrays or
        lists) needed to build this feature
      batch: (bool, default=True) Whether to batch the output

    Returns:
      feed_dict: Dictionary of placeholders to values for this feature
    """
    feed_dict = {}
    for _, value in self.features.items():
      feed_dict.update(value.get_feed_dict(placeholders, value_dict))
    return feed_dict

  @abc.abstractmethod
  def gen_sample(self, name, index):
    """Generate a sample for this dataset.

    This can either generate synthetically example or load appropriate data
    (e.g. images) to store into a tfrecord for fast loading.

    Args:
      name: name of the mode we are in (e.g. 'train', 'test')
      index: number identifer for this particular sample

    Returns:
      features: Dictionary of key strings to values of this sample
    """
    return {name: index}

  def process_features(self, loaded_features):
    """Prepare features for storing into a tfrecord.

    This can either generate synthetically example or load appropriate data
    (e.g. images) to store into a tfrecord for fast loading.

    Args:
      name: name of the mode we are in (e.g. 'train', 'test')

    Returns:
      features: Dictionary of key strings to values of this sample
    """
    features = {}
    for key, feat in self.features.items():
      features.update(feat.get_feature_write(loaded_features[key]))
    return features

  ######### Generic methods ##########
  # Hopefully after this point you won't have to subclass any of these
  def get_parser_op(self):
    """Returns function that parses a tfrecord Example.

    This can be with a `tf.data` dataset for parsing a tfrecord.

    Returns:
      parser_op: function taking in a record and retruning a parsed dictionary
    """
    keys_to_features = {}
    for _, value in self.features.items():
      keys_to_features.update(value.get_feature_read())

    def parser_op(record):
      example = tf.parse_single_example(record, keys_to_features)
      return {
          k: v.tensors_to_item(example)
          for k, v in self.features.items()
      }

    return parser_op

  def load_batch(self, name, batch_size, shuffle_data=True, repeat=None):
    """Return batch loaded from this dataset from the tfrecords of mode `name`

    This is the primary function used in training.

    Args:
      name: name of the mode we are in (e.g. 'train', 'test')
      batch_size: size (>= 1) of the batch to load
      shuffle_data: (boolean, Default= True) Whether to shuffle data or not
      repeat: Number of times to repeat the dataset, `None` if looping forever.
        Default= `None`

    Returns:
      features: Dictionary of key strings to values of this sample
    """
    data_sources = glob.glob(
        os.path.join(self.data_dir, name, '*.tfrecords'))
    if shuffle_data:
      np.random.shuffle(data_sources)  # Added to help the shuffle
    # Build dataset provider
    dataset = tf.data.TFRecordDataset(data_sources)
    dataset = dataset.map(self.get_parser_op())
    dataset = dataset.repeat(repeat)
    if shuffle_data:
      dataset = dataset.shuffle(buffer_size=5 * batch_size)
      # dataset = dataset.prefetch(buffer_size=batch_size)

    iterator = dataset.make_one_shot_iterator()
    batch = []
    for _ in range(batch_size):
      batch.append(iterator.get_next())

    # Constructing output sample using known order of the keys
    sample = {}
    for key, value in self.features.items():
      sample[key] = value.stack([batch[b][key] for b in range(batch_size)])
    return sample

  # TODO: Make hooks to make this more general
  def convert_dataset(self, name, num_entries):
    """Writes out tfrecords using `gen_sample` for mode `name`

    This is the primary function for generating the dataset. It saves out the
    tfrecords in `os.path.join(self.data_dir, name)`. Displays a progress
    bar to show progress. Be careful with memory usage on disk.

    Args:
      name: name of the mode we are in (e.g. 'train', 'test')
      num_entries: number of entries (>= 1) to generate
    """
    self.sizes[name] = num_entries
    fname = '{:03d}.tfrecords'
    outfile = lambda idx: os.path.join(self.data_dir, name,
                                       fname.format(idx))
    if not os.path.isdir(os.path.join(self.data_dir, name)):
      os.makedirs(os.path.join(self.data_dir, name))

    print('Writing dataset to {}/{}'.format(self.data_dir, name))
    writer = None
    record_idx = 0
    file_idx = self.MAX_IDX + 1
    for index in tqdm.tqdm(range(self.sizes[name])):
      if file_idx > self.MAX_IDX:
        file_idx = 0
        if writer:
          writer.close()
        writer = tf.python_io.TFRecordWriter(outfile(record_idx))
        record_idx += 1
      loaded_features = self.gen_sample(name, index)
      features = self.process_features(loaded_features)
      example = tf.train.Example(features=tf.train.Features(feature=features))
      writer.write(example.SerializeToString())
      file_idx += 1

    if writer:
      writer.close()
    # And save out a file with the creation time for versioning
    timestamp_file = '{}_timestamp.txt'.format(name)
    timestamp_str = 'TFrecord created {}'.format(datetime.datetime.now())
    with open(os.path.join(self.data_dir, timestamp_file), 'w') as date_file:
      date_file.write(timestamp_str)

  def create_np_dataset(self, name, num_entries):
    """Writes out `npz` files using `gen_sample` for mode `name`

    This function generates the dataset in numpy form. This is in case you need
    to use placeholders. Displays a progress bar to show progress. Be careful
    with memory usage on disk.

    Args:
      name: name of the mode we are in (e.g. 'train', 'test')
      num_entries: number of entries (>= 1) to generate
    """
    self.sizes[name] = num_entries
    fname = '{:04d}.npz'
    outfile = lambda idx: os.path.join(self.data_dir, name,
                                       fname.format(idx))
    if not os.path.isdir(os.path.join(self.data_dir, name)):
      os.makedirs(os.path.join(self.data_dir, name))
    print('Writing dataset to {}'.format(os.path.join(self.data_dir,
                                                      name)))
    for index in tqdm.tqdm(range(num_entries)):
      features = self.gen_sample(name, index)
      npz_dict = {}
      for key, feat in self.features.items():
        npz_dict.update(feat.npz_value(features[key]))
      np.savez(outfile(index), **npz_dict)

    # And save out a file with the creation time for versioning
    timestamp_file = 'np_test_timestamp.txt'
    with open(os.path.join(self.data_dir, timestamp_file), 'w') as date_file:
      contents = 'Numpy Dataset created {}'.format(datetime.datetime.now())
      date_file.write(contents)

  def load_npz_file(self, name, index):
    """
    """
    fname = os.path.join(mydataset.data_dir, name, '{:04d}.npz')
    with open(fname.format(iteration), 'r') as npz_file:
      npz_dict = dict(np.load(npz_file))
    return npz_dict



