# -*- coding: utf-8 -*-


import math
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import init_ops

import config
from utils import load_feature_config, get_vocab_dict
from base.layers.core import DeepNet


class BaseCTRModel(tf.keras.models.Model):
    def __init__(self):
        super().__init__()
        self.costs = None
        self.global_step = None

        self.labels, self.cat_features, self.con_features, self.bool_features = load_feature_config()
        self.vocab_dict = get_vocab_dict(config.training_feature_info_file, self.cat_features)
        self.input_layer = self.build_input_layer()
        self.predict_layer = self.build_predictions()

    def BN(self, fv):
        with tf.name_scope('BN'):
            fv = tf.keras.layers.BatchNormalization(fused=True,
                                                    renorm=self.flags.renorm)(fv, training=self.is_training, )
            return fv

    def concat(self, inputs):
        return tf.concat(inputs, -1)

    def build_deep(self, hidden=None, activation=tf.nn.relu):
        return DeepNet(hidden, activation, sparse=False, droprate=1 - self.flags.keep_prob, flags=self.flags)

    def build_dense_layer(self, fv):
        if self.flags.use_bn:
            return self.BN(tf.math.log1p(fv))
        else:
            return tf.math.log1p(fv)

    def build_predictions(self):
        prediction = tf.keras.layers.Dense(
            1, activation=tf.nn.sigmoid)
        return prediction

    def build_network(self, features, is_training=None):
        """
        must defined in subclass
        """
        raise NotImplementedError("build_network: not implemented!")

    def call(self, inputs, is_training=None, mask=None):
        features = self.build_features(inputs)
        logit = self.build_network(features, is_training)
        preds = self.predict_layer(logit)
        return preds

    def build_input_layer(self):
        feature_columns = []

        for feature in self.con_features:
            feature_columns.append(tf.feature_column.numeric_column(feature))

        for feature in self.bool_features:
            feature_columns.append(tf.feature_column.numeric_column(feature))

        for feature in self.cat_features:
            cat_values_list = [int(x) if x else -1 for x in self.vocab_dict[feature]]
            cat_value_to_index_layer = tf.feature_column.categorical_column_with_vocabulary_list(feature,
                                                                                                 vocabulary_list=cat_values_list,
                                                                                                 num_oov_buckets=1)
            embedding_dim = len(cat_values_list)
            if embedding_dim > 10:
                embedding_dim = math.ceil(math.log2(embedding_dim) * 8)

            # 构造嵌入层
            glorot = np.sqrt(2.0 / (len(cat_values_list) + embedding_dim))
            # 默认`truncated_normal_initializer`with mean `0.0` and standard deviation `1 / sqrt(dimension)`
            cat_embedding_layer = tf.feature_column.embedding_column(categorical_column=cat_value_to_index_layer,
                                                                     dimension=embedding_dim, combiner='sqrtn',
                                                                     initializer=init_ops.truncated_normal_initializer(
                                                                         mean=0.0, stddev=glorot))
            feature_columns.append(cat_embedding_layer)

        input_layer = tf.keras.layers.DenseFeatures(feature_columns)
        return input_layer

    def build_wide_layer(self):
        feature_columns = []

        for feature in self.con_features:
            feature_columns.append(tf.feature_column.numeric_column(feature))

        for feature in self.bool_features:
            feature_columns.append(tf.feature_column.numeric_column(feature))

        for feature in self.cat_features:
            cat_values_list = [int(x) for x in self.vocab_dict[feature]]
            cat_value_to_index_layer = tf.feature_column.categorical_column_with_vocabulary_list(feature,
                                                                                                 vocabulary_list=cat_values_list,
                                                                                                 num_oov_buckets=1)
            embedding_dim = len(cat_values_list)
            if embedding_dim > 10:
                embedding_dim = math.ceil(math.log2(embedding_dim) * 8)

            # 构造嵌入层
            glorot = np.sqrt(2.0 / (len(cat_values_list) + embedding_dim))
            # 默认`truncated_normal_initializer`with mean `0.0` and standard deviation `1 / sqrt(dimension)`
            cat_embedding_layer = tf.feature_column.embedding_column(categorical_column=cat_value_to_index_layer,
                                                                     dimension=embedding_dim, combiner='sqrtn',
                                                                     initializer=init_ops.truncated_normal_initializer(
                                                                         mean=0.0, stddev=glorot))
            feature_columns.append(cat_embedding_layer)

        input_layer = tf.keras.layers.DenseFeatures(feature_columns)
        return input_layer

    def build_features(self, inputs):
        features = self.input_layer(inputs)
        return features
