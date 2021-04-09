# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from base_trainable import BaseTrainable
from base_model import BaseCTRModel

from base.layers.core import Linear
import config


class ESMM(BaseTrainable, BaseCTRModel):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        share_hidden = config.share_deep_layers
        hidden = config.deep_layers
        self.deep_share_block = self.build_deep(hidden=share_hidden)

        self.deep_ctr_block0 = self.build_deep(hidden=hidden)
        self.deep_cvr_block0 = self.build_deep(hidden=hidden)

        self.deep_ctr_block1 = self.build_deep(hidden=hidden)
        self.deep_cvr_block1 = self.build_deep(hidden=hidden)

        self.deep_ctr_block2 = self.build_deep(hidden=hidden)
        self.deep_cvr_block2 = self.build_deep(hidden=hidden)

        self.deep_ctr_block3 = self.build_deep(hidden=hidden)
        self.deep_cvr_block3 = self.build_deep(hidden=hidden)

        self.deep_ctr_block4 = self.build_deep(hidden=hidden)
        self.deep_cvr_block4 = self.build_deep(hidden=hidden)

    def build_loss(self):
        return {"ctr": "binary_crossentropy", "cvr": "binary_crossentropy"}

    def build_loss_weight(self):
        return {"ctr": 1.0, "cvr": 1.0}

    def build_networks(self, features, is_training=None):
        share_output = self.deep_share_block(features, is_training)

        ctr_logit0 = self.deep_ctr_block0(share_output, is_training)
        cvr_logit0 = self.deep_cvr_block0(share_output, is_training)

        ctr_logit1 = self.deep_ctr_block1(share_output, is_training)
        cvr_logit1 = self.deep_cvr_block1(share_output, is_training)

        ctr_logit2 = self.deep_ctr_block2(share_output, is_training)
        cvr_logit2 = self.deep_cvr_block2(share_output, is_training)

        ctr_logit3 = self.deep_ctr_block3(share_output, is_training)
        cvr_logit3 = self.deep_cvr_block3(share_output, is_training)

        ctr_logit4 = self.deep_ctr_block4(share_output, is_training)
        cvr_logit4 = self.deep_cvr_block4(share_output, is_training)

        return ctr_logit0, cvr_logit0, ctr_logit1, cvr_logit1, ctr_logit2, cvr_logit2, ctr_logit3, cvr_logit3, ctr_logit4, cvr_logit4

    def call(self, inputs, is_training=None):
        features, step_level = self.build_features(inputs)
        ctr_logit0, cvr_logit0, ctr_logit1, cvr_logit1, ctr_logit2, cvr_logit2, ctr_logit3, cvr_logit3, ctr_logit4, cvr_logit4 = self.build_networks(features, is_training)

        self.ctr_pred0 = self.predict_layer(ctr_logit0)
        self.cvr_pred0 = self.predict_layer(cvr_logit0)
        self.ctr_pred0 =  tf.identity(self.ctr_pred0, name="ctr_label0")
        self.ctcvr_pred0 = tf.multiply(self.ctr_pred0, self.cvr_pred0, name="cvr_label0")

        self.ctr_pred1 = self.predict_layer(ctr_logit1)
        self.cvr_pred1 = self.predict_layer(cvr_logit1)
        self.ctr_pred1 =  tf.identity(self.ctr_pred1, name="ctr_label1")
        self.ctcvr_pred1 = tf.multiply(self.ctr_pred1, self.cvr_pred1, name="cvr_label1")

        self.ctr_pred2 = self.predict_layer(ctr_logit2)
        self.cvr_pred2 = self.predict_layer(cvr_logit2)
        self.ctr_pred2 =  tf.identity(self.ctr_pred2, name="ctr_label2")
        self.ctcvr_pred2 = tf.multiply(self.ctr_pred2, self.cvr_pred2, name="cvr_label2")

        self.ctr_pred3 = self.predict_layer(ctr_logit3)
        self.cvr_pred3 = self.predict_layer(cvr_logit3)
        self.ctr_pred3 =  tf.identity(self.ctr_pred3, name="ctr_label3")
        self.ctcvr_pred3 = tf.multiply(self.ctr_pred3, self.cvr_pred3, name="cvr_label3")

        self.ctr_pred4 = self.predict_layer(ctr_logit4)
        self.cvr_pred4 = self.predict_layer(cvr_logit4)
        self.ctr_pred4 =  tf.identity(self.ctr_pred4, name="ctr_label4")
        self.ctcvr_pred4 = tf.multiply(self.ctr_pred4, self.cvr_pred4, name="cvr_label4")

        self.ctr_pred = tf.concat([self.ctr_pred0, self.ctr_pred1, self.ctr_pred2, self.ctr_pred3, self.ctr_pred4], axis=1)
        self.ctcvr_pred = tf.concat([self.ctcvr_pred0, self.ctcvr_pred1, self.ctcvr_pred2, self.ctcvr_pred3, self.ctcvr_pred4], axis=1)
        step_index = tf.one_hot(tf.cast(tf.squeeze(step_level, axis=1), dtype=tf.int32), depth=5)
        self.ctr_pred = tf.reduce_sum(tf.multiply(self.ctr_pred, step_index), axis=1)
        self.ctcvr_pred = tf.reduce_sum(tf.multiply(self.ctcvr_pred, step_index), axis=1)
        return {"ctr": self.ctr_pred, "cvr": self.ctcvr_pred}



if __name__ == "__main__":
    print("ok")
