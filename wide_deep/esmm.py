# -*- coding: utf-8 -*-

import tensorflow as tf

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
        self.deep_ctr_block = self.build_deep(hidden=hidden)
        self.deep_cvr_block = self.build_deep(hidden=hidden)

    def build_loss(self):
        return {"ctr": "binary_crossentropy", "cvr": "binary_crossentropy"}

    def build_loss_weight(self):
        return {"ctr": 1.0, "cvr": 1.0}

    def build_networks(self, features, is_training=None):
        share_output = self.deep_share_block(features, is_training)
        ctr_logit = self.deep_ctr_block(share_output, is_training)
        cvr_logit = self.deep_cvr_block(share_output, is_training)
        return ctr_logit, cvr_logit

    def call(self, inputs, is_training=None):
        ## print(inputs)
        features = self.build_features(inputs)
        ctr_logit, cvr_logit = self.build_networks(features, is_training)

        self.ctr_pred = self.predict_layer(ctr_logit)
        self.cvr_pred = self.predict_layer(cvr_logit)

        self.ctr_pred =  tf.identity(self.ctr_pred, name="ctr_label")
        self.ctcvr_pred = tf.multiply(self.ctr_pred, self.cvr_pred, name="cvr_label")
        return {"ctr": self.ctr_pred, "cvr": self.ctcvr_pred, "output_0": self.ctcvr_pred}



if __name__ == "__main__":
    print("ok")
