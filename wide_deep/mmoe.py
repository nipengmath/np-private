# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow.keras.backend import expand_dims,repeat_elements,sum
from tensorflow.keras.initializers import VarianceScaling

from base_trainable import BaseTrainable
from base_model import BaseCTRModel

from base.layers.core import Linear
import config

from tensorflow.keras import layers, Model, initializers, regularizers, activations, constraints, Input


class MMoE(BaseTrainable, BaseCTRModel):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        # Hidden nodes parameter
        self.units = config.units
        self.num_experts = config.num_experts
        self.num_tasks = config.num_tasks

        # Weight parameter
        self.expert_kernels = None
        self.gate_kernels = None
        self.expert_kernel_initializer = initializers.get(config.expert_kernel_initializer)
        self.gate_kernel_initializer = initializers.get(config.gate_kernel_initializer)
        self.expert_kernel_regularizer = regularizers.get(config.expert_kernel_regularizer)
        self.gate_kernel_regularizer = regularizers.get(config.gate_kernel_regularizer)
        self.expert_kernel_constraint = constraints.get(config.expert_kernel_constraint)
        self.gate_kernel_constraint = constraints.get(config.gate_kernel_constraint)

        # Activation parameter
        #self.expert_activation = activations.get(expert_activation)
        self.expert_activation = config.expert_activation
        self.gate_activation = config.gate_activation

        # Bias parameter
        self.expert_bias = None
        self.gate_bias = None
        self.use_expert_bias = config.use_expert_bias
        self.use_gate_bias = config.use_gate_bias
        self.expert_bias_initializer = initializers.get(config.expert_bias_initializer)
        self.gate_bias_initializer = initializers.get(config.gate_bias_initializer)
        self.expert_bias_regularizer = regularizers.get(config.expert_bias_regularizer)
        self.gate_bias_regularizer = regularizers.get(config.gate_bias_regularizer)
        self.expert_bias_constraint = constraints.get(config.expert_bias_constraint)
        self.gate_bias_constraint = constraints.get(config.gate_bias_constraint)

        # Activity parameter
        self.activity_regularizer = regularizers.get(config.activity_regularizer)

        self.expert_layers = []
        self.gate_layers = []
        for i in range(self.num_experts):
            self.expert_layers.append(layers.Dense(self.units, activation=self.expert_activation,
                                                   use_bias=self.use_expert_bias,
                                                   kernel_initializer=self.expert_kernel_initializer,
                                                   bias_initializer=self.expert_bias_initializer,
                                                   kernel_regularizer=self.expert_kernel_regularizer,
                                                   bias_regularizer=self.expert_bias_regularizer,
                                                   activity_regularizer=None,
                                                   kernel_constraint=self.expert_kernel_constraint,
                                                   bias_constraint=self.expert_bias_constraint))
        for i in range(self.num_tasks):
            self.gate_layers.append(layers.Dense(self.num_experts, activation=self.gate_activation,
                                                 use_bias=self.use_gate_bias,
                                                 kernel_initializer=self.gate_kernel_initializer,
                                                 bias_initializer=self.gate_bias_initializer,
                                                 kernel_regularizer=self.gate_kernel_regularizer,
                                                 bias_regularizer=self.gate_bias_regularizer, activity_regularizer=None,
                                                 kernel_constraint=self.gate_kernel_constraint,
                                                 bias_constraint=self.gate_bias_constraint))
        self.tower_layers = []
        self.output_layers = []
        for i in range(self.num_tasks):
            self.tower_layers.append(layers.Dense(units=8,activation='relu',kernel_initializer=VarianceScaling()))
            self.output_layers.append(layers.Dense(units=1, activation='softmax', kernel_initializer=VarianceScaling()))

    def build_loss(self):
        return {"ctr": "binary_crossentropy", "cvr": "binary_crossentropy"}

    def build_loss_weight(self):
        return {"ctr": 1.0, "cvr": 1.0}

    def build_mmoe(self, features, is_training=None):
        expert_outputs, gate_outputs, final_outputs = [], [], []
        for expert_layer in self.expert_layers:
            expert_output = expand_dims(expert_layer(features), axis=2)
            expert_outputs.append(expert_output)
        expert_outputs = tf.concat(expert_outputs,2)

        for gate_layer in self.gate_layers:
            gate_outputs.append(gate_layer(features))

        for gate_output in gate_outputs:
            expanded_gate_output = expand_dims(gate_output, axis=1)
            weighted_expert_output = expert_outputs * repeat_elements(expanded_gate_output, self.units, axis=1)
            final_outputs.append(sum(weighted_expert_output, axis=2))

        # 返回的矩阵维度 num_tasks * batch * units
        return final_outputs

    def build_networks(self, features, is_training=None):
        output_layers = []
        mmoe_layers = self.build_mmoe(features)
        output_info = [(1, 'ctr'), (1, 'cvr')]

        for index, task_layer in enumerate(mmoe_layers):
            tower_layer = self.tower_layers[index](task_layer)
            output_layer = self.output_layers[index](tower_layer)
            output_layers.append(output_layer)
        ctr_pred = output_layers[0]
        cvr_pred = output_layers[1]
        return ctr_pred, cvr_pred

    def call(self, inputs, is_training=None):
        features = self.build_features(inputs)
        self.ctr_pred, self.cvr_pred = self.build_networks(features, is_training)

        #self.ctr_pred = self.predict_layer(ctr_logit)
        #self.cvr_pred = self.predict_layer(cvr_logit)

        self.ctr_pred =  tf.identity(self.ctr_pred, name="ctr_label")
        self.ctcvr_pred = tf.multiply(self.ctr_pred, self.cvr_pred, name="cvr_label")

        return {"ctr": self.ctr_pred, "cvr": self.ctcvr_pred}




if __name__ == "__main__":
    print("ok")
