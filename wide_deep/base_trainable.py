#  -*- coding: utf-8 -*-

import os
import time
from datetime import datetime

import tensorflow as tf
import tensorflow_addons as tfa
from absl import flags, logging

from callbacks import LearningRateScheduler, CSVLogger, LossAndErrorPrintingCallback

import config
from utils import load_feature_config


class BaseTrainable(object):
    def __init__(self):
        super().__init__()
        self.seed_everything()
        if not os.path.exists(config.summaries_dir):
            os.makedirs(config.summaries_dir)
        logging.get_absl_handler().use_absl_log_file(
            program_name='DeePray',
            log_dir=config.summaries_dir
        )
        logging.info(' {} Initialize training'.format(
            time.strftime("%Y%m%d %H:%M:%S")))

        self.flags = config
        self.batch_size = self.flags.batch_size
        self.max_patient_passes = self.flags.patient_valid_passes
        self.labels, self.cat_features, self.con_features, self.bool_features = load_feature_config()
        self.metrics_object = self.build_metrics()
        self.loss_object = self.build_loss()
        self.loss_weight = self.build_loss_weight()

    def seed_everything(self, seed=10):
        tf.random.set_seed(seed)
        # random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        # np.random.seed(seed)

    def build_loss(self):
        return tf.keras.losses.BinaryCrossentropy()

    def build_loss_weight(self):
        return None

    def build_metrics(self):
        metrics = []
        metrics.append(tf.keras.metrics.AUC())
        metrics.append(tf.keras.metrics.BinaryAccuracy())
        return metrics

    def parser(self, record):
        feature_map = {}
        for key in self.bool_features:
            feature_map[key] = tf.io.FixedLenFeature([1], tf.float32)
        for key in self.con_features:
           feature_map[key] = tf.io.FixedLenFeature([1], tf.float32)
        for key in self.cat_features:
            feature_map[key] = tf.io.VarLenFeature(tf.int64)

        parsed = tf.io.parse_single_example(record, features=feature_map)
        feature = {}
        for key in parsed:
            v = parsed[key]

            ## 特殊处理离散值
            if key in self.cat_features:
                v = tf.reshape(v.values, shape=[1, -1])
            feature[key] = v

        ## label
        label = tf.io.parse_single_example(record, features={
            "ctr_label": tf.io.FixedLenFeature([1], tf.float32),
            "cvr_label": tf.io.FixedLenFeature([1], tf.float32)
        })
        ctr_label =  tf.identity(label["ctr_label"], name="ctr_label")
        cvr_label =  tf.identity(label["cvr_label"], name="cvr_label")
        return feature, {"ctr": ctr_label, "cvr": cvr_label}

    def tfrecord_pipeline(self, tfrecord_files, batch_size,
                          epochs, shuffle=True):
        dataset = tf.data.TFRecordDataset(
            [tfrecord_files], compression_type=None) \
            .map(self.parser,
                 num_parallel_calls=tf.data.experimental.AUTOTUNE if self.flags.parallel_parse is None else self.flags.parallel_parse)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.flags.shuffle_buffer)
        dataset = dataset.repeat(epochs) \
            .batch(batch_size) \
            .prefetch(buffer_size=self.flags.prefetch_buffer)
        return dataset

    def create_train_data_iterator(self):
        self.train_iterator = self.tfrecord_pipeline(
            self.flags.train_tfrecord_path, self.flags.batch_size, epochs=1
        )
        self.valid_iterator = self.tfrecord_pipeline(
            self.flags.valid_tfrecord_path, self.flags.batch_size, epochs=1, shuffle=False
        )

    def train(self, model):
        self.create_train_data_iterator()
        optimizer = self.build_optimizer()
        model.compile(optimizer=optimizer,
                      loss=self.loss_object,
                      loss_weights=self.loss_weight,
                      metrics=self.metrics_object)
        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=self.flags.summaries_dir),
            CSVLogger(self.flags.summaries_dir + '/log.csv', append=True, separator=','),
            LossAndErrorPrintingCallback()
        ]
        if self.flags.profile_batch:
            tb_callback = tf.keras.callbacks.TensorBoard(log_dir=self.flags.summaries_dir,
                                                         profile_batch=self.flags.profile_batch)
            callbacks.append(tb_callback)
        if self.flags.patient_valid_passes:
            EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_ctr_auc',
                                                             patience=self.flags.patient_valid_passes,
                                                             mode='min',
                                                             restore_best_weights=True)
            callbacks.append(EarlyStopping)
        if self.flags.checkpoint_path:
            # Create a callback that saves the model's weights
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.flags.checkpoint_path,
                                                             save_weights_only=True,
                                                             monitor='val_loss',
                                                             mode='auto',
                                                             save_best_only=True)
            callbacks.append(cp_callback)
        if self.flags.lr_schedule:
            callbacks.append(LearningRateScheduler(self.lr_schedule))
        history = model.fit(self.train_iterator, validation_data=self.valid_iterator,
                            epochs=self.flags.epochs, callbacks=callbacks)
        return history

    def _mylog(self, r):
        return tf.math.log(tf.math.maximum(r, tf.constant(1e-18)))

    def build_optimizer(self):
        if self.flags.optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam
        elif self.flags.optimizer == "adadelta":
            optimizer = tf.keras.optimizers.Adadelta
        elif self.flags.optimizer == "adagrad":
            optimizer = tf.keras.optimizers.Adagrad
        elif self.flags.optimizer == "lazyadam":
            optimizer = tfa.optimizers.LazyAdam
        elif self.flags.optimizer == "ftrl":
            optimizer = tf.keras.optimizers.Ftrl
        elif self.flags.optimizer == "sgd":
            optimizer = tf.keras.optimizers.SGD
        elif self.flags.optimizer == "RMSprop":
            optimizer = tf.keras.optimizers.RMSprop
        else:
            raise ValueError('--optimizer {} was not found.'.format(self.flags.optimizer))
        return optimizer(learning_rate=self.flags.learning_rate)

    def lr_schedule(self, epoch, lr):
        """Helper function to retrieve the scheduled learning rate based on epoch."""
        LR_SCHEDULE = config.LR_SCHEDULE
        if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
            return lr
        for i in range(len(LR_SCHEDULE)):
            if epoch == LR_SCHEDULE[i][0]:
                return LR_SCHEDULE[i][1]
        return lr
