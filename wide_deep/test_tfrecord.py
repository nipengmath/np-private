# -*- coding: utf-8 -*-


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from base_trainable import BaseTrainable


base_trainer = BaseTrainable()


def test1():
    filenames = ["./data/train.tfrecord"]
    raw_dataset = tf.data.TFRecordDataset(filenames)
    for raw_record in raw_dataset.take(10):
        #print(raw_record)
        x = base_trainer.parser(raw_record)
        print(x)

def test2():
    base_trainer.create_train_data_iterator()
    train_iter = base_trainer.train_iterator
    for feature_batch, label_batch in train_iter.take(1):
        print("==1", feature_batch)
        ctr_3 = feature_batch["ctr_3"]
        key_menu1 = feature_batch["key_menu1"]
        print("==1", ctr_3.shape)
        print("==2", key_menu1.shape)


if __name__ == "__main__":
    print("ok")
    #test1()
    test2()
