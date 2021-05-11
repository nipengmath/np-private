# -*- coding: utf-8 -*-

## 特征配置文件
feature_config_file = "/mount_nas/newsugg/workspace/nlp/model/search_product_24/feature_config_info.json"
training_feature_info_file = "/mount_nas/newsugg/workspace/nlp/model/search_product_24/training_feature_info.json"

## cuda gpu
gpus = "1"  #"0,1"


## 训练数据
train_path = "/mount_nas/newsugg/workspace/nlp/model/search_product_24/train.json"
valid_path = "/mount_nas/newsugg/workspace/nlp/model/search_product_24/test.json"
#train_path = "/mount_nas/newsugg/workspace/nlp/model/search_product_24/sample.json"
#valid_path = "/mount_nas/newsugg/workspace/nlp/model/search_product_24/sample.json"

train_tfrecord_path = "./data/train.tfrecord"
valid_tfrecord_path = "./data/valid.tfrecord"

## 处理训练数据的进程数
num_workers = 10

## 模型

#model_name = "lr"
model_name = "esmm"

#deep_layers = [1024, 1024, 1024]
deep_layers = [512, 256, 128]
share_deep_layers = [512, 256, 128]
batch_size = 1024
learning_rate = 0.0005
epochs = 5


## 训练参数
lr_schedule = False
LR_SCHEDULE = [
    # (epoch to start, learning rate) tuples
    (2, 0.01), (5, 0.001), (20, 0.0001)
]

# ["adam", "adagrad", "adadelta", "lazyadam", "sgd", "RMSprop", "ftrl"],
optimizer = "adam"
patient_valid_passes = 0
profile_batch = None
summaries_dir = "summary"

# tfrecord
parallel_parse = 10
shuffle_buffer = 100 * batch_size
prefetch_buffer = 10 * batch_size

use_bn = False
res_deep = False
renorm = False
l1 = 0.1
l2 = 0.1
keep_prob = 1.0
embedding_size = 32
emb_size_factor = 6


## save model
checkpoint_path = "./saved_model/ckpt_model/cp-{epoch:04d}.ckpt"
pb_path = "./saved_model/pb_model"
