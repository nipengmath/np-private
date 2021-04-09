# -*- coding: utf-8 -*-

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import config
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus # 使用 GPU 0，1

sess_config = tf.compat.v1.ConfigProto()
sess_config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=sess_config)

from absl import app, flags

from build_model import init_model


def main():
    ## 初始话模型
    model = init_model(config.model_name)
    ## 模型训练
    model.train(model)
    model.save(config.pb_path)


if __name__ == "__main__":
    print("ok")
    main()
