# -*- coding: utf-8 -*-


import codecs, json


import config
from converter import Json2TFRecord


def main():
    converter = Json2TFRecord()
    converter(config.train_path, config.train_tfrecord_path)
    converter(config.valid_path, config.valid_tfrecord_path)



if __name__ == "__main__":
    print("ok")
    main()
