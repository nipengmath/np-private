# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
import multiprocessing
import sys, os
import codecs
import json
import time
from datetime import datetime
from multiprocessing import Process, Queue

import tensorflow as tf


from utils import load_feature_config
import config


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(pathname)s:%(lineno)d - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Json2TFRecord(object):
    def __init__(self):
        self.labels, self.cat_features, self.con_features, self.bool_features = load_feature_config()
        self.in_queue = Queue()
        self.out_queue = Queue()

    def __call__(self, in_file, out_file):
        def parsing_loop():
            while True:  # loop.
                raw_record = self.in_queue.get()  # read from in_queue.
                if isinstance(raw_record, str):
                    break
                features = {}  # dict for all feature columns and target column.
                for item in raw_record:
                    tmp = raw_record[item]
                    if item in self.cat_features:
                        # 离散特征值，是个列表的列表
                        tmp = tmp[0]
                        features[item] = self._int64_feature(tmp)
                    elif item in self.con_features:
                        # 连续特征值，是个浮点数
                        features[item] = self._float_feature(tmp)
                    elif item in self.bool_features:
                        # 布尔特征值，是个列表
                        features[item] = self._float_feature(tmp)
                    elif item in self.labels:
                        # 标签特征值，是个数
                        tmp = float(tmp)
                        features[item] = self._float_feature(tmp)

                # create an instance of tf.Example.
                example = tf.train.Example(features=tf.train.Features(feature=features))
                # serialize the tf.Example to string.
                raw_example = example.SerializeToString()

                # write the serialized tf.Example out.
                self.out_queue.put(raw_example)

        def writing_loop():
            writer = tf.io.TFRecordWriter(out_file)
            sample_count = 0
            while True:
                raw_example = self.out_queue.get()  # read from out_queue.
                logging.debug('writing_loop raw_example:{}'.format(raw_example))
                if isinstance(raw_example, str):
                    break
                writer.write(raw_example)  # write it out.
                sample_count += 1
                if not sample_count % 10000:
                    logging.info('%s Processed %d examples' % (datetime.now(), sample_count))
                    sys.stdout.flush()
            writer.close()  # close the writer.
            logging.info('%s >>>> Processed %d examples <<<<' % (datetime.now(), sample_count))
            self.sample_cnt = sample_count
            sys.stdout.flush()

        start_time = time.time()
        # start parsing processes.
        num_parsers = config.num_workers
        parsers = []
        for i in range(num_parsers):
            p = Process(target=parsing_loop)
            parsers.append(p)
            p.start()

        # start writing process.
        writer = Process(target=writing_loop)
        writer.start()
        # logging.info('%s >>>> BEGIN to feed input file %s <<<<' % (datetime.now(), self.path))

        with codecs.open(in_file) as f:
            for line in f:
                raw_record = json.loads(line)
                self.in_queue.put(raw_record)  # write to in_queue.
        # terminate and wait for all parsing processes.
        for i in range(num_parsers):
            self.in_queue.put("DONE")
        for i in range(num_parsers):
            parsers[i].join()

        # terminate and wait for the writing process.
        self.out_queue.put("DONE")
        writer.join()
        end_time = time.time()
        total_time = (end_time - start_time)
        logging.info('%s >>>> END of consuming input file %s <<<<' % (datetime.now(), out_file))
        sys.stdout.flush()

    @staticmethod
    def _int64_feature(value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def _float_feature(value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def _bytes_feature(value):
        ## print("==1", value)
        ## return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value,]))
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=tf.convert_to_tensor(value).numpy()))

    def write_feature_map(self, dateframe, path):
        with open(path, 'a') as f:
            for item in self.CATEGORY_FEATURES:
                f.writelines(','.join([str(dateframe[item].max()), item, 'CATEGORICAL\n']))
            for item in self.NUMERICAL_FEATURES:
                f.write(','.join(['1', item, 'NUMERICAL\n']))
            for item in self.VARIABLE_FEATURES:
                pass
                # f.write(','.join(['1', item, 'VARIABLE\n']))
            for item in self.LABEL:
                f.write(','.join([str(dateframe[item].nunique()), item, 'LABEL\n']))
