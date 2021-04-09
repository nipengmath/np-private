# -*- coding: utf-8-*-


import codecs, json

import config


def load_feature_config():
    """
    输入：type=0 离散；type=1 连续；type=2 布尔
    """
    labels = ["ctr_label", "cvr_label"]
    cat_features = []
    con_features = []
    bool_features = []

    path = config.feature_config_file
    with codecs.open(path) as f:
        data = json.loads(f.read())
    conf = data["model_config"]
    for feature, info in conf.items():
        _type = info["type"]
        if _type == 0:
            cat_features.append(feature)
        elif _type == 1:
            con_features.append(feature)
        elif _type == 2:
            bool_features.append(feature)
    return labels, cat_features, con_features, bool_features


def get_vocab_dict(training_feature_info_path, cat_features):
    with open(training_feature_info_path, 'r') as f:
        info = json.load(f)

    config = info["model_config"]
    vocab_dict = {}
    for cat in cat_features:
        value_list = config[cat]["categorical_string_set"]
        vocab_dict[cat] = value_list
    return vocab_dict


if __name__ == "__main__":
    load_feature_config()
