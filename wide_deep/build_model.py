# -*- coding: utf-8 -*-


from esmm import ESMM
from mmoe import MMoE


def init_model(model_name):
    if model_name == "esmm":
        model = ESMM()
    if model_name == "mmoe":
        model = MMoE()
    return model
