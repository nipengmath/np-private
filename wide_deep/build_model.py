# -*- coding: utf-8 -*-


from lr import LR
from esmm import ESMM


def init_model(model_name):
    if model_name == "lr":
        model = LR()
    if model_name == "esmm":
        model = ESMM()
    return model
