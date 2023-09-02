#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import torch
import pickle

import matplotlib.pyplot as plt

from detectron2.evaluation import inference_context
from ReLA2.model_instance import get_model, raw_data2feature
from detectron2.engine import default_argument_parser

# TODO: image needs to be source image

def get_src_input():
    image_file_path = '/root/autodl-tmp/data/tmp/feature.pkl'
    feature_dic = pickle.load(open(image_file_path, 'rb'))
    input_dic = {
        'image': feature_dic['image'],
        'text': feature_dic['text_input'],
    }
    print("target: ", input_dic['text'])
    return input_dic

artificial_args = """--config-file
configs/referring_swin_base.yaml
--num-gpus
1
--dist-url
auto
--eval-only
MODEL.WEIGHTS
/root/autodl-tmp/wwx/Models/GRES/gres_swin_base.pth
OUTPUT_DIR
/root/autodl-tmp/output/alan/GRES"""

for argument in artificial_args.split('\n'):
    sys.argv.append(argument)

args, unknown = default_argument_parser().parse_known_args()
GRES_model, cfg = get_model(args)


input_dic = get_src_input()

inference_context(GRES_model)
with torch.no_grad():
    feature_dic = raw_data2feature(cfg, input_dic)
    outputs = GRES_model([feature_dic])
    for res, masker in zip(outputs['images'], outputs['mask']):
        res = res * masker
        fig = plt.figure()
        plt.tight_layout()
        plt.imshow(res.squeeze(0).to("cpu").permute(1, 2, 0).detach().numpy())
        plt.title(f"The image is multiplied by target_masks. target: {input_dic['text']}")
        plt.xticks([])
        plt.yticks([])
        plt.show()
