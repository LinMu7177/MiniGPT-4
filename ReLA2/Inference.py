#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import sys
from detectron2.evaluation import inference_context
from detectron2.engine import default_argument_parser
from ReLA2.model_instance import get_model, raw_data2feature


# TODO: image needs to be source image
class GRESModelContainer:
    def __init__(self):
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

        argv_backup = copy.deepcopy(sys.argv)
        sys.argv = []
        sys.argv.append('/root/autodl-tmp/codes/MiniGPT-4/ReLA2/demo.py')
        for argument in artificial_args.split('\n'):
            sys.argv.append(argument.lstrip())

        args, unknown = default_argument_parser().parse_known_args()
        args.config_file = 'ReLA2/' + args.config_file

        self.GRES_model, _ = get_model(args)
        sys.argv = argv_backup
        # inference_context(self.GRES_model)
