#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle
from typing import Dict

import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from transformers import BertTokenizer

from gres_model.data.dataset_mappers.refcoco_mapper import build_transform_test

try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings

    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import detectron2.utils.comm as comm

from detectron2.config import get_cfg
from detectron2.engine import (
    default_argument_parser,
    default_setup,
)

from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

# MaskFormer
from gres_model import (
    add_maskformer2_config,
    add_refcoco_config
)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_refcoco_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="referring")
    return cfg


def raw_data2feature(cfg, sample: Dict):
    max_tokens = cfg.REFERRING.MAX_TOKENS
    bert_type = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(bert_type)

    image = sample['image']
    dataset_dict = {}
    # TODO: get image file_path?
    # image = utils.read_image(dataset_dict["file_name"], format=self.img_format)

    tfm_gens = build_transform_test(cfg)

    # TODO: get padding mask
    # by feeding a "segmentation mask" to the same transforms
    padding_mask = np.ones(image.shape[:2])

    image, transforms = T.apply_transform_gens(tfm_gens, image)
    # the crop transformation has default padding value 0 for segmentation
    padding_mask = transforms.apply_segmentation(padding_mask)
    padding_mask = ~ padding_mask.astype(bool)

    # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
    # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
    # Therefore it's important to use torch.Tensor.
    dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
    dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

    sentence_raw = sample['text_input']
    attention_mask = [0] * max_tokens
    padded_input_ids = [0] * max_tokens

    input_ids = tokenizer.encode(text=sentence_raw, add_special_tokens=True)

    input_ids = input_ids[:max_tokens]
    padded_input_ids[:len(input_ids)] = input_ids

    attention_mask[:len(input_ids)] = [1] * len(input_ids)

    dataset_dict['lang_tokens'] = torch.tensor(padded_input_ids).unsqueeze(0)
    dataset_dict['lang_mask'] = torch.tensor(attention_mask).unsqueeze(0)

    return dataset_dict


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    cfg = setup(args)
    image_file_path = '/root/autodl-tmp/data/tmp/feature.pkl'
    feature_dic = pickle.load(open(image_file_path, 'rb'))
    dataset_dict = raw_data2feature(cfg, feature_dic)
    assert torch.equal(dataset_dict['image'], feature_dic['convert_image'])
    assert torch.equal(dataset_dict['lang_tokens'], feature_dic['lang_tokens'])
    assert torch.equal(dataset_dict['lang_mask'], feature_dic['lang_mask'])
