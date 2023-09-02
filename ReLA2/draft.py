#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from detectron2.evaluation import inference_context
import matplotlib.pyplot as plt

from demo import GRES_model

input_file_path = '/root/autodl-tmp/data/tmp/batched_inputs.pt'

inference_context(GRES_model)
with torch.no_grad():
    batch_input = torch.load(input_file_path)
    outputs = GRES_model(batch_input)
    for res, masker in zip(outputs['images'], outputs['mask']):
        res = res * masker
        fig = plt.figure()
        plt.tight_layout()
        plt.imshow(res.squeeze(0).to("cpu").permute(1, 2, 0).detach().numpy())
        plt.title("The image is multiplied by target_masks.")
        plt.xticks([])
        plt.yticks([])
        plt.show()
