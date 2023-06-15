import os
import logging
import warnings

from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from minigpt4.common.registry import registry
from minigpt4.datasets.datasets.coco_vqa_datasets import COCOVQADataset, COCOVQAEvalDataset

# @registry.register_builder("coco_vqa")
# class COCOVQABuilder(BaseDatasetBuilder):
#     train_dataset_cls = COCOVQADataset
#     eval_dataset_cls = COCOVQAEvalDataset
#
#     DATASET_CONFIG_DICT = {
#         "default": "configs/datasets/coco/defaults_vqa.yaml",
#         "eval": "configs/datasets/coco/eval_vqa.yaml",
#     }

@registry.register_builder("coco_vqa")
class COCOVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOVQADataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_vqa.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'vqa_train.json')],
            vis_root=os.path.join(storage_path, ''),
        )

        return datasets
