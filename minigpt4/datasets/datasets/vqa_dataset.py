from pathlib import Path

from PIL import Image
from detectron2.data import transforms as T

from minigpt4.datasets.datasets.caption_datasets import CaptionDataset
from ReLA2.gres_model import RefCOCOMapper
from ReLA2.gres_model.data.datasets.grefcoco import load_grefcoco_json


def build_image_transform(image_size: int = 480):
    augmentation = []

    augmentation.extend([
        T.Resize((image_size, image_size))
    ])

    return augmentation


class CCVQADataset(CaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        # TODO need to change
        refer_root = Path('/root/autodl-tmp/data/VQA_Data/raw_VizWiz/GRES_data/')
        dataset_name = 'grefcoco'
        splitby = 'unc'
        split = 'train'
        is_train = split == 'train'
        image_root = refer_root / 'images/train2014'
        self.dataset_gres_style = load_grefcoco_json(refer_root, dataset_name, splitby, split, image_root)

        tfm_gens = build_image_transform(480)
        image_format = 'RGB'
        bert_type = 'bert-base-uncased'
        max_tokens = 20

        self.ref_coco_mapper = RefCOCOMapper(
            is_train=is_train,
            tfm_gens=tfm_gens,
            image_format=image_format,
            bert_type=bert_type,
            max_tokens=max_tokens,
        )

        del self.img_ids
        del self.annotation
        del self.vis_root

    def get_focus_feature_dic(self, index):
        dataset_dict = self.dataset_gres_style[index]
        dic = self.ref_coco_mapper(dataset_dict)
        return dic

    def __getitem__(self, index):
        # TODO this assumes image input, not general enough
        focus_feature_dic = self.get_focus_feature_dic(index)
        image = Image.open(focus_feature_dic['file_name']).convert("RGB")
        image = self.vis_processor(image)

        return {
            "image": image,
            "image_id": focus_feature_dic['image_id'],
            'focus_image': focus_feature_dic['image'],
            'focus_lang_tokens': focus_feature_dic['lang_tokens'],
            'focus_lang_mask': focus_feature_dic['lang_mask'],
            'query': focus_feature_dic['sentence']['raw'],
            'answer': focus_feature_dic['most_common_answer'],
            # 'answers': focus_feature_dic['answers'],
        }

    def __len__(self):
        return len(self.dataset_gres_style)
