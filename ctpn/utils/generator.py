# -*- coding: utf-8 -*-
"""
   File Name：     generator
   Description :  生成器
   Author :       mick.yi
   date：          2019/3/14
"""
import numpy as np
from ..utils import image_utils, np_utils


def generator(image_annotations, max_output_dim, max_gt_num, batch_size):
    image_length = len(image_annotations)
    while True:
        ids = np.random.choice(image_length, batch_size, replace=False)
        batch_images = []
        batch_images_meta = []
        batch_gt_boxes = []
        batch_gt_class_ids = []
        for id in ids:
            image_annotation = image_annotations[id]
            image, image_meta, gt_boxes = image_utils.load_image_gt(id,
                                                                    image_annotation['image_path'],
                                                                    max_output_dim,
                                                                    image_annotation['boxes'])
            batch_images.append(image)
            batch_images_meta.append(image_meta)
            class_ids = image_annotation['labels']
            batch_gt_boxes.append(np_utils.pad_to_fixed_size(gt_boxes, max_gt_num))
            batch_gt_class_ids.append(
                np_utils.pad_to_fixed_size(np.expand_dims(np.array(class_ids), axis=1), max_gt_num))

        # 返回结果
        yield {"input_image": np.asarray(batch_images),
               "input_image_meta": np.asarray(batch_images_meta),
               "gt_class_ids": np.asarray(batch_gt_class_ids),
               "gt_boxes": np.asarray(batch_gt_boxes)}, None