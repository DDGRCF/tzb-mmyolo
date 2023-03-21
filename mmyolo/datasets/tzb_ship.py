# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from mmdet.datasets.api_wrappers import COCO
from mmdet.datasets import CocoDataset



@DATASETS.register_module()
class TzbShipDataset(CocoDataset):
    """Dataset for tzb-ship."""

    METAINFO = {
        'classes':
        ('ship', ),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (246, 0, 122), (191, 162, 208)]
    }
    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True

