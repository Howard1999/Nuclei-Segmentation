# The new config inherits a base config to highlight the necessary modification
_base_ = 'mmdetection/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)),
    rpn_head=dict(
        anchor_generator=dict(
            scales=[1]
        )))

# Modify dataset related settings
dataset_dir = '../../coco/normal/'
dataset_type = 'COCODataset'
classes = ('cell',)
data = dict(
    train=dict(
        img_prefix=dataset_dir + 'train/',
        classes=classes,
        ann_file=dataset_dir + 'train.json'),
    val=dict(
        img_prefix=dataset_dir + 'val/',
        classes=classes,
        ann_file=dataset_dir + 'val.json'),
    test=dict(
        img_prefix=dataset_dir + 'test/',
        classes=classes,
        ann_file=dataset_dir + 'test.json'))