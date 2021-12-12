# The new config inherits a base config to highlight the necessary modification
_base_ = '/mmdetection/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=2),
        mask_head=dict(num_classes=2)),
    rpn_head=dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[7],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64])),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                gpu_assign_thr=500,
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1))),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=800,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))

# Modify dataset related settings
dataset_dir = '/mnt/coco/unsharp/'
dataset_type = 'CocoDataset'
classes = ('background','nuclei')
data = dict(
    images_per_gpu=1,
    works_per_gpu=1,
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
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=24)
# load_from="/mnt/mask_r_cnn_checkpoint/latest.pth"