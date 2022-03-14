_base_ = [
    '_base_/models/centernet_resnet18_140e_coco.py',
    '_base_/datasets/coco_detection.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py'
]

# optimizer
model = dict(
    # pretrained='https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s12.pth.tar', # for old version of mmdetection 
    backbone=dict(
        type='poolformer_s12_feat',
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', 
            checkpoint=\
                'https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s12.pth.tar', 
            ),
    ),
    neck=dict(
        type='CTResNetNeck',
        in_channel=512,
        num_deconv_filters=(256, 128, 64),
        num_deconv_kernels=(4, 4, 4),
        use_dcn=True
    ),
    #init_cfg=dict(
    #    type='Pretrained', 
    #    checkpoint=\
    #        './work_dirs/centernet_poolformer_s12_fpn_1x_coco/epoch_12.pth',
    #),
)

#optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
