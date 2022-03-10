# model settings

model = dict(
    type='SingleStageDetector',
    #pretrained='open-mmlab://vgg16_caffe',
    backbone=dict(
    #    type='SSDVGG',
        #input_size=input_size,
        #depth=16,
        fork_feat=True,
        #ceil_mode=True,
        out_indices=(3, 4),
        #out_feature_indices=(22, 34),
        #l2_norm_scale=20)
    ),
    neck=dict(
        type='SSDNeck',
        in_channels=(80, 320),
        out_channels=(80, 320, 512, 256, 256, 128),
        level_strides=(2, 2, 2, 2),
        level_paddings=(1, 1, 1, 1),
        l2_norm_scale=None,
        use_depthwise=True,
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.03),
        act_cfg=dict(type='ReLU6'),
        init_cfg=dict(type='TruncNormal', layer='Conv2d', std=0.03)),
    bbox_head=dict(
        type='SSDHead',
        in_channels=(80, 320, 512, 256, 256, 128),
        num_classes=80,
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            basesize_ratio_range=(0.15, 0.9),
            strides=[16, 32, 64, 128, 256, -1],
            ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2]],
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2])),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.,
            ignore_iof_thr=-1,
            gt_max_assign_all=False),
        smoothl1_beta=1.,
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        nms=dict(type='nms', iou_threshold=0.45),
        min_bbox_size=0,
        score_thr=0.02,
        max_per_img=200))
cudnn_benchmark = True
