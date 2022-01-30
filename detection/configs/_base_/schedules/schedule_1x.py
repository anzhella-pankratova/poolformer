# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=28)

# from centernet config

# optimizer
# Based on the default settings of modern detectors, the SGD effect is better
# than the Adam in the source code, so we use SGD default settings and
# if you use adam+lr5e-4, the map is 29.1.
#optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
# Based on the default settings of modern detectors, we added warmup settings.
#lr_config = dict(
#    policy='step',
#    warmup='linear',
#    warmup_iters=1000,
#    warmup_ratio=1.0 / 1000,
#    step=[18, 24])  # the real step is [18*5, 24*5]
#runner = dict(type='EpochBasedRunner', max_epochs=28)  # the real epoch is 28*5=140
