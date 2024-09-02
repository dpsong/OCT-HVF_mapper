cudnn_benchmark = True
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
fp16 = dict(loss_scale='dynamic')

model = dict(type='OCTEVA3D',  slice_dim=16, num_classes=2, grad_checkpointing=True)
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys={
            'patch_embed.': dict(lr_mult=0.1, decay_mult=1.),
            'blocks.': dict(lr_mult=0.1, decay_mult=1.),
            'norm.': dict(lr_mult=0.1, decay_mult=1.),
            'rope.': dict(lr_mult=0.1, decay_mult=1.),
        }))
optimizer_config = dict(grad_clip=100.)
runner = dict(type='EpochBasedRunner', max_epochs=10)
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(type='RepeatDataset',
               times=1,
               dataset=dict(
                   type='OCTVFDataset',
                   data_root='data/example/',
                   split='data/TrainVal_split/train_split.json',
                   num_classes=2,
                   slice_dim=16,
                   reg_targets_dim=54,
                   cls_targets_dim=52,
                   num_thread=1,
                   dataAug=True,
               )),
    val=dict(type='OCTVFDataset',
             data_root='data/example/',
             split='data/TrainVal_split/val_split.json',
             reg_targets_dim=54,
             cls_targets_dim=52,
             num_classes=2,
             slice_dim=16),
)
evaluation = dict(interval=1,
                  less_keys=['mad'],
                  greater_keys=['acc'],
                  save_best='mad')
lr_config = dict(policy='CosineAnnealing',
                 min_lr=0,
                 by_epoch=False,
                 warmup='linear',
                 warmup_ratio=0.001,
                 warmup_iters=100,
                 warmup_by_epoch=False)
optimizer_config = dict()
checkpoint_config = dict(by_epoch=True, interval=1)
log_config = dict(
    interval=25,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])
