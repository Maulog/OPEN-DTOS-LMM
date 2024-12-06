training_args = dict(
    # run
    output_dir=None,  # required. must be filled by derived configs.
    overwrite_output_dir=True, # True是从头训练  False是继续训练
    report_to='none',
    seed=42,
 
    # datasets
    remove_unused_columns=False,

    # train
    do_train=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=2,
    learning_rate=2e-5,
    lr_scheduler_type='cosine',
    weight_decay=1e-4,
    warmup_ratio=0.03,
    evaluation_strategy='no',

    # train ddp
    tf32=True,
    bf16=True,
    gradient_checkpointing=True,
    # deepspeed="config/_base_/train/zero1.json",

    # train logging
    logging_steps=100,
    save_strategy='steps',
    save_steps=10000,
    save_total_limit=4, # 每次保存的最近保存数

    # eval and predict
    do_eval=False,
    do_predict=False,
    predict_with_generate=False,
    per_device_eval_batch_size=1,
    dataloader_num_workers=4, # 16太高了，感觉内存消耗高（会爆内存），二阶段的6线程也会爆内存
)
