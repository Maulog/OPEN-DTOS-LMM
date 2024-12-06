training_args = dict(
    # run
    output_dir=None,  # required. must be filled by derived configs.
    overwrite_output_dir=False,
    report_to='none',
    seed=42,

    # datasets
    remove_unused_columns=False,
    
    # # ddp
    # deepspeed="config/_base_/train/zero3.json",

    # logging
    logging_steps=1,

    # eval and predict
    do_eval=False,
    do_predict=False,
    do_multi_predict=True,
    tf32=True,
    fp16=False, # False
    fp16_full_eval=False,
    bf16=True,
    bf16_full_eval=True, # True
    predict_with_generate=True,
    per_device_eval_batch_size=1,
    dataloader_num_workers=2,
)
