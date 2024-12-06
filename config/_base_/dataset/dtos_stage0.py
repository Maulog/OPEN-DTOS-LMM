_base_ = ['DEFAULT_TRAIN_DATASET.py']

data_args = dict(
    # demo
    # query = "<image> , <image> , <image> , Please describe these three images separately.",
    # image_files = ["demo/two_people_sport.jpg", "demo/cow_and_woman.jpg", 
    #                 "demo/furniture.jpg"],
    # video_file = None,
    
    query = "<video> Please describe this video.",
    image_files = None,
    video_file = 'demo/musical.mov', # 'demo/cook.mov', 
    
    num_video_frames = 6,
    
    sep = ",",
    
    #dataset
    train = None,
    validation=None,
    test=None,
    

    # compute_metric
    compute_metric=None,

    # # padding collator kwargs
    # collator_kwargs=dict(
    #     padding=True,
    #     max_length=1024,
    # ),

    # generate config
    gen_kwargs=dict(
        temperature = 0.2,
        top_p = None,
        num_beams = 1,
        max_new_tokens = 1024,
    ),
)
