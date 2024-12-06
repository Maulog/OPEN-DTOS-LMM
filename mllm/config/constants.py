#########dtos_lmm==VILA##########
CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>" # different from root

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

#########dtos_loc##########
VIDEO_TOKEN_INDEX = -300
DEFAULT_VIDEO_TOKEN = "<video>"
VIDEO_PLACEHOLDER = "<video-placeholder>"
DEFAULT_MOMENT_TOKEN = "<rec>" # *
DEFAULT_VI_START_TOKEN = "<vi_start>" # *
DEFAULT_VI_END_TOKEN = "<vi_end>" # *
DEFAULT_MO_START_TOKEN = "<rec_start>" # *
DEFAULT_MO_END_TOKEN = "<rec_end>"

#########dtos_seg##########
# DEFAULT_SEG_TOKEN = "<seg>"
DEFAULT_BOX_TOKEN = "<box>" # combine to <seg>
DEFAULT_POS_TOKEN = "<pos>" # combine to <seg>
DEFAULT_NEG_TOKEN = "<neg>" # combine to <seg>

DEFAULT_TGT_START_TOKEN = "<tgt_start>"
DEFAULT_TGT_END_TOKEN = "<tgt_end>"
DEFAULT_VALID_TOKEN = "<valid>"

DEFAULT_MOM_START_TOKEN = "<moment_start>"
DEFAULT_MOM_END_TOKEN = "<moment_end>"

DEFAULT_TGT_FIRST_TOKEN = "<tgt_1>"
DEFAULT_TGT_SECOND_TOKEN = "<tgt_2>"
DEFAULT_TGT_THIRD_TOKEN = "<tgt_3>"
DEFAULT_TGT_FOURTH_TOKEN = "<tgt_4>"
DEFAULT_TGT_FIFTH_TOKEN = "<tgt_5>"
DEFAULT_TGT_SIXTH_TOKEN = "<tgt_6>"
DEFAULT_TGT_SEVENTH_TOKEN = "<tgt_7>"
DEFAULT_TGT_EIGHTH_TOKEN = "<tgt_8>"
DEFAULT_TGT_NINTH_TOKEN = "<tgt_9>"
DEFAULT_TGT_TENTH_TOKEN = "<tgt_10>"