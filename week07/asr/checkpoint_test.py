from collections import OrderedDict
from pprint import pprint
import torch
from natsort import natsorted

from src.utils import get_encoder_description

if __name__ == "__main__":
    ckpt = torch.load(
        "/mnt/c/Users/Kirill/Documents/repos/speech-tech-mipt/week07/asr/data/q5x5_ru_stride_4_crowd_epoch_4_step_9794.ckpt",
        map_location="cpu")
    pprint(get_encoder_description(OrderedDict(natsorted(ckpt['state_dict'].items(), key=lambda x: x[0]))))
