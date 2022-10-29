from collections import OrderedDict

import torch


def find_nth_occurrence(haystack: str, needle: str, n: int) -> int:
    parts = haystack.split(needle, n)
    if len(parts) < n + 1:
        return -1
    return len(haystack) - len(parts[-1]) - len(needle)


def map_model_state_dict(state_dict):
    res = OrderedDict()
    for k, v in state_dict.items():
        if "encoder" in k:
            block_index = k[find_nth_occurrence(k, ".", 2) + 1: find_nth_occurrence(k, ".", 3)]
            block_key = k[find_nth_occurrence(k, ".", 3) + 1:]

            if "conv" in block_key:
                sub_block_index = block_key[
                                  find_nth_occurrence(block_key, ".", 1) + 1:find_nth_occurrence(block_key, ".", 2)
                                  ]
                sub_block_key = block_key[find_nth_occurrence(block_key, ".", 2) + 1:]

                new_key = f"encoder.layers.{block_index}.mainline.{sub_block_index}.{sub_block_key}"
                res[new_key] = v
            elif "res" in block_key:
                sub_block_index = block_key[
                                  find_nth_occurrence(block_key, ".", 1) + 1:find_nth_occurrence(block_key, ".", 2)
                                  ]
                sub_block_key = block_key[find_nth_occurrence(block_key, ".", 2) + 1:]

                new_key = f"encoder.layers.{block_index}.residual.{sub_block_index}.{sub_block_key}"
                res[new_key] = v
        else:
            res[k] = v

    return res


def get_encoder_description(obj):
    if torch.is_tensor(obj):
        return obj.size()

    if isinstance(obj, dict):
        res = OrderedDict()
        for k in obj:
            if "encoder" in k:
                res[k] = get_encoder_description(obj[k])

        return res

    return obj
