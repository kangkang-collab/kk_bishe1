import torch
from mmengine.dataset import default_collate

def my_collate(batch):
    # 保持 points 为 list，不进行堆叠
    for item in batch:
        if "points" in item:
            item["points"] = item["points"]

    # 使用 MMEngine 的默认 collate 处理其他字段
    return default_collate(batch)
