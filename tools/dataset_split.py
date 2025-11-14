import os
import random

imageset_dir = "../data/custom_av/ImageSets"

# 기존 train.txt 읽기
with open(os.path.join(imageset_dir, "train.txt"), "r") as f:
    file_ids = [line.strip() for line in f.readlines() if line.strip()]

# 섞기
random.seed(42)
random.shuffle(file_ids)

# 7:3 분할
split_ratio = 0.7
split_index = int(len(file_ids) * split_ratio)
train_ids = file_ids[:split_index]
val_ids = file_ids[split_index:]

# 저장
with open(os.path.join(imageset_dir, "train.txt"), "w") as f:
    f.write("\n".join(train_ids))

with open(os.path.join(imageset_dir, "val.txt"), "w") as f:
    f.write("\n".join(val_ids))

print(f"✅ 완료: train={len(train_ids)} / val={len(val_ids)}")
