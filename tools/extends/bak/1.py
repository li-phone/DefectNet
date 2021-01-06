import shutil
import os
from tqdm import tqdm
import glob

dir_id = 1
imgs = glob.glob(r"E:\data\images\detections\fabric_defects\coco_fabric\plain_fabric\fabric\*\*.jpg")
imgs.sort()
for i in tqdm(range(len(imgs))):
    src = imgs[i]
    dst = os.path.join(
        r"E:\data\images\detections\fabric_defects\coco_fabric\plain_fabric\fabric\trainval.part.{}".format(
            str(dir_id)),
        os.path.basename(src))
    if (i + 1) % 2500 == 0:
        dir_id += 1
    shutil.move(src, dst)
