# 阿秋
# 2024/3/28  18:33
import os
from PIL import Image
import numpy as np
import random

train_split = 0.6
val_split = 0.2
path = "/your_dir"
train_txt = "/media/ps/qiu/WHDLD/segmentaion_sets/train.txt"
val_txt = "/media/ps/qiu/WHDLD/segmentaion_sets/val.txt"
test_txt = "/media/ps/qiu/WHDLD/segmentaion_sets/test.txt"

image_path = os.path.join(path, 'Labels')

lis = os.listdir(image_path)
select_train = random.sample(lis, int(len(os.listdir(image_path))*train_split))
remain = [i for i in lis if i not in select_train]
select_val = random.sample(remain, int(len(os.listdir(image_path))*val_split))
select_test = [i for i  in lis if i not in select_train and i not in select_val]


with open(train_txt, 'r') as file:
    head = file.read()
    if head:
        with open(train_txt, 'w', encoding='utf-8') as empty:
            empty.write('')
            for str1 in select_train:
                empty.write(str1.split('.')[0] + '\n')
with open(val_txt, 'r') as file:
    head = file.read()
    if head:
        with open(val_txt, 'w', encoding='utf-8') as empty:
            empty.write('')
            for str1 in select_val:
                empty.write(str1.split('.')[0] + '\n')
with open(test_txt, 'r') as file:
    head = file.read()
    if head:
        with open(test_txt, 'w', encoding='utf-8') as empty:
            empty.write('')
            for str1 in select_test:
                empty.write(str1.split('.')[0] + '\n')


