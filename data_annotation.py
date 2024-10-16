import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

trainval_percent    = 0.8
train_percent       = 0.75

dataset_path      = 'landcover.ai' #'WHLDLD'
input_size = 256

def GraytoRGB():
    label_to_color = {
    0: (0, 0, 0),       # block  background
    1: (255, 0, 0),     # red    buliding
    2: (0, 255, 0),     # green  woodland
    3: (0, 255, 255),   # buld   water
    }
    
    # 将label_to_color字典转换为调色板列表
    palette = [0] * 256 * 3
    for label, color in label_to_color.items():
        palette[label * 3:label * 3 + 3] = color
    
    input_folder = "/yourdirs/landcover.ai/Labels_m/"
    output_folder = "/yourdirs/landcover.ai/Labels/"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    file_list = [f for f in os.listdir(input_folder) if f.endswith('.png')]

    for filename in tqdm(file_list, desc="Processing images"):
        input_image_path = os.path.join(input_folder, filename)
        output_image_path = os.path.join(output_folder, filename)

        gray_image = Image.open(input_image_path).convert('L')

        palette_image = gray_image.convert('P')
        palette_image.putpalette(palette)

        palette_image.save(output_image_path)

if __name__ == "__main__":
    if dataset_path == 'WHDLD':
        random.seed(0)
        print("Generate txt in ImageSets.")
        segfilepath     = os.path.join(dataset_path, 'Labels')
        saveBasePath    = os.path.join(dataset_path, "segmentaion_sets")
        
        temp_seg = os.listdir(segfilepath)
        total_seg = []
        for seg in temp_seg:
            if seg.endswith(".png"):
                total_seg.append(seg)
    
        # 4940
        num     = len(total_seg)
        list    = range(num)
        tv      = int(num*trainval_percent)
        tr      = int(tv*train_percent)  
        trainval= random.sample(list,tv)  
        train   = random.sample(trainval,tr)  
        
        print("train and val size",tv)
        print("traub suze",tr)
        ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
        with open(os.path.join(saveBasePath,'trainval.txt'), 'r') as file:
            head = file.read()
            if head:
                ftrainval.write('')
    
        ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
        with open(os.path.join(saveBasePath,'test.txt'), 'r') as file:
            head = file.read()
            if head:
                ftest.write('')
    
        ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
        with open(os.path.join(saveBasePath,'train.txt'), 'r') as file:
            head = file.read()
            if head:
                ftrain.write('')
    
        fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')
        with open(os.path.join(saveBasePath,'val.txt'), 'r') as file:
            head = file.read()
            if head:
                fval.write('')
    
        
        for i in list:  
            name = total_seg[i][:-4]+'\n'  
            if i in trainval:  
                ftrainval.write(name)  
                if i in train:  
                    ftrain.write(name)  
                else:  
                    fval.write(name)  
            else:  
                ftest.write(name)
        # with open(os.path.join(saveBasePath,'trainval.txt'), 'r') as file:
        #     print(file.readlines())
        ftrainval.close()  
        ftrain.close()  
        fval.close()  
        ftest.close()
        print("Generate txt in ImageSets done.")
        
    if dataset_path == 'landcover.ai':
        random.seed(0)
        print("Generate txt in ImageSets.")
        
        GraytoRGB()
        print("Convert successfully.")
        
        segfilepath     = os.path.join(dataset_path, 'Labels')
        saveBasePath    = os.path.join(dataset_path, "segmentaion_sets")
        
        temp_seg = os.listdir(segfilepath)
        total_seg = []
        for seg in temp_seg:
            if seg.endswith(".png"):
                total_seg.append(seg)
    
        # 
        num     = len(total_seg)
        print(num)
        list    = range(num)
        tv      = int(num*trainval_percent)
        tr      = int(tv*train_percent)  
        trainval= random.sample(list,tv)  
        train   = random.sample(trainval,tr)  
        
        print("train and val size",tv)
        print("traub suze",tr)
        ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
        with open(os.path.join(saveBasePath,'trainval.txt'), 'r') as file:
            head = file.read()
            if head:
                ftrainval.write('')
    
        ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
        with open(os.path.join(saveBasePath,'test.txt'), 'r') as file:
            head = file.read()
            if head:
                ftest.write('')
    
        ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
        with open(os.path.join(saveBasePath,'train.txt'), 'r') as file:
            head = file.read()
            if head:
                ftrain.write('')
    
        fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')
        with open(os.path.join(saveBasePath,'val.txt'), 'r') as file:
            head = file.read()
            if head:
                fval.write('')
    
        
        for i in list:  
            name = total_seg[i][:-4]+'\n'  
            if i in trainval:  
                ftrainval.write(name)  
                if i in train:  
                    ftrain.write(name)  
                else:  
                    fval.write(name)  
            else:  
                ftest.write(name)
        # with open(os.path.join(saveBasePath,'trainval.txt'), 'r') as file:
        #     print(file.readlines())
        ftrainval.close()  
        ftrain.close()  
        fval.close()  
        ftest.close()
        print("Generate txt in ImageSets done.")
    

    print("Check datasets format, this may take a while.")
    classes_nums        = np.zeros([input_size], dtype=int)
    for i in tqdm(list):
        name            = total_seg[i]
        png_file_name   = os.path.join(segfilepath, name)
        if not os.path.exists(png_file_name):
            raise ValueError("Label image %s not detected, please check if the file exists in the specific path and if the suffix is png."%(png_file_name))
        png             = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            print("Labeled images %s have a shape of %s and are not greyscale or eight-bit colour images."%(name, str(np.shape(png))))
            print("The label image needs to be either greyscale or octet colour, and the value of each pixel point of the label is the category to which the pixel point belongs."%(name, str(np.shape(png))))

        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=input_size)
            
    print("Prints the value and number of pixel points.")
    print('-' * 37)
    print("| %15s | %15s |"%("Key", "Value"))
    print('-' * 37)
    for i in range(input_size):
        if classes_nums[i] > 0:
            print("| %15s | %15s |"%(str(i), str(classes_nums[i])))
            print('-' * 37)
    
    if classes_nums[255] > 0 and classes_nums[0] > 0 and np.sum(classes_nums[1:255]) == 0:
        print("It was detected that the value of the pixel point in the label contains only 0 and 255, and that there is an error in the data format.")
    elif classes_nums[0] > 0 and np.sum(classes_nums[1:]) == 0:
        print("There is an error in the data format, please double check the dataset format.")