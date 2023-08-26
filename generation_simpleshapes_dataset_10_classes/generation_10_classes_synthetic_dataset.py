import numpy as np
from matplotlib import pyplot as plt
import io
import cv2
from pathlib import Path
import os
import shutil
import argparse

from shapes_generation import triangle_with_color, rectangle_with_color
from shapes_generation import circle_with_color, hexagon_with_color, trapezoid_with_color
from shapes_generation import star_with_color, cross_with_color, heart_with_color
from shapes_generation import moon_with_color, cloud_with_color


parser = argparse.ArgumentParser()
parser.add_argument('--path_to_backgrounds', type=str, dafault='/home/e_radionova/Datasets/SimpleShapes_10_classes/backgrounds_300x400',
                    help='Path to set of backgrounds')
args = parser.parse_args()

# Support dicts
dict_colors = {'green': (1,128,1), 
               'black': (1, 1, 1), 
               'blue': (1,1,255), 
               'brown': (139,69,19), 
               'yellow': (255,255,1), 
               'deeppink': (255,20,147), 
               'orange': (255,165,1), 
               'purple': (138,43,226), 
               'white': (254,254,254),
               'pink': (255,192,203)
              }

figures_dct = {'triangle': triangle_with_color,
              'rectangle': rectangle_with_color,
              'circle': circle_with_color,
              'hexagon': hexagon_with_color,
              'trapezoid': trapezoid_with_color,
              'star': star_with_color,
              'cross': cross_with_color,
              'moon': moon_with_color,
              'cloud': cloud_with_color}

# Functions
def get_set_colors(dict_colors, background_names):
    color_count = {key:[] for key in dict_colors.keys()}
    for color in dict_colors.keys():
        for back_name in background_names:
            if color in back_name:
                color_count[color].append(back_name.split('.')[0])
            else:
                if 'multi' in back_name:
                    color_count['deeppink'].append(back_name.split('.')[0])
    color_count['deeppink'] = list(np.unique(color_count['deeppink']))
    return color_count

def generate_figures(dict_colors, color_count, imgs_folder='./raw_data/images', masks_foled='./raw_data/masks'):
    for color in dict_colors.keys():
        lst_backs = color_count[color]
        for name_back_color in lst_backs:
            for idx in range(NUM_PER_CLASS):
                for key_shape in figures_dct.keys():
                    img, mask = figures_dct[key_shape](WIDTH, HEIGHT, color)
                    np.save(f'{imgs_folder}/{name_back_color}_{key_shape}_{idx}.npy', img)
                    cv2.imwrite(f'{masks_foled}/{name_back_color}_{key_shape}_{idx}.png', mask)
    print('Generation is finished')
    
def connect_imgs_with_back(path_background, path_front_img):
    background = cv2.imread(path_background)
    front_img = np.load(path_front_img)
    front_img = cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB)
    name_img = Path(path_front_img).stem
    print(name_img)
    img_common = np.zeros_like(background)
    for i in range(front_img.shape[0]):
        for j in range(front_img.shape[1]):
            for ch in range(front_img.shape[2]):
                if front_img[i][j][ch] != 0:
                    img_common[i][j][ch] = front_img[i][j][ch]
                else:
                    img_common[i][j][ch] = background[i][j][ch]
    cv2.imwrite(f'./dataset/images/{name_img}.jpg', img_common)
    
# Main code for generation
NUM_PER_CLASS = 10 # 48 backgrounds and 10 shapes
WIDTH = 300
HEIGHT = 400

folders_pathes = ['./raw_data/images', './raw_data/masks', './dataset/images/', './dataset/masks/']
for path in folders_pathes:
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)

backgrounds_folder = Path(args.path_to_backgrounds)
background_names = np.sort(os.listdir(backgrounds_folder)).tolist()

color_count = get_set_colors(dict_colors, background_names)
generate_figures(dict_colors, color_count)

imgs_folder = './raw_data/images'
for color in dict_colors.keys():
    for color_name in color_count[color]:  
        pathes_color = []
        for figure in figures_dct.keys():
            for j in range(NUM_PER_CLASS):
                pathes_color.append(str(imgs_folder) + f'/{color_name}_{figure}_{j}.npy')
                
        for path in pathes_color:
            connect_imgs(f'{str(backgrounds_folder)}/{color_name}.jpg', path)
    
masks_folder = Path('./raw_data/masks')
dataset_masks = Path('./dataset/masks/')
mask_files = sorted(os.listdir(masks_folder))
for file in mask_files: 
    shutil.copyfile(masks_foled / file, dataset_masks / file)
