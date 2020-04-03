# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 20:19:09 2020

Purpose is to read text from a scarbble board. Using screenshots from the Scrabble Go app, and a game I was playing.

@author: 61920832 - Mithra Kruthiventi
"""

from PIL import Image
import pytesseract
import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb
import math

pytesseract.pytesseract.tesseract_cmd= r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Then, build the small 10x10x3 squares, filled with the respective color. You can use NumPy to easily fill the squares with the color:

def cropper(image):
    cropy=[]
    for m in range(image.shape[0]):
        cnt = 0
        for mx in range(image.shape[1]):
            if(image[m,mx]==255):
                cnt= cnt+1
        if(cnt>= 0.3*image.shape[1]):
            cropy.append(m)
    
    cropx=[]
    for m in range(image.shape[1]):
        cnt = 0
        for my in range(image.shape[0]):
            if(image[my,m]==255):
                cnt= cnt+1
        if(cnt>= 0.3*image.shape[1]):
            cropx.append(m)
    return cropy, cropx

def masker(rgb_im):
    hsv_im = cv2.cvtColor(rgb_im, cv2.COLOR_RGB2HSV)
    
    light_orange = (0, 0, 255)
    dark_orange = (160, 20, 255)
    
    lo_square = np.full((10, 10, 3), light_orange, dtype=np.uint8) / 255.0
    do_square = np.full((10, 10, 3), dark_orange, dtype=np.uint8) / 255.0
    
    mask = cv2.inRange(hsv_im, light_orange, dark_orange)
    
    plt.subplot(1, 2, 1)
    plt.imshow(mask, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(rgb_im)
    plt.show()
    
    return mask

im= cv2.imread("C://Users//61920832//Desktop//Scrabble//nirvana.jpeg")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

mask1 = masker(im)

cropy, cropx = cropper(mask1)

new=im[min(cropy):max(cropy),min(cropx):max(cropx)]
plt.imshow(new)

new_pil= Image.fromarray(new)

length_x, width_y = new_pil.size
factor = min(1, float(1024.0 / length_x))
size = int(factor * length_x), int(factor * width_y)
im_resized = new_pil.resize(size, Image.ANTIALIAS)
im_resized.save("C://Users//61920832//Desktop//Scrabble//nirvana_temp.jpeg", dpi=(300, 300))

base_board=cv2.imread("C://Users//61920832//Desktop//Scrabble//nirvana_temp.jpeg")
base_board = cv2.cvtColor(base_board, cv2.COLOR_BGR2RGB)

mask2 = masker(base_board)
cry, crx = cropper(mask2)

# plt.imshow(base_board)
# for c in range(len(crx)-1):
#     if(crx[c]+1<crx[c+1] or crx[c]-1>crx[c-1]):
#         plt.plot([crx[c], crx[c]],[0, base_board.shape[1]], 'k-', lw=1)
# for c in range(len(cry)-1):
#     if(cry[c]+1<cry[c+1] or cry[c]-1>cry[c-1]):
#         plt.plot([0, base_board.shape[0]],[cry[c], cry[c]], 'k-', lw=1)
# plt.show()

co_x, co_y = [],[]
for c in range(len(crx)-1):
    if(crx[c]+1<crx[c+1] or crx[c]-1>crx[c-1]):
        co_x.append(crx[c])
for c in range(len(cry)-1):
    if(cry[c]+1<cry[c+1] or cry[c]-1>cry[c-1]):
        co_y.append(cry[c])

def aread(y_1,y_2,x_1,x_2):
    return math.sqrt(pow((y_2-y_1),2) + pow((x_2-x_1),2))

def area_filter(co_x, co_y):
    im_map =[]
    coordinates=[]
    ster = 0
    for x in range(len(co_x)-1):
        for y in range(len(co_y)-1):
            if(aread(co_y[y],co_y[y+1],co_x[x],co_x[x+1])>70):
                im_map.append(base_board[co_y[y]:co_y[y+1],co_x[x]:co_x[x+1]])
                coordinates.append([ster,y,y+1,x,x+1, co_y[y],co_y[y+1],co_x[x],co_x[x+1],aread(co_y[y],co_y[y+1],co_x[x],co_x[x+1])])
                ster = ster + 1
    return im_map, coordinates

im_m, coords=area_filter(co_x, co_y)

print(len(im_m))
print(len(coords))

# taken from https://gist.github.com/mattjmorrison/932345
def trim(im, border):
    bg = Image.new(im.mode, im.size, border)
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

def image_to_texts(im):
    new_size = tuple(4*x for x in im.size)
    im = im.resize(new_size, Image.ANTIALIAS)

    enhancer = ImageEnhance.Contrast(im)
    im = enhancer.enhance(2)
    im = im.convert('L')
    im = ImageOps.invert(im)
    im = im.point(lambda x: 0 if x < 100 else 255)
    im = im.filter(ImageFilter.SMOOTH_MORE)
    im = trim(im, 255)

    return im, pytesseract.image_to_string(im, lang='eng', config='--psm 6')

tiles , letters = [],[]
for t in range(len(im_m)):
    Image.fromarray(im_m[t]).save('C://Users//61920832//Desktop//Scrabble//temp.png', dpi=(300,300))
    tile= cv2.imread("C://Users//61920832//Desktop//Scrabble//temp.png")
    tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
    im_t, txt_t = image_to_texts(Image.fromarray(tile[5:46,]))
    tiles.append(txt_t)
    letters.append(im_t)
    im_t.save('C://Users//61920832//Desktop//Scrabble//images//'+str(t)+"".join([character for character in txt_t if character.isalnum()])+'tile.jpeg', dpi=(300,300))
    print(t)
