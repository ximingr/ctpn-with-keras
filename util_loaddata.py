
from __future__ import absolute_import, division, print_function

import cv2
import numpy as np

import sys
import os

from util_cache_func import cache_it

'''
read image from array.

#read the data from the file
with open(somefile, 'rb') as infile:
     buf = infile.read()

#use numpy to construct an array from the bytes
x = np.fromstring(buf, dtype='uint8')

#decode the array into an image
img = cv2.imdecode(x, cv2.IMREAD_UNCHANGED)

#show it
cv2.imshow("some window", img)
cv2.waitKey(0)
'''


@cache_it()
def load_image_file(file, max_h=None, max_w=None):

    image = cv2.imread( file )

    h = image.shape[0]
    w = image.shape[1]

    ratio = 1.0
    if max_h and max_w and h > max_h:
        ratio = max_h/h

        w = int(w*ratio)
        h = int(h*ratio)

        image = cv2.resize(image, (w, h))

    return ratio, image

# 小文件，压缩的意义不大
@cache_it(compress=False)
def _load_label_file(file, ratio = 1.0):

    boxes = []

    with open( file ) as fh:
        data_lines = fh.readlines()

        for line in data_lines:
            line_data = line.strip().split(",")
            label = line_data[-1]
            line_data = line_data[:8]
            if len(line_data) != 8:
                print("unknown format", file, line)
                continue

            box = []
            for off in range(0,8,2):
                x, y = line_data[off:off+2]
                x = int(x) * ratio
                y = int(y) * ratio
                box.append( (int(x), int(y)) )

            boxes.append([box, label])

    return boxes

@cache_it()
def load_data_folder(folder):
    '''
    input:
        "folder" which contains
            - .txt for box&label
            - .jpg for images

    return: list of,

        key (filename without path or extension), data_image, list of box&label
    '''
    data_file = {}
    label_file = {}

    for root, _, files in os.walk(folder, topdown=False):

        for name in files:
            if name.endswith(".txt"):
                label_file[name[:-4]] = root
            elif name.endswith(".jpg"):
                data_file[name[:-4]] = root

    results = []
    for ele in data_file:
        if ele not in label_file:
            print("WARNING: {} is not found with a label.".format(ele), file=sys.stderr)
            continue

        _, data = load_image_file( os.sep.join([data_file[ele], ele+".jpg"]) )

        boxes = _load_label_file( os.sep.join([label_file[ele], ele+".txt"]) )

        results.append([ele, data, boxes])

    return results


if __name__ == "__main__":
    # res = load_data_folder("/data/9.work/ctpn-data/ICDAR2019/0325-task1-subset1")
    # print( len(res) )
    res = load_data_folder("/data/9.work/ctpn-data/ICDAR2019/0325-task1-sub000")
    print( len(res) )

