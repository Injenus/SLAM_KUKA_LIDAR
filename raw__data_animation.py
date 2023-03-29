from PIL import Image
import cv2
import numpy as np
import glob

res_file_name = 'all_data.csv'
direc = 'frame by frame/'

frameSize = (500, 500)
out = cv2.VideoWriter('MOVE.avi', cv2.VideoWriter_fourcc(*'DIVX'), 60,
                      frameSize)
for filename in glob.glob(direc + '*.jpg'):
    img = cv2.imread(filename, IMREAD_REDUCED_GRAYSCALE_2=16)
    out.write(img)
out.release()

frames = []
for i in range(1600):
    frame = Image.open(direc + '{}_{}.jpg'.format(res_file_name[:-4], i))
    frames.append(frame)
frames[0].save(
    direc + 'MOVE.gif',
    save_all=True,
    append_images=frames[1:],
    optimize=True,
    duration=150,
    loop=0
)
