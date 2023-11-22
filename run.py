#Prepare assets
#If you ran out of RAM this means that the video is too large. You can shorten it above.

center_video_to_body = False 
crop_video_to_body = False 
video_crop_expansion_factor = 1.05 
center_image_to_body = True 
crop_image_to_body = False 
image_crop_expansion_factor = 1.05 
video_crop_expansion_factor = max(video_crop_expansion_factor, 1)
image_crop_expansion_factor = max(image_crop_expansion_factor, 1)

import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
from IPython.display import HTML, clear_output
import cv2
import shutil
import os
import warnings
from IPython.display import HTML, clear_output
from base64 import b64encode

warnings.filterwarnings("ignore")

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def fix_dims(im):
    if im.ndim == 2:
        im = np.tile(im[..., None], [1, 1, 3])
    return im[...,:3]

def get_crop(im, center_body=True, crop_body=True, expansion_factor=1, rects=None):
    im = fix_dims(im)
    if (center_body or crop_body) and rects is None:
        rects, _ = hog.detectMultiScale(im, winStride=(4, 4),padding=(8,8), scale=expansion_factor)
    if (center_body or crop_body) and rects is not None and len(rects):
        x0,y0,w,h = sorted(rects, key=lambda x: x[2]*x[3])[-1]
        if crop_body:
            x0 += w//2-h//2
            x1 = x0+h
            y1 = y0+h
        else:
            img_h,img_w = im.shape[:2]
            x0 += (w-img_h)//2
            x1 = x0+img_h
            y0 = 0
            y1 = img_h
    else:
        h,w = im.shape[:2]
        x0 = (w-h)//2
        x1 = (w+h)//2
        y0 = 0
        y1 = h
    return int(x0),int(x1),int(y0),int(y1)

def pad_crop_resize(im, x0=None, x1=None, y0=None, y1=None, new_h=256, new_w=256):
    im = fix_dims(im)
    h,w = im.shape[:2]
    if x0 is None:
      x0 = 0
    if x1 is None:
      x1 = w
    if y0 is None:
      y0 = 0
    if y1 is None:
      y1 = h
    if x0<0 or x1>w or y0<0 or y1>h:
        im = np.pad(im, pad_width=[(max(-y0,0),max(y1-h,0)),(max(-x0,0),max(x1-w,0)),(0,0)], mode='edge')
    return resize(im[max(y0,0):y1-min(y0,0),max(x0,0):x1-min(x0,0)], (new_h, new_w))

image_path = './data/images/jumong_body.jpg'
video_path = './data/video/supershy_shorts_28s.mp4'

source_image = imageio.imread(image_path)
source_image = pad_crop_resize(source_image, *get_crop(source_image, center_body=center_image_to_body, crop_body=crop_image_to_body, expansion_factor=image_crop_expansion_factor))
imageio.imwrite('./data/images/crop.jpg', (source_image*255).astype(np.uint8))

#shutil.rmtree('C:/Users/ysdoh/OneDrive/바탕 화면/대학/2023-2/Track_project/imspector/impersonator/data/images', ignore_errors=True)
#os.makedirs('C:/Users/ysdoh/OneDrive/바탕 화면/대학/2023-2/Track_project/imspector/impersonator/data/images')


with imageio.get_reader(video_path, format='mp4') as reader:
  fps = reader.get_meta_data()['fps']

  driving_video = []
  rects = None
  try:
      for i,im in enumerate(reader):
          if not crop_video_to_body:
              break
          rects, _ = hog.detectMultiScale(im, winStride=(4, 4),padding=(8,8), scale=video_crop_expansion_factor)
          if rects is not None and len(rects):
              break
      x0,x1,y0,y1 = get_crop(im, center_body=center_video_to_body, crop_body=crop_video_to_body, expansion_factor=video_crop_expansion_factor, rects=rects)
      reader.set_image_index(0)
      for j,im in enumerate(reader):
          driving_video.append(pad_crop_resize(im,x0,x1,y0,y1))
          imageio.imwrite(os.path.join('./data/images/','%05d.jpg'%j), (driving_video[-1]*255).astype(np.uint8))
  except RuntimeError:
      pass

def vid_display(source, driving, generated=None):
    fig = plt.figure(figsize=(8 + 4 * (generated is not None), 6))

    ims = []
    for i in range(len(driving)):
        cols = [source]
        cols.append(driving[i])
        if generated is not None:
            cols.append(generated[i])
        im = plt.imshow(np.concatenate(cols, axis=1), animated=True)
        plt.axis('off')
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)
    plt.close()
    return ani

clear_output()
if rects is not None and len(rects):
    print('first found body in frame %d'%i)
print('number of frames: %d'%j)
#HTML(vid_display(source_image, driving_video).to_html5_video())


##### Animate

bg_ks = 13 #@param {type:"slider", min:1, max:25, step:2}
ft_ks = 3 #@param {type:"slider", min:1, max:25, step:2}
has_detector = True #@param {type:"boolean"}
post_tune = True #@param {type:"boolean"}
front_warp = True #@param {type:"boolean"}
cam_strategy = 'smooth' #@param ['smooth', 'source', 'copy'] {type:"raw"}

run = 'python run_imitator.py --gpu_ids 0 --model imitator --output_dir ./outputs/results --src_path ./data/images/crop.jpg --tgt_path /content/images --save_res --bg_ks %d --ft_ks %d %s %s %s --cam_strategy %s'%(
    bg_ks, ft_ks, '--has_detector' if has_detector else '', '--post_tune' if post_tune else '', '--front_warp' if front_warp else '', cam_strategy)

os.system(run)