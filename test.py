import os
import datetime

def data_preprocessing(img_path, video_path, rec_path):
    new_path = img_path.replace("image.png", "").strip()
    print(new_path)
    with open(new_path + 'cover.txt', 'w') as f:
        f.write(img_path)
        f.write(video_path)
        f.write(rec_path)
        f.close
    
    return new_path + 'cover.txt'