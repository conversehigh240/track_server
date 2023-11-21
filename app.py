from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import yt_dlp
import os
import test
import datetime
import shutil

app = Flask(__name__)

@app.route('/ai/saveFile', methods = ['GET', 'POST'])
def saveFile():
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    path = 'static/' + timestamp + '/'
    if not (os.path.isdir(path)):
        os.makedirs(path)

    if request.method == 'POST':
        img = request.files['img']
        img_name, img_extension = os.path.splitext(img.filename)
        img_extension = '.png'
        img.save(path + secure_filename('image') + img_extension)

        vid = request.files['vid']
        vid_name, vid_extension = os.path.splitext(vid.filename)
        vid_extension = '.mp4'
        vid.save(path + secure_filename('video') + vid_extension)
        
        rec = request.files['rec']
        rec_name, rec_extension = os.path.splitext(rec.filename)
        rec_extension = '.m4a'
        rec.save(path + secure_filename('record') + rec_extension)

        img_path = path + 'image.png'
        video_path = path + 'video.mp4'
        rec_path = path + 'record.m4a'

        if os.path.isfile(img_path) and os.path.isfile(video_path) and os.path.isfile(rec_path):
            test.data_preprocessing(img_path, video_path, rec_path)

        if os.path.exists(path + 'cover.txt'):
            os.remove(img_path)
            os.remove(video_path)
            os.remove(rec_path)
            
        return "completed"
    

@app.route('/ai/saveFile-YouTube', methods = ['GET', 'POST'])
def saveYouTube():
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    path = 'static/' + timestamp + '/'
    if not (os.path.isdir(path)):
        os.makedirs(path)

    if request.method == 'POST':
        img = request.files['img']
        img_name, img_extension = os.path.splitext(img.filename)
        img_extension = '.png'
        img.save(path + secure_filename('image') + img_extension)

        url = request.form['url']
        ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best',
        'outtmpl': path + 'video.mp4'
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        rec = request.files['rec']
        rec_name, rec_extension = os.path.splitext(rec.filename)
        rec_extension = '.m4a'
        rec.save(path + secure_filename('record') + rec_extension)

        img_path = path + 'image.png'
        video_path = path + 'video.mp4'
        rec_path = path + 'record.m4a'

        if os.path.isfile(img_path) and os.path.isfile(video_path) and os.path.isfile(rec_path):
            test.data_preprocessing(img_path, video_path, rec_path)

        if os.path.exists('static/cover/test.txt'):
            shutil.rmtree(path)

    return "completed"
    

@app.route('/ai/deleteCover', methods=['GET'])
def deleteCover():
    os.remove('static/cover.mp4')
    return "covered video deleted successfully"

if __name__ == '__main__':
    app.run(host = '0.0.0.0')