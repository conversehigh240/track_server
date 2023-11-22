from flask import Flask, request, session
from werkzeug.utils import secure_filename
import yt_dlp
import os
import test
import datetime
import shutil

app = Flask(__name__)
app.secret_key = '8GAHkf10u2'

@app.route('/ai/saveFile', methods = ['GET', 'POST'])
def saveFile():
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    path = 'static/' + timestamp
    if not (os.path.isdir(path)):
        os.makedirs(path)

    session['my_path'] = path

    if request.method == 'POST':
        img = request.files['img']
        img_name, img_extension = os.path.splitext(img.filename)
        img_extension = '.png'
        img.save(path + '/' + secure_filename('image') + img_extension)

        vid = request.files['vid']
        vid_name, vid_extension = os.path.splitext(vid.filename)
        vid_extension = '.mp4'
        vid.save(path + '/' + secure_filename('video') + vid_extension)
        
        rec = request.files['rec']
        rec_name, rec_extension = os.path.splitext(rec.filename)
        rec_extension = '.m4a'
        rec.save(path + '/' + secure_filename('record') + rec_extension)

        img_path = path + '/image.png'
        video_path = path + '/video.mp4'
        rec_path = path + '/record.m4a'

        if os.path.isfile(img_path) and os.path.isfile(video_path) and os.path.isfile(rec_path):
            test.data_preprocessing(img_path, video_path, rec_path)
            
        return "completed"
    

@app.route('/ai/saveFile-YouTube', methods = ['GET', 'POST'])
def saveYouTube():
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    path = 'static/' + timestamp
    if not (os.path.isdir(path)):
        os.makedirs(path)

    session['my_path'] = path

    if request.method == 'POST':
        img = request.files['img']
        img_name, img_extension = os.path.splitext(img.filename)
        img_extension = '.png'
        img.save(path + '/' + secure_filename('image') + img_extension)

        url = request.form['url']
        ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best',
        'outtmpl': path + '/video.mp4'
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        rec = request.files['rec']
        rec_name, rec_extension = os.path.splitext(rec.filename)
        rec_extension = '.m4a'
        rec.save(path + '/' + secure_filename('record') + rec_extension)

        img_path = path + '/image.png'
        video_path = path + '/video.mp4'
        rec_path = path + '/record.m4a'

        if os.path.isfile(img_path) and os.path.isfile(video_path) and os.path.isfile(rec_path):
            test.data_preprocessing(img_path, video_path, rec_path)

    return "completed"

@app.route('/ai/checkFile', methods=['GET'])
def checkFile():
    if 'my_path' in session:
        path = session['my_path']
        cover_path = path + '/cover.txt'

        if os.path.exists(cover_path):
            return "True"
        else:
            return "False"
    else:
        return "path error"
    
@app.route('/ai/getFile', methods=['GET'])
def getFile():
    path = session['my_path']
    cover_path = path + '/video.mp4'

    return "http://163.180.160.104:5001/" + cover_path
    

@app.route('/ai/deleteFile', methods=['GET'])
def deleteCover():
    path = session['my_path']
    shutil.rmtree(path)
    
    return "file deleted successfully"

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port=5001)