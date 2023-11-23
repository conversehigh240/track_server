from pytube import YouTube
import os
from moviepy.editor import VideoFileClip
from pydub import AudioSegment

PATH = 'Data'
# img_path = PATH + '/' + 'wheein_photo.png'
cap_path = 'video/blue_check.mp4'
video_url = 'https://www.youtube.com/shorts/7n1kJvwXqt8' #유튜브 링크
voi_path = 'raw_data'

#유튜브 링크로 커버할 영상 다운받기
def download_video(video_url, video_name):
    yt = YouTube(video_url)
    yt.streams.filter(res='720p', file_extension='mp4').first().download(output_path=cap_path, filename=video_name)

def cap_sep(PATH, cap):
    #1) 커버할 영상 소리 제거(mp4 -> mp4)
    os.makedirs(PATH, exist_ok=True)
    new_clip = cap.without_audio()
    new_clip.write_videofile(PATH + '/video_without_sound.mp4')

    #2) 커버할 영상에서 음성 추출(mp4 -> mp3)
    os.makedirs(PATH + '/cover_music', exist_ok=True)
    cap.audio.write_audiofile(PATH + '/cover_music/cover_audio.mp3')

# download_video(video_url, 'ive_baddie.mp4') #영상 제목 설정
cap = VideoFileClip(cap_path)
cap_sep('video', cap)

# files
src = "video/cover_music/cover_audio.mp3"
dst = "video/cover_music/cover_audio.wav"

# convert wav to mp3
audSeg = AudioSegment.from_mp3(src)
audSeg.export(dst, format="wav")