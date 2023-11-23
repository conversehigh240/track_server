from types import SimpleNamespace
from main import inference

# configure setting
configures = {
    'model_path'            :   'exp/combsub-test/model_last.pt', # 추론에 사용하고자 하는 모델, 바로위에서 학습한 모델을 가져오면댐
    'input'                 :   'video/cover_music/cover_audio.wav', # 추론하고자 하는 노래파일의 위치 - 님들이 바꿔야댐 
    'output'                :   'output/jumong_bluecheck.wav',  # 결과물 파일의 위치
    'device'                :   'cpu',
    'spk_id'                :   '1', 
    'spk_mix_dict'          :   'None', 
    'key'                   :   '0', 
    'enhance'               :   'true' , 
    'pitch_extractor'       :   'crepe' ,
    'f0_min'                :   '50' ,
    'f0_max'                :   '1100',
    'threhold'              :   '-60',
    'enhancer_adaptive_key' :   '0'
}
cmd = SimpleNamespace(**configures)

inference(cmd)

