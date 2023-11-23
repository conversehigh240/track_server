from preprocess import preprocess
from ddsp.vocoder import F0_Extractor, Volume_Extractor, Units_Encoder
from diffusion.vocoder import Vocoder
import os
import librosa
import torch
from logger import utils
from tqdm import tqdm
from glob import glob
from pydub import AudioSegment
from logger.utils import traverse_dir
from pydub import AudioSegment
import os
import torchaudio
from sep_wav import demucs
from sep_wav import audio_norm
import subprocess
from sep_wav import get_ffmpeg_args
import subprocess
from draw import main
from preprocess import preprocess
from train import ddsp_train
import multiprocessing
import argparse

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Cuda setting
    device = 'cpu'

    # configure loading
    args = utils.load_config('./configs/combsub.yaml')

    # set path
    # MP4_DATA_PATH   = 'preprocess/mp4'
    ORIGINAL_PATH   = 'preprocess/original/'
    DEMUCS_PATH     = 'preprocess/demucs/'
    NORM_PATH       = 'preprocess/norm/'
    TEMP_LOG_PATH   = 'temp_ffmpeg_log.txt'  # ffmpeg의 무음 감지 로그의 임시 저장 위치
    TENSOR_2D = 'preprocess/2d_tensor/'

    def convert_m4a_to_wav(m4a_path, wav_path):
        # Make sure the output directory exists
        os.makedirs(wav_path, exist_ok=True)

        # List all m4a files in the input directory
        m4a_files = [f for f in os.listdir(m4a_path) if f.endswith('.m4a')]

        for m4a_file in m4a_files:
            # Construct the full paths
            m4a_full_path = os.path.join(m4a_path, m4a_file)
            wav_filename = os.path.splitext(m4a_file)[0] + '.wav'
            wav_full_path = os.path.join(wav_path, wav_filename)

            # Convert m4a to wav
            sound = AudioSegment.from_file(m4a_full_path, format='m4a')
            sound.export(wav_full_path, format='wav')

            print(f"Converted {m4a_file} to {wav_filename}")

    # Example usage
    m4a_path = 'raw_data'
    wav_path = 'preprocess/original'
    convert_m4a_to_wav(m4a_path, wav_path)

    def load_and_pad_audio(file_path, target_channels):
        waveform, sample_rate = torchaudio.load(file_path)

        # Check the number of channels
        num_channels = waveform.shape[0]

        # If the audio has fewer channels than the target, pad with zeros
        if num_channels < target_channels:
            padding_channels = target_channels - num_channels
            padding = torch.zeros((padding_channels, waveform.shape[1]))
            waveform = torch.cat([waveform, padding], dim=0)

        return waveform

    def save_audio(waveform, file_path, sample_rate):
        torchaudio.save(file_path, waveform, sample_rate)

    def process_folder(input_folder, output_folder, target_channels=2, sample_rate=44100):
        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # List all wav files in the input folder
        wav_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]

        for wav_file in wav_files:
            # Construct the full paths for input and output files
            input_file_path = os.path.join(input_folder, wav_file)
            output_file_path = os.path.join(output_folder, wav_file)

            # Load and pad the audio file
            waveform = load_and_pad_audio(input_file_path, target_channels)

            # Save the modified audio file
            save_audio(waveform, output_file_path, sample_rate)

    # Example usage
    input_folder = ORIGINAL_PATH
    output_folder = TENSOR_2D
    process_folder(input_folder, output_folder)

    demucs(TENSOR_2D, DEMUCS_PATH)

    for filepath in tqdm(glob(DEMUCS_PATH+"*.wav"), desc="노멀라이징 작업 중..."):
        filename = os.path.splitext(os.path.basename(filepath))[0]
        out_filepath = os.path.join(NORM_PATH, filename) + ".wav"
        audio_norm(filepath, out_filepath, sample_rate = 44100)

    for filepath in tqdm(glob(NORM_PATH+"*.wav"), desc="음원 자르는 중..."):
        duration = librosa.get_duration(filename=filepath)
        max_last_seg_duration = 0
        sep_duration_final = 15
        sep_duration = 15

        while sep_duration > 4:
            last_seg_duration = duration % sep_duration
            if max_last_seg_duration < last_seg_duration:
                max_last_seg_duration = last_seg_duration
                sep_duration_final = sep_duration
            sep_duration -= 1

        filename = os.path.splitext(os.path.basename(filepath))[0]
        out_filepath = os.path.join(args.data.train_path,"audio", f"{filename}-%04d.wav")
        subprocess.run(f'ffmpeg -i "{filepath}" -f segment -segment_time {sep_duration_final} "{out_filepath}" -y', capture_output=True, shell=True)

    for filepath in tqdm(glob(args.data.train_path+"/audio/*.wav"), desc="무음 제거 중..."):
        if os.path.exists(TEMP_LOG_PATH):
            os.remove(TEMP_LOG_PATH)

        ffmpeg_arg = get_ffmpeg_args(filepath)
        subprocess.run(ffmpeg_arg, capture_output=True, shell=True)

        start = None
        end = None

        with open(TEMP_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if "lavfi.silence_start" in line:
                    start = float(line.split("=")[1])
                if "lavfi.silence_end" in line:
                    end = float(line.split("=")[1])

        if start != None:
            if start == 0 and end == None:
                os.remove(filepath)
            
    if os.path.exists(TEMP_LOG_PATH):
            os.remove(TEMP_LOG_PATH)

    main()

    # get data
    sample_rate = args.data.sampling_rate
    hop_size = args.data.block_size

    # initialize f0 extractor
    f0_extractor = F0_Extractor(
                        args.data.f0_extractor, 
                        args.data.sampling_rate, 
                        args.data.block_size, 
                        args.data.f0_min, 
                        args.data.f0_max)

    # initialize volume extractor
    volume_extractor = Volume_Extractor(args.data.block_size)

    # initialize mel extractor
    mel_extractor = None
    if args.model.type == 'Diffusion':
        mel_extractor = Vocoder(args.vocoder.type, args.vocoder.ckpt, device = device)
        if mel_extractor.vocoder_sample_rate != sample_rate or mel_extractor.vocoder_hop_size != hop_size:
            mel_extractor = None
            print('Unmatch vocoder parameters, mel extraction is ignored!')

    # initialize units encoder
    if args.data.encoder == 'cnhubertsoftfish':
        cnhubertsoft_gate = args.data.cnhubertsoft_gate
    else:
        cnhubertsoft_gate = 10             
    units_encoder = Units_Encoder(
                        args.data.encoder, 
                        args.data.encoder_ckpt, 
                        args.data.encoder_sample_rate, 
                        args.data.encoder_hop_size, 
                        device = device)    

    # preprocess training set
    preprocess(args.data.train_path, f0_extractor, volume_extractor, mel_extractor, units_encoder, sample_rate, hop_size, device = device)

    # preprocess validation set
    preprocess(args.data.valid_path, f0_extractor, volume_extractor, mel_extractor, units_encoder, sample_rate, hop_size, device = device)
    
    def parse_args():
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-c",
            "--config",
            type=str,
            default='./configs/combsub.yaml',  # 기본값으로 사용할 파일의 경로
            help="path to the config file")
        return parser.parse_args()
    
    cmd = parse_args()
    args = utils.load_config(cmd.config)
    print(' > config:', cmd.config)
    print(' >    exp:', args.env.expdir)

    ddsp_train(args)

    # from types import SimpleNamespace
    # from main import inference
    # # configure setting
    # configures = {
    #     'model_path'            :   'exp/combsub-test/model_best.pt', # 추론에 사용하고자 하는 모델, 바로위에서 학습한 모델을 가져오면댐
    #     'input'                 :   'video/cover_music/cover_audio.wav', # 추론하고자 하는 노래파일의 위치 - 님들이 바꿔야댐 
    #     'output'                :   'output/jumong_bluecheck.wav',  # 결과물 파일의 위치
    #     'device'                :   'cuda',
    #     'spk_id'                :   '1', 
    #     'spk_mix_dict'          :   'None', 
    #     'key'                   :   '0', 
    #     'enhance'               :   'true' , 
    #     'pitch_extractor'       :   'crepe' ,
    #     'f0_min'                :   '50' ,
    #     'f0_max'                :   '1100',
    #     'threhold'              :   '-60',
    #     'enhancer_adaptive_key' :   '0'
    # }
    # cmd = SimpleNamespace(**configures)

    # inference(cmd)

