import os
import argparse
import torch
import librosa
import time
from scipy.io.wavfile import write
import soundfile as sf
from tqdm import tqdm
from pathlib import Path

# set paths
home_path = str(Path.home())
work_path = os.path.join(home_path, 'code', 'repo', 'free-vc')
if os.getcwd() != work_path:
    os.chdir(work_path)
print('current path: {}'.format(os.getcwd()))

import utils
from models import SynthesizerTrn
from mel_processing import mel_spectrogram_torch
from wavlm import WavLM, WavLMConfig
from speaker_encoder.voice_encoder import SpeakerEncoder
import logging
logging.getLogger('numba').setLevel(logging.WARNING)

from scripts.utils import empty_dir

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpfile", type=str, default="configs/freevc.json", help="path to json config file")
    parser.add_argument("--ptfile", type=str, default="checkpoints/freevc.pth", help="path to pth file")
    parser.add_argument("--txtpath", type=str, default="convert.txt", help="path to txt file")
    parser.add_argument("--outdir", type=str, default="output/freevc", help="path to output dir")
    parser.add_argument("--use_timestamp", default=False, action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # runtime mode
    args = parse_args()

    # # interactive mode
    # args = argparse.ArgumentParser()

    # recording_id = 'MARCHE_AssessmentTacticalEnvironment'
    # voice = 'dmytro'
    # stress = 'dictionary'
    # args.hpfile = os.path.join(work_path, 'configs', 'freevc.json')
    # args.ptfile = os.path.join(work_path, 'checkpoints', 'freevc.pth')
    # args.txtpath = os.path.join(work_path, 'txtfiles', '{}_{}-{}.txt'.format(recording_id, voice, stress))
    # args.outdir = os.path.join(work_path, 'outputs', recording_id, '{}-{}'.format(voice, stress))
    # args.use_timestamp = False

    # print arguments
    print('config file: {}'.format(args.hpfile))
    print('model file: {}'.format(args.ptfile))
    print('text file: {}'.format(args.txtpath))
    print('output dir: {}'.format(args.outdir))
    print('use timestamp: {}'.format(args.use_timestamp))

    os.makedirs(args.outdir, exist_ok=True)
    hps = utils.get_hparams_from_file(args.hpfile)

    print("Loading model...")
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    _ = net_g.eval()
    print("Loading checkpoint...")
    _ = utils.load_checkpoint(args.ptfile, net_g, None, True)

    print("Loading WavLM for content...")
    cmodel = utils.get_cmodel(0)

    if hps.model.use_spk:
        print("Loading speaker encoder...")
        smodel = SpeakerEncoder('speaker_encoder/ckpt/pretrained_bak_5805000.pt')

    print("Processing text...")
    titles, srcs, tgts = [], [], []
    with open(args.txtpath, "r") as f:
        for rawline in f.readlines():
            title, src, tgt = rawline.strip().split("|")
            titles.append(title)
            srcs.append(src)
            tgts.append(tgt)

    print("Synthesizing...")
    with torch.no_grad():
        for line in tqdm(zip(titles, srcs, tgts)):
        # for line in zip(titles, srcs, tgts):
            title, src, tgt = line

            # tgt
            wav_tgt, _ = librosa.load(tgt, sr=hps.data.sampling_rate)
            wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
            if hps.model.use_spk:
                g_tgt = smodel.embed_utterance(wav_tgt)
                g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).cuda()
            else:
                wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).cuda()
                mel_tgt = mel_spectrogram_torch(
                    wav_tgt,
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax
                )

            # src
            wav_src, _ = librosa.load(src, sr=hps.data.sampling_rate)
            wav_src = torch.from_numpy(wav_src).unsqueeze(0).cuda()
            c = utils.get_content(cmodel, wav_src)

            if hps.model.use_spk:
                audio = net_g.infer(c, g=g_tgt)
            else:
                audio = net_g.infer(c, mel=mel_tgt)
            audio = audio[0][0].data.cpu().float().numpy()

            # write audio to file
            if args.use_timestamp:
                timestamp = time.strftime("%m-%d_%H-%M", time.localtime())
                out_wavname = "{}.wav".format(timestamp+"_"+title)
            else:
                out_wavname = f"{title}.wav"
            out_wavfile = os.path.join(args.outdir, out_wavname)    
            # # option 1: write using scipy (precision: 25-bit, sample encoding: 32-bit floating point PCM)    
            # write(out_wavfile, hps.data.sampling_rate, audio)
            # option 2: write using soundfile (precision: 16-bit, sample encoding: 16-big signed integer PCM)
            sf.write(out_wavfile, audio, hps.data.sampling_rate)