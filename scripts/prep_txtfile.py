# prepare the text file which contains title|src-voice-wav (take content from)|tgt-voice-wav (take voice from)
#
# Zhenhao Ge, 2024-06-16

import os
from pathlib import Path
import glob
import argparse

# set paths
home_path = str(Path.home())
work_path = os.path.join(home_path, 'code', 'repo', 'free-vc')
if os.getcwd() != work_path:
    os.chdir(work_path)
print('current path: {}'.format(os.getcwd()))

def filter_path(paths, keywords):
    for kw in keywords:
        paths = [f for f in paths if kw not in f]
    return paths

def get_wavs(wav_path, keywords):

    wavs = sorted(glob.glob(os.path.join(wav_path, '*.wav')))
    wavs = filter_path(wavs, keywords)
    # num_wavs = len(wavs)
    # print('# of wav files: {}'.format(num_wavs))
    return wavs

def check_pairs(src_wavs, tgt_wavs):
    """check src and tgt wavs from their filename to ensure they have the same timestamps"""

    # check # of wavs
    num_src_wavs = len(src_wavs)
    num_tgt_wavs = len(tgt_wavs)
    assert num_src_wavs == num_tgt_wavs, \
        '# of src and tgt wavs mismatch: {} vs. {}!'.format(num_src_wavs, num_tgt_wavs)

    for i, (path1, path2) in enumerate(zip(src_wavs, tgt_wavs)):
        filename1 = os.path.basename(path1)
        filename2 = os.path.basename(path2)
        if filename1 != filename2:
            raise Exception('{}/{}: filename mis-match: {} vs. {}'.format(i+1, num_src_wavs, filename1, filename2))
            return False
        else:
            return True    

def parse_args():
    usage = 'prepare the text file which contains 3 parts: title, source-wav-path, target-wav-path'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--src-path', type=str, help='path to the source wavs')
    parser.add_argument('--tgt-path', type=str, help='path to the target wavs')
    parser.add_argument('--txt-file', type=str, help='output txt file')
    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    # args = argparse.ArgumentParser()

    # recording_id = 'MARCHE_AssessmentTacticalEnvironment'
    # voice = 'dmytro'
    # stress = 'dictionary'

    # args.src_path = os.path.join(home_path, 'code', 'repo', 'ukr-tts', 'outputs', 'sofw', 'espnet',
    #     recording_id, '{}-{}'.format(voice, stress))
    # args.tgt_path = os.path.join(home_path, 'code', 'repo', 'ukr-tts', 'data', recording_id, 'segments')

    # txt_filename = '{}_{}-{}.txt'.format(recording_id, voice, stress)
    # args.txt_file = os.path.join(work_path, 'txtfiles', txt_filename)

    # localize arguments
    src_path = args.src_path
    tgt_path = args.tgt_path
    txt_file = args.txt_file

    # check file existence
    assert os.path.isdir(src_path), 'path to the source wav does not exist!'.format(args.src_path)
    assert os.path.isdir(tgt_path), 'path to the target wav does not exist!'.format(args.tgt_path)

    # print out arguments
    print(f'source path: {src_path}')
    print(f'target path: {tgt_path}')
    print(f'text file: {txt_file}')

    # set wav file filter keywords
    keywords = ['_converted', '16000']

    # get source wav files
    src_wavs = get_wavs(src_path, keywords)
    num_src_wavs = len(src_wavs)
    print('# of the source wav files: {}'.format(num_src_wavs))

    # get target wav files
    tgt_wavs = get_wavs(tgt_path, keywords)
    num_tgt_wavs = len(tgt_wavs)
    print('# of the target wav files: {}'.format(num_tgt_wavs))

    # check if src_wav and tgt_wav come in pairs
    pair_status = check_pairs(src_wavs, tgt_wavs)

    # prepare rows including 3 columns (id, src_wav, tgt_wav)
    rows = ['' for _ in range(num_src_wavs)]
    for i in range(num_src_wavs):
        title = os.path.splitext(os.path.basename(src_wavs[i]))[0]
        rows[i] = '|'.join([title, src_wavs[i], tgt_wavs[i]])

    # write the output txt file
    with open(txt_file, 'w') as f:
        f.writelines('\n'.join(rows) + '\n')
    print('wrote txt file: {}'.format(txt_file))