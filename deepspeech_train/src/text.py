import csv
import os
import re
import shutil

from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError, CouldntEncodeError
import soundfile

from .utils.measure import timer, counter
from .utils.coloured import Print


def text_filter(t: str) -> str:
    t = t.strip().lower()
    t = re.sub("[^a-z ']", ' ', t)
    t = re.sub(' +', ' ', t)
    return t.strip()


def make_test_csv(path2out):
    with open(path2out, 'r', encoding='utf-8') as r:
        with open(path2out.replace('train.csv', 'test.csv'), 'w', encoding='utf-8') as w:
            lines = r.read().splitlines()
            w.writelines(f'{lines[0]}\n{lines[1]}\n')


@timer
def make_train_csv(path2audio, path2audio2out, path2meta, path2out, buffer):
    shutil.rmtree(path2audio2out, ignore_errors=True)
    os.makedirs(path2audio2out, exist_ok=True)
    audio_dict = {''.join(i.split('.')[:-1]): f'{path2audio}/{i}' for i in os.listdir(path2audio)}
    with open(path2meta, 'r', encoding='UTF-8') as reader:
        with open(path2out, 'w', newline='') as csvfile:
            fieldnames = ['wav_filename', 'wav_filesize', 'transcript']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            lines = reader.read().strip().splitlines()
            for c, line in enumerate(lines, 1):
                try:
                    name, text = line.split('|')[0], line.split('|')[1]
                    if name in audio_dict.keys():
                        try:
                            audio_segment = AudioSegment.from_file(
                                audio_dict[name], format=audio_dict[name].split('.')[-1])
                            audio_segment = audio_segment.set_frame_rate(16000)
                            sound = audio_segment.split_to_mono()
                            if len(sound) > 1:
                                audio_segment = sound[0]
                            audio_segment.export(f'{buffer}', format='wav')
                            data, samplerate = soundfile.read(f'{buffer}')
                            path_out = f'{path2audio2out}/{name}.wav'
                            soundfile.write(path_out, data, samplerate, subtype='PCM_16')
                            size = os.stat(path_out).st_size
                            writer.writerow(
                                {'wav_filename': f'{name}.wav', 'wav_filesize': size, 'transcript': text_filter(text)}
                            )
                            Print('b([INFO]) w(Processing audio: {} remaining: {})'.format(name, counter(len(lines) - c, 5)))
                        except (CouldntDecodeError, CouldntEncodeError) as err:
                            Print(f'm([Error]) w({err})')
                except IndexError as err:
                    Print(f'm([Error]) w({err})')

