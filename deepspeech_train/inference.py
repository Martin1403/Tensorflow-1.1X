import deepspeech
import numpy as np
import wave

from src.utils.coloured import Print
from src.utils.measure import timer


class DeepSpeech:
    def __init__(self, model: str, scorer: str) -> None:
        self.model = deepspeech.Model(model)
        self.model.enableExternalScorer(scorer)
        self.model.setScorerAlphaBeta(alpha=0.931289039105002, beta=1.1834137581510284)

    @timer
    def transcribe(self, audio: str) -> str:
        w = wave.open(audio, 'r')
        wav = bytearray()
        for i in range(w.getnframes()):
            wav.extend(w.readframes(i))
        data = np.frombuffer(wav, np.int16)
        return self.model.stt(data)


if __name__ == '__main__':
    test = "checkpoint/buffer.wav"
    deep = DeepSpeech(model='graph/output_graph.pbmm', scorer='graph/output_graph.scorer')
    Print(f"g([INFERENCE]) w(From file: {test} => {deep.transcribe(audio=test)})")
