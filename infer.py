import argparse

import torch
from scipy.io.wavfile import write

import commons
import utils
from models import SynthesizerTrn
from text import text_to_sequence
from text.symbols import symbols


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-s", "--sid", default=0, type=int)
    parser.add_argument("-t", "--text", required=True)
    parser.add_argument("-o", "--output_path", default="infer/test.wav")
    args = parser.parse_args()
    return args


class vits:
    def __init__(self, checkpoint_path, config_path):
        self.hps = utils.get_hparams_from_file(config_path)
        self.spk_count = self.hps.data.n_speakers
        self.net_g = SynthesizerTrn(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model
        ).cuda()
        _ = self.net_g.eval()
        _ = utils.load_checkpoint(checkpoint_path, self.net_g, None)

    def get_text(self, text, hps):
        text_norm = text_to_sequence(text, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def infer(self, text, output_path, spk_id=0):
        stn_tst = self.get_text(text, self.hps)
        with torch.no_grad():
            x_tst = stn_tst.cuda().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
            sid = torch.LongTensor([spk_id]).cuda()
            audio = (
                self.net_g.infer(
                    x_tst,
                    x_tst_lengths,
                    sid=sid,
                    noise_scale=0.667,
                    noise_scale_w=0.8,
                    length_scale=1,
                )[0][0, 0]
                .data.cpu()
                .float()
                .numpy()
            )
        write(output_path, self.hps.data.sampling_rate, audio)


def infer(model, config, sid, text, output_path):
    tts = vits(model, config)
    tts.infer(text, output_path, sid)


if __name__ == "__main__":
    args = get_args()

    infer(args.model, args.config, args.sid, args.text, args.output_path)
