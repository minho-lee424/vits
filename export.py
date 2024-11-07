import argparse
import json
import os
from typing import Optional

import onnx
import torch
from onnxsim import simplify

import utils
from models import SynthesizerTrn
from text.symbols import symbols


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("-c", "--config_path", default="config.json")
    parser.add_argument("-o", "--output_path", default="model.onnx")
    args = parser.parse_args()
    return args


def export_onnx(model_path, config_path, output_path):
    hps = utils.get_hparams_from_file(config_path)

    model_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )

    num_symbols = model_g.n_vocab
    num_speakers = model_g.n_speakers

    # Inference only
    model_g.eval()

    utils.load_checkpoint(model_path, model_g, None)

    with torch.no_grad():
        model_g.dec.remove_weight_norm()

    def infer_forward(text, text_lengths, scales, sid=None):
        noise_scale = scales[0]
        length_scale = scales[1]
        noise_scale_w = scales[2]
        audio = model_g.infer(
            text,
            text_lengths,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
            sid=sid,
        )[0].unsqueeze(1)

        return audio

    model_g.forward = infer_forward

    dummy_input_length = 50
    sequences = torch.randint(
        low=0, high=num_symbols, size=(1, dummy_input_length), dtype=torch.long
    )
    sequence_lengths = torch.LongTensor([sequences.size(1)])

    sid: Optional[torch.LongTensor] = None
    if num_speakers > 1:
        sid = torch.LongTensor([0])

    # noise, noise_w, length
    scales = torch.FloatTensor([0.667, 1.0, 0.8])
    dummy_input = (sequences, sequence_lengths, scales, sid)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Export
    torch.onnx.export(
        model=model_g,
        args=dummy_input,
        f=output_path,
        verbose=False,
        opset_version=13,
        input_names=["input", "input_lengths", "scales", "sid"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "phonemes"},
            "input_lengths": {0: "batch_size"},
            "output": {0: "batch_size", 1: "time"},
        },
    )

    # load your predefined ONNX model
    model = onnx.load(output_path)

    # convert model
    model_simp, check = simplify(model)

    assert check, "Simplified ONNX model could not be validated"

    onnx.save(model_simp, output_path)

    with open(config_path, "r") as f:
        config = json.load(f)

    with open(output_path + ".json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    args = get_args()

    export_onnx(args.model_path, args.config_path, args.output_path)
