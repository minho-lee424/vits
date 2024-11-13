import argparse

import torch
from torch import nn

import utils
from models import SynthesizerTrn
from text.symbols import symbols


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-n", "--num_speaker", required=True, type=int)
    args = parser.parse_args()
    return args


def extend_se(model, config, num_speaker):
    checkpoint_dict = torch.load(model, map_location="cpu")
    hps = utils.get_hparams_from_file(config)
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    utils.load_checkpoint(model, net_g, None)

    with torch.no_grad():
        new_embedding = nn.Embedding(num_speaker, hps.model.gin_channels)
        nn.init.normal_(new_embedding.weight, 0.0, hps.model.gin_channels**-0.5)
        for i in range(hps.data.n_speakers):
            new_embedding.weight[i] = net_g.emb_g.weight[torch.LongTensor([i])]
        net_g.emb_g = new_embedding
        net_g.n_speakers = num_speaker

    state_dict = net_g.state_dict()
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        checkpoint_dict["optimizer"]["param_groups"][0]["lr"],
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    for param_group in optim_g.param_groups:
        param_group["initial_lr"] = hps.train.learning_rate

    torch.save(
        {
            "model": state_dict,
            "iteration": checkpoint_dict["iteration"],
            "optimizer": optim_g.state_dict(),
            "learning_rate": checkpoint_dict["learning_rate"],
        },
        model,
    )


if __name__ == "__main__":
    args = get_args()

    extend_se(args.model, args.config, args.num_speaker)
