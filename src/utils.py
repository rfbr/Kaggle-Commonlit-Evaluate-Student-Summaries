from argparse import ArgumentParser
from datetime import datetime

import yaml
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import TensorBoardLogger

from src.config import CFG


def prepare_loggers_and_callbacks(
    timestamp, encoder_name, fold, monitors=[], patience=None, tensorboard=False, save_weights_only=False
):
    save_path = CFG.model_path / timestamp
    callbacks = [LearningRateMonitor(logging_interval='step')]
    if "/" in encoder_name:
        encoder_name = encoder_name.replace("/", "_")
    if patience:
        callbacks.append(EarlyStopping("valid_mcrmse", patience=patience))
    for monitor, mode, suffix in monitors:
        if suffix is not None and suffix != "":
            filename = "{epoch:02d}-{valid_mcrmse:.4f}" + f"_{suffix}"
        else:
            filename = "{epoch:02d}-{valid_mcrmse:.4f}"
        checkpoint = ModelCheckpoint(
            dirpath=save_path / encoder_name / f"fold_{fold}",
            filename=filename,
            monitor=monitor,
            mode=mode,
            save_weights_only=save_weights_only,
        )
        callbacks.append(checkpoint)
    loggers = []
    if tensorboard:
        tb_logger = TensorBoardLogger(save_dir=save_path, name=encoder_name, version=f"fold_{fold}")
        loggers.append(tb_logger)
    return loggers, callbacks


def apply_differential_lr(model, weight_decay=1, lr_head=1e-3, lr_transformer=3e-5, lr_decay=1, min_lr=1e-7):
    opt_params = []
    no_decay = ['bias', 'gamma', 'beta', 'transformer.embeddings', 'LayerNorm']
    nb_blocks = len(model.transformer.encoder.layer)
    for n, p in model.named_parameters():
        wd = 0 if any(nd in n for nd in no_decay) else weight_decay
        if "transformer" in n and "pooler" not in n:
            lr_ = lr_transformer
            if "transformer.embeddings" in n:
                lr_ = lr_transformer * lr_decay ** (nb_blocks)
            else:
                for i in range(nb_blocks):
                    if f"layer.{i}." in n:
                        lr_ = lr_transformer * lr_decay ** (nb_blocks - 1 - i)
                        break
        else:
            lr_ = lr_head
        opt_params.append(
            {
                "params": [p],
                "weight_decay": wd,
                'lr': max(min_lr, lr_),
            }
        )
        # print(n, lr_, wd)
    return opt_params


def prepare_args(config_path=CFG.config_path):
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        action="store",
        dest="config",
        help="Configuration scheme",
    )

    parser.add_argument(
        "--timestamp",
        action="store",
        dest="timestamp",
        help="Timestamp for versioning",
        default=str(datetime.now().strftime("%Y%m%d-%H%M%S")),
        type=str,
    )

    parser.add_argument(
        "--seed",
        action="store",
        dest="seed",
        help="Random seed",
        default=42,
        type=int,
    )

    parser.add_argument(
        "--logging",
        dest="logging",
        action="store_true",
        help="Flag to log to TensorBoard (on by default)",
    )
    parser.set_defaults(logging=True)
    args = parser.parse_args()
    # Lookup the config from the YAML file and set args
    with open(config_path, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
        for k, v in cfg[args.config].items():
            setattr(args, k, v)

    return args


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze(model):
    for p in model.parameters():
        p.requires_grad = True


def freeze_layer(model, name):
    for n, p in list(model.named_parameters()):
        if name in n:
            p.requires_grad = False


def unfreeze_layer(model, name):
    for n, p in list(model.named_parameters()):
        if name in n:
            p.requires_grad = True


def unfreeze_last_n_layers(model, n):
    max_layers = len(model.encoder.layer)
    for i in range(max_layers - n, max_layers):
        for p in model.encoder.layer[i].parameters():
            p.requires_grad = True
