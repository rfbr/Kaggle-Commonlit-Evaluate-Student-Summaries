import math

import pytorch_lightning as pl
from pytorch_lightning.callbacks import StochasticWeightAveraging

from src.config import CFG
from src.datasets import CommonLitDataModule
from src.models import CommonLitModel
from src.utils import prepare_loggers_and_callbacks


def run_all(fold, args):
    pl.seed_everything((args.seed + fold))
    save_path = CFG.model_path / args.timestamp / args.model_name / f"fold_{fold}"
    data_module = CommonLitDataModule(fold, **args.__dict__)
    num_training_steps = math.ceil(data_module.num_training_samples / args.batch_size * args.max_epochs)
    model = CommonLitModel(num_training_steps=num_training_steps, **args.__dict__)
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        precision=args.precision,
        val_check_interval=args.val_check_interval,
        accumulate_grad_batches=args.accumulate_grad_batches,
        max_epochs=args.max_epochs,
        devices=[1],
        limit_val_batches=0,
        num_sanity_val_steps=0,
        default_root_dir=save_path
        # deterministic=True
        # gradient_clip_val=1,
    )

    folder = args.model_name
    if "/" in folder:
        folder = folder.replace("/", "_")
    save_path = CFG.model_path / args.timestamp / folder / f"fold_{fold}"
    data_module.tokenizer.save_pretrained(save_path)
    model.config.to_json_file(str(save_path / "config.json"))
    trainer.fit(model, datamodule=data_module)


def run_fold(fold, args):
    pl.seed_everything((args.seed + fold))
    monitor_list = [("valid_mcrmse", "min", None)]
    loggers, callbacks = prepare_loggers_and_callbacks(
        args.timestamp,
        args.model_name,
        fold,
        monitors=monitor_list,
        tensorboard=args.logging,
        save_weights_only=True,
    )
    if args.swa:
        # swa = StochasticWeightAveraging(args.swa_lr, swa_epoch_start=2, annealing_epochs=3, annealing_strategy='cos')
        swa = StochasticWeightAveraging(args.swa_lr, swa_epoch_start=0.75)
        callbacks.append(swa)
    data_module = CommonLitDataModule(fold, **args.__dict__)
    num_training_steps = math.ceil(data_module.num_training_samples / args.batch_size * args.max_epochs)
    model = CommonLitModel(num_training_steps=num_training_steps, **args.__dict__)
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        precision=args.precision,
        val_check_interval=args.val_check_interval,
        accumulate_grad_batches=args.accumulate_grad_batches,
        max_epochs=args.max_epochs,
        logger=loggers,
        callbacks=callbacks,
        devices=[1],
        # deterministic=True
        # gradient_clip_val=1,
    )

    folder = args.model_name
    if "/" in folder:
        folder = folder.replace("/", "_")
    save_path = CFG.model_path / args.timestamp / folder / f"fold_{fold}"
    data_module.tokenizer.save_pretrained(save_path)
    model.config.to_json_file(str(save_path / "config.json"))

    trainer.fit(model, datamodule=data_module)
    res = trainer.validate(model, data_module, ckpt_path="best")
    return res[0]['valid_mcrmse']
