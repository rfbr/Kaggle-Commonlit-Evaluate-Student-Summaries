import pytorch_lightning as pl
import torch
from torch import nn, optim
from transformers import AutoConfig, AutoModel, get_cosine_schedule_with_warmup

from src.pooling import AttentionPooling, GeMText
from src.utils import apply_differential_lr, freeze, unfreeze_last_n_layers


class AttentionBlock(nn.Module):
    def __init__(self, in_features, middle_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.middle_features = middle_features
        self.out_features = out_features
        self.W = nn.Linear(in_features, middle_features)
        self.V = nn.Linear(middle_features, out_features)

    def forward(self, features, mask):
        att = torch.tanh(self.W(features))
        score = self.V(att)
        score = score + mask.unsqueeze(-1)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector


class AttentionHead(nn.Module):
    def __init__(self, embedding_size, attention_dim=512):
        super().__init__()
        self.ln = nn.LayerNorm(embedding_size)
        self.attention_block = AttentionBlock(embedding_size, attention_dim, 1)
        self.ffn = nn.Linear(embedding_size, 1)

    def forward(self, x, mask):
        att_mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        x = self.ln(x)
        x = self.attention_block(x, att_mask)
        x = self.ffn(x)
        return x


class MCRMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.content_rmse = nn.MSELoss()
        self.wording_rmse = nn.MSELoss()

    def forward(self, content_pred, wording_pred, content_true, wording_true):
        content_loss = torch.sqrt(self.content_rmse(content_pred, content_true))
        wording_loss = torch.sqrt(self.wording_rmse(wording_pred, wording_true))
        return (content_loss + wording_loss) / 2


class PoolinConfig:
    hiddendim_fc = 512
    dropout = 0


class CommonLitModel(pl.LightningModule):
    def __init__(
        self,
        model_name,
        lr_transformer=2e-5,
        lr_head=1e-4,
        weight_decay=1,
        lr_decay=0.5,
        min_lr=1e-8,
        betas=(0.9, 0.999),
        warmup=0,
        num_training_steps=1000,
        apply_dif_lr=True,
        remove_dropout=True,
        max_length=512,
        last_n_layers=5,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.max_position_embeddings = max_length
        if remove_dropout:
            self.config.hidden_dropout_prob = 0
            self.config.attention_probs_dropout_prob = 0
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)

        freeze(self.transformer)
        unfreeze_last_n_layers(self.transformer, last_n_layers)
        # self.content_head = AttentionHead(self.config.hidden_size)
        # self.wording_head = AttentionHead(self.config.hidden_size)

        self.content_pooling = GeMText(self.config.hidden_size, dim=1, eps=1e-6, p=3)
        self.content_head = nn.Linear(self.config.hidden_size, 1)
        self.wording_pooling = GeMText(self.config.hidden_size, dim=1, eps=1e-6, p=3)
        self.wording_head = nn.Linear(self.config.hidden_size, 1)

        self.loss_fn = MCRMSELoss()

    def forward(self, ids, masks, seps):
        output = self.transformer(input_ids=ids, attention_mask=masks, output_hidden_states=True)
        x = output['last_hidden_state']
        for i, s in enumerate(seps):
            masks[i, :s] = 0
        # output = self.content_head(x, masks)
        # return output[:, 0][:, None], output[:, 1][:, None]
        # return self.content_head(x, masks), self.wording_head(x, masks)
        return self.content_head(self.content_pooling(x, masks)), self.wording_head(self.wording_pooling(x, masks))

    def training_step(self, batch, batch_idx):
        ids = batch['ids']
        masks = batch['masks']
        content_target = batch['content'][:, None]
        wording_target = batch['wording'][:, None]
        sep = batch['sep']
        content_pred, wording_pred = self.forward(ids, masks, sep)
        loss = self.loss_fn(content_pred, wording_pred, content_target, wording_target)
        self.training_step_outputs.append(loss)
        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack([x for x in self.training_step_outputs]).mean()
        self.log("train_mcrmse", avg_loss, prog_bar=True, sync_dist=True)
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        ids = batch['ids']
        masks = batch['masks']
        sep = batch['sep']
        content_target = batch['content'][:, None]
        wording_target = batch['wording'][:, None]
        content_pred, wording_pred = self.forward(ids, masks, sep)
        self.validation_step_outputs.append(
            {
                "content_pred": content_pred,
                "wording_pred": wording_pred,
                "content_true": content_target,
                "wording_true": wording_target,
            }
        )

    def on_validation_epoch_end(self):
        content_preds = torch.cat([x['content_pred'] for x in self.validation_step_outputs], dim=0)
        wording_preds = torch.cat([x['wording_pred'] for x in self.validation_step_outputs], dim=0)
        content_targets = torch.cat([x['content_true'] for x in self.validation_step_outputs], dim=0)
        wording_targets = torch.cat([x['wording_true'] for x in self.validation_step_outputs], dim=0)
        loss = self.loss_fn(content_preds, wording_preds, content_targets, wording_targets)
        self.log_dict({"valid_mcrmse": loss}, prog_bar=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        if self.hparams.apply_dif_lr:
            parameters = apply_differential_lr(
                self,
                self.hparams.weight_decay,
                self.hparams.lr_head,
                self.hparams.lr_transformer,
                self.hparams.lr_decay,
                self.hparams.min_lr,
            )
        else:
            parameters = self.parameters()
        optimizer = torch.optim.AdamW(
            parameters, lr=self.hparams.lr_head, betas=self.hparams.betas, weight_decay=self.hparams.weight_decay
        )
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=self.hparams.num_training_steps, eta_min=self.hparams.min_lr
        # )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup,
            num_training_steps=self.hparams.num_training_steps,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
