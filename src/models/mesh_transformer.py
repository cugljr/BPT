import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange, repeat, pack
from typing import Dict, Any, Optional, Callable
from x_transformers import Decoder
from x_transformers.autoregressive_wrapper import top_k
from tqdm import tqdm
from lightning.pytorch import LightningModule
from src.models.miche_conditioner import PointConditioner
from src.models.ptv2_conditioner import PTV2Conditioner
from src.utils.helper import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from warmup_scheduler import GradualWarmupScheduler
from torch.utils import checkpoint as cp


# auto-regressive Transformer
class MeshTransformer(LightningModule):

    def __init__(
        self,
        decoder_config: Dict[str, Any],
        batch_size: int = 16,
        block_size: int = 8,
        offset_size: int = 16,
        pad_id: int = -1,
        conditioner_type: str = "miche",
        miche_path: str = "",
        miche_ckpt_path: str = "",
        miche_config_path: str = "",
        miche_freeze: bool = True,
        miche_disable_checkpoint: bool = True,
        ptv2_repo_path: str = "",
        ptv2_encoder_ckpt: str = "",
        ptv2_freeze: bool = True,
        max_seq_len: int = 1500,
        learning_rate: float = 3e-4,
        eta_min: float = 1e-4,
        warmup_steps: int = 1600,
        cosine_steps: int = 16000,
    ):
        super(MeshTransformer, self).__init__()
        self.save_hyperparameters()

        dim = decoder_config["dim"]
        depth = decoder_config["depth"]
        heads = decoder_config["heads"]
        dropout = decoder_config["dropout"]
        attn_flash = decoder_config["attn_flash"]
        ff_glu = decoder_config["ff_glu"]
        attn_qk_norm = decoder_config["attn_qk_norm"]
        cross_attn_num_mem_kv = decoder_config["cross_attn_num_mem_kv"]

        # block_ids, offset_ids, special_block_ids
        self.block_size = block_size
        self.offset_size = offset_size
        self.block_val = block_size**3
        self.offset_val = offset_size**3

        self.vocab_size = self.block_val + self.offset_val + self.block_val
        self.eos_token_id = self.vocab_size
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps
        self.cosine_steps = cosine_steps

        self.sp_block_embed = nn.Parameter(torch.randn(1, dim))
        self.block_embed = nn.Parameter(torch.randn(1, dim))
        self.offset_embed = nn.Parameter(torch.randn(1, dim))
        self.sos_embed = nn.Parameter(torch.randn(dim))
        self.token_embed = nn.Embedding(self.vocab_size + 1, dim)
        self.abs_pos_emb = nn.Embedding(max_seq_len, dim)
        self.conditioner_type = conditioner_type

        if conditioner_type == "miche":
            self.conditioner = PointConditioner(
                miche_path=miche_path,
                miche_ckpt_path=miche_ckpt_path if miche_ckpt_path else None,
                miche_config_path=miche_config_path if miche_config_path else None,
                feature_dim=dim,
                freeze=miche_freeze,
                disable_checkpoint=miche_disable_checkpoint,
            )
        elif conditioner_type == "ptv2":
            self.conditioner = PTV2Conditioner(
                repo_path=ptv2_repo_path,
                encoder_ckpt_path=ptv2_encoder_ckpt,
                feature_dim=dim,
                freeze=ptv2_freeze,
            )
        else:
            raise ValueError(f"Unsupported conditioner_type: {conditioner_type}")

        # autoregressive attention network
        self.decoder = Decoder(
            dim=dim,
            depth=depth,
            heads=heads,
            attn_flash=attn_flash,
            attn_dropout=dropout,
            ff_dropout=dropout,
            cross_attend=True,
            cross_attn_dim_context=dim,
            cross_attn_num_mem_kv=cross_attn_num_mem_kv,  # needed for preventing nan when dropping out text condition
            attn_qk_norm=attn_qk_norm,
            ff_glu=ff_glu
        )
        self.to_logits = nn.Linear(dim, self.vocab_size + 1)

    def forward(
        self,
        codes: Tensor,
        pc_xyz: Tensor = None,
        cond_embeds: Tensor = None,
        return_loss: bool = True,
        return_cache: bool = False,
        append_eos: bool = True,
        cache: Tensor = None,
    ):
        # handle conditions
        if cond_embeds is None:
            cond_embeds = self.conditioner(pc_xyz)

        # prepare mask for position embedding of block and offset tokens
        block_mask = (0 <= codes) & (codes < self.block_val)
        offset_mask = (self.block_val <= codes) & (
            codes < self.block_val + self.offset_val
        )
        sp_block_mask = (self.block_val + self.offset_val <= codes) & (
            codes < self.block_val + self.offset_val + self.block_val
        )

        # get some variable
        batch, seq_len = codes.shape

        assert (
            seq_len <= self.max_seq_len
        ), f"received codes of length {seq_len} but needs to be less than {self.max_seq_len}"

        # auto append eos token
        if append_eos:
            code_lens = ((codes == self.pad_id).cumsum(dim=-1) == 0).sum(dim=-1)
            codes = F.pad(codes, (0, 1), value=0)  # value=-1
            batch_arange = torch.arange(batch, device=self.device)
            batch_arange = rearrange(batch_arange, "... -> ... 1")
            code_lens = rearrange(code_lens, "... -> ... 1")
            codes[batch_arange, code_lens] = self.eos_token_id

        # if returning loss, save the labels for cross entropy
        if return_loss:
            assert seq_len > 0
            codes, labels = codes[:, :-1], codes

        # token embed
        codes = codes.masked_fill(codes == self.pad_id, 0)
        codes = self.token_embed(codes)

        # codebook embed + absolute positions
        seq_arange = torch.arange(codes.shape[-2], device=self.device)
        codes = codes + self.abs_pos_emb(seq_arange)

        # add positional embedding for block and offset token
        block_embed = repeat(self.block_embed, "1 d -> b n d", n=seq_len, b=batch)
        offset_embed = repeat(self.offset_embed, "1 d -> b n d", n=seq_len, b=batch)
        codes[block_mask] += block_embed[block_mask]
        codes[offset_mask] += offset_embed[offset_mask]

        sp_block_embed = repeat(self.sp_block_embed, "1 d -> b n d", n=seq_len, b=batch)
        codes[sp_block_mask] += sp_block_embed[sp_block_mask]

        # auto prepend sos token
        sos = repeat(self.sos_embed, "d -> b d", b=batch)
        codes, _ = pack([sos, codes], "b * d")

        attended, intermediates_with_cache = self.decoder(
            codes,
            cache=cache,
            return_hiddens=True,
            context=cond_embeds,
            context_mask=None,
        )

        # logits
        logits = self.to_logits(attended)

        if not return_loss:
            if not return_cache:
                return logits
            return logits, intermediates_with_cache

        # loss
        labels = labels.contiguous().view(-1).long()
        logits = logits.contiguous().view(-1, self.vocab_size + 1)
        loss_ce = F.cross_entropy(logits, labels, ignore_index=self.pad_id)
        acc = accuracy(logits, labels, ignore_label=self.pad_id)
        return loss_ce, acc

    def training_step(self, batch):
        codes = batch["codes"]
        pc_xyz = batch["pc_xyz"]

        loss_ce, acc = self(codes, pc_xyz)

        self.log(
            "train/loss_ce",
            loss_ce,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            "train/acc",
            acc,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        return loss_ce

    def validation_step(self, batch):
        codes = batch["codes"]
        pc_xyz = batch["pc_xyz"]

        loss_ce, acc = self(codes, pc_xyz)

        self.log(
            "val/loss_ce",
            loss_ce,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            "val/acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

    def configure_optimizers(self) -> Dict[str, Any]:
        """Method to create optimizer and learning rate scheduler

        Returns:
            dict: A dictionary with optimizer and learning rate scheduler
        """
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.1,
            betas=(0.9, 0.95),
        )
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.cosine_steps, eta_min=self.eta_min
        )
        scheduler_warmup = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=self.warmup_steps,
            after_scheduler=scheduler,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler_warmup,
                "interval": "step",
            },
        }

    def generate(
        self,
        pc_xyz: Tensor,
        prompt: Tensor = None,
        batch_size: int = 1,
        max_seq_len: int = 1500,
        top_k: int = 0,
        top_p: float = 1.0,
        temperature: float = 1.0,
        cache_kv: bool = True,
    ):
        # encode point cloud
        cond_embeds = self.conditioner(pc_xyz)
        codes = default(
            prompt, torch.empty((batch_size, 0), dtype=torch.int32, device=self.device)
        )
        curr_length = codes.shape[-1]
        cache = None

        # predict tokens auto-regressively
        for i in tqdm(
            range(curr_length, max_seq_len),
            desc=f"Process",
            dynamic_ncols=True,
            leave=False,
        ):
            output = self(
                codes,
                return_loss=False,
                return_cache=cache_kv,
                append_eos=False,
                cond_embeds=cond_embeds,
                cache=cache,
            )

            if cache_kv:
                logits, cache = output
            else:
                logits = output

            # sample code from logits
            logits = logits[:, -1]
            logits = joint_filter(logits, k=top_k, p=top_p)
            probs = F.softmax(logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)
            codes, _ = pack([codes, sample], "b *")

            # check for all rows to have [eos] to terminate
            is_eos_codes = codes == self.eos_token_id
            if is_eos_codes.any(dim=-1).all():
                break

        # mask out to padding anything after the first eos
        mask = is_eos_codes.float().cumsum(dim=-1) >= 1
        codes = codes.masked_fill(mask, self.pad_id)
        return codes
