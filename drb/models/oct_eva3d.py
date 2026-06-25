import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import timm
from mmcv.runner import auto_fp16


from timm.layers import apply_keep_indices_nlc
from timm.models.layers import trunc_normal_
from .base import Base
from .builder import MODELS

class FuseSliceLayer(nn.Module):

    def __init__(self, emb_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attn = nn.MultiheadAttention(emb_dim,
                                          32,
                                          dropout=0.5,
                                          batch_first=False)

    def forward(self, x):
        x, _ = self.attn(x, x, x)
        return x

@MODELS.register_module()
class OCTEVA3D(Base):

    def __init__(self, reg_targets_dim=54, cls_targets_dim=52, slice_dim=16, num_classes=2, grad_checkpointing=False, *args, **kw_args) -> None:
        super().__init__(*args, **kw_args)
        self.reg_targets_dim = reg_targets_dim
        self.cls_targets_dim = cls_targets_dim
        self.slice_dim = slice_dim  
        self.num_classes = num_classes
        self.fp16_enabled = True

        eva = timm.models.eva02_large_patch14_224(pretrained=True)
        self.cls_token = eva.cls_token
        self.pos_embed = eva.pos_embed
        self.patch_embed = eva.patch_embed
        _proj = eva.patch_embed.proj
        patch_proj = nn.Conv2d(1,
                               _proj.out_channels,
                               kernel_size=_proj.kernel_size,
                               stride=_proj.stride,
                               bias=(_proj.bias is not None))
        patch_proj.weight.data = _proj.weight.data.sum(dim=1, keepdim=True)
        patch_proj.bias.data = _proj.bias.data
        self.patch_embed.proj = patch_proj
        self.blocks = eva.blocks

        self.pos_drop = eva.pos_drop
        self.rope = eva.rope
        self.patch_drop = eva.patch_drop
        self.grad_checkpointing = grad_checkpointing
        self.norm = eva.norm

        self.slice_embed = nn.Parameter(torch.empty((self.slice_dim, eva.num_features)))
        self.reg_token = nn.Parameter(torch.empty(eva.num_features))
        self.class_token = nn.Parameter(torch.empty(eva.num_features))
        self.age_embed = nn.Embedding(120, eva.num_features)
        self.slice_fuse = nn.Sequential(
            *[FuseSliceLayer(eva.num_features) for _ in range(3)])

        self.reg_head = nn.Linear(eva.num_features, self.reg_targets_dim)
        self.cls_head = nn.Linear(eva.num_features, self.cls_targets_dim * self.num_classes)

    def init_weights(self):
        nn.init.normal_(self.slice_embed)
        nn.init.normal_(self.reg_token)
        nn.init.normal_(self.class_token)
        trunc_normal_(self.reg_head.weight, std=.02)
        nn.init.constant_(self.reg_head.bias, 0)
        trunc_normal_(self.cls_head.weight, std=.02)
        nn.init.constant_(self.cls_head.bias, 0)

    @auto_fp16(apply_to=('img', ))
    def forward_train(self, img, target, weight_indices, age, **kw_args): # age,
        B, _, F, H, W = img.shape
        pred = self.forward_test(img, age) # , age
        reg_pred = (pred['reg'] * 40 + 20).reshape(-1)  # (B * reg_targets_dim,) 
        cls_pred = pred['cls'].reshape(-1, self.num_classes)  #(B * cls_targets_dim, num_classes)

        CE = nn.CrossEntropyLoss(reduction='none')
        with torch.no_grad():
            target = target.detach()
        reg_target = (target[:, 0 : self.reg_targets_dim] * 40 + 20).reshape(-1).to(
            img.dtype)  # num 
        reg_indices = weight_indices[:, 0 : self.reg_targets_dim].reshape(-1)
        reg_n_count = (reg_indices == 0).sum().item()
        reg_def_count = (reg_indices == 1).sum().item()

        cls_target = target[:, self.reg_targets_dim : self.reg_targets_dim + self.cls_targets_dim].to(torch.long).reshape(-1) 
        cls_indices = weight_indices[:, self.reg_targets_dim : self.reg_targets_dim + self.cls_targets_dim].reshape(-1)
        cls_n_count = (cls_indices == 0).sum().item()
        cls_def_count = (cls_indices == 1).sum().item()


        reg_loss = (reg_pred - reg_target).abs()
        def_reg_loss = reg_loss[reg_indices == 1].sum()

        if self.reg_targets_dim==54:
            # add 2 blind points
            reg_n_count = reg_n_count + (reg_indices == -1).sum().item()
            n_reg_loss = reg_loss[reg_indices == 0].sum() + reg_loss[reg_indices == -1].sum()
        else:
            n_reg_loss = reg_loss[reg_indices == 0].sum()

        cls_loss = CE(cls_pred, cls_target)
        n_cls_loss = cls_loss[cls_indices == 0].sum()
        def_cls_loss = cls_loss[cls_indices == 1].sum()

        if cls_n_count > 0:
            n_cls_loss = n_cls_loss / cls_n_count
        if cls_def_count > 0:
            def_cls_loss = def_cls_loss / cls_def_count

        if reg_n_count > 0:
            n_reg_loss = n_reg_loss / reg_n_count
        if reg_def_count > 0:
            def_reg_loss = def_reg_loss / reg_def_count

        w_n_cls = 12.5
        w_abn_cls = 5
        
        loss = {
            'n_reg_loss': n_reg_loss,
            'n_cls_loss': n_cls_loss * w_n_cls,
            'def_reg_loss': def_reg_loss,
            'def_cls_loss': def_cls_loss * w_abn_cls
        }
        return loss
    
    def forward_test(self, img, age, **kw_args): #age,
        B, _, F, H, W = img.shape
        img = img.view(B * F, 1, H, W)

        x = self.patch_embed(img)
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        # apply abs position embedding
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        # obtain shared rotary position embedding and apply patch dropout
        rot_pos_embed = self.rope.get_embed() if self.rope is not None else None
        if self.patch_drop is not None:
            x, keep_indices = self.patch_drop(x)
            if rot_pos_embed is not None and keep_indices is not None:
                rot_pos_embed = apply_keep_indices_nlc(x, rot_pos_embed, keep_indices)

        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, rope=rot_pos_embed, use_reentrant=False)
            else:
                x = blk(x, rope=rot_pos_embed)

        x = self.norm(x)

        _, L, C = x.shape
        x = x.view(B, F, L, C)
        x = x + self.slice_embed.view(1, F, 1, C)
        x = x.permute(1, 2, 0, 3).reshape(L * F, B, C)
        x = torch.cat([
            x,  
            self.age_embed(age).unsqueeze(0),
            self.reg_token.view(1, 1, C).repeat(1, B, 1),
            self.class_token.view(1, 1, C).repeat(1, B, 1)
        ],
                      dim=0)
        for blk in self.slice_fuse:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x)
            else:
                x = blk(x)

        reg_pred = self.reg_head(x[-2])  
        cls_pred = self.cls_head(x[-1]).view(-1, self.cls_targets_dim, self.num_classes)
        return dict(reg=reg_pred, cls=cls_pred, md=0)
