import torch
from torch import nn
from einops import rearrange
from torch.distributions import Normal


from src.models.utils import TimeEncoding
from src.models.transformer_layers import CrossAttentionLayer, FFLayer
from src.models.embedders import Embedder


class VAEEncoder(nn.Module):
    # Encodes future trajectory into latent
    def __init__(
            self,
            vae_dim,
            T,
            emb_dim,
            num_heads,
            dropout,
            tx_hidden_size,
            activation,
            norm_first,
            num_vae_enc_layers,
    ):
        super().__init__()
        self.vae_dim = vae_dim
        self.T = T
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.tx_hidden_size = tx_hidden_size
        self.activation = activation
        self.norm_first = norm_first
        self.num_vae_enc_layers = num_vae_enc_layers
        self.wps_embedder = Embedder(4, self.emb_dim, expand_theta=True, ignore_pos=False, layer_norm=not self.norm_first)
        self.vae_time_pe = TimeEncoding(self.emb_dim, dropout=self.dropout, max_len=self.T)

        self.vae_cross_layers = []
        for _ in range(self.num_vae_enc_layers):
            vae_cross_layer = CrossAttentionLayer(
                d_model=self.emb_dim,
                nhead=self.num_heads,
                dropout=self.dropout,
                activation=self.activation,
                norm_first=self.norm_first,
                batch_first=True,
                mem_norm=self.norm_first,
            )
            self.vae_cross_layers.append(vae_cross_layer)
        self.vae_cross_layers = nn.ModuleList(self.vae_cross_layers)

        self.vae_ff_layers = []
        for _ in range(self.num_vae_enc_layers):
            vae_ff_layer = FFLayer(
                d_model=self.emb_dim,
                dropout=self.dropout,
                dim_feedforward=self.tx_hidden_size,
                activation=self.activation,
                norm_first=self.norm_first,
            )
            self.vae_ff_layers.append(vae_ff_layer)
        self.vae_ff_layers = nn.ModuleList(self.vae_ff_layers)

        vae_z_dist_layers = []
        if self.norm_first:
            vae_z_dist_layers.append(nn.LayerNorm(self.emb_dim))
        vae_z_dist_layers.append(nn.Linear(self.emb_dim, 2 * self.vae_dim))
        self.vae_z_dist = nn.Sequential(*vae_z_dist_layers)

        z_emb_layers = []
        z_emb_layers.append(nn.Linear(self.vae_dim, self.emb_dim))
        if not self.norm_first:
            z_emb_layers.append(nn.LayerNorm(self.emb_dim))
        self.z_emb = nn.Sequential(*z_emb_layers)

    def vae_cross_fn(self, agents_emb, wps_emb, wps_masks, cross_layer, ff_layer):
        #  :param agents_emb: (B, A, d)
        #  :param wps_emb: (B, T, A, d)
        #  :param wps_masks: (B, T, A)
        #  :return: (B, A, d)
        B, A, d = agents_emb.shape
        _, T, _, _ = wps_emb.shape

        agents_emb = rearrange(agents_emb, 'b a d -> (b a) 1 d')

        wps_emb = rearrange(wps_emb, 'b t a d -> (b a) t d')
        wps_masks = rearrange(wps_masks, 'b t a -> (b a) t')
        wps_masks = torch.where(wps_masks.all(dim=-1, keepdims=True), torch.zeros_like(wps_masks), wps_masks)
        wps_cross_atten_emb = cross_layer(
            agents_emb,
            wps_emb,
            memory_key_padding_mask=wps_masks)
        wps_ff_emb = ff_layer(
            wps_cross_atten_emb,
        )
        wps_atten_emb = rearrange(wps_ff_emb, '(b a) 1 d -> b a d', b=B, a=A)
        return wps_atten_emb

    def forward(self, agents_emb, agents_masks, gt_wps, wps_mask):
        B, T, A = wps_mask.shape
        wps_emb = self.wps_embedder(gt_wps)
        wps_emb = wps_emb + rearrange(self.vae_time_pe(rearrange(wps_emb, 'b t a d -> (b a) t d')), '(b a) t d -> b t a d', b=B, a=A)

        vae_emb = agents_emb

        for d in range(self.num_vae_enc_layers):
            vae_emb = self.vae_cross_fn(
                vae_emb,
                wps_emb,
                wps_mask.clone(),
                cross_layer=self.vae_cross_layers[d],
                ff_layer=self.vae_ff_layers[d],
            )

        pred = self.vae_z_dist(vae_emb)
        mean = pred[..., :self.vae_dim]
        std = torch.exp(pred[..., self.vae_dim:])
        dist = Normal(mean, std)
        z = dist.rsample()
        return z, dist

    def embed_z(self, z):
        z_emb = self.z_emb(z)
        return z_emb
