import torch
from torch import nn


class Embedder(nn.Module):
    # MLP embedder
    def __init__(self, inp_dim, emb_dim, ignore_pos=True, expand_theta=False, layer_norm=False,):
        super().__init__()
        self.ignore_pos = ignore_pos
        if self.ignore_pos:
            assert(not expand_theta)
            self.inp_dim = inp_dim - 3
        else:
            self.inp_dim = inp_dim
        self.emb_dim = emb_dim
        self.expand_theta = expand_theta
        self.layer_norm = layer_norm
        if self.expand_theta:
            embedder = [nn.Linear(self.inp_dim + 1, self.emb_dim)]
        else:
            embedder = [nn.Linear(self.inp_dim, self.emb_dim)]

        embedder.append(nn.ReLU())
        if self.layer_norm:
            embedder.append(nn.LayerNorm(self.emb_dim))

        self.embedder = nn.Sequential(*embedder)

    def forward(self, x):
        pos = x[..., :2]
        theta = x[..., 2:3]
        features = x[..., 3:]
        if self.ignore_pos:
            inp = features
        elif self.expand_theta:
            inp = torch.cat([pos, features, torch.cos(theta), torch.sin(theta)], dim=-1)
        else:
            inp = x

        y = self.embedder(inp)
        return y


class RelativeEmbedder(nn.Module):
    # MLP embedder that operates on relative poses
    def __init__(self, emb_dim, expand_theta=True, layer_norm=False,):
        super().__init__()

        self.inp_dim = 3
        self.emb_dim = emb_dim
        self.expand_theta = expand_theta
        self.layer_norm = layer_norm
        if self.expand_theta:
            embedder = [nn.Linear(self.inp_dim + 1, self.emb_dim)]
        else:
            embedder = [nn.Linear(self.inp_dim, self.emb_dim)]

        embedder.append(nn.ReLU())
        if self.layer_norm:
            embedder.append(nn.LayerNorm(self.emb_dim))

        self.embedder = nn.Sequential(*embedder)

    def forward(self, y, x, return_features=False):
        # x: B x ... x N x 3
        # y: B x ... x M x 3 or B x ... x N x M x 3

        # rel_pos: B x ... x N x M x 2
        if len(y.shape) == len(x.shape):
            rel_pos = y[..., :2].unsqueeze(-3) - x[..., :2].unsqueeze(-2)
            rel_theta = y[..., 2:3].unsqueeze(-3) - x[..., 2:3].unsqueeze(-2)
        else:
            assert(len(y.shape) == (len(x.shape) + 1))
            rel_pos = y[..., :2] - x[..., :2].unsqueeze(-2)
            rel_theta = y[..., 2:3] - x[..., 2:3].unsqueeze(-2)

        theta = x[..., 2]

        # rot_matrix: B x ... x N x 1 x 2 x 2
        rot_matrix = torch.stack([
            torch.stack([torch.cos(theta), -torch.sin(theta)], dim=-1),
            torch.stack([torch.sin(theta),  torch.cos(theta)], dim=-1)
        ], dim=-2)[..., None, :, :]

        # transformed_pos: B x ... x N x M x 2
        transformed_pos = torch.matmul(rel_pos[..., None, :2], rot_matrix).squeeze(-2)

        if self.expand_theta:
            inp = torch.cat([transformed_pos, torch.cos(rel_theta), torch.sin(rel_theta)], dim=-1)
        else:
            inp = torch.cat([transformed_pos, rel_theta], dim=-1)

        res = self.embedder(inp)
        if return_features:
            return res, inp
        else:
            return res


class RelativeFeatureEmbedder(nn.Module):
    # MLP embedder that operates on relative poses and additional relative features
    def __init__(self, inp_dim, emb_dim, expand_theta=True, layer_norm=False,):
        super().__init__()

        self.inp_dim = inp_dim
        self.emb_dim = emb_dim
        self.expand_theta = expand_theta
        self.layer_norm = layer_norm
        if self.expand_theta:
            embedder = [nn.Linear(self.inp_dim + 1, self.emb_dim)]
        else:
            embedder = [nn.Linear(self.inp_dim, self.emb_dim)]

        embedder.append(nn.ReLU())
        if self.layer_norm:
            embedder.append(nn.LayerNorm(self.emb_dim))

        self.embedder = nn.Sequential(*embedder)

    def forward(self, y, x, relative_features, return_features=False):
        # x: B x ... x N x 3
        # y: B x ... x M x 3 or B x ... x N x M x 3

        # rel_pos: B x ... x N x M x 2
        if len(y.shape) == len(x.shape):
            rel_pos = y[..., :2].unsqueeze(-3) - x[..., :2].unsqueeze(-2)
            rel_theta = y[..., 2:3].unsqueeze(-3) - x[..., 2:3].unsqueeze(-2)
        else:
            assert(len(y.shape) == (len(x.shape) + 1))
            rel_pos = y[..., :2] - x[..., :2].unsqueeze(-2)
            rel_theta = y[..., 2:3] - x[..., 2:3].unsqueeze(-2)

        theta = x[..., 2]

        # rot_matrix: B x ... x N x 1 x 2 x 2
        rot_matrix = torch.stack([
            torch.stack([torch.cos(theta), -torch.sin(theta)], dim=-1),
            torch.stack([torch.sin(theta),  torch.cos(theta)], dim=-1)
        ], dim=-2)[..., None, :, :]

        # transformed_pos: B x ... x N x M x 2
        transformed_pos = torch.matmul(rel_pos[..., None, :2], rot_matrix).squeeze(-2)

        if self.expand_theta:
            inp = torch.cat([transformed_pos, torch.cos(rel_theta), torch.sin(rel_theta), relative_features], dim=-1)
        else:
            inp = torch.cat([transformed_pos, rel_theta, relative_features], dim=-1)

        res = self.embedder(inp)
        if return_features:
            return res, inp
        else:
            return res
