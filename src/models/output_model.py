import torch
from torch import nn


class OutputModel(nn.Module):
    # Produces the parameters of a GMM distribution.
    # Initiallly adapted from https://github.com/roggirg/AutoBots
    def __init__(
            self,
            emb_dim=64,
            dist_dim=2,
            T=1,
            min_std=0.01,
            num_hidden=0,
            layer_norm=True,
            dropout=0.0,
            out_mean=None,
            out_std=None,
            output_logit=True,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.dist_dim = dist_dim
        self.T = T
        self.min_std = min_std
        self.num_hidden = num_hidden
        self.layer_norm = layer_norm
        self.output_logit = output_logit
        if out_mean is None:
            self.out_mean = nn.Parameter(torch.zeros(self.dist_dim), requires_grad=False)
        else:
            self.out_mean = nn.Parameter(out_mean, requires_grad=False)

        if out_std is None:
            self.out_std = nn.Parameter(torch.ones(self.dist_dim), requires_grad=False)
        else:
            self.out_std = nn.Parameter(out_std, requires_grad=False)

        modules = []
        if layer_norm:
            modules.append(nn.LayerNorm(self.emb_dim))

        for _ in range(self.num_hidden):
            modules.append(nn.Linear(self.emb_dim, self.emb_dim))
            modules.append(nn.ReLU())
            if dropout > 0.0:
                modules.append(nn.Dropout(dropout))

        modules.append(nn.Linear(self.emb_dim, self.T * 2 * self.dist_dim))

        self.output_model = nn.Sequential(*modules)

        if self.output_logit:
            logit_modules = []
            if layer_norm:
                logit_modules.append(nn.LayerNorm(self.emb_dim))

            for _ in range(self.num_hidden):
                logit_modules.append(nn.Linear(self.emb_dim, self.emb_dim))
                logit_modules.append(nn.ReLU())
                if dropout > 0.0:
                    logit_modules.append(nn.Dropout(dropout))

            logit_modules.append(nn.Linear(self.emb_dim, 1))

            self.logit_model = nn.Sequential(*logit_modules)

    def forward(self, agent_decoder_state):
        start_shape = agent_decoder_state.shape[:-1]
        x = agent_decoder_state.reshape((-1, self.emb_dim))

        out = self.output_model(x).reshape((*start_shape, self.T, 2*self.dist_dim))
        mean = out[..., :self.dist_dim]
        log_std = out[..., self.dist_dim:2*self.dist_dim]

        if self.out_std is not None:
            mean = mean * self.out_std
        if self.out_mean is not None:
            mean = mean + self.out_mean

        std = self.min_std + torch.exp(log_std)
        if self.out_std is not None:
            std = std * self.out_std

        if self.output_logit:
            logit = self.logit_model(x).reshape((*start_shape, 1))

            return torch.cat([mean, std], dim=-1), logit
        else:
            return torch.cat([mean, std], dim=-1)
