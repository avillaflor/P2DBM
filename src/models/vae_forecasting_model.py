import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from einops import rearrange, reduce
from torch.distributions import Normal, kl_divergence


from src.models.vae_encoder import VAEEncoder
from src.models.scene_encoder import SceneEncoder
from src.models.output_model import OutputModel
from src.models.transformer_layers import CrossAttentionLayer, FFLayer
from src.models.utils import nll_pytorch_dist, get_cosine_schedule_with_warmup, PIDController
from src.carla.features.utils import TOWNS
from src.carla.features.carla_map_features import CarlaMapFeatures


class VAEForecastingModel(pl.LightningModule):
    # CVAE Forecasting model with Transformer-based Encoder/Decoder structure
    def __init__(
            self,
            vae_dim=32,
            vae_beta=0.01,
            num_vae_enc_layers=4,
            num_map_pts=20000,
            num_local_pts=50,
            num_route_pts=20,
            route_downsample=10,
            T=8,
            H=1,
            f=1,
            dt=0.5,
            emb_dim=128,
            num_enc_layers=4,
            num_dec_layers=4,
            num_map_enc_layers=0,
            num_heads=8,
            tx_hidden_factor=4,
            activation="gelu",
            dropout=0.1,
            lr=2e-4,
            betaW=0.95,
            lr_decay=0.,
            warmup_steps=0,
            min_std=0.01,
            wd=1.e-1,
            norm_first=True,
            carla_maps_path='maps/',
            max_token_distance=50.,
            max_z_distance=7.5,
            pid_type='scenarios',
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_map_pts = num_map_pts
        self.num_local_pts = num_local_pts
        self.num_route_pts = num_route_pts
        self.T = T  # prediction horizon
        self.H = H  # history
        self.f = f  # frequency
        self.dt = dt
        self.route_downsample = route_downsample
        self.emb_dim = emb_dim
        self.vae_dim = vae_dim
        self.vae_beta = vae_beta
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.num_map_enc_layers = num_map_enc_layers
        self.num_vae_enc_layers = num_vae_enc_layers
        self.num_heads = num_heads
        self.tx_hidden_size = tx_hidden_factor * self.emb_dim
        self.activation = activation
        self.dropout = dropout
        self.lr = lr
        self.betaW = betaW
        self.lr_decay = lr_decay
        self.warmup_steps = warmup_steps
        self.wd = wd
        self.norm_first = norm_first

        self.min_std = min_std

        self.max_token_distance = max_token_distance
        self.carla_maps_dict = {}
        for town in TOWNS:
            self.carla_maps_dict[town] = CarlaMapFeatures(
                town,
                map_data_path=carla_maps_path,
                torch_device=self.device,
                max_token_distance=max_token_distance,
                max_obs_distance=2. * max_token_distance + (30. * self.dt * self.T),
                max_z_distance=max_z_distance,
                max_map_pts=self.num_map_pts,
            )

        self._dynamic_feature_names = ['vehicle_features', 'vehicle_masks', 'light_features', 'light_masks', 'walker_features', 'walker_masks']

        self.create_embedders()
        self.create_encoder()
        self.create_decoder()
        self.create_output_model()
        self.create_vae_encoder()

        self.apply(self._init_weights)

        self.pid_type = pid_type

        if self.pid_type == 'fast':
            self.fast_turn_controller = PIDController(K_P=3.0, K_I=0., K_D=0.3, n=20)
            self.fast_speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=20)
        elif self.pid_type == 'scenarios':
            self.fast_turn_controller = PIDController(K_P=1.4, K_I=0., K_D=0., n=20)
            self.fast_speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=20)
        else:
            raise NotImplementedError

    def create_embedders(self):
        self.scene_encoder = SceneEncoder(self.emb_dim, self.num_local_pts, layer_norm=not self.norm_first, use_route=True)

    def create_encoder(self):
        self.enc_cross_layers = []
        for _ in range(self.num_enc_layers):
            enc_cross_layer = CrossAttentionLayer(
                d_model=self.emb_dim,
                nhead=self.num_heads,
                dropout=self.dropout,
                activation=self.activation,
                norm_first=self.norm_first,
                batch_first=True,
                mem_norm=self.norm_first,
            )
            self.enc_cross_layers.append(enc_cross_layer)
        self.enc_cross_layers = nn.ModuleList(self.enc_cross_layers)

        self.enc_ff_layers = []
        for _ in range(self.num_enc_layers):
            enc_ff_layer = FFLayer(
                d_model=self.emb_dim,
                dropout=self.dropout,
                dim_feedforward=self.tx_hidden_size,
                activation=self.activation,
                norm_first=self.norm_first,
            )
            self.enc_ff_layers.append(enc_ff_layer)
        self.enc_ff_layers = nn.ModuleList(self.enc_ff_layers)

    def create_decoder(self):
        self.dec_cross_layers = []
        for _ in range(self.num_dec_layers):
            self.dec_cross_layers.append(
                CrossAttentionLayer(
                    d_model=self.emb_dim,
                    nhead=self.num_heads,
                    dropout=self.dropout,
                    activation=self.activation,
                    norm_first=self.norm_first,
                    batch_first=True,
                    mem_norm=self.norm_first,
                ))
        self.dec_cross_layers = nn.ModuleList(self.dec_cross_layers)

        self.dec_self_layers = []
        for _ in range(self.num_dec_layers):
            self.dec_self_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=self.emb_dim,
                    nhead=self.num_heads,
                    dropout=self.dropout,
                    dim_feedforward=self.tx_hidden_size,
                    activation=self.activation,
                    norm_first=self.norm_first,
                    batch_first=True,
                ))
        self.dec_self_layers = nn.ModuleList(self.dec_self_layers)

    def create_output_model(self):
        self.output_model = OutputModel(
            emb_dim=self.emb_dim,
            dist_dim=4,
            T=self.T,
            min_std=self.min_std,
            layer_norm=self.norm_first,
            dropout=self.dropout,
            output_logit=False,
        )

    def create_vae_encoder(self):
        # Doesn't need to be relative
        self.vae_encoder = VAEEncoder(
            vae_dim=self.vae_dim,
            T=self.T,
            emb_dim=self.emb_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            tx_hidden_size=self.tx_hidden_size,
            activation=self.activation,
            norm_first=self.norm_first,
            num_vae_enc_layers=self.num_vae_enc_layers,
        )

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _get_map_features(self, obs):
        refs = obs['ref']
        towns = obs['town']

        B = len(towns)

        map_features = torch.zeros((B, self.num_map_pts, 10), dtype=torch.float32, device=self.device)
        map_masks = torch.ones((B, self.num_map_pts), dtype=bool, device=self.device)

        for town in np.unique(towns):
            idxs = np.where(town == towns)[0]
            if isinstance(town, bytes):
                town_map_features, town_map_masks = self.carla_maps_dict[town.decode('utf-8')].get_model_features(refs[idxs])
            else:
                town_map_features, town_map_masks = self.carla_maps_dict[town].get_model_features(refs[idxs])
            map_features[idxs] = town_map_features
            map_masks[idxs] = town_map_masks
        return map_features, map_masks

    def process_observations(self, obs):
        agents_features = obs['vehicle_features'][:, -1]
        agents_masks = ~obs['vehicle_masks'][:, -1]

        if 'map_features' not in obs:
            map_features, map_masks = self._get_map_features(obs)
            map_masks = ~map_masks
        else:
            map_features = obs['map_features']
            map_masks = ~obs['map_masks']

        map_mask_out = (~map_masks).any(dim=0)
        map_features = map_features[:, map_mask_out]
        map_masks = map_masks[:, map_mask_out]

        light_features = obs['light_features'][:, -1]
        light_masks = ~obs['light_masks'][:, -1]

        light_mask_out = (~light_masks).any(dim=0)
        light_features = light_features[:, light_mask_out]
        light_masks = light_masks[:, light_mask_out]

        stop_features = obs['stop_features']
        stop_masks = ~obs['stop_masks']

        stop_mask_out = (~stop_masks).any(dim=0)
        stop_features = stop_features[:, stop_mask_out]
        stop_masks = stop_masks[:, stop_mask_out]

        walker_features = obs['walker_features'][:, -1]
        walker_masks = ~obs['walker_masks'][:, -1]

        walker_mask_out = (~walker_masks).any(dim=0)
        walker_features = walker_features[:, walker_mask_out]
        walker_masks = walker_masks[:, walker_mask_out]

        route_features = obs['route_features']
        route_masks = ~obs['route_masks']

        ref = agents_features[:, 0, :3]
        rel_pos = route_features[..., :2] - ref[..., :2].unsqueeze(1)
        theta = ref[..., 2]
        rot_matrix = torch.stack([
            torch.stack([torch.cos(theta), -torch.sin(theta)], dim=-1),
            torch.stack([torch.sin(theta),  torch.cos(theta)], dim=-1)
        ], dim=-2)
        transformed_pos = torch.matmul(rel_pos, rot_matrix)

        route_dists = transformed_pos.norm(dim=-1)
        route_masks = route_masks | (route_dists > 1.) | (transformed_pos[..., 0] < 0.)
        route_dists = torch.where(
            route_masks,
            torch.full_like(route_dists, torch.inf),
            route_dists)

        route_inds = torch.topk(route_dists, self.num_route_pts, largest=False).indices[:, self.route_downsample-1::self.route_downsample]
        route_features = route_features[torch.arange(route_inds.shape[0]).unsqueeze(1), route_inds]
        route_masks = route_masks[torch.arange(route_inds.shape[0]).unsqueeze(1), route_inds]

        return agents_features, agents_masks, light_features, light_masks, stop_features, stop_masks, walker_features, walker_masks, route_features, route_masks, map_features, map_masks

    def enc_fn(
            self,
            agents_emb,
            agents_to_agents_emb,
            agents_to_agents_masks,
            scene_to_agents_emb,
            scene_to_agents_masks,
            cross_layer,
            ff_layer,
    ):
        #  :param agents_emb: (B, A, d)
        #  :param agents_to_agents_masks: (B, A, A)
        #  :param agents_to_agents_emb: (B, A, A, d)
        #  :param scene_to_agents_emb: (B, A, P, d)
        #  :param scene_to_agents_masks: (B, A, P)
        #  :return: (B, A, d)
        B, A, P, d = scene_to_agents_emb.shape

        src_agents_emb = rearrange(agents_emb, 'b a d -> (b a) 1 d')

        cross_agents_emb = rearrange(agents_emb, 'b a d -> b 1 a d') + agents_to_agents_emb
        cross_agents_emb = rearrange(cross_agents_emb, 'b a1 a2 d -> (b a1) a2 d')
        cross_agents_masks = rearrange(agents_to_agents_masks, 'b a1 a2 -> (b a1) a2')

        cross_scene_emb = rearrange(scene_to_agents_emb, 'b a p d -> (b a) p d')
        cross_scene_masks = rearrange(scene_to_agents_masks, 'b a p -> (b a) p')

        cross_total_emb = torch.cat([cross_agents_emb, cross_scene_emb], dim=1)
        cross_total_masks = torch.cat([cross_agents_masks, cross_scene_masks], dim=1)
        cross_total_masks = torch.where(cross_total_masks.all(dim=-1, keepdims=True), torch.zeros_like(cross_total_masks), cross_total_masks)

        src_agents_emb = cross_layer(
            src_agents_emb,
            cross_total_emb,
            memory_key_padding_mask=cross_total_masks)

        src_agents_emb = ff_layer(src_agents_emb)

        res_agents_emb = rearrange(src_agents_emb, '(b a) 1 d -> b a d', b=B, a=A)
        return res_agents_emb

    def dec_fn(
            self,
            out_emb,
            agents_emb,
            agents_to_agents_emb,
            agents_to_agents_masks,
            scene_to_agents_emb,
            scene_to_agents_masks,
            cross_layer,
            self_layer,
    ):
        #  :param out_emb: (B, N, M, d)
        #  :param agents_emb: (B, A, d)
        #  :param agents_to_agents_emb: (B, N, A, d)
        #  :param agents_to_agents_masks: (B, N, A)
        #  :param scene_to_agents_emb: (B, N, P, d)
        #  :param scene_to_agents_masks: (B, N, P)
        #  :return: (B, N, M, d)
        B, N, M, d = out_emb.shape

        src_out_emb = rearrange(out_emb, 'b n m d -> (b n) m d')

        cross_agents_emb = rearrange(agents_emb, 'b n d -> b 1 n d') + agents_to_agents_emb
        cross_agents_emb = rearrange(cross_agents_emb, 'b n a d -> (b n) a d')
        cross_agents_masks = rearrange(agents_to_agents_masks, 'b n a -> (b n) a')

        cross_scene_emb = rearrange(scene_to_agents_emb, 'b n p d -> (b n) p d')
        cross_scene_masks = rearrange(scene_to_agents_masks, 'b n p -> (b n) p')

        cross_total_emb = torch.cat([cross_agents_emb, cross_scene_emb], dim=1)
        cross_total_masks = torch.cat([cross_agents_masks, cross_scene_masks], dim=1)
        cross_total_masks = torch.where(cross_total_masks.all(dim=-1, keepdims=True), torch.zeros_like(cross_total_masks), cross_total_masks)

        src_out_emb = cross_layer(
            src_out_emb,
            cross_total_emb,
            memory_key_padding_mask=cross_total_masks)

        src_out_emb = self_layer(
            src_out_emb,
        )

        src_out_emb = rearrange(src_out_emb, '(b n) m d -> b n m d', b=B, n=N)
        return src_out_emb

    def get_encoding(self, features):
        agents_features, agents_masks, light_features, light_masks, stop_features, stop_masks, walker_features, walker_masks, route_features, route_masks, map_features, map_masks = features

        scene_to_agents_emb, scene_to_agents_masks, agents_emb, agents_to_agents_emb, agents_to_agents_masks, route_to_agents_emb, route_to_agents_masks = self.scene_encoder(
            map_features,
            map_masks,
            light_features,
            light_masks,
            stop_features,
            stop_masks,
            walker_features,
            walker_masks,
            agents_features,
            agents_masks,
            route_features,
            route_masks,
        )

        # Process through encoder
        for i in range(self.num_enc_layers):
            agents_emb = self.enc_fn(
                agents_emb,
                agents_to_agents_emb,
                agents_to_agents_masks,
                scene_to_agents_emb,
                scene_to_agents_masks,
                self.enc_cross_layers[i],
                self.enc_ff_layers[i],
            )

        scene_to_agents_emb = torch.cat([
            scene_to_agents_emb,
            route_to_agents_emb,
        ], dim=2)
        scene_to_agents_masks = torch.cat([
            scene_to_agents_masks,
            route_to_agents_masks,
        ], dim=2)

        return agents_emb, agents_to_agents_emb, agents_to_agents_masks, scene_to_agents_emb, scene_to_agents_masks

    def get_vae_encoding(self, agents_emb, agents_masks, gt_wps, wps_mask):
        z, dist = self.vae_encoder(agents_emb, agents_masks, gt_wps, wps_mask)
        return z.unsqueeze(1), dist

    def get_decoding(self, z, agents_emb, agents_to_agents_emb, agents_to_agents_masks, scene_to_agents_emb, scene_to_agents_masks):
        B, M, A, _ = z.shape

        z_emb = self.vae_encoder.embed_z(z)
        out_seq = rearrange(agents_emb, 'b a d -> b a 1 d') + rearrange(z_emb, 'b m a d -> b a m d')

        for d in range(self.num_dec_layers):
            out_seq = self.dec_fn(
                out_seq,
                agents_emb,
                agents_to_agents_emb,
                agents_to_agents_masks,
                scene_to_agents_emb,
                scene_to_agents_masks,
                self.dec_cross_layers[d],
                self.dec_self_layers[d],
            )

        return out_seq

    def get_output(self, out_seq):
        B, A, M, _ = out_seq.shape
        outputs = self.output_model(rearrange(out_seq, 'b a m d -> (b a) m 1 d'))
        outputs = rearrange(outputs, '(b a) m 1 t d -> b m t a d', b=B, a=A)
        return outputs

    def get_pred_next_wps(self, outputs):
        output_thetas = torch.cumsum(outputs[..., 2], dim=2)
        output_speeds = torch.cumsum(outputs[..., 3:4], dim=2)

        thetas = output_thetas - outputs[..., 2]

        rot_matrix = torch.stack([
            torch.stack([ torch.cos(thetas), torch.sin(thetas)], dim=-1),
            torch.stack([-torch.sin(thetas), torch.cos(thetas)], dim=-1)
        ], dim=-2)

        output_pos_diffs = torch.matmul(outputs[..., None, :2], rot_matrix).squeeze(-2)
        output_pos = torch.cumsum(output_pos_diffs, dim=2)

        outputs_mean = torch.cat([output_pos, output_thetas.unsqueeze(-1), output_speeds], dim=-1)

        outputs_std = outputs[..., 4:]
        pred_next_wps = torch.cat([outputs_mean, outputs_std], dim=-1)
        return pred_next_wps

    def forward(self, obs, gt_wps, wps_mask):
        features = self.process_observations(obs)
        agents_emb, agents_to_agents_emb, agents_masks, scene_to_agents_emb, scene_to_agents_masks = self.get_encoding(features)
        z, dist = self.get_vae_encoding(agents_emb, agents_masks, gt_wps, wps_mask)
        out_seq = self.get_decoding(z, agents_emb, agents_to_agents_emb, agents_masks, scene_to_agents_emb, scene_to_agents_masks)
        outputs = self.get_output(out_seq)
        pred_next_wps = self.get_pred_next_wps(outputs)
        return pred_next_wps, dist, features

    def inference_forward(self, obs, z):
        features = self.process_observations(obs)
        agents_emb, agents_to_agents_emb, agents_masks, scene_to_agents_emb, scene_to_agents_masks = self.get_encoding(features)
        out_seq = self.get_decoding(z, agents_emb, agents_to_agents_emb, agents_masks, scene_to_agents_emb, scene_to_agents_masks)
        outputs = self.get_output(out_seq)
        pred_next_wps = self.get_pred_next_wps(outputs)
        return pred_next_wps, features

    def _compute_regression_loss(self, pred, gt):
        regression_loss = nll_pytorch_dist(pred, gt, dist='normal')
        mse = F.mse_loss(pred[..., :gt.shape[-1]], gt, reduction='none')
        return regression_loss, mse

    def _compute_loss(self, pred_wps, dists, gt_wps, wps_mask):
        B, _, A = wps_mask.shape
        shaped_pred_wps = rearrange(pred_wps, 'b 1 t a d -> b t a d')
        shaped_gt_wps = gt_wps
        shaped_wps_regression_loss, shaped_wps_mse_errors = self._compute_regression_loss(shaped_pred_wps, shaped_gt_wps)

        time_wps_regression_loss = rearrange(shaped_wps_regression_loss, 'b t a -> (b a) t')
        time_wps_mse_errors = rearrange(shaped_wps_mse_errors, 'b t a d -> (b a) t d')
        picked_time_wps_mse_errors = rearrange(shaped_wps_mse_errors, 'b t a d -> (b a) t d')
        w_mask = rearrange(wps_mask, 'b t a -> (b a) t')
        time_count = torch.clamp(torch.sum(w_mask, dim=1), min=1.)

        wps_regression_loss = torch.sum(time_wps_regression_loss * w_mask, dim=1) / time_count
        picked_wps_mse_errors = torch.sum(time_wps_mse_errors * w_mask.unsqueeze(-1), dim=1) / time_count.unsqueeze(-1)
        picked_wps_mse_errors = rearrange(picked_wps_mse_errors, '(b a) d -> b a d', b=B, a=A)

        agents_wps_m = wps_mask.any(dim=1)
        agents_wps_loss = rearrange(wps_regression_loss, '(b a) -> b a', b=B, a=A)

        ego_wps_loss = agents_wps_loss[:, :1][agents_wps_m[:, :1]].mean()
        other_wps_loss = agents_wps_loss[:, 1:][agents_wps_m[:, 1:]].mean()

        wps_loss = 0.5 * (ego_wps_loss + other_wps_loss)

        kl_target = Normal(torch.zeros_like(dists.loc), torch.ones_like(dists.scale))
        unmasked_kl_loss = kl_divergence(dists, kl_target).sum(dim=-1)

        kl_mask = wps_mask.any(dim=1)
        ego_kl_loss = unmasked_kl_loss[:, :1][kl_mask[:, :1]].mean()
        other_kl_loss = unmasked_kl_loss[:, 1:][kl_mask[:, 1:]].mean()
        kl_loss = 0.5 * (ego_kl_loss + other_kl_loss)

        loss = wps_loss + self.vae_beta * kl_loss

        agents_wps_m = reduce(wps_mask, 'b t a -> b a 1', 'max')
        ego_wps_mse_errors = torch.sum(picked_wps_mse_errors[:, 0] * agents_wps_m[:, 0], dim=0) / torch.clamp(agents_wps_m[:, 0].sum(), min=1.)
        other_wps_mse_errors = torch.sum(picked_wps_mse_errors[:, 1:] * agents_wps_m[:, 1:], dim=(0, 1)) / torch.clamp(agents_wps_m[:, 1:].sum(), min=1.)
        wps_mse_errors = torch.sum(picked_wps_mse_errors * agents_wps_m, dim=(0,1)) / torch.clamp(agents_wps_m.sum(), min=1.)
        time_wps_mse = torch.sum(picked_time_wps_mse_errors * rearrange(wps_mask, 'b t a -> (b a) t 1'), dim=0) / torch.clamp(reduce(wps_mask, 'b t a -> t 1', 'sum'), min=1.)

        wps_errors = torch.sqrt(wps_mse_errors)
        time_wps_errors = torch.sqrt(time_wps_mse)
        ego_wps_errors = torch.sqrt(ego_wps_mse_errors)
        other_wps_errors = torch.sqrt(other_wps_mse_errors)
        return loss, wps_loss, kl_loss, wps_errors, time_wps_errors, ego_wps_errors, other_wps_errors

    def run_step(self, batch, prefix):
        full_obs = batch.get_obs(ref_t=self.H-1, f=self.f, device=self.device)
        obs = {}
        for k in full_obs:
            if k in self._dynamic_feature_names:
                obs[k] = full_obs[k][:, :self.H]
            else:
                obs[k] = full_obs[k]

        veh_masks = obs['vehicle_masks'].any(dim=1).any(dim=0)
        obs['vehicle_features'] = obs['vehicle_features'][:, :, veh_masks]
        obs['vehicle_masks'] = obs['vehicle_masks'][:, :, veh_masks]

        gt_wps, wps_mask = batch.get_traj(ref_t=self.H-1, f=self.f)
        gt_wps = gt_wps[:, self.H:self.H+self.T, :, :4].to(self.device)
        wps_mask = wps_mask[:, self.H:self.H+self.T].to(self.device)

        gt_wps = gt_wps[:, :, veh_masks]
        wps_mask = wps_mask[:, :, veh_masks]

        pred_wps, dists, features = self.forward(obs, gt_wps, wps_mask)

        loss, wps_loss, kl_loss, wps_errors, time_wps_errors, ego_wps_errors, other_wps_errors = self._compute_loss(pred_wps, dists, gt_wps, wps_mask)
        return loss, pred_wps, features

    def training_step(self, batch, batch_idx):
        loss, *_ = self.run_step(batch, prefix='train')

        return loss

    def validation_step(self, batch, batch_idx):
        wp_loss, _, features = self.run_step(batch, prefix='val')
        return wp_loss

    def configure_optimizers(self):
        all_params = set(self.parameters())
        wd_params = set()
        for (name, m) in self.named_modules():
            if isinstance(m, nn.Linear):
                wd_params.add(m.weight)
        no_wd_params = all_params - wd_params
        main_optimizer = torch.optim.AdamW(
            [{'params': list(no_wd_params), 'weight_decay': 0}, {'params': list(wd_params), 'weight_decay': self.wd}],
            lr=self.lr,
            betas=(0.9, self.betaW))
        self.scheduler = get_cosine_schedule_with_warmup(main_optimizer, self.warmup_steps, self.trainer.estimated_stepping_batches, final_factor=self.lr_decay)
        main_d = {"optimizer": main_optimizer, "lr_scheduler": {"scheduler": self.scheduler, "interval": "step"}}
        return main_d

    @torch.inference_mode()
    def get_action(self, full_obs):
        obs = {}
        for k in full_obs:
            if k in self._dynamic_feature_names:
                obs[k] = full_obs[k][:, -self.H:]
            else:
                obs[k] = full_obs[k]

        veh_masks = obs['vehicle_masks'].any(dim=1).any(dim=0)
        obs['vehicle_features'] = obs['vehicle_features'][:, :, veh_masks]
        obs['vehicle_masks'] = obs['vehicle_masks'][:, :, veh_masks]

        pred_wps, logits, features = self.forward(obs)

        if logits is None:
            ego_waypoints = pred_wps[0, 0, :, 0, :]
        else:
            ego_logits = logits[0, :, 0]
            ego_labels = ego_logits.argmax()
            ego_wp_diffs = pred_wps[0, :, :, 0, :]
            ego_waypoints = ego_wp_diffs[ego_labels]

        ego_speed = obs['vehicle_features'][0, -1, 0, 3]

        actions = self.control_pid(ego_waypoints, ego_speed)
        return actions

    def control_pid(self, waypoints, curr_speed):
        # converts predictions to actions
        wps = waypoints[:, :2].data.cpu().numpy() * self.max_token_distance
        speed = curr_speed.data.cpu().numpy() * self.max_token_distance

        desired_speed = speed + waypoints[0, 3].data.cpu().numpy() * self.max_token_distance

        if self.pid_type == 'fast':
            brake_speed = 0.4
            brake = desired_speed < brake_speed or ((speed / desired_speed) > 1.1)

            delta = np.clip(desired_speed - speed, 0.0, 0.25)
            throttle = self.fast_speed_controller.step(delta)
            throttle = np.clip(throttle, 0.0, 0.75)
            if brake:
                gas = -1.0
            else:
                gas = throttle

            aim = wps[0]

            angle = np.arctan2(aim[1], aim[0]) * 2. / np.pi
            if desired_speed < brake_speed:
                angle = 0.0
            steer = self.fast_turn_controller.step(angle)
            steer = np.clip(steer, -1.0, 1.0)
        elif self.pid_type == 'scenarios':
            brake_speed = 0.4
            brake = desired_speed < brake_speed or ((speed / desired_speed) > 1.1)

            delta = np.clip(desired_speed - speed, 0.0, 0.25)
            throttle = self.fast_speed_controller.step(delta)
            throttle = np.clip(throttle, 0.0, 0.75)
            if brake:
                gas = -0.25
            else:
                gas = throttle

            aim = wps[0]
            angle = np.arctan2(aim[1], aim[0]) * 2. / np.pi
            if desired_speed < brake_speed:
                angle = 0.0
            steer = self.fast_turn_controller.step(angle)
            steer = np.clip(steer, -1.0, 1.0)
        else:
            raise NotImplementedError

        return np.array((steer, gas))
