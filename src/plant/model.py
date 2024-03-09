import logging
import numpy as np
import torch
import torch.nn as nn
from einops import repeat
from transformers import (
    AutoConfig,
    AutoModel,
)
from rdp import rdp
import math


from src.plant.pid_controller import PIDController


logger = logging.getLogger(__name__)


class HFLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.object_types = 2 + 1  # vehicles, route +1 for padding and wp embedding
        self.num_attributes = 6  # x,y,yaw,speed/id, extent x, extent y

        precisions = [
            self.config.pre_training.get("precision_speed", 4),
            self.config.pre_training.get("precision_pos", 4),
            self.config.pre_training.get("precision_pos", 4),
            self.config.pre_training.get("precision_angle", 4),
            self.config.pre_training.get("precision_pos", 4),
            self.config.pre_training.get("precision_pos", 4),
        ]

        self.vocab_size = [2**i for i in precisions]

        # model
        config = AutoConfig.from_pretrained(
            self.config.network.hf_checkpoint
        )  # load config from hugging face model
        n_embd = config.hidden_size
        self.model = AutoModel.from_config(config=config)

        self.cls_emb = nn.Parameter(
            torch.randn(1, self.num_attributes)
        )  # +1 because at this step we still have the type indicator

        # token embedding
        self.tok_emb = nn.Linear(self.num_attributes, n_embd)
        # object type embedding
        self.obj_token = nn.ParameterList(
            [
                nn.Parameter(torch.randn(1, self.num_attributes))
                for _ in range(self.object_types)
            ]
        )
        self.obj_emb = nn.ModuleList(
            [nn.Linear(self.num_attributes, n_embd) for _ in range(self.object_types)]
        )
        self.drop = nn.Dropout(self.config.network.embd_pdrop)

        # one head for each attribute type -> we have different precision per attribute
        self.heads = nn.ModuleList(
            [
                nn.Linear(n_embd, self.vocab_size[i])
                for i in range(self.num_attributes)
            ]
        )

        # wp (CLS) decoding
        self.wp_head = nn.Linear(n_embd, 64)
        self.wp_decoder = nn.GRUCell(input_size=4, hidden_size=65)
        self.wp_relu = nn.ReLU()
        self.wp_output = nn.Linear(65, 2)

        # PID controller
        # changed for our setup
        self.turn_controller = PIDController(K_P=1.4, K_I=0., K_D=0., n=20)
        self.speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=20)

        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = torch.nn.Linear
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith("_ih") or pn.endswith("_hh"):
                    # all recurrent weights will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("_emb") or "_token" in pn:
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer


    def forward(self, full_obs, device='cpu', return_targets=False):
        B = full_obs['vehicle_features'].shape[0]

        route_dists = full_obs['route_features'].norm(dim=-1)
        route_masks = full_obs['route_masks'] & (route_dists <= 1.)
        route_dists = torch.where(
            route_masks,
            route_dists,
            torch.full_like(route_dists, torch.inf),
        )

        route_inds = torch.topk(route_dists, 30, largest=False).indices
        route_features = full_obs['route_features'][torch.arange(route_inds.shape[0]).unsqueeze(1), route_inds]
        route_masks = route_masks[torch.arange(route_inds.shape[0]).unsqueeze(1), route_inds]

        processed_route_features = []
        processed_route_masks = []
        tps = []
        for i in range(B):
            r_features = route_features[i][route_masks[i]]
            if r_features.shape[0] > 0:
                tps.append(r_features[:10][-1, :2] * 50.)
                route_pts = r_features[..., :2].cpu().numpy() * 50.
                route_lw = r_features[..., 3].cpu().numpy() * 50.

                shortened_route = rdp(route_pts, epsilon=0.5)

                vectors = shortened_route[1:] - shortened_route[:-1]
                midpoints = shortened_route[:-1] + vectors / 2.
                norms = np.linalg.norm(vectors, axis=1)
                angles = np.arctan2(vectors[:, 1], vectors[:, 0])

                route_x = midpoints[:, 0]
                route_y = midpoints[:, 1]
                route_yaw = angles
                route_w = route_lw[:route_x.shape[0]]
                route_h = norms
                inp_route_features = np.stack([
                    route_x,
                    route_y,
                    route_yaw,
                    route_w,
                    route_h
                ], axis=-1)
                data_route_features = []
                for rf in inp_route_features:
                    if rf[-1] > 10.:
                        number_of_points = math.ceil(rf[-1] / 10) - 1
                        x1 = rf[0] + 0.5 * rf[-1] * math.cos(rf[2])
                        y1 = rf[1] + 0.5 * rf[-1] * math.sin(rf[2])
                        x0 = rf[0] - 0.5 * rf[-1] * math.cos(rf[2])
                        y0 = rf[1] - 0.5 * rf[-1] * math.sin(rf[2])
                        xs = np.linspace(x0, x1, number_of_points + 2)
                        ys = np.linspace(y0, y1, number_of_points + 2)
                        for j in range(number_of_points + 1):
                            new_rf = np.array([
                                0.5 * (xs[j] + xs[j+1]),
                                0.5 * (ys[j] + ys[j+1]),
                                rf[2],
                                rf[3],
                                rf[4] / (number_of_points + 1),
                            ])
                            data_route_features.append(new_rf)
                            if len(data_route_features) >= 2:
                                break
                    else:
                        data_route_features.append(rf)
                        if len(data_route_features) >= 2:
                            break

                data_route_features = torch.tensor(np.array(data_route_features[:2]), dtype=torch.float32, device=device)
                pad_len = 2 - data_route_features.shape[0]
                data_route_masks = torch.cat([torch.ones(data_route_features.shape[0], dtype=bool, device=device), torch.zeros(pad_len, dtype=bool, device=device)], dim=0)
                data_route_features = torch.cat([data_route_features, torch.zeros((pad_len, 5), dtype=torch.float32, device=device)], dim=0)
            else:
                tps.append(torch.zeros(2, dtype=torch.float32, device=device))
                data_route_features = torch.zeros((2, 5), dtype=torch.float32, device=device)
                data_route_masks = torch.zeros(2, dtype=bool, device=device)

            processed_route_features.append(data_route_features)
            processed_route_masks.append(data_route_masks)

        processed_route_features = torch.stack(processed_route_features, dim=0)
        processed_route_masks = torch.stack(processed_route_masks, dim=0)
        tps = torch.stack(tps, dim=0)
        processed_route_features = torch.cat([repeat(torch.arange(2, device=device), 'n -> b n 1', b=B), processed_route_features], dim=-1)

        # vehicle features
        vehicle_features = full_obs['vehicle_features'][:, 0]
        vehicle_masks = full_obs['vehicle_masks'][:, 0]

        veh_mask_out = vehicle_masks.any(dim=0)
        processed_vehicle_masks = vehicle_masks[:, veh_mask_out]
        vehicle_features = vehicle_features[:, veh_mask_out]

        vehicle_speed = vehicle_features[:, :, 3] * 50.
        vehicle_x = vehicle_features[:, :, 0] * 50.
        vehicle_y = vehicle_features[:, :, 1] * 50.
        vehicle_yaw = vehicle_features[:, :, 2]
        vehicle_w = vehicle_features[:, :, 5] * 50.
        vehicle_l = vehicle_features[:, :, 4] * 50.

        processed_vehicle_features = torch.stack([
            vehicle_speed,
            vehicle_x,
            vehicle_y,
            vehicle_yaw,
            vehicle_w,
            vehicle_l,
        ], dim=-1)

        cls_obj_embedding = self.obj_emb[2](self.obj_token[2])
        cls_embeddings = repeat(self.tok_emb(self.cls_emb) + cls_obj_embedding, '1 d -> b 1 d', b=B)

        vehicle_obj_embedding = self.obj_emb[0](self.obj_token[0])
        vehicle_embeddings = self.tok_emb(processed_vehicle_features) + vehicle_obj_embedding

        route_obj_embedding = self.obj_emb[1](self.obj_token[1])
        route_embeddings = self.tok_emb(processed_route_features) + route_obj_embedding

        embedding = torch.cat([cls_embeddings, vehicle_embeddings, route_embeddings], dim=1)
        # embedding dropout
        x = self.drop(embedding)

        # Transformer Encoder; use embedding for hugging face model and get output states and attention map
        output = self.model(**{"inputs_embeds": embedding}, output_attentions=True)
        x, attn_map = output.last_hidden_state, output.attentions

        num_vehicles = processed_vehicle_masks.shape[1]
        logit_embedding = x[:, 1:num_vehicles+1]

        targets_masks = full_obs['vehicle_masks'][:, :2, veh_mask_out].all(dim=1)

        logits = [
            self.heads[i](logit_embedding)[targets_masks] for i in range(self.num_attributes)
        ]


        if return_targets:
            next_features = full_obs['vehicle_features'][:, 1, veh_mask_out]

            next_speed = next_features[:, :, 3] * 50.
            next_x = next_features[:, :, 0] * 50.
            next_y = next_features[:, :, 1] * 50.
            next_yaw = next_features[:, :, 2]
            next_w = next_features[:, :, 5] * 50.
            next_l = next_features[:, :, 4] * 50.

            processed_next_features = torch.stack([
                next_speed,
                next_x,
                next_y,
                next_yaw,
                next_w,
                next_l,
            ], dim=-1)
            targets = self.quantize_data(processed_next_features)
            targets = targets[targets_masks]

        # get waypoint predictions
        z = self.wp_head(x[:, 0, :])
        # add traffic ligth flag
        z = torch.cat((z, torch.zeros_like(z[:, :1])), 1)

        output_wp = list()

        # initial input variable to GRU
        x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype)
        x = x.type_as(z)

        # autoregressive generation of output waypoints
        for _ in range(self.config.training.pred_len):
            x_in = torch.cat([x, tps], dim=1)
            z = self.wp_decoder(x_in, z)
            dx = self.wp_output(z)
            x = dx + x
            output_wp.append(x)

        pred_wp = torch.stack(output_wp, dim=1)

        if return_targets:
            # target wps
            target_wp = full_obs['vehicle_features'][:, 1:1+self.config.training.pred_len, 0, :2] * 50.
            target_wp_masks = full_obs['vehicle_masks'][:, 1:1+self.config.training.pred_len, 0]

            pred_wp = pred_wp[target_wp_masks]
            target_wp = target_wp[target_wp_masks]

        if return_targets:
            return logits, targets, pred_wp, target_wp, attn_map
        else:
            return logits, pred_wp, attn_map

    def quantize_data(self, data):
        normed_speed = data[..., 0] / 50.
        normed_x = (data[..., 1] + 50.) / 100.
        normed_y = (data[..., 2] + 50.) / 100.
        normed_yaw = (data[..., 3] + math.pi) / (2. * math.pi)
        normed_w = data[..., 4] / 50.
        normed_l = data[..., 5] / 50.
        normed_data = torch.stack([
            normed_speed,
            normed_x,
            normed_y,
            normed_yaw,
            normed_w,
            normed_l,
        ], dim=-1)
        normed_data = torch.clamp(normed_data, min=0., max=1.)
        rounded_data = normed_data * (torch.tensor(self.vocab_size, device=data.device) - 1)
        quantized_data = rounded_data.round().long()
        return quantized_data

    def control_pid(self, waypoints, velocity, is_stuck=False):
        """Predicts vehicle control with a PID controller.
        Args:
            waypoints (tensor): output of self.plan()
            velocity (tensor): speedometer input
        """
        assert waypoints.size(0) == 1
        waypoints = waypoints[0].data.cpu().numpy()

        speed = velocity[0].data.cpu().numpy()

        desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0
        if is_stuck:
            desired_speed = np.array(4.0) # default speed of 14.4 km/h

        brake = desired_speed < 0.4 or (speed / desired_speed) > 1.1

        delta = np.clip(desired_speed - speed, 0.0, 0.25)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)
        throttle = throttle if not brake else 0.0
        aim = (waypoints[1] + waypoints[0]) / 2.0
        angle = np.degrees(np.arctan2(aim[1], aim[0])) / 90
        # changed for our setup
        if desired_speed < 0.4:
            angle = 0.0
        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        return steer, throttle, brake
