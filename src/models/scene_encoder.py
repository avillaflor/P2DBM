import torch
from torch import nn
from einops import repeat


from src.models.embedders import Embedder, RelativeEmbedder, RelativeFeatureEmbedder


class SceneEncoder(nn.Module):
    # Handles the initial emebdding of the different entities in the scene
    def __init__(self, emb_dim, num_local_pts=50, layer_norm=False, use_route=False):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_local_pts = num_local_pts
        self.layer_norm = layer_norm
        self.use_route = use_route

        self.map_embedder = Embedder(7, self.emb_dim, layer_norm=self.layer_norm)
        self.map_to_agents_embedder = RelativeFeatureEmbedder(4, self.emb_dim, layer_norm=self.layer_norm)

        self.lights_embedder = Embedder(8, self.emb_dim, layer_norm=self.layer_norm)
        self.lights_to_agents_embedder = RelativeEmbedder(self.emb_dim, layer_norm=self.layer_norm)

        self.stops_embedder = Embedder(5, self.emb_dim, layer_norm=self.layer_norm)
        self.stops_to_agents_embedder = RelativeEmbedder(self.emb_dim, layer_norm=self.layer_norm)

        self.walkers_embedder = Embedder(6, self.emb_dim, layer_norm=self.layer_norm)
        self.walkers_to_agents_embedder = RelativeEmbedder(self.emb_dim, layer_norm=self.layer_norm)

        if self.use_route:
            self.route_embedder = Embedder(4, self.emb_dim, layer_norm=self.layer_norm)
            self.route_to_agents_embedder = RelativeEmbedder(self.emb_dim, layer_norm=self.layer_norm)

        self.ego_agents_embedder = Embedder(7, self.emb_dim, layer_norm=self.layer_norm)
        self.agents_embedder = Embedder(7, self.emb_dim, layer_norm=self.layer_norm)
        self.agents_to_agents_embedder = RelativeEmbedder(self.emb_dim, layer_norm=self.layer_norm)

    def forward(
            self,
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
            route_features=None,
            route_masks=None,
    ):
        B, A = agents_masks.shape

        map_to_agents_distances = torch.norm(map_features[..., :2].unsqueeze(-3) - agents_features[..., :2].unsqueeze(-2), dim=-1)
        map_to_agents_distances = torch.where(
            map_masks.unsqueeze(-2),
            torch.ones_like(map_to_agents_distances) * torch.inf,
            map_to_agents_distances)

        _, closest_inds = torch.topk(map_to_agents_distances, self.num_local_pts, largest=False)

        map_to_agents_features = map_features[torch.arange(B).view((B, 1, 1)), closest_inds]
        map_to_agents_masks = map_masks[torch.arange(B).view((B, 1, 1)), closest_inds]

        map_to_agents_egoagent_distances = map_to_agents_distances[:, 0][torch.arange(B).view((B, 1, 1)), closest_inds]
        map_to_agents_unobserved = (map_to_agents_egoagent_distances > 1.).unsqueeze(-1).float()
        map_to_agents_emb, map_to_agents_pos = self.map_to_agents_embedder(map_to_agents_features, agents_features, map_to_agents_unobserved, return_features=True)

        map_emb = self.map_embedder(map_to_agents_features[..., :7])
        map_to_agents_emb = map_to_agents_emb + map_emb
        map_to_agents_masks = map_to_agents_masks | (map_to_agents_pos[..., :2].norm(dim=-1) > 1.)

        lights_emb = self.lights_embedder(light_features)
        lights_to_agents_emb, lights_to_agents_pos = self.lights_to_agents_embedder(light_features, agents_features, return_features=True)
        lights_to_agents_emb = lights_to_agents_emb + lights_emb.unsqueeze(1)
        lights_to_agents_masks = light_masks.unsqueeze(1) | (lights_to_agents_pos[..., :2].norm(dim=-1) > 1.)

        stops_emb = self.stops_embedder(stop_features)
        stops_to_agents_emb, stops_to_agents_pos = self.stops_to_agents_embedder(stop_features, agents_features, return_features=True)
        stops_to_agents_emb = stops_to_agents_emb + stops_emb.unsqueeze(1)
        stops_to_agents_masks = stop_masks.unsqueeze(1) | (stops_to_agents_pos[..., :2].norm(dim=-1) > 1.)

        walkers_emb = self.walkers_embedder(walker_features)
        walkers_to_agents_emb, walkers_to_agents_pos = self.walkers_to_agents_embedder(walker_features, agents_features, return_features=True)
        walkers_to_agents_emb = walkers_to_agents_emb + walkers_emb.unsqueeze(1)
        walkers_to_agents_masks = walker_masks.unsqueeze(1) | (walkers_to_agents_pos[..., :2].norm(dim=-1) > 1.)

        scene_to_agents_emb = torch.cat([map_to_agents_emb, lights_to_agents_emb, stops_to_agents_emb, walkers_to_agents_emb], dim=2)
        scene_to_agents_masks = torch.cat([map_to_agents_masks, lights_to_agents_masks, stops_to_agents_masks, walkers_to_agents_masks], dim=2)

        ego_agents_emb = self.ego_agents_embedder(agents_features[:, :1])
        other_agents_emb = self.agents_embedder(agents_features[:, 1:])
        agents_emb = torch.cat([ego_agents_emb, other_agents_emb], dim=1)

        agents_to_agents_emb, agents_to_agents_pos = self.agents_to_agents_embedder(agents_features, agents_features, return_features=True)
        agents_to_agents_masks = agents_masks.unsqueeze(1) | (agents_to_agents_pos[..., :2].norm(dim=-1) > 1.)

        if self.use_route:
            assert(route_features is not None)
            assert(route_masks is not None)
            route_emb = self.route_embedder(route_features)
            route_to_agents_emb, route_to_agents_pos = self.route_to_agents_embedder(route_features, agents_features[:, :1], return_features=True)
            route_to_agents_emb = route_to_agents_emb + route_emb.unsqueeze(1)
            route_to_agents_emb = torch.cat([route_to_agents_emb, repeat(torch.zeros_like(route_to_agents_emb), 'b 1 ... -> b a ...', a=A-1)], dim=1)
            route_to_agents_masks = torch.cat([route_masks.unsqueeze(1), repeat(torch.ones_like(route_masks), 'b ... -> b a ...', a=A-1)], dim=1)
            return scene_to_agents_emb, scene_to_agents_masks, agents_emb, agents_to_agents_emb, agents_to_agents_masks, route_to_agents_emb, route_to_agents_masks
        else:
            return scene_to_agents_emb, scene_to_agents_masks, agents_emb, agents_to_agents_emb, agents_to_agents_masks
