from src.carla.merge_scenarios.agents.navigation import planner


class ControlledAgent:
    def __init__(
            self,
            config,
            world,
            vehicle,
            source_transform,
            destination_transform,
            **kwargs):

        map = world.get_map()

        self.global_planner = planner.GlobalPlanner()
        self.destination_transform = destination_transform
        dense_waypoints = self.global_planner.trace_route(map, source_transform, self.destination_transform)
        self.global_planner.set_global_plan(dense_waypoints)
