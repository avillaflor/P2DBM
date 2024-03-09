''' Planner '''
import math
import numpy as np
from collections import deque
import carla


from src.carla.merge_scenarios.agents.navigation.global_route_planner import GlobalRoutePlanner
from src.carla.merge_scenarios.agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO


class GlobalPlanner():

    def __init__(self):
        self._grp = None
        self._hop_resolution = 2.0
        self._min_distance_percentage = 0.9
        # queue with tuples of (waypoint, RoadOption)
        self._waypoints_queue = deque(maxlen=20000)
        self._waypoints_queue_old = deque(maxlen=20000)
        self.dist_to_trajectory = 0
        self.second_last_waypoint = None
        self.last_waypoint = None
        self._min_distance = self._hop_resolution * self._min_distance_percentage

    @property
    def hop_resolution(self):
        return self._hop_resolution

    def trace_route(self, map, start_transform, destination_transform):
        """
        This method sets up a global router and returns the optimal route
        from start_waypoint to end_waypoint
        """

        start_waypoint, end_waypoint = map.get_waypoint(start_transform.location), map.get_waypoint(destination_transform.location)

        # Setting up global router
        if self._grp is None:
            dao = GlobalRoutePlannerDAO(map, self._hop_resolution)
            grp = GlobalRoutePlanner(dao)
            grp.setup()
            self._grp = grp

        # Obtain route plan
        route = self._grp.trace_route(
            start_waypoint.transform.location,
            end_waypoint.transform.location)

        return route

    def _compute_distances_between_waypoints(self, current_plan):
        modified_plan = []
        last_waypoint = current_plan[-1][0]
        dist = 0
        for elem in reversed(current_plan):
            waypoint, unk = elem
            dist += last_waypoint.transform.location.distance(waypoint.transform.location)
            modified_plan.append((waypoint, unk, dist))
            last_waypoint = waypoint
        modified_plan.reverse()
        return modified_plan

    def set_global_plan(self, current_plan):
        self._waypoints_queue.clear()
        prev_wp = None
        modified_plan = self._compute_distances_between_waypoints(current_plan)
        for elem in modified_plan:
            if not self._same_waypoint(elem[0], prev_wp):
                self._waypoints_queue.append(elem)
                self._waypoints_queue_old.append(elem)
            prev_wp = elem[0]

    def _waypoints_to_list(self):
        wp_list = []
        for waypoint in self._waypoints_queue:
            wp_list.append([waypoint[0].transform.location.x, waypoint[0].transform.location.y, waypoint[0].transform.rotation.yaw])

        return wp_list

    def get_next_orientation(self, vehicle_transform, num_next_waypoints=5):

        next_waypoints_angles = []
        next_waypoints_vectors = []
        next_waypoints = []
        max_index = -1
        for i, (waypoint, _, dist) in enumerate(self._waypoints_queue):
            dist_i = waypoint.transform.location.distance(vehicle_transform.location)
            if dist_i < self._min_distance:
                max_index = i

        q_len = len(self._waypoints_queue)
        if max_index >= 0:
            for i in range(max_index + 1):
                waypoint, _, dist= self._waypoints_queue.popleft()

                if i == q_len - 1:
                    self.last_waypoint = waypoint
                elif i == q_len - 2:
                    self.second_last_waypoint = waypoint

        for i, (waypoint, _, dist) in enumerate(self._waypoints_queue):
            if i > num_next_waypoints - 1:
                break
            dot, angle, w_vec = self._get_dot_product_and_angle(vehicle_transform, waypoint)

            if len(next_waypoints_angles) == 0:
                next_waypoints_angles = [angle]
                next_waypoints = [waypoint]
                dist_to_goal = dist
                next_waypoints_vectors = [w_vec]
            else:
                next_waypoints_angles.append(angle)
                next_waypoints.append(waypoint)
                next_waypoints_vectors.append(w_vec)


        next_waypoints_angles_array = np.array(next_waypoints_angles)
        if len(next_waypoints_angles) > 0:
            angle = np.mean(next_waypoints_angles_array)
        else:
            print("No next waypoint found!")
            dist_to_goal = 0
            angle = 0

        if len(next_waypoints) > 1:
            self.dist_to_trajectory = self._get_point_to_line_distance(
                vehicle_transform,
                next_waypoints[0],
                next_waypoints[1])
        elif len(next_waypoints) > 0:
            self.dist_to_trajectory = self._get_point_to_line_distance(
                vehicle_transform,
                self.second_last_waypoint,
                next_waypoints[0])

        else:
            # Reached near last waypoint
            # use second_last_waypoint
            print("Needed to use second_last_waypoint")
            if self.second_last_waypoint is not None and self.last_waypoint is not None:
                self.dist_to_trajectory = self._get_point_to_line_distance(
                    vehicle_transform,
                    self.second_last_waypoint,
                    self.last_waypoint)
                next_waypoints = [self.second_last_waypoint, self.last_waypoint]
            else:
                self.dist_to_trajectory = 0
                next_waypoints = [self.last_waypoint]

        return angle, self.dist_to_trajectory, dist_to_goal, next_waypoints, next_waypoints_angles, next_waypoints_vectors, self._waypoints_to_list()

    def _get_dot_product_and_angle(self, vehicle_transform, waypoint):

        v_begin = vehicle_transform.location
        v_end = v_begin + carla.Location(
            x=math.cos(math.radians(vehicle_transform.rotation.yaw)),
            y=math.sin(math.radians(vehicle_transform.rotation.yaw)))

        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
        w_vec = np.array([waypoint.transform.location.x - v_begin.x, waypoint.transform.location.y - v_begin.y, 0.0])
        dot = np.dot(w_vec, v_vec)
        angle = math.acos(np.clip(np.dot(w_vec, v_vec) / (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))

        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            angle *= -1.0

        return dot, angle, w_vec[:2]

    def _get_point_to_line_distance(self, vehicle_transform, waypoint1, waypoint2):
        point = np.array([vehicle_transform.location.x, vehicle_transform.location.y])
        point1_on_line = np.array([waypoint1.transform.location.x, waypoint1.transform.location.y])
        point2_on_line = np.array([waypoint2.transform.location.x, waypoint2.transform.location.y])
        return self._get_point_to_line_distance_helper(point, point1_on_line, point2_on_line)

    def _get_point_to_line_distance_helper(self, point, point1_on_line, point2_on_line):
        a_vec = point2_on_line - point1_on_line
        b_vec = point - point1_on_line
        # returning signed distance
        return np.cross(a_vec, b_vec) / np.linalg.norm(a_vec)

    def _same_waypoint(self, waypoint1, waypoint2):

        if waypoint1 is None or waypoint2 is None:
            return True
        x1 = waypoint1.transform.location.x
        y1 = waypoint1.transform.location.y
        x2 = waypoint2.transform.location.x
        y2 = waypoint2.transform.location.y

        return (x1 == x2) and (y1 == y2)
