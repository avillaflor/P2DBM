def compute_done_condition(prev_episode_measurements, curr_episode_measurements, config):
    # Episode termination conditions
    success = curr_episode_measurements["distance_to_goal"] < config.scenario_config.dist_for_success

    # Check if static threshold reach, always False if static is disabled
    static = (curr_episode_measurements["static_steps"] > config.scenario_config.max_static_steps) and \
        not config.scenario_config.disable_static

    # Collision Reward
    is_collision = False
    lane_change = False
    obs_collision = (curr_episode_measurements["num_collisions"] - prev_episode_measurements["num_collisions"]) > 0
    is_collision = obs_collision

    # count out_of_road also as a collision
    if not config.obs_config.disable_lane_invasion_sensor:
        is_collision = obs_collision or curr_episode_measurements["out_of_road"]

        # count any lane change also as a collision
        lane_change = curr_episode_measurements['num_lane_intersections'] > 0
        is_collision = is_collision or lane_change

    curr_episode_measurements['obs_collision'] = obs_collision
    curr_episode_measurements['lane_change'] = lane_change
    curr_episode_measurements["is_collision"] = is_collision

    # Check if collision, always False if collision is disabled
    collision = curr_episode_measurements["is_collision"] and not config.scenario_config.disable_collision
    maxStepsTaken = curr_episode_measurements["num_steps"] > config.scenario_config.max_steps
    offlane = (curr_episode_measurements['num_lane_intersections'] > 0) and not config.obs_config.disable_lane_invasion_sensor

    if success:
        termination_state = 'success'
    elif collision:
        if curr_episode_measurements['obs_collision']:
            termination_state = 'obs_collision'
        elif not config.obs_config.disable_lane_invasion_sensor and curr_episode_measurements["out_of_road"]:
            termination_state = 'out_of_road'
        elif not config.obs_config.disable_lane_invasion_sensor and curr_episode_measurements['lane_change']:
            termination_state = 'lane_invasion'
        else:
            termination_state = 'unexpected_collision'
    elif offlane:
        termination_state = 'offlane'
    elif static:
        termination_state = 'static'
    elif maxStepsTaken:
        termination_state = 'max_steps'
    else:
        termination_state = 'none'

    curr_episode_measurements['termination_state'] = termination_state
    done = success or collision or offlane or static or maxStepsTaken
    return done
