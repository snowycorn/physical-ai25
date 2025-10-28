import numpy as np
from PIL import Image
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import os
import sys
import argparse
import shutil
import json


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def transform_depth(image):
    depth_img = (image / 10 * 255).astype(np.uint8)
    return depth_img

def transform_semantic(semantic_obs):
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)
    return semantic_img

def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #depth snesor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #semantic snesor
    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec, semantic_sensor_spec]
    
    # move_forward: 0.05 meter each step
    # turn_left/right: 1 degrees each step
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=settings["forward_step"])
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=settings["rotate_step"])
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=settings["rotate_step"])
        )
    }
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def get_yaw_from_quat(quat):
    """Convert quaternion to yaw angle (in radians)."""
    w, x, y, z = quat.w, quat.x, quat.y, quat.z
    yaw = np.arctan2(2 * (w * y + x * z), 1 - 2 * (y * y + z * z))
    return yaw

def load_semantic_id_to_label(file_path):
    with open(file_path, "r") as f:
        info = json.load(f)
        id_to_label = np.array(info["id_to_label"])
        print(len(id_to_label))
    return id_to_label

def navigatePath(path, target_label, test_scene, info_semantic, save_video_path, alpha=0.5, forward_step=0.02, rotate_step=2):
    sim_settings = {
        "scene": test_scene,  # Scene path
        "default_agent": 0,  # Index of the default agent
        "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
        "width": 512,  # Spatial resolution of the observations
        "height": 512,
        "sensor_pitch": 0,  # sensor pitch (x rotation in rads)
        "forward_step": forward_step,
        "rotate_step": rotate_step
    }

    
    id_to_label = load_semantic_id_to_label(info_semantic)
    
    cfg = make_simple_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)
    
    # initialize an agent
    agent = sim.initialize_agent(sim_settings["default_agent"])
    
    # Set agent state
    agent_state = habitat_sim.AgentState()
    start_position = path[0]
    agent_state.position = np.array([start_position[0], 0.0, start_position[1]])
    agent.set_state(agent_state)
    
    # obtain the default, discrete actions that an agent can perform
    # default action space contains 3 actions: move_forward, turn_left, and turn_right
    action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
    print("Discrete action space: ", action_names)
    
    # --- initialize VideoWriter ---
    height = sim_settings["height"]
    width = sim_settings["width"]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(save_video_path, fourcc, 30, (width, height))
    
    # go through the path
    for next_position in path: 
        agent_state = agent.get_state()
        agent_position = agent_state.position[[0, 2]]
               
        while True:  
            agent_state = agent.get_state()
            
            agent_position = agent_state.position[[0, 2]]
            agent_yaw = get_yaw_from_quat(agent_state.rotation)
            print("agent position: ", agent_position)
            print("agent rotation: ", np.degrees(agent_yaw))
            
            # agent.position 和 next_position 夠接近時結束
            if np.linalg.norm(agent_position - next_position) < forward_step:
                break
            
            direction = next_position - agent_state.position[[0, 2]]
            target_yaw = np.arctan2(-direction[0], -direction[1])
            print("target_position", )
            print("direction", direction)
            print("target_rotation", np.degrees(target_yaw))
            
            yaw_diff = (target_yaw - agent_yaw + np.pi) % (2 * np.pi) - np.pi
            # 檢查 agent.rotation 是否面對 next_position，若否 action = "turn_left" 或 "turn_right"
            if abs(yaw_diff) > np.deg2rad(rotate_step):
                action = "turn_left" if yaw_diff > 0 else "turn_right"
            # 若是，action = "move_forward"
            else:
                action = "move_forward"

            # step the simulator
            observations = sim.step(action)
            rgb_img = observations["color_sensor"]
            semantic_obs = observations["semantic_sensor"]

            # create target mask
            semantic_id = id_to_label[semantic_obs]
            target_mask = (semantic_id == target_label).astype(np.uint8) * 255
            mask_rgb = np.zeros_like(rgb_img)
            mask_rgb[:, :, 0] = target_mask  # Red channel highlight
            # cv2.imshow("mask", transform_rgb_bgr(mask_rgb))
            
            # blend images by addWeighted
            blended = cv2.addWeighted(rgb_img, 1.0, mask_rgb, alpha, 0)
            blended_bgr = transform_rgb_bgr(blended)
            
            cv2.imshow("RGB", blended_bgr)
            cv2.waitKey(1)
            
            # write frame to video 
            video_writer.write(blended_bgr)
            
    # release video writer
    video_writer.release()
    print(f"Navigation video saved as {save_video_path}")