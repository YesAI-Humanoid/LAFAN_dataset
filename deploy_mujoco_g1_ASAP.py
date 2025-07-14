import time
import mujoco.viewer
import mujoco
import numpy as np
import torch
import onnx
import onnxruntime
import yaml
import os


def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "configs/g1_mujoco_g1_ASAP.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    policy_path = config['paths']['policy_path']
    xml_path = config['paths']['xml_path']

    print("policy_path: ", policy_path)
    print("xml_path   : ", xml_path)

    # define context variables from config
    control_decimation = config['simulation']['control_decimation']
    simulation_dt = config['simulation']['simulation_dt']
    simulation_duration = config['simulation']['simulation_duration']
    num_actions = config['simulation']['num_actions']
    history_length = config['simulation']['history_length']

    # Load scaling factors from config
    scaling_factors = config['scaling_factors']

    counter = 0  # Initialize counter for control decimation
    action = np.zeros(num_actions, dtype=np.float32)
    default_angles = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                               -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                               0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0], dtype=np.float32)


    kps = np.array([100,  # left_hip_yaw_joint
                    100,  # left_hip_roll_joint
                    100,  # left_hip_pitch_joint
                    200,  # left_knee_joint
                    20,  # left_ankle_pitch_joint
                    20,  # left_ankle_roll_joint

                    # Right leg (6 joints)
                    100,  # right_hip_yaw_joint
                    100,  # right_hip_roll_joint
                    100,  # right_hip_pitch_joint
                    200,  # right_knee_joint
                    20,  # right_ankle_pitch_joint
                    20,  # right_ankle_roll_joint

                    # Waist (3 joints)
                    400,  # waist_yaw_joint
                    400,  # waist_roll_joint
                    400,  # waist_pitch_joint

                    # Left arm (4 joints)
                    90,  # left_shoulder_pitch_joint
                    60,  # left_shoulder_roll_joint
                    20,  # left_shoulder_yaw_joint
                    60,  # left_elbow_joint

                    # Right arm (4 joints)
                    90,  # right_shoulder_pitch_joint
                    60,  # right_shoulder_roll_joint
                    20,  # right_shoulder_yaw_joint
                    60,  # right_elbow_joint
                    ], dtype=np.float32)


    kds = np.array([2.5,  # left_hip_yaw_joint
                    2.5,  # left_hip_roll_joint
                    2.5,  # left_hip_pitch_joint
                    5.0,  # left_knee_joint
                    0.2,  # left_ankle_pitch_joint
                    0.1,  # left_ankle_roll_joint

                    # Right leg (6 joints)
                    2.5,  # right_hip_yaw_joint
                    2.5,  # right_hip_roll_joint
                    2.5,  # right_hip_pitch_joint
                    5.0,  # right_knee_joint
                    0.2,  # right_ankle_pitch_joint
                    0.1,  # right_ankle_roll_joint

                    # Waist (3 joints)
                    5.0,  # waist_yaw_joint
                    5.0,  # waist_roll_joint
                    5.0,  # waist_pitch_joint

                    # Left arm (4 joints)
                    2.0,  # left_shoulder_pitch_joint
                    1.0,  # left_shoulder_roll_joint
                    0.4,  # left_shoulder_yaw_joint
                    1.0,  # left_elbow_joint

                    # Right arm (4 joints)
                    2.0,  # right_shoulder_pitch_joint
                    1.0,  # right_shoulder_roll_joint
                    0.4,  # right_shoulder_yaw_joint
                    1.0,  # right_elbow_joint
                    ], dtype=np.float32)

    tau_limit = np.array([88, 88, 88, 139, 50, 50,
                          88, 88, 88, 139, 50, 50,
                          88, 50, 50,
                          25, 25, 25, 25,
                          25, 25, 25, 25], dtype=np.float32)

    dof_pos_limit_up = np.array([2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618,
                                 2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618,
                                 2.618, 0.52, 0.52,
                                 2.6704, 2.2515, 2.618, 2.0944,
                                 2.6704, 2.2515, 2.618, 2.0944], dtype=np.float32)

    dof_pos_limit_low = np.array([-2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618,
                                  -2.5307, -2.9671, -2.7576, -0.087267, -0.87267, -0.2618,
                                  -2.618, -0.52, -0.52,
                                  -3.0892, -1.5882, -2.618, -1.0472,
                                  -3.0892, -2.2515, -2.618, -1.0472], dtype=np.float32)

    target_dof_pos = default_angles.copy()
    ref_motion_phase = 0.0

    ### ------------------------  History buffer dimensions: ------------------------ ###
    # Root linear velocity: 3 values × history_length timesteps
    lin_vel_buf = np.zeros(3 * history_length, dtype=np.float32)

    # Root angular velocity: 3 values × history_length timesteps
    ang_vel_buf = np.zeros(3 * history_length, dtype=np.float32)

    # Projected gravity: 3 values × history_length timesteps
    proj_g_buf = np.zeros(3 * history_length, dtype=np.float32)

    # Joint positions: num_actions joints × history_length timesteps
    dof_pos_buf = np.zeros(num_actions * history_length, dtype=np.float32)

    # Joint velocities: num_actions joints × history_length timesteps
    dof_vel_buf = np.zeros(num_actions * history_length, dtype=np.float32)

    # Previous actions: num_actions values × history_length timesteps
    action_buf = np.zeros(num_actions * history_length, dtype=np.float32)

    # Motion phase: 1 value × history_length timesteps
    ref_motion_phase_buf = np.zeros(1 * history_length, dtype=np.float32)

    # Current frame dimensions:
    # - base_ang_vel:       3 dimensions (x, y, z angular velocity)
    # - projected_gravity:  3 dimensions (gravity vector in robot frame)
    # - dof_pos:           23 dimensions (one for each joint position)
    # - dof_vel:           23 dimensions (one for each joint velocity)
    # - actions:           23 dimensions (previous actions for each joint)
    # - ref_motion_phase:   1 dimension  (phase variable for synchronized movement)

    # Total: 3 + 3 + 23 + 23 + 23 + 1 = 76 dimensions
    ### ------------------------------------------------------------------------------- ###
    # load onnx model
    onnx_model = onnx.load(policy_path)
    ort_session = onnxruntime.InferenceSession(policy_path)
    input_name = ort_session.get_inputs()[0].name

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            tau = np.clip(tau, -tau_limit, tau_limit)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:]  # Joint positions
                dqj = d.qvel[6:]  # Joint velocities
                quat = d.qpos[3:7]  # Root orientation (quaternion)
                lin_vel = d.qvel[:3]  # Root linear velocity
                ang_vel = d.qvel[3:6]  # Root angular velocity

                # Apply scaling factors from config
                projected_gravity = get_gravity_orientation(quat)
                dof_pos = qj * scaling_factors['dof_pos']
                dof_vel = dqj * scaling_factors['dof_vel']
                base_ang_vel = ang_vel * scaling_factors['base_ang_vel']
                base_lin_vel = lin_vel * scaling_factors['base_lin_vel']
                ref_motion_phase += scaling_factors['ref_motion_phase_increment']

                history_obs_buf = np.concatenate((action_buf,
                                                  ang_vel_buf,
                                                  dof_pos_buf,
                                                  dof_vel_buf,
                                                  proj_g_buf,
                                                  ref_motion_phase_buf), axis=-1, dtype=np.float32)
                obs_buf = np.concatenate((action,
                                          base_ang_vel,
                                          dof_pos,
                                          dof_vel,
                                          history_obs_buf,
                                          projected_gravity,
                                          np.array([ref_motion_phase])), axis=-1, dtype=np.float32)

                # update history
                ang_vel_buf = np.concatenate((base_ang_vel, ang_vel_buf[:-3]), axis=-1, dtype=np.float32)
                lin_vel_buf = np.concatenate((base_lin_vel, lin_vel_buf[:-3]), axis=-1, dtype=np.float32)
                proj_g_buf = np.concatenate((projected_gravity, proj_g_buf[:-3]), axis=-1, dtype=np.float32)
                dof_pos_buf = np.concatenate((dof_pos, dof_pos_buf[:-num_actions]), axis=-1, dtype=np.float32)
                dof_vel_buf = np.concatenate((dof_vel, dof_vel_buf[:-num_actions]), axis=-1, dtype=np.float32)
                action_buf = np.concatenate((action, action_buf[:-num_actions]), axis=-1, dtype=np.float32)
                ref_motion_phase_buf = np.concatenate((np.array([ref_motion_phase]), ref_motion_phase_buf[:-1]),
                                                      axis=-1, dtype=np.float32)

                obs_tensor = torch.from_numpy(obs_buf).unsqueeze(0).cpu().numpy()
                action = np.squeeze(ort_session.run(None, {input_name: obs_tensor})[0])
                # transform action to target_dof_pos
                target_dof_pos = action * scaling_factors['action'] + default_angles

                # target_dof_pos = np.clip(target_dof_pos, dof_pos_limit_low, dof_pos_limit_up)
                target_dof_pos = np.clip(target_dof_pos, dof_pos_limit_low, dof_pos_limit_up)
            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()


            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
