"""
This script runs Deep Q-Network RL algorithm for control
of stormwater systems using a SWMM model as the environment

Author: Benjamin Bowes
Date: May 10, 2019

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pyswmm import Simulation, Nodes, Links
from smart_stormwater_rl.RL_DDPG.actor_critic import Actor
from smart_stormwater_rl.pyswmm_utils import save_state, save_action

swmm_inp = "C:/Users/Ben Bowes/PycharmProjects/smart_stormwater_rl/swmm_input_files/simple_2_ctl_smt.inp"
save_model_name = "saved_model_188"  # init model name
actor_dir = "smart_stormwater_rl/RL_DDPG/saved_models_actor"
out_dir = "smart_stormwater_rl/RL_DDPG/saved_swmm_output"
lr = 0.00005
tau = 0.001

# Initialize input states TODO dynamically read from swmm input file
temp_height = np.zeros(2, dtype='int32')  # St1.depth, St2.depth
temp_valve = np.zeros(2, dtype='int32')  # R1.current_setting, R2.current_setting
# input_states = np.append(temp_height, temp_valve)
input_states = temp_height

# Allocate actions and set range
action_space = 2  # number of structures to control
act_range = np.asarray([0., 1.])

# Initialize Actor, returns two networks: Actor and target
actor = Actor(input_states.shape, action_space, act_range, 0.1 * lr, tau)

# Load model weights from saved actor
actor.load_weights(os.path.join(actor_dir, save_model_name + "_actor.h5"))

# init lists to store values for plotting
St1_depth = []
St2_depth = []
J3_depth = []
St1_flooding = []
St2_flooding = []
J3_flooding = []
R1_position = []
R2_position = []

# initialize simulation
sim = Simulation(swmm_inp)  # read input file
control_time_step = 900  # control time step in seconds
sim.step_advance(control_time_step)  # set control time step
node_object = Nodes(sim)  # init node object
St1 = node_object["St1"]
St2 = node_object["St2"]
J3 = node_object["J3"]
node_list = [St1, St2, J3]

# Change pond depth if desired
# St1.full_depth = 4
# St2.full_depth = 4

link_object = Links(sim)  # init link object
R1 = link_object["R1"]
R2 = link_object["R2"]
valve_list = [R1, R2]

step_count = 0
# for step in sim:
sim.start()
sim_len = sim.end_time - sim.start_time
num_steps = int(sim_len.total_seconds()/control_time_step)
while step_count <= num_steps - 1:  # loop through all steps in the simulation
    # print("step_count: ", step_count)
    # print("current sim time: ", sim.current_time)
    step_count += 1

    if step_count >= num_steps:
        break
    else:
        # initialize valve settings
        if sim.current_time == sim.start_time:
            R1.target_setting = 0.5
            R2.target_setting = 0.5

        # construct current system states as inputs
        temp_height = np.asarray([St1.depth, St2.depth])
        temp_flood = np.asarray([St1.flooding, St2.flooding, J3.flooding])
        temp_valve = np.asarray([R1.current_setting, R2.current_setting])
        node_states = np.append(temp_height, temp_flood)
        # input_states = np.append(node_states, temp_valve).reshape(1, len(node_states) + len(temp_valve))
        input_states = temp_height
        # print(input_states)
        # print("valve: ", temp_valve)

        # record values
        St1_depth.append(St1.depth)
        St2_depth.append(St2.depth)
        J3_depth.append(J3.depth)
        St1_flooding.append(St1.flooding)
        St2_flooding.append(St2.flooding)
        J3_flooding.append(J3.flooding)
        R1_position.append(R1.current_setting)
        R2_position.append(R2.current_setting)

        # Select action according to the current policy (Actor weights)
        action = actor.predict(input_states)  # one action for each controllable structure
        print(action)
        action = np.clip(action, act_range[0], act_range[1])  # make sure actions in correct range

        # Implement Action
        for i, j in enumerate(valve_list):
            j.target_setting = action[0][i]

        # Execute selected actions
        sim.__next__()  # advances swmm model by one step
        # print("step_count: ", step_count)
        # print("current sim time: ", sim.current_time)

# close simulation at end of episode
sim.report()
sim.close()

# plot results from simulation
plt.subplot(2, 2, 1)
plt.plot(St1_depth)
plt.ylim(0, 5)
plt.title('St1_depth')
plt.ylabel("ft")
plt.xlabel("time step")

plt.subplot(2, 2, 2)
plt.plot(St2_depth)
plt.ylim(0, 5)
plt.title('St2_depth')
plt.ylabel("ft")
plt.xlabel("time step")

plt.subplot(2, 2, 3)
plt.plot(J3_depth)
plt.ylim(0, 2)
plt.title('J3_depth')
plt.ylabel("ft")
plt.xlabel("time step")

# bar graph for total flooding
plt.subplot(2, 2, 4)
plt.bar([0, 1, 2], [sum(St1_flooding), sum(St2_flooding), sum(J3_flooding)], tick_label=["St1", "St2", "J3"])
plt.ylim(0)
plt.title('total_flooding')
plt.ylabel("10^3 cubic feet")

plt.tight_layout()
plt.show()
# plt.savefig("smart_stormwater_rl/RL_DDPG/plots/ddpg_model_results_" + str(best_episode) + rwd + ".png", dpi=300)
# plt.close()

# # plot rewards and actions
# plt.plot(R1_position)
# plt.plot(R2_position, linestyle='--')
# plt.ylim(0, 1)
# plt.ylabel("orifice position")
# plt.xlabel("time step")
# plt.tight_layout()
# # plt.savefig("smart_stormwater_rl/RL_DDPG/plots/ddpg_model_rewards_" + str(num_episodes) + rwd + "epi" +
# #             str(best_episode) + ".png", dpi=300)
# # plt.close()
# plt.show()
