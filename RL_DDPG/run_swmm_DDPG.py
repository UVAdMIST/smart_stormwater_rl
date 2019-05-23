"""
This script runs Deep Q-Network RL algorithm for control
of stormwater systems using a SWMM model as the environment

Author: Benjamin Bowes
Date: May 10, 2019

"""

import os
import numpy as np
import itertools
import matplotlib.pyplot as plt
from datetime import timedelta
from pyswmm import Simulation, Nodes, Links
from smart_stormwater_rl.RL_DDPG.actor_critic import Actor, Critic, build_network
from smart_stormwater_rl.replay_memory import ReplayMemoryAgent
from smart_stormwater_rl.reward_functions import reward_function2 as reward_function  # specify reward function to use
from smart_stormwater_rl.pyswmm_utils import save_state, save_action

num_episodes = 10000  # number of times to repeat simulation
rewards_episode_tracker = []  # init rewards of each episode
flood_episode_tracker = []  # init flooding of each episode
swmm_inp = "C:/Users/Ben Bowes/PycharmProjects/smart_stormwater_rl/swmm_input_files/simple_2_ctl_smt.inp"
save_model_name = "saved_model_"  # init model name
model_dir = "smart_stormwater_rl/RL_DDPG/saved_models"
reward_dir = "smart_stormwater_rl/RL_DDPG/saved_model_rewards"
rwd = "rwd2"  # name of reward function used

# Initialize input states
temp_height = np.zeros(5)  # St1.depth, St1.flooding, St2.depth, St2.flooding, J3.flooding
temp_valve = np.zeros(2)  # three possible actions for each valve (-0.05, same, +0.05)
input_states = np.append(temp_height, temp_valve)

# Allocate actions
temp_acts = itertools.product(range(3), repeat=len(temp_valve))
temp_acts = list(temp_acts)
# action_space = np.asarray([[-1 if j == 0 else 1 for j in i] for i in temp_acts])
action_space = np.asarray([[-1 if j == 2 else j for j in i] for i in temp_acts])

# initialize action randomly
action = np.random.randint(0, len(action_space))

# Initialize Actor, two networks: Actor and target
# model = target = build_network(len(input_states), len(temp_acts), 2, 100, 'relu', 0.0)


# Initialize Critic, two networks: Critic and target


# Replay Memory
replay = ReplayMemoryAgent(len(input_states), 1000000)

# Deep Q learning agent
prof_x = DeepQAgent(model, target, len(input_states), replay.replay_memory, epsi_greedy)

# init lists to store values for plotting
St1_depth = []
St2_depth = []
J3_depth = []
St1_flooding = []
St2_flooding = []
J3_flooding = []
R1_position = []
R2_position = []

episode = 0
t_epsi = 0
while episode < num_episodes:  # loop through all episodes
    if episode % 100 == 0:
        print("episode: ", episode)
        print("t_epsi", t_epsi)
    # load model weights if not first episode
    if episode != 0:
        model.load_weights(os.path.join(model_dir, save_model_name))

    St1_depth_episode = []
    St2_depth_episode = []
    J3_depth_episode = []
    St1_flooding_episode = []
    St2_flooding_episode = []
    J3_flooding_episode = []
    R1_position_episode = []
    R2_position_episode = []

    episode += 1

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

    # Simulation Tracker
    reward_sim = []
    flood_sim = []

    # temp_height = np.asarray([St1.initial_depth, St2.initial_depth])
    # temp_flood = np.asarray([0, 0, 0])

    step_count = 0
    # for step in sim:
    sim.start()
    sim_len = sim.end_time - sim.start_time
    num_steps = int(sim_len.total_seconds()/control_time_step)
    # print(num_steps)
    end_time = sim.end_time - timedelta(seconds=1)
    while step_count <= num_steps - 1:  # loop through all steps in the simulation
        # print("step_count: ", step_count)
        # print("current sim time: ", sim.current_time)
        t_epsi += 1
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
            input_states = np.append(node_states, temp_valve).reshape(1, len(node_states) + len(temp_valve))
            # print(input_states)
            # print("valve: ", temp_valve)

            # record values
            St1_depth_episode.append(St1.depth)
            St2_depth_episode.append(St2.depth)
            J3_depth_episode.append(J3.depth)
            St1_flooding_episode.append(St1.flooding)
            St2_flooding_episode.append(St2.flooding)
            J3_flooding_episode.append(J3.flooding)
            R1_position_episode.append(R1.current_setting)
            R2_position_episode.append(R2.current_setting)

            # Select action according to the current policy (Actor weights) and exploration noise
            action = critic.ac_model.predict_on_batch(input_states)  # one action for each controllable structure

            # Implement Action


            # Execute selected actions
            sim.__next__()  # advances swmm model by one step
            # print("step_count: ", step_count)
            # print("current sim time: ", sim.current_time)

            # Observe next state
            temp_new_height = np.asarray([St1.depth, St2.depth])
            temp_new_flood = np.asarray([St1.flooding, St2.flooding, J3.flooding])
            temp_new_valve = np.asarray([R1.current_setting, R2.current_setting])
            node_new_states = np.append(temp_new_height, temp_new_flood)
            input_new_states = np.append(node_new_states, temp_new_valve).reshape(1, len(node_new_states) +
                                                                                  len(temp_new_valve))
            # print("new state", input_new_states)
            # print("new valve: ", temp_new_valve)

            # Observe reward
            reward = reward_function(temp_new_height, temp_new_flood)
            reward_sim.append(reward)
            # print("reward: ", reward)

            # add to replay
            replay.replay_memory_update(input_states, input_new_states, reward, action, False)

            # Train
            if episode % 100 == 0:
                update = False
                # update = True
                if t_epsi % 1000 == 0:
                    update = True
                prof_x.train_q(update)

    # close simulation at end of episode
    sim.report()
    sim.close()

    # Store reward values
    rewards_episode_tracker.append(np.mean(np.asarray(reward_sim)))

    if episode != 1:
        if rewards_episode_tracker[-1] > max(rewards_episode_tracker[:-1]):
            best_episode = episode
            St1_depth = St1_depth_episode
            St2_depth = St2_depth_episode
            J3_depth = J3_depth_episode
            St1_flooding = St1_flooding_episode
            St2_flooding = St2_flooding_episode
            J3_flooding = J3_flooding_episode
            R1_position = R1_position_episode
            R2_position = R2_position_episode

            out_states = [St1_depth, St2_depth, J3_depth, St1_flooding, St2_flooding, J3_flooding]
            out_actions = [R1_position, R2_position]

    save_model_name = "saved_model_" + str(episode)
    model.save(os.path.join(model_dir, save_model_name))  # save neural network model

np.save(os.path.join(reward_dir, save_model_name + "_rewards"), rewards_episode_tracker)  # save all mean rewards
save_state_name = "DDPG_" + str(num_episodes) + "_states_" + rwd
save_action_name = "DDPG_" + str(num_episodes) + "_actions_" + rwd
save_state(out_states, save_state_name)
save_action(out_actions, save_action_name)


# plot results from last episode
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
# plt.show()
plt.savefig("smart_stormwater_rl/RL_DDPG/plots/dqn_model_results_" + str(best_episode) + rwd + ".png", dpi=300)
plt.close()

plt.plot(rewards_episode_tracker)
plt.ylabel("average reward")
plt.xlabel("episode")
plt.tight_layout()
plt.savefig("smart_stormwater_rl/RL_DDPG/plots/dqn_model_rewards_" + str(num_episodes) + rwd + ".png", dpi=300)
plt.close()
