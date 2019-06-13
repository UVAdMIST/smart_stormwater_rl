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
from smart_stormwater_rl.RL_DDPG.actor_critic import Actor, Critic
from smart_stormwater_rl.replay_memory import ReplayMemoryAgent, random_indx, create_minibatch
from smart_stormwater_rl.reward_functions import reward_function2 as reward_function  # specify reward function to use
from smart_stormwater_rl.pyswmm_utils import OrnsteinUhlenbeckProcess, save_state, save_action, gen_noise

num_episodes = 10000  # number of times to repeat simulation
rewards_episode_tracker = []  # init rewards of each episode
flood_episode_tracker = []  # init flooding of each episode
swmm_inp = "C:/Users/Ben Bowes/PycharmProjects/smart_stormwater_rl/swmm_input_files/simple_2_ctl_smt.inp"
save_model_name = "saved_model_"  # init model name
actor_dir = "smart_stormwater_rl/RL_DDPG/saved_models_actor"
critic_dir = "smart_stormwater_rl/RL_DDPG/saved_models_critic"
reward_dir = "smart_stormwater_rl/RL_DDPG/saved_model_rewards"
out_dir = "smart_stormwater_rl/RL_DDPG/saved_swmm_output"
rwd = "rwd2"  # name of reward function for labeling plots/data
gamma = 0.99
lr = 0.00005
tau = 0.001
batch_size = 100

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

# Initialize Critic, returns two networks: Critic and target
critic = Critic(input_states.shape, action_space, lr, tau)

# Replay Memory
replay = ReplayMemoryAgent(len(input_states), action_space, 1000000)

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
        actor.load_weights(os.path.join(actor_dir, save_model_name + "_actor.h5"))
        critic.load_weights(os.path.join(critic_dir, save_model_name + "_critic.h5"))

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

    # Initialize Noise Process
    noise = OrnsteinUhlenbeckProcess(size=action_space)

    step_count = 0
    # for step in sim:
    sim.start()
    sim_len = sim.end_time - sim.start_time
    num_steps = int(sim_len.total_seconds()/control_time_step)
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
            # input_states = np.append(node_states, temp_valve).reshape(1, len(node_states) + len(temp_valve))
            input_states = temp_height
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
            # (sigmoid in last layer of actor should give range of actions from 0 to 1)
            action = actor.predict(input_states)  # one action for each controllable structure
            # print(action)
            # add exploration noise and make sure actions are within range
            # action = np.clip(action + gen_noise(episode, action_space), act_range[0], act_range[1])
            # noise_t = np.array([noise.generate(t_epsi), noise.generate(t_epsi)], 'float32').transpose()
            noise_t = noise.generate(1)
            expo = np.exp(-episode*3/num_episodes)
            action = np.clip(action + noise_t * expo, act_range[0], act_range[1])
            # action = np.clip(action + (np.random.random(1) / episode), act_range[0], act_range[1])  # try random exploration
            # if np.random.random(1) < 0.1:
            #     action = np.array(np.random.random(2), 'float32', ndmin=2)
            # action = np.clip(action, act_range[0], act_range[1])

            # Implement Action
            for i, j in enumerate(valve_list):
                j.target_setting = action[0][i]
            # R1.target_setting = action[0][0]
            # R2.target_setting = action[1][1]

            # Execute selected actions
            sim.__next__()  # advances swmm model by one step
            # print("step_count: ", step_count)
            # print("current sim time: ", sim.current_time)

            # Observe next state
            temp_new_height = np.asarray([St1.depth, St2.depth])
            temp_new_flood = np.asarray([St1.flooding, St2.flooding, J3.flooding])
            temp_new_valve = np.asarray([R1.current_setting, R2.current_setting])
            node_new_states = np.append(temp_new_height, temp_new_flood)
            # input_new_states = np.append(node_new_states, temp_new_valve).reshape(1, len(node_new_states) +
            #                                                                       len(temp_new_valve))
            input_new_states = temp_new_height
            # print("new state", input_new_states)
            # print("new valve: ", temp_new_valve)

            # Observe reward
            reward = reward_function(temp_new_height, temp_new_flood)
            reward_sim.append(reward)
            # print("reward: ", reward)

            # add to replay
            replay.replay_memory_update(input_states, input_new_states, reward, action, False)

            # Sample minibatch from memory
            rnd_indx = random_indx(batch_size, replay.replay_memory['states'].data().shape[0])
            minibatch = create_minibatch(rnd_indx, replay, batch_size, action_space)
            batch_states = minibatch['states']
            batch_new_states = minibatch['states_new']
            batch_actions = minibatch['actions']
            batch_rewards = minibatch['rewards']
            batch_terminal = minibatch['terminal']

            # Predict target q-values using target network (critic takes [state, action] as input)
            q_values = critic.target_predict([batch_new_states, actor.target_predict(batch_new_states)])

            # calculate critic targets using the Bellman equation
            critic_target = np.asarray(q_values)
            for i in range(q_values.shape[0]):
                if batch_terminal[i]:
                    critic_target[i] = batch_rewards[i]
                else:
                    critic_target[i] = batch_rewards[i] + gamma * q_values[i]

            # Train both networks on sampled batch, update target networks
            # Train critic
            critic.train_on_batch(batch_states, batch_actions, critic_target)

            # Q-Value Gradients under Current Policy
            actions_for_grads = actor.model.predict(batch_states)
            grads = critic.gradients(batch_states, actions_for_grads)  # changed from batch_actions to actions_for_grads

            # Train actor
            actor.train(batch_states, actions_for_grads, np.array(grads).reshape((-1, action_space)))  # changed from batch_actions to actions_for_grads

            # Transfer weights to target networks at rate Tau
            actor.transfer_weights()
            critic.transfer_weights()

    # close simulation at end of episode
    sim.report()
    sim.close()

    # Store reward values
    rewards_episode_tracker.append(np.mean(np.asarray(reward_sim)))

    if episode != 1:
        if rewards_episode_tracker[-1] >= max(rewards_episode_tracker[:-1]):
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

    # save neural network models
    save_model_name = "saved_model_" + str(episode)
    actor.save(os.path.join(actor_dir, save_model_name))
    critic.save(os.path.join(critic_dir, save_model_name))

np.save(os.path.join(reward_dir, save_model_name + "_rewards"), rewards_episode_tracker)  # save all mean rewards
save_state_name = "DDPG_" + str(num_episodes) + "_states_" + rwd
save_action_name = "DDPG_" + str(num_episodes) + "_actions_" + rwd
save_state(out_states, os.path.join(out_dir, save_state_name + "states.csv"))
save_action(out_actions, os.path.join(out_dir, save_action_name + "actions.csv"))


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
plt.savefig("smart_stormwater_rl/RL_DDPG/plots/ddpg_model_results_" + str(best_episode) + rwd + ".png", dpi=300)
plt.close()

# plot rewards and actions
plt.subplot(2, 1, 1)
plt.plot(rewards_episode_tracker)
plt.ylabel("average reward")
plt.xlabel("episode")

plt.subplot(2, 1, 2)
plt.plot(R1_position)
plt.plot(R2_position, linestyle='--')
plt.ylim(0, 1)
plt.ylabel("orifice position")
plt.xlabel("time step")
plt.tight_layout()
plt.savefig("smart_stormwater_rl/RL_DDPG/plots/ddpg_model_rewards_" + str(num_episodes) + rwd + "epi" +
            str(best_episode) + ".png", dpi=300)
plt.close()
