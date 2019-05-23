"""
Created by Benjamin Bowes, 4-19-19
This script records depth and flood values at each swmm model time step and plots them.
"""

from pyswmm import Simulation, Nodes, Links, Subcatchments
import matplotlib.pyplot as plt
from smart_stormwater_rl.pyswmm_utils import save_out

control_time_step = 900  # control time step in seconds
swmm_inp = "RL_Class_S19/rl_project/simple_2_ctl_smt.inp"  # swmm input file

St1_depth = []
St2_depth = []
J3_depth = []
St1_flooding = []
St2_flooding = []
J3_flooding = []

with Simulation(swmm_inp) as sim:  # loop through all steps in the simulation
    sim.step_advance(control_time_step)
    node_object = Nodes(sim)  # init node object
    St1 = node_object["St1"]
    St2 = node_object["St2"]
    J3 = node_object["J3"]

    St1.full_depth = 4
    St2.full_depth = 4

    link_object = Links(sim)  # init link object
    R1 = link_object["R1"]
    R2 = link_object["R2"]

    subcatchment_object = Subcatchments(sim)
    S1 = subcatchment_object["S1"]
    S2 = subcatchment_object["S2"]

    for step in sim:
        St1_depth.append(St1.depth)
        St2_depth.append(St2.depth)
        J3_depth.append(J3.depth)
        St1_flooding.append(St1.flooding)
        St2_flooding.append(St2.flooding)
        J3_flooding.append(J3.flooding)

out_lists = [St1_depth, St2_depth, J3_depth, St1_flooding, St2_flooding, J3_flooding]

save_out(out_lists, "Uncontrolled_smallpond")

# plot results
plt.subplot(2, 2, 1)
plt.plot(St1_depth)
plt.title('St1_depth')
plt.ylabel("ft")
plt.xlabel("time step")

plt.subplot(2, 2, 2)
plt.plot(St2_depth)
plt.title('St2_depth')
plt.ylabel("ft")
plt.xlabel("time step")

plt.subplot(2, 2, 3)
plt.plot(J3_depth)
plt.title('J3_depth')
plt.ylabel("ft")
plt.xlabel("time step")

# bar graph for total flooding
plt.subplot(2, 2, 4)
plt.bar([0, 1, 2], [sum(St1_flooding), sum(St2_flooding), sum(J3_flooding)], tick_label=["ST1", "St2", "J3"])
plt.title('total_flooding')
plt.ylabel("10^3 cubic feet")

plt.tight_layout()
# plt.show()
plt.savefig("RL_Class_S19/rl_project/plots/baseline_model_results_smallpond.png", dpi=300)
plt.close()
