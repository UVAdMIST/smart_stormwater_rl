import matplotlib.pyplot as plt
import pandas as pd
import os

# Read output files from different model tests
df_dict = {}
for file in os.listdir("RL_Class_S19/rl_project/saved_swmm_output"):
    file_name = str(file).split('.')[0]
    df = pd.read_csv(os.path.join("RL_Class_S19/rl_project/saved_swmm_output", file))
    df_dict.update({file_name: df})

# plot results
fig, axs = plt.subplots(2, 2, figsize=(6, 6))

width = 0.2
N = 0
plt_count = 0
styles = ['-', '-.', '--']
for key, value in df_dict.items():
    # print(key)
    if key in ['DQN_1000_smallpond_rwd1', 'DQN_1000_smallpond_rwd2', 'Uncontrolled_smallpond']:
        axs[0, 0].plot(value["St1_depth"], linestyle=styles[plt_count], label=key)
        axs[0, 1].plot(value["St2_depth"], linestyle=styles[plt_count], label=key)
        axs[1, 0].plot(value["J3_depth"], linestyle=styles[plt_count], label=key)
        axs[1, 1].bar([0 + width * plt_count, 2 + width * plt_count, 4 + width * plt_count],
                      [sum(value["St1_flooding"]), sum(value["St2_flooding"]), sum(value["J3_flooding"])],
                      width, align='center', label=key)
        plt_count += 1
    N += 1

axs[0, 0].set_ylim(0, 5)
axs[0, 1].set_ylim(0, 5)
axs[1, 0].set_ylim(0, 2.25)
axs[1, 1].set_ylim(0)
axs[0, 0].set_title('St1 Depth')
axs[0, 0].set_ylabel("ft")
axs[0, 0].set_xlabel("time step")
axs[0, 1].set_title('St2 Depth')
axs[0, 1].set_ylabel("ft")
axs[0, 1].set_xlabel("time step")
axs[1, 0].set_title('J3 Depth')
axs[1, 0].set_ylabel("ft")
axs[1, 0].set_xlabel("time step")
axs[1, 1].set_title('Total Flooding')
axs[1, 1].set_ylabel("10^3 cubic feet")
axs[1, 1].set_xticks([width, 2 + width, 4 + width])
axs[1, 1].set_xticklabels(("ST1", "St2", "J3"))

handles, labels = axs[1, 1].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0))
fig.tight_layout()
fig.subplots_adjust(bottom=0.2, hspace=0.4)
# plt.show()
plt.savefig("RL_Class_S19/rl_project/plots/model_comparison1000_smallpond.png", dpi=300)
plt.close()
