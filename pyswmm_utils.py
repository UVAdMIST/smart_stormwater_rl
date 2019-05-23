"""
Helper functions for controlling SWMM simulations with RL

Written by Benjamin Bowes, May 6, 2019
"""

import pandas as pd
import os


def get_control_structures(inp_file, structure_type):
    """
    returns list of swmm structures to be controlled

    inp_file = path to swmm input file
    structure_type = section of input file to search for (ex. "[ORIFICES]")
    """
    start_line = None
    end_line = None

    with open(inp_file, 'r') as tmp_file:
        lines = tmp_file.readlines()

    struct_list = []
    for i, j in enumerate(lines):
        if j.startswith(structure_type):
            start_line = i
            for k, l in enumerate(lines[i + 1:]):
                if l.startswith("["):
                    end_line = k + i
                    break
            if not end_line:
                end_line = len(lines)

    for m in range(start_line + 3, end_line, 1):
        struct = lines[m].split(" ")[0]
        struct_list.append(struct)

    return struct_list


def save_state(out_lists, model_name):
    # saves state for plotting depths and flooding
    cols = ["St1_depth", "St2_depth", "J3_depth", "St1_flooding", "St2_flooding", "J3_flooding"]
    out_df = pd.DataFrame(out_lists).transpose()
    out_df.columns = cols
    out_df.to_csv(os.path.join("smart_stormwater_rl/saved_swmm_output", model_name + "states.csv"), index=False)


def save_action(out_lists, model_name):
    # saves action for plotting policy
    cols = ["R1", "R2"]
    out_df = pd.DataFrame(out_lists).transpose()
    out_df.columns = cols
    out_df.to_csv(os.path.join("smart_stormwater_rl/saved_swmm_output", model_name + "actions.csv"), index=False)
