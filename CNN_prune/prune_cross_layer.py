import os
import sys

sys.path.insert(0, os.getcwd())

from configs.configs import *
from utils.utils import swapaxes, buildPreLayerSelectedFM
from ordering import cmi
import numpy as np
from cutoff_point.scree_test import select

P = 100  # permutaion number
eta = 0.05  # permutation significant level

# buildPreLayerSelectedFM(pruned_fm, layer_index=32, choose_before_index=192)

selected_fm_before = [
    # pruned_fm[32]["select_fm"],
]

extLabels = np.load(PATH_LABEL)
input = np.load(PATH_INPUT)
data = np.load(PATH_DATA_TRAIN)

extLabels = swapaxes(extLabels)
input = swapaxes(input)
data = swapaxes(data)

CMI_vector, MI_XT_vector, MI_TY_vector, ordered_fm, p_value_vector = (
    cmi.filter_number_selection(
        data,
        extLabels,
        input,
        P,
        eta,
        max_iters=None,
        list_selected_fm_before=selected_fm_before,
        pruning_method=PRUNING_METHOD
    )
)

selected_fm_topK = select(
    pruned_fm, TARGET_LAYER, ordered_fm, topk=3, order_pruning=order_pruning, selected_fm=select_for_pruning
)

data_to_print = [
    (f"'select':", "[", ", ".join(str(x) for x in ordered_fm), "],"),
    (f"'p_value_vector':", "[", ", ".join(str(x) for x in p_value_vector), "],"),
    (f"'CMI_vector':", "[", ", ".join(str(x) for x in CMI_vector), "],"),
    (f"'scree_test':", "[", ", ".join(str(x) for x in selected_fm_topK), "],"),
]

# Writing to file
file_path = PATH_OUTPUT_PRUNE_FM
with open(file_path, "w") as file:
    for line in data_to_print:
        file.write(" ".join(line) + "\n")
