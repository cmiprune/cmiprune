import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
from ordering import cmi as per_layer_cmi # 'filter_number_selection', @arg 'list_selected_fm_before' equals to empty list
from results.bidir_perlayer_scree import pruned_fm
from utils.utils import swapaxes, buildPreLayerSelectedFM, getFmPath, getOutputPathForPerlayerCMI
from configs.configs import *


P = 100  # permutaion number
eta = 0.05  # permutation significant level

for i in range(len(LIST_CONV2D_LAYERS)):
  # load data
  extLabels = np.load(PATH_LABEL)
  input = np.load(PATH_INPUT)
  data = np.load(getFmPath(LIST_CONV2D_LAYERS[i]))

  extLabels = swapaxes(extLabels)
  input = swapaxes(input)
  data = swapaxes(data)

  CMI_vector, MI_XT_vector, MI_TY_vector, ordered_fm, p_value_vector = (
      per_layer_cmi.filter_number_selection(
          data,
          extLabels,
          input,
          P,
          eta,
          max_iters=None,
          list_selected_fm_before=[],
          pruning_method=PRUNING_METHOD
      )
  )
  
  if SCREE in PRUNING_METHOD:
    from cutoff_point.scree_test import select
    selected_fm_topK = select(
        pruned_fm, LIST_CONV2D_LAYERS[i], ordered_fm, topk=3, order_pruning=None, selected_fm=None
    )
    cut_points, _ = selected_fm_topK
    cutoff_approach = "Scree test"
  elif XMEANS in PRUNING_METHOD:
    from cutoff_point.xmeans import select
    cut_points = select(
        pruned_fm, LIST_CONV2D_LAYERS[i], ordered_fm, topk=None, order_pruning=None, selected_fm=None
    )
    cutoff_approach = "X-Means"
  elif PTEST in PRUNING_METHOD:
    cut_points = [len(CMI_vector)]
    cutoff_approach = "Permutation test"
  else:
    assert False, "Unknown pruning method"

  data_to_print = [
      (f"'select':", "[", ", ".join(str(x) for x in ordered_fm), "],"),
      (f"'arr_p_values':", "[", ", ".join(str(x) for x in CMI_vector), "],"),
      (f"'CMI_vector':", "[", ", ".join(str(x) for x in CMI_vector), "],"),
      (f"'cut_points-{cutoff_approach}':", "[", ", ".join(str(x) for x in cut_points), "],"),
  ]

  # Writing to file
  file_path = getOutputPathForPerlayerCMI(LIST_CONV2D_LAYERS[i])
  with open(file_path, "w") as file:
    for line in data_to_print:
      file.write(" ".join(line) + "\n")
