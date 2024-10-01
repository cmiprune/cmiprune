import numpy as np

from configs.configs import PATH_PREFIX_FM, PATH_OUTPUT_PER_LAYER

def getFmPath(layer_index):
  return f'{PATH_PREFIX_FM}{layer_index}.npy'

def getOutputPathForPerlayerCMI(layer_index, template=PATH_OUTPUT_PER_LAYER):
  return template.format(layer_index)

def swapaxes(data):
  '''
  Swap axes to match in python
  '''
  if data.ndim == 4:
    data = np.swapaxes(data, 0, 3)
    data = np.swapaxes(data, 1, 2)
    data = np.swapaxes(data, 2, 3)
  elif data.ndim == 2:
    data = np.swapaxes(data, 0, 1)
  return data


def buildPreLayerSelectedFM(pruned_fm, layer_index, choose_before_index):
  pruned_fm[layer_index]['select_cutoff'] = np.array(
    pruned_fm[layer_index]['select'][:choose_before_index])
  
  path_fm = getFmPath(layer_index)
  fm = swapaxes(np.load(path_fm))
  pruned_fm[layer_index]['select_fm'] = fm[:, pruned_fm[layer_index]['select_cutoff'], :, :]
