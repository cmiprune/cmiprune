from configs.constants import *

''' ***** BEGIN Configuration ***** '''
PRUNING_METHOD = BIDIR + COMPACT_CMI + PTEST
TARGET_LAYER = 2

PATH_DATA = 'dataset'
PATH_OUTPUT_PRUNE_FM = f'results/fm_layers/bidir_full_scree_{TARGET_LAYER}.txt'
PATH_OUTPUT_PER_LAYER = 'results/fm_layers/bidir_perlayer_scree{}.txt'

PATH_PREFIX_FM = f'{PATH_DATA}/vgg16_conv2d_train_cmi_'
PATH_LABEL = f'{PATH_DATA}/vgg16_extLabels_train_cmi.npy'
PATH_INPUT = f'{PATH_DATA}/vgg16_input_train_cmi.npy'
PATH_DATA_TRAIN = f'{PATH_DATA}/vgg16_conv2d_train_cmi_{TARGET_LAYER}.npy'
PATH_CIFAR = f'{PATH_DATA}/CIFAR10_data/'
''' ***** END Configuration ***** '''

if PRUNING_METHOD == BIDIR + PER_LAYER_CMI + SCREE:
  from results.bidir_perlayer_scree import pruned_fm
elif PRUNING_METHOD == BIDIR + PER_LAYER_CMI + XMEANS:
  from results.bidir_perlayer_xmeans import pruned_fm
elif PRUNING_METHOD == BIDIR + PER_LAYER_CMI + PTEST:
  from results.bidir_perlayer_ptest import pruned_fm
  
elif PRUNING_METHOD == BIDIR + FULL_CMI + SCREE:
  from results.bidir_full_scree import pruned_fm, order_pruning, select_for_pruning
elif PRUNING_METHOD == BIDIR + COMPACT_CMI + SCREE:
  from results.bidir_compact_scree import pruned_fm, order_pruning, select_for_pruning

elif PRUNING_METHOD == FORWARD + COMPACT_CMI + SCREE:
  from results.forward_compact_scree import pruned_fm, order_pruning, select_for_pruning
elif PRUNING_METHOD == FORWARD + FULL_CMI + SCREE:
  from results.forward_full_scree import pruned_fm, order_pruning, select_for_pruning

elif PRUNING_METHOD == BIDIR + COMPACT_CMI + XMEANS:
  from results.bidir_compact_xmeans import pruned_fm, order_pruning, select_for_pruning
elif PRUNING_METHOD == BIDIR + COMPACT_CMI + PTEST:
  from results.bidir_compact_ptest import pruned_fm, order_pruning, select_for_pruning