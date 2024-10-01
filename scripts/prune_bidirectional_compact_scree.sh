# Edit file configs:
# - Set PRUNING_METHOD = BIDIR + COMPACT_CMI + SCREE
# - Set TARGET_LAYER to layer to be pruned, must be a value in LIST_CONV2D_LAYERS, file configs/constants.py
# Edit file results/bidir_compact_scree.py:
# - Set 'order_pruning' to contains only 1 previously pruned layer
# - Set 'select_for_pruning' to list of 1 corresponding number of selection
python CNN_prune/prune_cross_layer.py