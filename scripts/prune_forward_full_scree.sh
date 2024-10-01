# Edit file configs:
# - Set PRUNING_METHOD = FORWARD + FULL_CMI + SCREE
# - Set TARGET_LAYER to layer to be pruned, must be a value in LIST_CONV2D_LAYERS, file configs/constants.py
# Edit file results/forward_full_scree.py:
# - Set 'order_pruning' = LIST_CONV2D_LAYERS (ordered list of layers to be pruned)
# - Set 'select_for_pruning' to list of corresponding number of selection for each layer

python CNN_prune/prune_cross_layer.py