# Pruning Deep Convolutional Neural Network Using Conditional Mutual Information

Code implementation of **Pruning Deep Convolutional Neural Network Using Conditional Mutual Information**

## Table of Content
<!-- [TOC] -->
<div class="toc">

</div>


<!-- ## Introduction -->


## Basic Usage
### Prerequisites
The code has the following dependencies:

- python 3.8
- numpy 1.26.4
- torch 2.1.2
- torchmetrics 1.3.0
- torchvision 0.16.2

### Configs
Edit the file `/configs/configs.py` to set the parameters for the experiment.

- **`PATH_DATA`**: define the location where you save the [dataset](#dataset)

<!-- ### Folder structure -->

## Experiments 
<!-- #TODO: refer to the terminal for accuracy -->

### Bi-directional Pruning with Compact CMI and Scree test

#### Step 1: Determine layer with highest pruning percentage using *per-layer CMI* and *Scree test*
- 1.a: Edit file `/configs/configs.py`
  - Set PRUNING_METHOD = BIDIR + COMPACT_CMI + SCREE
  - Change PATH_OUTPUT_PER_LAYER to path to save results

- 1.b: Run following script:
```bash
sh scripts/prune_bidirectional_perlayer_scree.sh
```

#### Step 2: Prune CNN with Forward and Backward Computation

- 2.a: Edit file `/configs/configs.py`
  - Set `PRUNING_METHOD` = BIDIR + COMPACT_CMI + SCREE
  - Set `TARGET_LAYER` to layer to be pruned, must be a value in `LIST_CONV2D_LAYERS`, file `configs/constants.py`
  - Set `PATH_OUTPUT_PRUNE_FM` to path to save results

- 2.b: Edit file results `results/bidir_compact_scree.py`
  - Set 'order_pruning' to list of ordered list of layers to be pruned
  - Set 'select_for_pruning' to list of corresponding number of selection for each layer

- 2.c: Run following script:
```bash
sh scripts/prune_bidirectional_compact_scree.sh
```

#### Step 3: Prepare results for the next iteration

The script from previous step runs the `.py` file in `/CNN_prune`. First, it computes the CMI vectors and orders filters for a layer. After that, the script found the cut-off points and corresponding accuracy. The output will be saved in `PATH_OUTPUT_PRUNE_FM`, copy output to `bidir_compact_scree.py` in `/fm_layers`.

````python
pruned_fm = {
    32: {
        'total': 512,
        # Paste the result here    
        # Sample:
        '''
        'select': [ 349, 246, 45,... ],
        'p_value_vector': [ -1.0, -1.0, -1.0,... ],
        'CMI_vector': [ 0.007687962798611284, 0.007546566567393281, 0.007414801505923892,... ],
        'scree_test': [ [1, 0, 2], [-1.0, 0.5124378109452736, -1.0] ],
        '''
    },
    
}
````
In `bidir_compact_scree.py`, set `order_pruning` to contains only 1 previously pruned layer and set `select_for_pruning` to list of 1 corresponding number of selection.

````Python
order_pruning = [32]
select_for_pruning = [192]
````
In `prune_cross_layer.py`, add these lines of code to prune previous layers and prepare for next layer to be pruned.

````Python
# Prune previous layers: 32th layer
buildPreLayerSelectedFM(pruned_fm, layer_index=32, choose_before_index=192)

selected_fm_before = [
    pruned_fm[32]["select_fm"],
]
````

Change the `TARGET_LAYER` to the next layer to be pruned and run the script at step 2 again.

### Bi-directional Pruning with Full CMI and Scree test
#### Step 1 and 2: Similar to the [Bi-directional Pruning with Compact CMI and Scree test](#bi-directional-pruning-with-compact-cmi-and-scree-test)

#### Step 3: Prepare results for the next iteration

In `bidir_full_scree.py`, add layer to the ordered list to be pruned `order_pruning`  and the corresponding number of selection to `select_for_pruning`.

````Python
order_pruning = [32, 36]
select_for_pruning = [192, 201]
````
In `prune_cross_layer.py`, add these lines of code to prune previous layers and prepare for next layer to be pruned.

````Python
# Prune previous layers: 32th and 36th layer
buildPreLayerSelectedFM(pruned_fm, layer_index=32, choose_before_index=192)
buildPreLayerSelectedFM(pruned_fm, layer_index=36, choose_before_index=201)`

selected_fm_before = [
    pruned_fm[32]["select_fm"],
    pruned_fm[36]["select_fm"],
]
````

Change the `TARGET_LAYER` to the next layer to be pruned and run the script at step 2 again.

### Forward Pruning
The steps are similar to the [Bi-directional Pruning with Compact CMI and Scree test](#bi-directional-pruning-with-compact-cmi-and-scree-test) and [Bi-directional Pruning with Full CMI and Scree test](#bi-directional-pruning-with-full-cmi-and-scree-test), but only include step 2 and 3. The running script are as follows:
- For Compact CMI and Scree test: `prune_forward_compact_scree.sh`
- For Full CMI and Scree test: `prune_forward_full_scree.sh`

### Bi-directional Pruning with Compact CMI and X-Means
The steps are similar to [Bi-directional Pruning with Compact CMI and Scree test](#bi-directional-pruning-with-compact-cmi-and-scree-test) except for the script to run. The script for step 1 and 2 are:

```bash
sh scripts/prune_bidirectional_perlayer_xmeans.sh
```
```bash
sh scripts/prune_bidirectional_compact_xmeans.sh
```

### Bi-directional Pruning with Compact CMI and Permutation test
The steps are similar to [Bi-directional Pruning with Compact CMI and Scree test](#bi-directional-pruning-with-compact-cmi-and-scree-test) except for the script to run. The script for step 1 and 2 are:

```bash
sh scripts/prune_bidirectional_perlayer_ptest.sh
```
```bash
sh scripts/prune_bidirectional_compact_ptest.sh
```


## Dataset
The dataset was uploaded to this [link](https://drive.google.com/drive/folders/1pbTNWWCNBAZ9UyiLEYpiAS3qQAZB4MQc?usp=sharing).
