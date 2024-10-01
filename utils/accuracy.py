import os
import sys

sys.path.insert(0, os.getcwd())

from configs.configs import *
from torchvision import models
import torch
import torchvision.transforms as transforms
from torchvision import transforms, datasets
import numpy as np
import warnings
np.warnings = warnings

from copy import deepcopy
from torchmetrics import Accuracy

from dataset.cifar10_models.vgg import vgg16_bn


def calc_accuracy(pruned_fm, fm_idx, l_choose, order_layers, order_chosen):

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]
            ),
        ]
    )

    cifar10_train = datasets.CIFAR10(
        root=PATH_CIFAR, train=True, transform=preprocess, download=True
    )

    cifar10_test = datasets.CIFAR10(
        root=PATH_CIFAR, train=False, transform=preprocess, download=True
    )

    train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=200, shuffle=False)

    test_loader = torch.utils.data.DataLoader(
        cifar10_test, batch_size=256, shuffle=False, drop_last=True, pin_memory=True
    )

    model = vgg16_bn(pretrained=True).to(device)
    model.eval()  

    def test_accuracy(model, loader=test_loader, verbose=False):
        model.eval()
        accuracy_metric = Accuracy(task="multiclass", num_classes=10).to(device)

        # Calculate accuracy on the test dataset
        with torch.no_grad():
            for batch_data, batch_labels in loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                output = model(batch_data)
                predicted_classes = torch.argmax(output, dim=1)
                accuracy_metric(predicted_classes, batch_labels)

        # Get the overall accuracy
        accuracy = accuracy_metric.compute()
        if verbose:
            print(f"Accuracy on the test dataset: {np.round(accuracy.item(), 4)}")
        return accuracy

    

    def adjust_select(select):
        select = [e - 1 for e in select]
        return select

    def find_prune(select_indices, total):
        s = set(select_indices)
        prune_indices = [e for e in range(total) if e not in s]
        return prune_indices

    def custom_prune(layer, indices):
        # Prune the weights
        with torch.no_grad():
            # Zero out the weights for the specified indices
            layer.weight[indices, :, :, :] = 0
            # If the layer has a bias term, you can also zero it out
            if layer.bias is not None:
                layer.bias[indices] = 0

    def sample(
        model, fm_idx=2, beta=None, alpha=None, head=None, verbose=False, beta_idx=None
    ):
        """
        - beta: minimum threshold for CMI
        - beta_idx: maximum index for CMI on list of CMI
        - alpha: minimum threshold for p_value
        """

        model_t = deepcopy(model)
        layer_to_prune = model_t.features[fm_idx - 2]

        n_total = pruned_fm[fm_idx]["total"]

        select = np.array(pruned_fm[fm_idx]["select"])

        if head is not None:
            select = select[:head]
        if beta is not None:
            idx_beta = np.array(pruned_fm[fm_idx]["CMI_vector"]) >= beta
            select = np.array(pruned_fm[fm_idx]["select"])[idx_beta]
        if beta_idx is not None:
            select = np.array(pruned_fm[fm_idx]["select"])[:beta_idx]
        if alpha is not None:
            idx_alpha = (
                np.array(pruned_fm[fm_idx]["arr_p_values"]) <= alpha
            ) 
            select = np.array(pruned_fm[fm_idx]["select"])[idx_alpha]

        if verbose:
            print(f"select:{len(select)}/{n_total}, select:{select}")

        select = adjust_select(select)
        prune_indices = find_prune(select, n_total)
        custom_prune(layer_to_prune, prune_indices)

        return model_t
    
    model_t = vgg16_bn(pretrained=True).to(device)
    
    if (order_chosen):
        for i, (layer_idx, choose) in enumerate(zip(order_layers, order_chosen)):
            print(f"Pruning layer {layer_idx:02d} from index {choose:03d}")
            model_t = sample(model_t, fm_idx=layer_idx, beta_idx=choose)
    
    print(f"Pruning model - train accuracy")

    for i in l_choose:
        print(f"----- index: {i:03d} -----")
        model_main = sample(model_t, fm_idx=fm_idx, beta_idx=i)
        test_accuracy(model_main, loader=train_loader, verbose=True)

    print(f"Pruning model - test accuracy")
    
    for i in l_choose:
        print(f"----- index: {i:03d} -----")
        model_main = sample(model_t, fm_idx=fm_idx, beta_idx=i)
        test_accuracy(model_main, loader=test_loader, verbose=True)