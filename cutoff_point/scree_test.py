from utils.accuracy import calc_accuracy

def sort_with_indices(lst):
    # Enumerate the list to create tuples (value, index)
    indexed_lst = [(value, index) for index, value in enumerate(lst)]

    # Sort the list in descending order based on values
    sorted_lst = sorted(indexed_lst, key=lambda x: x[0], reverse=True)

    # Extract the indices from the sorted list
    sorted_indices = [index for _, index in sorted_lst]

    return sorted_indices


def select(pruned_fm, layer_idx, list_cmi, topk=3, order_pruning = None, selected_fm=None):
    assert len(list_cmi) >= 1

    list_QDA = [None for _ in list_cmi]
    list_QDA[0] = list_QDA[-1] = -1.0

    for i in range(1, len(list_cmi) - 1):
        if list_cmi[i] - list_cmi[i + 1] == 0:
            list_QDA[i] = 0
        else:
            list_QDA[i] = (list_cmi[i - 1] - list_cmi[i]) / (
                list_cmi[i] - list_cmi[i + 1]
            )

    sorted_indices = sort_with_indices(list_QDA)

    if topk is None:
        topk = len(list_QDA)
        
    calc_accuracy(pruned_fm, layer_idx, sorted_indices[:topk], order_pruning, selected_fm)

    return sorted_indices[:topk], list_QDA[:topk]
