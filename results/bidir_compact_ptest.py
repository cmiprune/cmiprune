file_description = 'Bi-directional pruning + Compact CMI + Scree test'

order_pruning = [32]
select_for_pruning = [192] # only 1 layer is allowed for compact CMI

pruned_fm = { # update at each pruning iteration
    2: {
        'total': 64,
        'select': [],
        'arr_p_values': [],
        'CMI_vector': [],
    },
    5: {
        'total': 64,
        'select': [],
        'arr_p_values': [],
        'CMI_vector': [],
    },
    9: {
        'total': 128,
        'select': [],
        'arr_p_values': [],
        'CMI_vector': [],
    },
    12: {
        'total': 128,
        'select': [],
        'arr_p_values': [],
        'CMI_vector': [],
    },
    16: {
        'total': 256,
        'select': [],
        'arr_p_values': [],
        'CMI_vector': [],
    },
    19: {
        'total': 256,
        'select': [],
        'arr_p_values': [],
        'CMI_vector': [],
    },
    22: {
        'total': 256,
        'select': [],
        'arr_p_values': [],
        'CMI_vector': [],
    },
    26: {
        'total': 512,
        'select': [],
        'arr_p_values': [],
        'CMI_vector': [],
    },
    29: {
        'total': 512,
        'select': [],
        'arr_p_values': [],
        'CMI_vector': [],
    },
    32: {
        'total': 512,
        'select': [],
        'arr_p_values': [],
        'CMI_vector': [],
    },
    36: {
        'total': 512,
        'select': [],
        'arr_p_values': [],
        'CMI_vector': [],
    },
    39: {
        'total': 512,
        'select': [],
        'arr_p_values': [],
        'CMI_vector': [],
    },
    42: {
        'total': 512,
        'select': [],
        'arr_p_values': [],
        'CMI_vector': [],
    },
}