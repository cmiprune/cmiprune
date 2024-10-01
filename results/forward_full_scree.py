from configs.configs import LIST_CONV2D_LAYERS
file_description = 'Forward pruning + Full CMI + Scree test'

order_pruning = LIST_CONV2D_LAYERS # [2, 5, 9, 12, 16, 19, 22, 26, 29, 32, 36, 39, 42]
select_for_pruning = [192, 201]

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