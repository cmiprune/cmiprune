import copy
import numpy as np
from configs.constants import PTEST

def guassianMatrix(X, sigma):
  G = np.matmul(X, X.T)
  K = 2 * G - np.diag(G).reshape((-1, 1)).T
  K = np.exp((1 / (2 * sigma**2)) * (K - np.diag(G).reshape((-1, 1))))

  return K


def S_one(A, alpha):
  eigenvalue, _ = np.linalg.eig(A)
  lx = np.abs(np.diag(eigenvalue))
  res = (1 / (1 - alpha)) * np.log(np.sum(lx**alpha))
  return res


def S_multiple(list_gram_matrices, var_shape0, alpha):
  if type(list_gram_matrices) is list:
    n = len(list_gram_matrices)
    T1 = list_gram_matrices[0]
    for i in range(1, n):
      T1 = T1 * list_gram_matrices[i] * var_shape0
    return S_one(T1, alpha)
  else:
    assert False, f"Error: S_multiple - type: {type(list_gram_matrices)}"


def make_list_gram_matrices(feature_maps, sigma, first_var):
  if first_var:
    if feature_maps.ndim == 2:
      T = np.real(guassianMatrix(feature_maps, sigma)) / feature_maps.shape[0]
      return [T]
    else:
      feature_maps = feature_maps.reshape((feature_maps.shape[0], -1))
      T = np.real(guassianMatrix(feature_maps, sigma)) / feature_maps.shape[0]
      return [T]
  elif feature_maps.ndim == 2:
    T = np.real(guassianMatrix(feature_maps, sigma)) / feature_maps.shape[0]
    return [T]
  elif feature_maps.ndim == 4:
    list_gram_matrices = [None for _ in range(feature_maps.shape[1])]
    for i in range(feature_maps.shape[1]):
      source = feature_maps[:, [i], :, :]
      source = source.reshape(
          (source.shape[0], source.shape[2] * source.shape[3])
      )
      list_gram_matrices[i] = (
          np.real(guassianMatrix(source, sigma)) / source.shape[0]
      )
    return list_gram_matrices
  else:
    assert False, f'Invalid dim of feature_maps: {feature_maps.ndim}'

def mutual_information_estimation(
    var1_list_gram_matrices, var2_list_gram_matrices, sigma1, sigma2, alpha
):
  var1_shape0 = var1_list_gram_matrices[0].shape[0]
  var2_shape0 = var2_list_gram_matrices[0].shape[0]

  H1 = S_multiple(var1_list_gram_matrices, var1_shape0, alpha)
  H2 = S_multiple(var2_list_gram_matrices, var2_shape0, alpha)
  H3 = S_multiple(
      var1_list_gram_matrices + var2_list_gram_matrices, var1_shape0, alpha
  )

  return H1 + H2 - H3

def conditional_mutual_information_estimation(
    var1_list_gram_matrices,
    var2_list_gram_matrices,
    var3_list_gram_matrices,
    # selected_filters_gram_matrices,
    sigma1,
    sigma2,
    alpha,
):
  """
  %% variable 1 is class labels
  %  variable 2 is non-selected filter set
  %  variable 3 is selected filter set
  CMI(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
  """
  var1_shape0 = var1_list_gram_matrices[0].shape[0]
  var2_shape0 = var2_list_gram_matrices[0].shape[0]
  var3_shape0 = var3_list_gram_matrices[0].shape[0]

  assert var1_shape0 == var2_shape0 and var1_shape0 == var3_shape0

  # Adding selected filters to var3
  # var3_list_gram_matrices += selected_filters_gram_matrices

  H_XZ = S_multiple(
      var1_list_gram_matrices + var3_list_gram_matrices, var1_shape0, alpha
  )
  H_YZ = S_multiple(
      var2_list_gram_matrices + var3_list_gram_matrices, var1_shape0, alpha
  )
  H_XYZ = S_multiple(
      var1_list_gram_matrices + var2_list_gram_matrices + var3_list_gram_matrices,
      var1_shape0,
      alpha,
  )
  H_Z = S_multiple(var3_list_gram_matrices, var1_shape0, alpha)
  return H_XZ + H_YZ - H_XYZ - H_Z

def permutation_test_CMI(
    extLabels_list_gram_matrices,
    data_remain_list_gram_matrices,
    data_select,
    sigma1,
    sigma2,
    alpha,
    CMI,
    P,
):
  data_select = copy.deepcopy(data_select)
  count = 0
  target_filter = data_select[:, [-1], :, :]
  s1, _, s3, s4 = target_filter.shape
  target_filter = target_filter.reshape((s1, s3 * s4))
  for i in range(P):
    # print(f'permutation_test_CMI: {i}')
    perm_filter = target_filter[np.random.permutation(s1), :]
    perm_filter = perm_filter.reshape((s1, 1, s3, s4))
    data_select[:, [-1], :, :] = perm_filter
    data_select_list_gram_matrices = make_list_gram_matrices(
        data_select, sigma2, False
    )

    CMI_per = conditional_mutual_information_estimation(
        extLabels_list_gram_matrices,
        data_remain_list_gram_matrices,
        data_select_list_gram_matrices,
        sigma1,
        sigma2,
        alpha,
    )

    if CMI_per <= CMI:
      count = count + 1

  p_value = count / P
  return p_value

def filter_number_selection(data, extLabels, input, P, eta, max_iters=None, list_selected_fm_before=[], pruning_method=None):
  selected_fm_before_gram_matrices = []
  for fm in list_selected_fm_before:
    sigma = fm.shape[0] ** (-1 / (4 + fm.shape[2] * fm.shape[3]))
    sigma = 5 * sigma
    fm_gram_matrices = make_list_gram_matrices(fm, sigma=sigma, first_var=False)
    selected_fm_before_gram_matrices.extend(fm_gram_matrices)

  CMI_vector = []
  MI_XT_vector = []
  MI_TY_vector = []
  p_value_vector = []

  sel_flag = []
  sel_remain = list(range(0, data.shape[1]))

  sigma1 = extLabels.shape[0] ** (-1 / (4 + extLabels.shape[1]))
  sigma2 = data.shape[0] ** (-1 / (4 + data.shape[2] * data.shape[3]))
  sigma3 = input.shape[0] ** (-1 / (4 + input.shape[2] * input.shape[3]))

  sigma1 = 5 * sigma1
  sigma2 = 5 * sigma2
  sigma3 = 5 * sigma3

  alpha = 1.01

  extLabels_list_gram_matrices = make_list_gram_matrices(
      extLabels, sigma1, True)
  input_list_gram_matrices = make_list_gram_matrices(input, sigma3, True)

  i = 0
  while i <= data.shape[1] - 1 - 1:
    print(f"Selecting the {i:02d}-th filter", flush=True)
    MI_vector = np.zeros((data.shape[1] - i,))

    for j in range(MI_vector.shape[0]):
      pick_filter = [*sel_flag, sel_remain[j]]
      # pick_filter = list(set(pick_filter))
      variable1 = data[:, pick_filter, :, :]
      variable1_list_gram_matrices = make_list_gram_matrices(
          variable1, sigma2, False
      )
      MI_vector[j] = mutual_information_estimation(
          extLabels_list_gram_matrices,
          selected_fm_before_gram_matrices + variable1_list_gram_matrices,
          sigma1,
          sigma2,
          alpha,
      )
    index = np.argmax(MI_vector)

    sel_flag.append(sel_remain[index])
    del sel_remain[index]
    del MI_vector

    data_sel_list_matrices = make_list_gram_matrices(
        data[:, sel_flag, :, :], sigma2, False
    )
    data_remain_list_matrices = make_list_gram_matrices(
        data[:, sel_remain, :, :], sigma2, False
    )

    MI_XT = mutual_information_estimation(
        input_list_gram_matrices, data_sel_list_matrices, sigma3, sigma2, alpha
    )
    MI_XT_vector.append(MI_XT)

    MI_TY = mutual_information_estimation(
        extLabels_list_gram_matrices, data_sel_list_matrices, sigma1, sigma2, alpha
    )
    MI_TY_vector.append(MI_TY)

    CMI = conditional_mutual_information_estimation(
        extLabels_list_gram_matrices,
        data_remain_list_matrices,
        selected_fm_before_gram_matrices + data_sel_list_matrices,
        sigma1,
        sigma2,
        alpha
    )

    if PTEST in pruning_method:
      p_value = permutation_test_CMI(
          extLabels_list_gram_matrices,
          data_remain_list_matrices,
          data[:, sel_flag, :, :],
          sigma1,
          sigma2,
          alpha,
          CMI,
          P,
      )
      CMI_vector.append(CMI)
      p_value_vector.append(p_value)
      if p_value >= eta:
        break
    else:
      p_value = -1.0
      CMI_vector.append(CMI)
      p_value_vector.append(p_value)

    i = i + 1

    if max_iters is not None and i > max_iters:
      break
    
  return CMI_vector, MI_XT_vector, MI_TY_vector, sel_flag, p_value_vector
