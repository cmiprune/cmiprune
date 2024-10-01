from utils.accuracy import calc_accuracy
from pyclustering.cluster.xmeans import xmeans, splitting_type, kmeans_plusplus_initializer, kmeans

def find_cut_points(clusters):
  cut_points = []
  sorted_clusters = sorted(clusters, key=lambda x: x[0])
  for i in range(len(sorted_clusters) - 1):
      first_of_next = sorted_clusters[i + 1][0]
      cut_points.append((first_of_next))
  return cut_points

def select(pruned_fm, layer_idx, list_cmi, topk=None, order_pruning=None, selected_fm=None):
  assert len(list_cmi) >= 1
  assert topk is None
  
  fm_n_converted = pruned_fm[layer_idx]['CMI_vector']
  layer_n = [[i, cmi_vector] for i, cmi_vector in enumerate(fm_n_converted)]
  
  amount_initial_centers = 2
  initial_centers = kmeans_plusplus_initializer(layer_n, amount_initial_centers, random_state=42).initialize()
  
  xmeans_instance = xmeans(layer_n, initial_centers, random_state=42)
  xmeans_instance.process()
  
  clusters = xmeans_instance.get_clusters()
  centers = xmeans_instance.get_centers()
  
  cut_points = find_cut_points(clusters)
  
  calc_accuracy(pruned_fm, layer_idx, cut_points, order_pruning, selected_fm)
  
  return cut_points