import numpy as np

def load_node_labels(netcc_file_path):
    with open(netcc_file_path) as f:
        lines = f.readlines()
        node_labels = lines[3].split('\t')
        node_labels = [i.strip() for i in node_labels]
    return node_labels

# x = load_node_labels('30AQ_01_rest_04_LPI_000.netcc')
# print(x)
# y = np.loadtxt('30AQ_01_rest_04_LPI_000.netcc', delimiter='\t')
# print(y.shape)