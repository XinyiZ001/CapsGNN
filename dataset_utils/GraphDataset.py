import os,random,json
import numpy as np
import networkx as nx
import pickle

class GraphDataset():
    def __init__(self,input_dir,extn,class_label_fname):
        extn = extn + '_pro'
        graphs_dataset = []
        self.input_dir = input_dir
        for file in os.listdir(input_dir):
            if file.endswith(extn):
                file_path = os.path.join(input_dir,file)
                graphs_dataset.append(file_path)
        self.graph_to_label = {line.rstrip().split('.')[0]:int(line.rstrip().split()[1]) for line in open(class_label_fname,'r')}
        self.num_classes = len(set(self.graph_to_label.values()))
        self.node_index = {}
        self.scan(graphs_dataset)
        self.graphs_dataset = graphs_dataset
        self.graph_read_index = 0

    def scan(self,graphs_dataset):
        for graph_file in graphs_dataset:
            with open(graph_file,'rb') as f:
                graph = pickle.load(f)
            break
        attri_len = [1] * len(graph['node_attri'])

        self.reconstruct_num = 0
        for graph_file in graphs_dataset:
            with open(graph_file,'rb') as f:
                graph = pickle.load(f)
            for attri_idx in range(len(attri_len)):
                node_attri = graph['node_attri'][attri_idx]
                if max(node_attri)+1>attri_len[attri_idx]:
                    attri_len[attri_idx] = max(node_attri)+1

        self.attri_len = attri_len
        self.reconstruct_num = len(graph['reconstruct'])


    def print_status(self):
        print("Dataset Information:")
        print(" Graph number: {}".format(len(self.graphs_dataset)))
        print(" Attribute Info:")
        for list_idx,attri_list in enumerate(self.attri_len):
            print("     Attri {} number: {}".format(list_idx,self.attri_len[list_idx]))
        print(" Number of Classes: {}".format(self.num_classes))

    def files_shuffle(self):
        random.shuffle(self.graphs_dataset_train)

    def files_shuffle_valid(self):
        random.shuffle(self.graphs_dataset_valid)

    def files_shuffle_test(self):
        random.shuffle(self.graphs_dataset_test)


    def data_gen(self, data_source, batch_size=1):
        if_epoch = 0
        index_start = self.graph_read_index
        index_end = min(self.graph_read_index + batch_size, len(data_source))
        self.graph_read_index += batch_size

        graph_files = data_source[index_start:index_end]
        max_node_num = 0
        node_attris = [];adj_mats = [];labels = []; reconstructs = []
        for graph_file_name in graph_files:
            graph_file_name = str(graph_file_name)
            graph_file = os.path.join(self.input_dir,graph_file_name)
            with open(graph_file,'rb') as f:
                graph = pickle.load(f)
            node_attri = graph['node_attri']
            if len(node_attri[0]) > max_node_num:
                max_node_num = len(node_attri[0])

            node_attris.append(node_attri)
            reconstructs.append(graph['reconstruct'])
            adj_mat = np.array(graph['adj_mat'].todense())
            adj_mats.append(adj_mat)

            labels.append(graph['label'])
        for b_idx in range(len(labels)):
            for a_idx in range(len(node_attris[0])):
                node_attris[b_idx][a_idx] = node_attris[b_idx][a_idx] + [0] * (max_node_num - len(node_attris[b_idx][a_idx]))
            adj_mat_temp = np.zeros((max_node_num, max_node_num))
            adj_mat_temp[:adj_mats[b_idx].shape[0], :adj_mats[b_idx].shape[1]] = adj_mats[b_idx]
            adj_mats[b_idx] = adj_mat_temp

        if index_end == len(data_source):
            self.graph_read_index = 0
            if_epoch = 1
        return node_attris, adj_mats, labels, reconstructs, if_epoch
