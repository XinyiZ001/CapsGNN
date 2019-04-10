import argparse, os, shutil, scipy
import numpy as np
import networkx as nx
import scipy.sparse as sp
import pickle,random

def node_attri():
    """
    define the input and reconstruct graph info
    :return:
    """
    attri_name = {
        'Label':{
            'channel':1,
            'if_input':True,
            'if_reconst':True
        },
        'degree':{
            'channel':1,
            'if_input':True,
            'if_reconst':True
        },
        'constant':{
            'channel': 1,
            'if_input':True,
            'if_reconst':False
        },
    }
    return attri_name


def settings():
    parser = argparse.ArgumentParser("Data_Preprocessing")

    parser.add_argument("--dataset_input_dir", type=str, default="../gexf_files/NCI1",help="Where gexf dataset is stored.")
    parser.add_argument("--output_data_dir", type=str, default="../data_plk/",help="Where output dataset is stored.")
    parser.add_argument("--pickle_v", type=int, default=2,help="version of pickle.")
    parser.add_argument("--x_fold", type=int, default=10,help="build train_test_split_index for x_fold." )
    parser.add_argument("--gen_split_file", type=bool, default=False, help="If generate split file")

    return parser.parse_args()


class GraphDataset():
    def __init__(self,input_dir,extn,class_label_fname,dataset_output_dir,attri_dict):
        graphs_dataset = []
        for file in os.listdir(input_dir):
            if file.endswith(extn):
                file_path = os.path.join(input_dir,file)
                graphs_dataset.append(file_path)
        self.attri_dict = attri_dict
        self.graph_to_label = {line.rstrip().split()[0]:int(line.rstrip().split()[1]) for line in open(class_label_fname,'r')}
        self.class_label_fname = class_label_fname
        self.num_classes = len(set(self.graph_to_label.values()))
        self.node_index = {}
        self.scan(graphs_dataset)
        self.graphs_dataset = graphs_dataset
        self.graph_read_index = 0
        self.graph_read_index_test = 0
        self.dataset_output_dir = dataset_output_dir
        data_frame = {
            'id_to_attri_maps': self.id_to_attri_maps,
            'attri_to_id_maps': self.attri_to_id_maps
        }
        ofname = input_dir + '_id_attri'
        with open(ofname, 'wb') as f:
            pickle.dump(data_frame, f, protocol=pickle_v)

    def scan(self,graphs_dataset):
        """
        Collect graph dataset info
        :param graphs_dataset:
        :return:
        """
        print('Dataset Scan:-----------------------')
        attri_to_id_maps = []
        label_to_id_map = {}
        id_to_label_map = []
        id_to_attri_maps = []
        degree_max = 0
        for label in self.graph_to_label.values():
            if label in set(id_to_label_map):
                continue
            label_to_id_map[label] = len(label_to_id_map)
            id_to_label_map.append(label)
        num_edge = []
        num_node = []
        for graph_idx,graph_file in enumerate(graphs_dataset):
            graph = nx.read_gexf(graph_file)
            num_edge.append(graph.number_of_edges())
            num_node.append(len(graph))
            attri_idx = 0; attri_idx_all = 0
            for attri_name in self.attri_dict:
                num_atti_channel = self.attri_dict[attri_name]['channel']
                if attri_name == 'degree':
                    continue
                for attri_idx_sub in range(num_atti_channel):
                    attri_idx_all = attri_idx_sub+attri_idx
                    nodes = graph.nodes(data=attri_name)

                    for node,attri in nodes:
                        if attri is None:
                            attri = 'None'
                            if attri_name != 'constant':
                                print('"None" appears !!')
                        if len(id_to_attri_maps) <= attri_idx_all:
                            id_to_attri_maps.append([])
                            attri_to_id_maps.append(dict())
                        if attri in set(id_to_attri_maps[attri_idx_all]):
                            continue
                        else:
                            id_to_attri_maps[attri_idx_all].append(attri)
                            attri_to_id_maps[attri_idx_all][attri] = len(attri_to_id_maps[attri_idx_all])
                attri_idx = attri_idx_all +1
            for node,degree in graph.degree():
                if degree > degree_max:
                    degree_max = degree

            if graph_idx % 10 == 0:
                print("Scaned file :----------------------- {}%".format(graph_idx*100.0/len(graphs_dataset)))
        num_edge = np.array(num_edge)
        print("Average number of edges: {}".format(np.average(num_edge)))
        print("Average number of nodes: {}".format(np.average(np.array(num_node))))

        for attri_name in self.attri_dict:
            if attri_name == 'degree':
                num_channel = self.attri_dict[attri_name]['channel']
                for channel_idx in range(num_channel):
                    if len(id_to_attri_maps) <= attri_idx+channel_idx:
                        id_to_attri_maps.append([])
                        attri_to_id_maps.append(dict())
                    attri_to_id_maps[attri_idx+channel_idx] = {i: i for i in range(degree_max+1)}
                    id_to_attri_maps[attri_idx+channel_idx] = list(range(degree_max+1))

        self.id_to_attri_maps = id_to_attri_maps
        self.attri_to_id_maps = attri_to_id_maps
        self.label_to_id_map = label_to_id_map
        self.id_to_label_map = id_to_label_map

    def print_status(self):
        print("Dataset Information:")
        print(" Graph number: {}".format(len(self.graphs_dataset)))
        print(" Attribute Info:")
        for list_idx,attri_list in enumerate(self.attri_to_id_maps):
            print("     Attri {} number: {}".format(list_idx+1,len(self.attri_to_id_maps[list_idx])))
        print(" Number of Classes: {}".format(self.num_classes))

    def data_gen(self,pickle_v,save):
        """
        Generate experimental data format and save to target dir.
        :param pickle_v: 2 & 3
        :param save: True
        :return:
        """
        reconst_index = self.build_reconst_index()
        reconst_index_str = [str(i) for i in reconst_index]
        print('reconstruct_index:{}'.format(' '.join(reconst_index_str)))

        input_index = self.buil_input_index()
        input_index_str = [str(i) for i in input_index]
        print('input_index:{}'.format(' '.join(input_index_str)))

        graph_name = []
        for graph_idx,graph_file in enumerate(self.graphs_dataset):
            fname = os.path.basename(graph_file)
            ofname = os.path.join(self.dataset_output_dir,fname.replace('.gexf','.gexf_pro'))
            node_attri_recon = []
            node_attri_input = []
            graph = nx.read_gexf(graph_file)
            nodelist = list(graph)

            for node in nodelist:
                attri_idx = 0
                for attri_name in self.attri_dict:
                    num_atti_channel = self.attri_dict[attri_name]['channel']
                    if attri_name == 'degree':
                        continue
                    try:
                        attri_value = graph.node[node][attri_name]
                    except:
                        attri_value = 'None'

                    for attri_idx_sub in range(num_atti_channel):
                        attri_idx_all = attri_idx_sub + attri_idx

                        if len(node_attri_recon) <= attri_idx_all:
                            node_attri_recon.append([])
                            node_attri_input.append([])
                        node_attri_recon[attri_idx_all].append(self.attri_to_id_maps[attri_idx_all][attri_value])
                        node_attri_input[attri_idx_all].append(self.attri_to_id_maps[attri_idx_all][attri_value])
                    attri_idx = attri_idx_all+1

                degree = graph.degree[node]

                for attri_name in self.attri_dict:
                    if attri_name == 'degree':
                        num_channel = self.attri_dict[attri_name]['channel']
                        for channel_idx in range(num_channel):
                            if len(node_attri_recon) <= attri_idx+channel_idx:
                                node_attri_recon.append([])
                                node_attri_input.append([])
                            node_attri_input[attri_idx+channel_idx].append(self.attri_to_id_maps[attri_idx+channel_idx][degree])
                            node_attri_recon[attri_idx+channel_idx].append(self.attri_to_id_maps[attri_idx+channel_idx][degree])

            adj_mat = nx.adjacency_matrix(graph).todense().astype(float)
            adj_mat += np.eye(adj_mat.shape[0], adj_mat.shape[1])
            adj_mat = sp.coo_matrix(adj_mat)


            graph_label = self.label_to_id_map[self.graph_to_label[os.path.basename(ofname).rstrip().split('_')[0]]]

            reconstruct_value = []
            for a_idx in reconst_index: # index of attri that you want to reconstruct
                reconstruct_value += [node_attri_recon[a_idx].count(i) for i in range(len(self.attri_to_id_maps[a_idx]))]

            input_value = []
            for a_idx in input_index:
                input_value.append(node_attri_input[a_idx])

            data_frame = {
                'adj_mat':adj_mat,
                'node_attri':input_value,
                'reconstruct':reconstruct_value,
                'label':graph_label,
            }
            graph_name.append(os.path.basename(ofname))
            if save:
                with open(ofname,'wb') as f:
                    pickle.dump(data_frame,f,protocol=pickle_v)
            if graph_idx % 10 == 0:
                print("processed file : {}%".format(graph_idx*100.0/len(self.graphs_dataset)))

        if save:
            shutil.copy(self.class_label_fname, self.dataset_output_dir+'.Labels')
        return graph_name

    def build_reconst_index(self):
        reconst_index = []
        attri_idx = 0
        for attri_name in self.attri_dict:
            num_atti_channel = self.attri_dict[attri_name]['channel']
            if_reconst = self.attri_dict[attri_name]['if_reconst']
            if attri_name == 'degree':
                continue
            for attri_idx_sub in range(num_atti_channel):
                attri_idx_all = attri_idx_sub + attri_idx
                if if_reconst:
                    reconst_index.append(attri_idx_all)
            attri_idx = attri_idx_all + 1

        for attri_name in self.attri_dict:
            if attri_name == 'degree':
                num_channel = self.attri_dict[attri_name]['channel']
                if_reconst = self.attri_dict[attri_name]['if_reconst']
                for channel_idx in range(num_channel):
                    attri_idx_all = attri_idx + channel_idx
                    if if_reconst:
                        reconst_index.append(attri_idx_all)

        return  reconst_index

    def buil_input_index(self):
        input_index = []
        attri_idx = 0
        for attri_name in self.attri_dict:
            num_atti_channel = self.attri_dict[attri_name]['channel']
            if_input = self.attri_dict[attri_name]['if_input']
            if attri_name == 'degree':
                continue
            for attri_idx_sub in range(num_atti_channel):
                attri_idx_all = attri_idx_sub + attri_idx
                if if_input:
                    input_index.append(attri_idx_all)
            attri_idx = attri_idx_all + 1

        for attri_name in self.attri_dict:
            if attri_name == 'degree':
                num_channel = self.attri_dict[attri_name]['channel']
                if_input = self.attri_dict[attri_name]['if_input']
                for channel_idx in range(num_channel):
                    attri_idx_all = attri_idx + channel_idx
                    if if_input:
                        input_index.append(attri_idx_all)

        return input_index


    def train_test_idx(self,x_fold,graph_name_list,pickle_v):

        print('Build test index ##################')
        remainder = len(graph_name_list)%x_fold
        test_graph_name_list = [[] for i in range(x_fold)]
        fold = 0

        random.shuffle(graph_name_list)
        for graph_name in graph_name_list:
            test_graph_name_list[fold].append(os.path.basename(graph_name))
            fold += 1
            if fold == x_fold:
                fold = 0

        train_test_split = []
        for i in range(x_fold):
            test_list = test_graph_name_list[i]
            if i == x_fold-1:
                i = -1
            val_list = test_graph_name_list[i+1]
            train_list = list(set(graph_name_list)-set(val_list)-set(test_list))
            data = {
                'train':train_list,
                'test':test_list,
                'val':val_list
            }
            train_test_split.append(data)
        fname = os.path.basename(self.dataset_output_dir)
        ofname = self.dataset_output_dir+'_train_test_split'

        with open(ofname, 'wb') as f:
            pickle.dump(train_test_split, f, protocol=pickle_v)

def pickle_version_convertor(dataset_dir,input_v,output_v):

    if os.path.isdir(dataset_dir):
        output_file_dir = dataset_dir + '_v_' + str(output_v)
        if not os.path.exists(output_file_dir):
            os.mkdir(output_file_dir)
        for file in os.listdir(dataset_dir):
            file_path = os.path.join(dataset_dir,file)
            output_file_path = os.path.join(output_file_dir,file)
            f_in = open(file_path,'rb')
            f_out= open(output_file_path,'wb')
            if input_v == 2 and output_v == 3:
                data_frame = pickle.load(f_in,encoding='latin1')
                pickle.dump(data_frame,f_out,protocol=output_v)
            elif input_v == 3 and output_v == 2:
                data_frame = pickle.load(f_in)
                pickle.dump(data_frame,f_out,protocol=output_v)
    else:
        file_path = dataset_dir
        output_file_path = file_path + '_v_' + str(output_v)
        f_in = open(file_path, 'rb')
        f_out = open(output_file_path, 'wb')
        if input_v == 2 and output_v == 3:
            data_frame = pickle.load(f_in, encoding='latin1')
            pickle.dump(data_frame, f_out, protocol=output_v)
        elif input_v == 3 and output_v == 2:
            data_frame = pickle.load(f_in)
            pickle.dump(data_frame, f_out, protocol=output_v)


if __name__ == '__main__':
    FLAGS = settings()
    dataset_input_dir = FLAGS.dataset_input_dir
    dataset_output_dir = os.path.join(FLAGS.output_data_dir,os.path.basename(dataset_input_dir))
    class_label_fname = FLAGS.dataset_input_dir + '.Labels'
    pickle_v = FLAGS.pickle_v
    x_fold = FLAGS.x_fold

    if not os.path.exists(dataset_output_dir):
        os.mkdir(dataset_output_dir)

    gd = GraphDataset(dataset_input_dir,extn = 'gexf',class_label_fname=class_label_fname,dataset_output_dir=dataset_output_dir, attri_dict=node_attri())
    gd.print_status()
    graph_name = gd.data_gen(pickle_v = pickle_v,save = True)
    if FLAGS.gen_split_file:
        gd.train_test_idx(x_fold, graph_name,pickle_v = pickle_v)


