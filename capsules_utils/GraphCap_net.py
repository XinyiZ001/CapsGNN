from capsules_utils.module import *


class CapsGNN_nets(object):
    '''
    Define the Capsule Graph Neural Network Structure
    '''

    def __init__(self,node_attris,num_classes,learning_rate,graph_embedding_size,node_embedding_size,iterations,net_structure,decay_step,lambda_val,reg_scale,noise,reconstruct_num,if_Attention,coordinate):
        self.node_attris = node_attris
        self.coordinate = coordinate
        self.if_Attention = if_Attention
        self.lambda_val = lambda_val
        self.node_embedding_size = node_embedding_size
        self.graph_embedding_size = graph_embedding_size
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.iterations = iterations
        self.position_emb_size= 2
        self.decay_step = decay_step
        self.net_structure = net_structure
        self.reg_scale = reg_scale
        self.noise = noise
        self.reconstruct_num = reconstruct_num
        self.net_construct()

    def net_construct(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.device('/cpu:0'):
                batch_adj_mats = tf.placeholder(tf.float32, shape=([None, None, None]))
                batch_labels = tf.placeholder(tf.int64, shape=([None, ]))
                if_train = tf.placeholder(tf.bool, shape=())

                node_indicator = tf.reduce_max(batch_adj_mats, axis=-1, keep_dims=True)  # (?, N, 1)

                node_input = []
                node_basic_embeddings = []
                node_basic_embeddings_selected = []

                for list_idx, attri_list in enumerate(self.node_attris):
                    node_input.append(tf.placeholder(tf.int32, shape=([None, None]), name='node_input_' + str(list_idx)))
                    node_basic_embeddings.append(tf.Variable(tf.random_normal(shape=[attri_list, self.node_embedding_size],mean=0.0, stddev=0.1),name='node_emb_' + str(list_idx)))
                    node_basic_embeddings_selected.append(tf.nn.embedding_lookup(node_basic_embeddings[list_idx], node_input[list_idx]) * node_indicator)

                node_basic_embeddings_selected = tf.stack(node_basic_embeddings_selected, axis=2)  # (?, N, C, d)
                node_basic_embeddings_selected = drop_channel(node_basic_embeddings_selected, dropout_rate=self.noise, name='drop_channel', if_train=if_train)
                reconstruct_value = tf.placeholder(tf.float32, shape=([None, self.reconstruct_num]))
                batch_trans_mats = adj_mats_normalize(batch_adj_mats,node_indicator)

                self.node_inputs = node_input
                self.adj_mats = batch_adj_mats
                self.labels = batch_labels
                self.if_train = if_train
                self.reconstruct_value = reconstruct_value

                multi_layer_nodes_embeddings = []
                channel_num = 0
                node_embeddings = node_basic_embeddings_selected

                # Node embedding Block
                for layer_idx, C_node_config in enumerate(self.net_structure['node_emb']):
                    # input->( ?, N, Ci, d) =====> output->(?,N,Co,d)
                    if layer_idx == 0:
                        Ci_node = len(self.node_attris)
                    else:
                        Ci_node = self.net_structure['node_emb'][layer_idx-1]
                    Co_node = self.net_structure['node_emb'][layer_idx]
                    name = 'node_embedding_' + str(layer_idx)
                    node_embeddings = node_embedding_gen(
                        inputs=node_embeddings,
                        trans_mats=batch_trans_mats,
                        N=tf.shape(batch_adj_mats)[1],
                        out_channel=Co_node,
                        in_channel=Ci_node,
                        node_embedding_size=self.node_embedding_size,name=name)  # (?, N, Co, d_out)

                    multi_layer_nodes_embeddings.append(node_embeddings)
                    channel_num += Co_node

                nets = tf.concat(multi_layer_nodes_embeddings, axis=2)  # (?, N, C_all, d)

                # Attetnion
                if self.if_Attention:
                    nets = nn_attention_layer(
                        inputs=nets,
                        batch_size=tf.shape(batch_trans_mats)[0] ,
                        mask = node_indicator,
                        name = 'attention',
                        emb_size=self.node_embedding_size,
                        channel_num = channel_num)

                # Graph embedding Block
                for layer_idx, out_graph_channel in enumerate(self.net_structure['graph_emb']):
                    if layer_idx == 0:
                        in_graph_channel = channel_num
                    else:
                        in_graph_channel = self.net_structure['graph_emb'][layer_idx-1]
                    name = 'graph_embedding_' + str(layer_idx)
                    nets, a_j = capsule_layer(
                        inputs=nets,
                        nodes_indicator=node_indicator,
                        position_emb_size=self.position_emb_size,
                        Ci = in_graph_channel,
                        Co = out_graph_channel,
                        iterations=self.iterations,
                        in_emb_size=self.node_embedding_size,
                        graph_emb_size=self.graph_embedding_size,
                        batch_size=tf.shape(batch_adj_mats)[0],
                        name= name,
                        coordinate=self.coordinate)  # (?, 1, C_all, d)
                self.graph_capsules = nets

                # Classification Block
                nets = class_capsules(
                    nets,
                    num_classes=self.num_classes, iterations=self.iterations,graph_emb_size=self.graph_embedding_size,
                    batch_size=tf.shape(batch_adj_mats)[0], name='class_capsule')
                self.classes_capsules = nets

                # Loss Function
                loss,loss_regular,loss_margin = loss_layer(
                    nets,
                    reconstruct_value=reconstruct_value,
                    label=batch_labels,
                    reg_scale = self.reg_scale,
                    lambda_val= self.lambda_val,
                    reconstruct_num = self.reconstruct_num,
                    num_class=self.num_classes,
                    batch_size=tf.shape(batch_adj_mats)[0],name = 'loss_layer')
                self.loss = loss
                self.loss_regular = loss_regular
                self.loss_margin = loss_margin

                error_sum,error_mean,not_equal = error_layer(
                    nets,
                    label=batch_labels,
                    batch_size=tf.shape(batch_adj_mats)[0],
                    num_class=self.num_classes,name = 'error_layer')
                self.error = error_sum
                self.not_equal = not_equal

                global_step = tf.Variable(0, trainable=False)
                learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                           global_step,self.decay_step, 0.9,
                                                           staircase=True)  # linear decay over time
                optimizer = tf.train.AdamOptimizer(learning_rate)
                self.optimizer = optimizer.minimize(loss, global_step=global_step)
                self.learning_step = learning_rate

                total_parameters = 0
                for variable in tf.trainable_variables():
                    # shape is an array of tf.Dimension
                    shape = variable.get_shape()
                    print("{}   :   {}".format(variable.name, shape))
                    variable_parameters = 1
                    for dim in shape:
                        variable_parameters *= dim.value
                    total_parameters += variable_parameters
                print(total_parameters)

        self.graph = graph
