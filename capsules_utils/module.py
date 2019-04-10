import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

epsilon = 1e-11
m_plus = 0.9
m_minus = 0.1
regular_sc = 1e-7
defaut_initializer = tf.contrib.layers.xavier_initializer()
# defaut_initializer = tf.random_normal_initializer(mean=0.0,stddev=0.01)


def drop_channel(inputs,dropout_rate,name,if_train):
    """
    :param inputs: (batch, N, C, d)
    :param dropout_rate: dropout_rate
    :param name:
    :return: (?, 1, C, d)
    """
    with tf.variable_scope(name):
        inputs = tf.layers.dropout(
            inputs,
            rate=dropout_rate,
            name=name,
            noise_shape=[tf.shape(inputs)[0], tf.shape(inputs)[1],tf.shape(inputs)[2],1],
            training=if_train)  # (batch, N, C, d)
    return inputs


def node_embedding_gen(inputs,trans_mats,N,in_channel, out_channel,node_embedding_size, name):
    """
    :param inputs: (?, N, Ci, d)
    :param trans_mats: (?, N, N)
    :param N: num_of_nodes
    :param out_channel: num_of_output_channels
    :param out_embedding_size: output_embedding_size
    :param name:
    :return: # (?, N, 1, d_out)
    """
    with tf.variable_scope(name) as scope:
        inputs = tf.reshape(inputs, shape=[tf.shape(inputs)[0], N, in_channel * node_embedding_size])  # (?, N, C*d)
        inputs = tf.expand_dims(inputs, axis=2)  # (?, N, 1, C*d)
        nets = tf.layers.conv2d(
            inputs,
            filters=node_embedding_size * out_channel,
            kernel_size=1,
            strides=1, padding='VALID', use_bias=True)  # (?, N, Co*d)
        nets = tf.squeeze(nets, axis=2)  # (?, N, C*d)
        nets = tf.matmul(trans_mats, nets)  # (?, N , Co*d)
        nets = tf.nn.tanh(nets)  # (?, N , Co*d)

        nets = tf.reshape(nets, shape=[-1, N, out_channel, node_embedding_size])  # (?, N, Co, d_out)

    tf.logging.info("{} output shape: {}".format(name, nets.get_shape()))
    return nets


def nn_attention_layer(inputs,batch_size,mask,name,emb_size,channel_num):
    """
    :param inputs: (batch, N, C, d)
    :param batch_size:
    :param mask: (batch, N, 1)
    :param name:
    :param emb_size: int(d)
    :param channel_num: int(C)
    :return: (batch, N, C, d)
    """

    N = tf.shape(inputs)[1]
    with tf.variable_scope(name) as scope:
        inputs_ = tf.reshape(inputs, shape=[batch_size * N, emb_size * channel_num])  # (?*N, C*d)
        atten = tf.layers.dense(inputs_, units=int(emb_size * channel_num / 16), activation=tf.nn.tanh)  # (?*N, C*d/16)
        atten = tf.layers.dense(atten, units=channel_num, activation=None)  # (?*N, C)
        atten = tf.reshape(atten, shape=[batch_size, N, channel_num, 1])  # (?, N, C, 1)
        atten = mask_softmax(atten, mask, dim=1)  # (batch, N, C, 1)

        input_scaled = tf.multiply(inputs, atten)  # (batch, N, C, 1)
        num_nodes = tf.expand_dims(tf.reduce_sum(mask, axis=1, keep_dims=True), axis=-1)
        input_scaled = input_scaled * num_nodes

    return input_scaled


def capsule_layer(inputs, Ci, Co, in_emb_size, graph_emb_size, iterations, position_emb_size,nodes_indicator, batch_size, coordinate, name):
    """
    :param inputs: (?, N, C, d)
    :param Ci:
    :param Co:
    :param in_emb_size:
    :param out_emb_size:
    :param iterations:
    :param position_emb_size:
    :param nodes_indicator: (?, N, 1)
    :param batch_size:
    :param name:
    :param shared:
    :param with_position: bool
    :return:
    """
    with tf.variable_scope(name) as scope:
        inputs_poses = inputs
        if coordinate:
            i_size = Ci-1
            out_emb_size = graph_emb_size+position_emb_size
        else:
            i_size = Ci
            out_emb_size = graph_emb_size
        o_size = Co  # Co
        in_emb_size = in_emb_size  # di
        N = tf.shape(inputs_poses)[1]

        with tf.variable_scope('votes') as scope:
            votes = mat_transform_with_coordinate(
                input=inputs_poses,
                Co=Co,
                in_emb_size=in_emb_size,
                out_emb_size=graph_emb_size,
                batch_size=batch_size,
                num_node=N,
                position_emb_size=position_emb_size,
                coordinate = coordinate
            )  # (batch, 1, 1, N, i_size, Co, d)
            votes = tf.reshape(votes,
                               shape=[batch_size, 1, 1, tf.shape(votes)[3] * tf.shape(votes)[4], o_size,out_emb_size])  # (batch, 1, 1, N*i_size, Co, d)
            tf.logging.info("{} votes shape: {}".format(name, votes.get_shape()))

        with tf.variable_scope('routing') as scope:
            num_nodes = tf.reduce_sum(nodes_indicator, axis=1, keep_dims=True)  # (?, #1, 1)
            num_nodes = num_nodes[:, tf.newaxis, tf.newaxis, :, tf.newaxis, :]  # (?, 1, 1, #1, 1, 1)
            b_IJ = tf.zeros(shape=[batch_size, 1, 1, N*i_size, Co, 1], dtype=np.float32)  # (?, 1, 1, N*Ci, Co, 1)
            v_j, a_j = routing_graph(
                votes=votes,
                b_Ij=b_IJ,
                num_nodes = num_nodes,
                iterations=iterations)  # (?, 1, 1, 1, Co, d)
            v_j = tf.reshape(v_j,shape=[batch_size,1,Co,out_emb_size])  # (?, 1, Co, d)
            a_j = tf.reshape(a_j,shape=[batch_size,1,Co,1])
    return v_j,a_j


def class_capsules(inputs_poses,graph_emb_size, num_classes, iterations, batch_size, name):
    """
    :param inputs: (?, 1, C, d)
    :param num_classes: 2
    :param iterations: 3
    :param batch_size: ?
    :param name:
    :return poses, activations: poses (?, num_classes, 1, d), activation (?, num_classes).
    """
    inputs_poses = inputs_poses
    inputs_shape = inputs_poses.get_shape()
    in_emb_size = int(inputs_shape[-1])    # d
    N = tf.shape(inputs_poses)[1]
    i_size = tf.shape(inputs_poses)[2]
    with tf.variable_scope(name) as scope:
        with tf.variable_scope('votes') as scope:
            votes = mat_transform_with_coordinate(
                input=inputs_poses,
                Co=num_classes,
                in_emb_size=in_emb_size,
                out_emb_size=graph_emb_size,
                batch_size=batch_size,
                num_node=N
            )  # (batch, 1, 1, 1, Ci, Co, d)
            votes = tf.squeeze(votes, axis=3)  # (?, 1, 1, Ci, Co, d)
            tf.logging.info("{} votes shape: {}".format(name,votes.get_shape()))

        with tf.variable_scope('routing') as scope:
            num_nodes = tf.ones(shape=[batch_size, 1, 1, 1, 1, 1], dtype=np.float32)
            b_IJ = tf.zeros(shape=[batch_size, 1, 1, i_size, num_classes, 1], dtype=np.float32)  # (?, 1, 1, N*Ci, Co, 1)
            v_j, a_j= routing_graph(votes=votes, b_Ij=b_IJ, num_nodes = num_nodes, iterations=iterations)  # (?, 1, 1, 1, Co, d)
            v_j = tf.reshape(v_j, shape=[batch_size, 1, num_classes, graph_emb_size])  # (?, 1, Co, d)
            a_j = tf.reshape(a_j, shape=[batch_size, 1, num_classes, 1])
        return v_j,a_j


def loss_layer(inputs,reconstruct_value,label,lambda_val,batch_size,num_class,reconstruct_num,name,reg_scale):
    """
    :param inputs: ((?, 1, C, d),(?, 1, C, 1))
    :param node_indicator: (?, N, 1)
    :param num_classes: 2
    :param iterations: 3
    :param batch_size: ?
    :param name:
    :return loss
    """
    v_j,a_j = inputs
    input_shape = v_j.get_shape()
    with tf.variable_scope(name) as scope:
        with tf.variable_scope('margin_loss') as scope:
            a_j = tf.reshape(a_j,shape=[batch_size,num_class,1,1])

            max_l = tf.square(tf.maximum(0., m_plus - a_j))
            max_r = tf.square(tf.maximum(0., a_j - m_minus))

            max_l = tf.reshape(max_l, shape=(batch_size, num_class))
            max_r = tf.reshape(max_r, shape=(batch_size, num_class))

            T_c = tf.one_hot(label,depth=num_class)
            L_c = T_c * max_l + lambda_val * (1 - T_c) * max_r

            margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))
        with tf.variable_scope('recons_loss') as scope:
            v_j = tf.reshape(v_j,shape=[batch_size,num_class,input_shape[-1]])

            T_c_ = T_c[:,:,tf.newaxis]
            correct_output = tf.multiply(v_j,T_c_)
            correct_output = tf.reduce_sum(correct_output,axis=1)

            decoded = tf.contrib.layers.fully_connected(correct_output,num_outputs= 128)
            decoded = tf.contrib.layers.fully_connected(decoded,num_outputs = reconstruct_num,activation_fn = None)

            decoded = tf.nn.sigmoid(decoded)

            neg_indicator = tf.where(reconstruct_value < 1e-5, tf.ones(tf.shape(reconstruct_value)), tf.zeros(tf.shape(reconstruct_value)))
            pos_indicator = 1- neg_indicator
            reconstruct_value_max = tf.reduce_max(reconstruct_value,axis=1,keep_dims=True)

            reconstruct_value = reconstruct_value/(reconstruct_value_max+epsilon)
            diff = tf.square(decoded - reconstruct_value)

            neg_loss = tf.reduce_max(diff * neg_indicator, axis=-1)
            pos_loss = tf.reduce_max(diff * pos_indicator, axis=-1)
            # neg_loss = tf.reduce_sum(diff*neg_indicator,axis=-1) / (tf.reduce_sum(neg_indicator*1.0,axis=-1)+epsilon)
            # pos_loss = tf.reduce_sum(diff*pos_indicator,axis=-1) / (tf.reduce_sum(pos_indicator*1.0,axis=-1)+epsilon)

            recon_loss = tf.reduce_mean(pos_loss+neg_loss)

        loss = margin_loss + recon_loss*reg_scale
    return loss,recon_loss, margin_loss


def error_layer(input,label,batch_size,num_class,name):
    """
    :param inputs: ((?, 1, C, d),(?, 1, C, 1))
    :param num_classes: 2
    :param iterations: 3
    :param batch_size: ?
    :param name:
    :return error
    """
    v_j,a_j = input# (?, 1, Co, d)
    with tf.variable_scope(name) as scope:
        result = tf.reshape(a_j,shape = [batch_size, num_class])
        pred = tf.argmax(result,axis=1)
        error = tf.cast(tf.not_equal(pred,label),dtype=tf.int32)
        error_sum = tf.reduce_sum(error)
        error_mean = tf.reduce_mean(error)
    return error_sum,error_mean,error


def mask_softmax(inputs,mask,dim):
    """
    :param inputs: (batch, N, C, 1)
    :param mask: (batch, N, 1)
    :param dim: does softmax along which axis
    :return: normalized attention (batch, N, C, 1)
    """
    with tf.variable_scope('bulid_mask') as scope:
        e_inputs = tf.exp(inputs) + epsilon # (batch, N, C, 1)
        mask = mask[:,:,:,tf.newaxis]  # (batch, N, 1, 1)
        mask = tf.tile(mask,multiples=[1,1,tf.shape(e_inputs)[2],1])  # (batch, N, C, 1)
        masked_e_inputs = tf.multiply(e_inputs,mask)  # (batch, N, C, 1)
        sum_col = tf.reduce_sum(masked_e_inputs,axis=dim,keep_dims=True) +epsilon  # (batch, 1, C, 1)
        result = tf.div(masked_e_inputs,sum_col)  # (batch, N, C, 1)
    return result


def adj_mats_normalize(batch_adj_mats, node_indicator):
    """
    :param batch_adj_mats: (batch, N, N)
    :param node_indicator: (batch, N, 1)
    :return: (batch, N, N)
    """
    with tf.variable_scope('adj_mats_normal') as scope:
        row_sum = tf.reduce_sum(batch_adj_mats,axis=-1,keep_dims=True)  # (batch , N, 1)
        row_sum = tf.where(node_indicator<1e-5,tf.ones(tf.shape(row_sum)),row_sum)
        r_inv = tf.pow(row_sum,-0.5)
        r_inv = tf.squeeze(r_inv * node_indicator,axis=-1)
        r_mat_inv = tf.matrix_diag(r_inv)  # (batch , N, N)
        batch_trans_mats = tf.matmul(tf.matmul(r_mat_inv,batch_adj_mats),r_mat_inv)

        return batch_trans_mats


def mat_transform_with_coordinate(input,Co,in_emb_size,out_emb_size, batch_size,num_node,position_emb_size = 0,coordinate = False):
    """
    :param input: (?, N, Ci, d)
    :param Co:
    :param in_emb_size:
    :param out_emb_size:
    :param batch_size:
    :param num_node:
    :param position_emb_size:
    :param corordinate:
    :return: (batch, 1, 1, N, Ci, Co, d)
    """

    if coordinate:

        input_shape = input.get_shape()
        Ci = input_shape[2]
        output = input[:, :, tf.newaxis, :, tf.newaxis, tf.newaxis, :]  # (batch, N, 1, Ci, 1, 1, 16)
        properties = output[:, :, :, :-1, :, :, :]  # (batch, N, 1, Ci-1 , 1, 1, 16)
        position = output[:, :, :, -1, :, :, :]  # (batch, N, 1, 1, 1, 16)
        position = tf.expand_dims(position, axis=3)  # (batch, N, 1, 1, 1, 1, 16)

        w_pro = slim.variable(
            'w_pro',
            shape=[1, 1, 1, Ci-1, Co, in_emb_size, out_emb_size], dtype=tf.float32,
            initializer=defaut_initializer,
            regularizer=slim.l2_regularizer(regular_sc))  # (1, 1, 1, Ci-1, Co, d, d)
        w_pro = tf.tile(w_pro, [batch_size, num_node, 1, 1, 1, 1, 1])  # (batch, N, 1, Ci-1, Co, d, d)
        w_pos = slim.variable(
            'w_pos',
            shape=[1, 1, 1, 1, Co, in_emb_size, position_emb_size], dtype=tf.float32,
            initializer=defaut_initializer,
            regularizer=slim.l2_regularizer(regular_sc))  # (1, 1, 1, 1, Co, d, d)
        w_pos = tf.tile(w_pos, [batch_size, num_node, 1, 1, 1, 1, 1])  # (batch, N, 1, 1, Co, d, d)

        properties = tf.tile(properties, [1, 1, 1, 1, Co, 1, 1])  # (batch, N, 1, Ci-1, Co, 1, d)
        position = tf.tile(position, [1, 1, 1, 1, Co, 1, 1])   # (batch, N, 1, 1, Co, 1, d)
        votes_properties = tf.matmul(properties, w_pro,name='Trans')  # (batch, N, 1, Ci-1 , Co, 1, d)
        votes_positions = tf.tile(tf.matmul(position,w_pos), multiples=[1,1,1,Ci-1,1,1,1])  # (batch, N, 1, Ci-1, Co, 1, d)
        votes = tf.concat([votes_properties,votes_positions],axis=-1)
        votes = tf.reshape(votes, [batch_size, 1, 1, num_node, Ci-1, Co, position_emb_size+out_emb_size])  # (batch, 1, 1, N, Ci-1, Co, d_pro+d_pos)
    else:
        input_shape = input.get_shape()
        Ci = input_shape[2]
        output = input[:, :, tf.newaxis, :, tf.newaxis, tf.newaxis, :]  # (batch, N, 1, Ci, 1, 1, 16)

        w = slim.variable(
            'w',
            shape=[1, 1, 1, Ci, Co, in_emb_size, out_emb_size], dtype=tf.float32,
            initializer=defaut_initializer,
            regularizer=slim.l2_regularizer(regular_sc))  # (1, 1, 1, Ci, Co, d, d)
        w = tf.tile(w, [batch_size, num_node, 1, 1, 1, 1, 1])  # (batch, N, 1, Ci, Co, d, d)

        output = tf.tile(output, [1, 1, 1, 1, Co, 1, 1])  # (batch, N, 1, Ci, Co, 1, d)

        votes = tf.matmul(output, w, name='Trans')  # (batch, N, 1, Ci, Co, 1, d)
        votes = tf.reshape(votes, [batch_size, 1, 1, num_node, Ci, Co, out_emb_size])  # (batch, 1, 1, N, Ci, Co, d)

    return votes


def routing_graph(votes,b_Ij,num_nodes, iterations = 3):
    """
    :param votes: (?, 1, 1, Ci, Co, d)
    :param b_Ij: (?, 1, 1, Ci, Co, 1)
    :param num_nodes: (?, 1, 1, #1, 1, 1)
    :param iterations: 3
    :return:
    """
    u_hat = votes
    u_hat_stopped = tf.stop_gradient(u_hat)
    for r_iter in range(iterations):
        with tf.variable_scope('iter_' + str(r_iter)) as scope:
            c_ij = tf.nn.softmax(b_Ij, dim=4)  # (?, 1, 1, Ci, Co, 1)
            if r_iter == iterations - 1:
                s_j = tf.multiply(c_ij, u_hat)
                s_j = tf.reduce_sum(s_j, axis=3, keep_dims=True)/num_nodes
                v_j, a_j = squash(s_j)  # (?, 1, 1, 1, Co, d)
            elif r_iter < iterations - 1:
                s_j = tf.multiply(c_ij, u_hat_stopped)
                s_j = tf.reduce_sum(s_j, axis=3, keep_dims=True)/num_nodes
                v_j, a_j = squash(s_j)  # (?, 1, 1, 1, Co, d)
                v_j = tf.tile(v_j, [1, 1, 1, tf.shape(votes)[3], 1, 1])  # (?, 1, 1, Ci, Co, d)
                u_produce_v = tf.reduce_sum(u_hat_stopped * v_j, axis=5,
                                            keep_dims=True)  # (?, 1, 1, Ci, Co, 1)
                b_Ij += u_produce_v

    return v_j,a_j


def squash(v_j, dim = -1):
    """
    :param v_j: (?, 1, 1, 1, Co, d)
    :param dim:
    :return:
    """
    vec_squared_norm = tf.reduce_sum(tf.square(v_j), dim, keep_dims=True)
    a_j =  vec_squared_norm / (1 + vec_squared_norm)
    scalar_factor = a_j / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * v_j  # element-wise
    return vec_squashed,a_j
