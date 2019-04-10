import numpy as np
import pickle
import os
from time import time

import tensorflow as tf
from tensorflow.python import debug as tf_debug

from dataset_utils.GraphDataset import GraphDataset
from config import settings,get_net_structure
from capsules_utils.GraphCap_net import CapsGNN_nets

def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    FLAGS = settings()
    np.random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)
    extn = '.gexf'
    class_labels_fname = FLAGS.dataset_dir + '.Labels'

    net_structure = get_net_structure()
    layer_depth = len(net_structure['node_emb'])
    layer_width = net_structure['node_emb'][0]
    num_graph_capsules = net_structure['graph_emb'][0]

    graph_emb_size = FLAGS.graph_embedding_size
    node_emb_size = FLAGS.node_embedding_size

    reg_scale = FLAGS.reg_scale
    batch_size = FLAGS.batch_size
    max_epoch = FLAGS.epochs
    noise = FLAGS.noise

    train_test_split_file = FLAGS.dataset_dir + '_train_test_split'

    error_file_name = '_'.join(
        ['log', os.path.basename(FLAGS.dataset_dir), 'bs', str(batch_size), 'epoch', str(max_epoch), 'lr',
         str(FLAGS.learning_rate), 'dc', str(FLAGS.decay_step), 'noise', str(noise), 'layer-depth',
         str(layer_depth), 'layer-width', str(layer_width), 'node-dim', str(node_emb_size), 'graph-dim',
         str(graph_emb_size), 'graph-cap', str(num_graph_capsules), 'reg-scal', str(reg_scale), 'atten',
         str(FLAGS.Attention), 'coordinate', str(FLAGS.coordinate), 'iter', str(FLAGS.iterations)])

    gd = GraphDataset(input_dir=FLAGS.dataset_dir,
                      extn=extn,
                      class_label_fname=class_labels_fname)
    gd.print_status()

    GraphNet = CapsGNN_nets(
        node_attris=gd.attri_len,
        num_classes=gd.num_classes,
        learning_rate=FLAGS.learning_rate,
        node_embedding_size=FLAGS.node_embedding_size,
        graph_embedding_size=FLAGS.graph_embedding_size,
        iterations=FLAGS.iterations,
        net_structure=get_net_structure(),
        decay_step=FLAGS.decay_step,
        reg_scale = reg_scale,
        noise = noise,
        reconstruct_num=gd.reconstruct_num,
        coordinate=FLAGS.coordinate,
        if_Attention=FLAGS.Attention,
        lambda_val=FLAGS.lambda_val)

    error_write_out = ''

    tf_config = tf.ConfigProto(allow_soft_placement=True)
    # tf_config.gpu_options.allow_growth = True
    with open(train_test_split_file,'rb') as f:
        train_test_split_groups = pickle.load(f)
    for train_test_split_groups_idx,groups_dict in enumerate(train_test_split_groups):
        gd.graphs_dataset_train = groups_dict['train']
        gd.graphs_dataset_valid = groups_dict['val']
        gd.graphs_dataset_test = groups_dict['test']
        print("Dataset: train: {}, valid: {}, test: {}".format(len(gd.graphs_dataset_train),len(gd.graphs_dataset_valid),len(gd.graphs_dataset_test)))
        with tf.Session(graph=GraphNet.graph,config=tf_config) as sess:

            init = tf.global_variables_initializer()
            sess.run(init)
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

            train_step = 0
            for i in range(FLAGS.epochs):
                loss_error_record = []
                loss = 0;error = 0;loss_margin = 0;loss_regular = 0
                step = 0;one_epoch = 0;processed_graph_num = 0
                t0 = time()
                gd.files_shuffle()
                while one_epoch == 0:
                    attri_descriptors, adj_mats, label, reconstructs, one_epoch = gd.data_gen(data_source=gd.graphs_dataset_train,batch_size=FLAGS.batch_size)
                    attri_descriptors = list(zip(*attri_descriptors))
                    feed_dict = dict()
                    for idx in range(len(attri_descriptors)):
                        feed_dict[GraphNet.node_inputs[idx]] = attri_descriptors[idx]
                    feed_dict[GraphNet.labels] = label
                    feed_dict[GraphNet.adj_mats] = adj_mats
                    feed_dict[GraphNet.reconstruct_value] = reconstructs
                    feed_dict[GraphNet.if_train] = True
                    _, loss_val, loss_regular_val, loss_margin_val, error_val, lr,not_equal = sess.run(
                        [GraphNet.optimizer,
                         GraphNet.loss,
                         GraphNet.loss_regular,
                         GraphNet.loss_margin,
                         GraphNet.error,
                         GraphNet.learning_step,
                         GraphNet.not_equal],
                        feed_dict=feed_dict)
                    print('gloabal_step:{}'.format(lr))
                    loss += loss_val; loss_margin+= loss_margin_val;loss_regular+=loss_regular_val
                    error += error_val
                    step += 1
                    processed_graph_num += len(label)
                    train_step += 1
                    if step % 1 == 0:
                        if step > 0:
                            average_loss = loss / step
                            average_loss_margin = loss_margin / step
                            average_loss_regular = loss_regular / step
                            tf.logging.info('Epoch : %d : Batch size: %d Average loss for step: %d : %f' % (i, len(label), step, average_loss))
                            tf.logging.info('   Margin loss : %f Regular loss: %f' % (average_loss_margin,average_loss_regular))

                epoch_time = time() - t0
                tf.logging.info('######################### TRAIN\tEpoch: %d : loss : %f, error : %f, time : %.2f sec.  #####################' % (i, loss_margin / step,error/(1.0*processed_graph_num), epoch_time))
                loss_error_record.extend([' train : ', str(loss_margin / step), str(error / (1.0 * processed_graph_num)), '\n'])

                if i % 1 == 0:
                    test_step = train_step
                    loss = 0; error = 0
                    one_epoch = 0;step = 0
                    gd.files_shuffle_test()
                    t0 = time()
                    while one_epoch == 0:
                        attri_descriptors, adj_mats,  label, reconstructs, one_epoch = gd.data_gen(data_source=gd.graphs_dataset_test,batch_size=FLAGS.batch_size)
                        attri_descriptors = list(zip(*attri_descriptors))
                        feed_dict = dict()
                        for idx in range(len(attri_descriptors)):
                            feed_dict[GraphNet.node_inputs[idx]] = attri_descriptors[idx]
                        feed_dict[GraphNet.labels] = label
                        feed_dict[GraphNet.adj_mats] = adj_mats
                        feed_dict[GraphNet.reconstruct_value] = reconstructs
                        feed_dict[GraphNet.if_train] = False
                        loss_val,error_val = sess.run(
                            [GraphNet.loss_margin,
                             GraphNet.error,],
                            feed_dict=feed_dict)
                        loss += loss_val; error += error_val; step += 1; test_step += 1
                    test_time = time()-t0
                    tf.logging.info('#########################  TEST\tEpoch: %d : loss : %f, error : %f, time : %.2f sec.  #####################' % (i, loss / step ,error/(1.0*len(gd.graphs_dataset_test)), test_time))
                    loss_error_record.extend(['test : ', str(loss / step), str(error / (1.0 * len(gd.graphs_dataset_test))), '\n'])
                if i % 1 == 0:
                    valid_step = train_step
                    loss = 0; error = 0
                    one_epoch = 0;step = 0
                    gd.files_shuffle_valid()
                    t0 = time()
                    while one_epoch == 0:
                        attri_descriptors, adj_mats,  label, reconstructs, one_epoch = gd.data_gen(data_source=gd.graphs_dataset_valid,batch_size=FLAGS.batch_size)
                        attri_descriptors = list(zip(*attri_descriptors))
                        feed_dict = dict()
                        for idx in range(len(attri_descriptors)):
                            feed_dict[GraphNet.node_inputs[idx]] = attri_descriptors[idx]
                        feed_dict[GraphNet.labels] = label
                        feed_dict[GraphNet.adj_mats] = adj_mats
                        feed_dict[GraphNet.reconstruct_value] = reconstructs
                        feed_dict[GraphNet.if_train] = False
                        loss_val,error_val = sess.run(
                            [GraphNet.loss_margin,
                             GraphNet.error],
                            feed_dict=feed_dict)
                        loss += loss_val; error += error_val; step += 1; valid_step += 1
                    valid_time = time()-t0
                    tf.logging.info('######################### Valid\tEpoch: %d : loss : %f, error : %f, time : %.2f sec.  #####################' % (i, loss / step ,error/(1.0*len(gd.graphs_dataset_valid)), valid_time))
                    loss_error_record.extend(['valid : ', str(loss / step), str(error / (1.0 * len(gd.graphs_dataset_valid))), '\n','__________', '\n'])
                error_write_out += ' '.join(loss_error_record)

                with open(error_file_name, 'w') as f:
                    f.write(error_write_out)
                if i == FLAGS.epochs - 1:
                    error_write_out += '###############################################\n'
                    with open(error_file_name, 'w') as f:
                        f.write(error_write_out)

                

if __name__ == '__main__':
    main()
