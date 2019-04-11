import argparse

def settings():
    parser = argparse.ArgumentParser("GraphClassification")

    parser.add_argument("--dataset_dir", type=str, default="data_plk/ENZYMES",help="Where dataset is stored.")

    parser.add_argument("--epochs", type=int, default=3000, help="epochs")

    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")

    parser.add_argument("--iterations", type=int, default=3, help="number of iterations of dynamic routing")

    parser.add_argument("--seed", type=int, default=12345, help="Initial random seed")

    parser.add_argument('-node_emb_size', "--node_embedding_size", default=8, type=int,help="Intended subgraph embedding size to be learnt")

    parser.add_argument('-graph_emb_size', "--graph_embedding_size", default=8, type=int,help="Intended graph embedding size to be learnt")

    parser.add_argument("--learning_rate", default=0.001, type=float,help="Learning rate to optimize the loss function")

    parser.add_argument("--decay_step", default=20000, type=float,help="Learning rate decay step")

    parser.add_argument("--lambda_val", default=0.5, type=float,help="Lambda factor for margin loss")

    parser.add_argument("--noise", default=0.3, type=float, help="dropout applied in input data")

    parser.add_argument("--Attention", default=True, type=bool, help="If use Attention module")

    parser.add_argument("--reg_scale", default=0.1, type=float, help="Regualar scale (reconstruction loss)")

    parser.add_argument("--coordinate", default=False, type=bool,help="If use Location record")

    return parser.parse_args()

def get_net_structure():
    net_structure = {
        'node_emb':[2,2,2,2,2], # num of channels in each layer of GCN
        'graph_emb':[16] # num of capsules in graph embedding layer
    }
    return net_structure
