# CapsGNN

This repository contains an official TensorFlow implementation of Capsule Graph Neural Network ([CapsGNN](https://openreview.net/forum?id=Byl8BnRcYm)).

The implementation of dynamic routing refers to the [[code]](https://github.com/naturomics/CapsNet-Tensorflow)

### Package Version

    networkx    2.2
    numpy       1.16.2
    scipy       1.2.1
    argparse    1.1
    tensorflow  1.12.1

### Basic Usage

#### Data Preparation

1. We provide the preprocessing program to generate specific experimental data format. The default raw data format should be `.gexf` (avalaible at [[gexf Dataset]](https://drive.google.com/drive/folders/1qXx-OZlJtgRYn579aQX13ou2hutqJz41?usp=sharing)). Each line of the label file represents a graph with the format <br/>
```
    xxx.gexf label
```
To generate experimental data format:

```
    $ python3 dataset_utils/preprocessing.py --dataset_input_dir graph_gexf/ENZYMES --output_data_dir data_plk --pickle_v 3 --x_fold 10 --gen_split_file True
```    

#### Execute
1. All the hyperparameters can be set in `config.py` and the training procedure can be executed through: 

```
    $ python3 main.py --dataset_dir data_plk/ENZYMES --epochs 3000 --lambda_val 0.5
```

### Citing
If you find *CapsGNN* is useful for your research, please consider citing the following paper:

	@inproceedings{xinyi2018capsule,
       title={Capsule Graph Neural Network},
       author={Zhang Xinyi and Lihui Chen},
       booktitle={International Conference on Learning Representations},
       year={2019},
       url={https://openreview.net/forum?id=Byl8BnRcYm},
      }
      
Please send any questions you might have about the codes and/or the algorithm to <xinyi001@e.ntu.edu.sg>.
