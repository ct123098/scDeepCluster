"""
Implementation of scDeepCluster for scRNA-seq data
"""

from network import SCDeepCluster, cluster_acc

from time import time
import numpy as np
from keras.models import Model
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input, GaussianNoise, Layer, Activation
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping

from sklearn.cluster import KMeans
from sklearn import metrics

import h5py
import scanpy.api as sc
from layers import ConstantDispersionLayer, SliceLayer, ColWiseMultLayer
from loss import poisson_loss, NB, ZINB
from preprocess import read_dataset, normalize
import tensorflow as tf

from numpy.random import seed
seed(2211)
from tensorflow import set_random_seed
set_random_seed(2211)



if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--data_file', default='data.h5')
    parser.add_argument('--maxiter', default=2e4, type=int)
    parser.add_argument('--pretrain_epochs', default=400, type=int)
    parser.add_argument('--gamma', default=1, type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=0, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='results/scDeepCluster')
    parser.add_argument('--ae_weight_file', default='ae_weights.h5')
    parser.set_defaults(early_stop=True)
    parser.add_argument('--no_early_stop', dest='early_stop', action='store_false')
    parser.add_argument('--name', default='None')

    args = parser.parse_args()

    # load dataset
    optimizer1 = Adam(amsgrad=True)
    optimizer2 = 'adadelta'

    # data_mat = h5py.File(args.data_file)
    # x = np.array(data_mat['X'])
    # y = np.array(data_mat['Y'])

    # # preprocessing scRNA-seq read counts matrix
    # adata = sc.AnnData(x)
    # adata.obs['Group'] = y

    adata = sc.read_10x_mtx(args.data_file, var_names='gene_symbols', cache=True)     
    adata = read_dataset(adata,
                     transpose=False,
                     test_split=False,
                     copy=True)
    adata = normalize(adata,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)
    y = None

    input_size = adata.n_vars

    print(adata.X.shape)
    # print(y.shape)

    x_sd = adata.X.std(0)
    x_sd_median = np.median(x_sd)
    print("median of gene sd: %.5f" % x_sd_median)


    if args.update_interval == 0:  # one epoch
        args.update_interval = int(adata.X.shape[0]/args.batch_size)
    print(args)


    # Define scDeepCluster model
    scDeepCluster = SCDeepCluster(dims=[input_size, 256, 64, 32], n_clusters=args.n_clusters, noise_sd=2.5)
    plot_model(scDeepCluster.model, to_file='scDeepCluster_model.png', show_shapes=True)
    print("autocoder summary")
    scDeepCluster.autoencoder.summary()
    print("model summary")
    scDeepCluster.model.summary()

    t0 = time()

    print("Initial:")
    # scDeepCluster.eval(adata.X, adata.obs.size_factors, y)

    # Pretrain autoencoders before clustering
    if args.ae_weights is None:
        scDeepCluster.pretrain(x=[adata.X, adata.obs.size_factors], y=adata.raw.X, batch_size=args.batch_size, epochs=args.pretrain_epochs, optimizer=optimizer1, ae_file=args.ae_weight_file)

    # begin clustering, time not include pretraining part.

    scDeepCluster.fit(x_counts=adata.X, sf=adata.obs.size_factors, y=y, raw_counts=adata.raw.X, batch_size=args.batch_size, tol=args.tol, maxiter=args.maxiter,
             update_interval=args.update_interval, ae_weights=args.ae_weights, save_dir=args.save_dir, loss_weights=[args.gamma, 1], optimizer=optimizer2, early_stop=args.early_stop)

    # Show the final results
    print("Final:")
    # scDeepCluster.eval(adata.X, adata.obs.size_factors, y)
    scDeepCluster.output(adata.X, adata.obs.size_factors, adata, "../../../log/{}.h5ad".format(args.name))
    # q, _ = scDeepCluster.model.predict([], verbose=0)
    # y_pred = q.argmax(1)
    # # evaluate the clustering performance
    # acc = np.round(cluster_acc(y, y_pred), 5)
    # nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    # ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    # print('Final: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))
    print('Clustering time: %d seconds.' % int(time() - t0))
