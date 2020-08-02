import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
from sklearn.metrics import roc_auc_score

MATRIX_JITTER = 1e-4
FLOAT_TYPE = tf.float32

def sigmoid( x):
    return 1.0 / ( 1 + np.exp( -x))

def metrics_rmse( y_true, y_pred):
    err  = y_true - y_pred
    se = err * err
    mse = np.mean( se)
    rmse = np.sqrt( mse)

    return rmse

def metrics_accuracy( y_true, y_pred):
    tp = np.sum( y_true == y_pred)
    N = len( y_true)
    acc = tp / N
    return acc

def metrics_auc( y_true, logits):
    auc = roc_auc_score(y_true,logits)
    return auc

def get_embeddings( U, ind):
    '''
    get embeddings
    :param U: list of embeddings [ nmod, num_item, rank]
    :param ind: index to embedings [ batch, nmod]
    :return: list of embeddings
    '''
    nmod = len(U)
    X = [tf.gather(U[k], ind[:, k]) for k in range(nmod)]
    return X

def get_uniq_factors( U, uniq_inds):
    num_mods = len( uniq_inds)
    return [ tf.gather( U[k], uniq_inds[k]) for k in range( num_mods)]


def sample_embeddings(U_mean, U_vec_std, inds, return_batch_info = False):
    '''
    Note this function operates by list
    '''
    num_mods = len( U_mean)

    batch_mean = [tf.gather(U_mean[k], inds[:, k]) for k in range(num_mods)]
    batch_vec_std = [tf.gather(U_vec_std[k], inds[:, k]) for k in range(num_mods)]

    gaussian_noise = [ tf.random.normal( tf.shape( batch_mean[k]),dtype=FLOAT_TYPE) for k in range( num_mods)]

    sampled_embeddings = [ batch_mean[k] + batch_vec_std[k] * gaussian_noise[k] for k in range( num_mods)]

    if return_batch_info:
        return sampled_embeddings, batch_mean, batch_vec_std
    else:
        return sampled_embeddings

def kernel_cross_tf(tf_Xm, tf_Xn, tf_log_amp, tf_log_lengthscale):
    '''
    :param tf_Xm: [m,d]
    :param tf_Xn: [n,d]
    :param tf_log_amp:
    :param tf_log_lengthscale:
    :return: K [ m,n]
    '''
    '''
    tf_Xm = tf.matmul(tf_Xm, tf.linalg.tensor_diag(1.0 / tf.exp(0.5 * tf.reshape(tf_log_lengthscale, [-1]))))
    tf_Xn = tf.matmul(tf_Xn, tf.linalg.tensor_diag(1.0 / tf.exp(0.5 * tf.reshape(tf_log_lengthscale, [-1]))))
    col_norm1 = tf.reshape(tf.reduce_sum(tf_Xm * tf_Xm, 1), [-1, 1])
    col_norm2 = tf.reshape(tf.reduce_sum(tf_Xn * tf_Xn, 1), [-1, 1])
    K = col_norm1 - 2.0 * tf.matmul(tf_Xm, tf.transpose(tf_Xn)) + tf.transpose(col_norm2)
    if return_raw_K:
        return tf.exp( -0.5 * K + tf_log_amp), K
    else:
        return tf.exp( -0.5 * K + tf_log_amp)
    '''
    lengthscale = 1.0 / tf.exp(tf_log_lengthscale)
    X = tf.expand_dims(tf_Xm, 1)
    Y = tf.expand_dims(tf_Xn, 0)
    K = (X - Y) *(X - Y) * tf.reshape( lengthscale, [-1])
    K = tf.reduce_sum(K, axis=-1)
    K = tf.exp(-0.5 * K + tf_log_amp)
    return K

def sample_sparse_f(mu_b, Ltril_b, Kmm, Knm, log_amp, jitter = 0.0, return_alpha = False, return_std = False):
    '''
    :param mu_b:
    :param Ltril_b:
    :param Kmm:
    :param Knm:
    :param log_amp:
    :param jitter:
    :param return_alpha:
    :return:
    '''
    #sample alpha
    z = tf.random.normal(mu_b.shape, dtype=FLOAT_TYPE)
    alpha = mu_b + Ltril_b @ z

    z_f = tf.random.normal( [tf.shape( Knm)[0], 1], dtype=FLOAT_TYPE)
    stdev = tf.sqrt( tf.exp( log_amp) + jitter - tf.reduce_sum( Knm * tf.transpose( tf.linalg.solve( Kmm, tf.transpose(Knm))), axis=1, keepdims=True ))
    f = Knm @ tf.linalg.solve(  Kmm,alpha)  + stdev*z_f

    if return_std:
        return f, stdev
    else:
        return f


def KL_pseudo_output( Kmm,Ltril, mu, k):
    '''
    return KL( q(alpha) || p( alpha))
    :param Kmm:
    :param Kmm_inv:
    :param Sig: Ltril@Ltril.T
    :param Ltril:
    :param mu: [ length, 1]
    :return:
    '''

    #KL = 0.5 * tf.linalg.trace( Kmm_inv @ ( Sig + mu@tf.transpose( mu)))
    KL = 0.5 * tf.linalg.trace( tf.linalg.solve( Kmm, Ltril@tf.transpose(Ltril) + mu@tf.transpose( mu) ))
    KL = KL  - k * 0.5  + 0.5 * tf.linalg.logdet( Kmm) - 0.5 * tf.reduce_sum( tf.log(  tf.linalg.diag_part( Ltril) ** 2) )
    return KL

def KL_Gaussian_Ltrig_tf( mu_0, Ltrig_0, mu_1, Ltrig_1, k):
    '''
    KL of two gaussian by Ltrig
    '''
    #KL = 0.5 * tf.linalg.trace( Kmm_inv @ ( Sig + mu@tf.transpose( mu)))

    Var_0 = Ltrig_0 @ tf.transpose( Ltrig_0) + tf.eye( k) * MATRIX_JITTER
    Var_1 = Ltrig_1 @ tf.transpose( Ltrig_1) + tf.eye( k) * MATRIX_JITTER

    KL = 0.5 * tf.linalg.trace( tf.linalg.solve( Var_1, Var_0 + ( mu_0 - mu_1)@tf.transpose( mu_0 - mu_1) ))
    KL = KL  - k * 0.5  + 0.5 * tf.reduce_sum( tf.log(  tf.linalg.diag_part( Ltrig_1) ** 2)) - 0.5 * tf.reduce_sum( tf.log(  tf.linalg.diag_part( Ltrig_0) ** 2))
    return KL

def KL_Gaussian_std_vec_tf( lst_mu_0, lst_std_vec_0, lst_mu_1, lst_std_vec_1, k, return_batch = False):
    '''
    lst_mu_0:[batch, rank]
    lst_std_vec_0:[ batch, rank]
    '''

    var_vec_0 = lst_std_vec_0 **2
    var_vec_1 = lst_std_vec_1 **2

    lst_KL = tf.reduce_sum( var_vec_0 / var_vec_1,axis=1,keepdims=True) + tf.reduce_sum( (lst_mu_0 - lst_mu_1) ** 2  / var_vec_1, axis=1,keepdims=True)
    lst_KL = 0.5 * ( lst_KL + tf.reduce_sum( tf.log( var_vec_1), axis=1, keepdims=True) - tf.reduce_sum( tf.log( var_vec_0), axis=1, keepdims = True) - k)

    if return_batch:
        return tf.reduce_sum( lst_KL), lst_KL
    else:
        return tf.reduce_sum( lst_KL)


def KL_Gaussian_scalar_std( lst_mu_0, lst_std_vec_0, lst_mu_1, lst_std_vec_1, k = 1):
    '''
    lst_mu_0:[batch, rank]
    lst_std_vec_0:[ batch, rank]
    '''

    var_vec_0 = lst_std_vec_0 **2
    var_vec_1 = lst_std_vec_1 **2

    lst_KL = var_vec_0 / var_vec_1 + (lst_mu_0 - lst_mu_1) ** 2  / var_vec_1
    lst_KL = 0.5 * ( lst_KL +tf.log( var_vec_1) - tf.log( var_vec_0) - k)

    return tf.reduce_sum( lst_KL)


def KL_log_normal( params0, params1 ):
    mu_0, var_0 = params0[0], params0[1]
    mu_1, var_1 = params1[0], params1[1]

    KL = 0.5 * tf.log( var_1 / var_0) + 0.5 / var_1 *( ( mu_0 - mu_1) **2 + var_0 - var_1)
    return KL



class DataGenerator:
    def __init__(self, X, y=None, shuffle = True):
        self.X = X
        self.y = y
        self.shuffle = shuffle
        #self.repeat = repeat

        self.num_elems = len(X)
        self.curr_idx = 0

        if self.shuffle:
            self.random_idx = np.random.permutation(self.num_elems)
        else:
            self.random_idx = np.arange( self.num_elems)


    def draw_last(self, return_idx = False):
        '''
        draw last batch sample
        :return:
        '''
        if self.y is not None:
            if return_idx:
                return self.X[self.last_arg_idx], self.y[self.last_arg_idx], self.last_arg_idx
            else:
                return self.X[self.last_arg_idx], self.y[self.last_arg_idx]
        else:
            if return_idx:
                return self.X[self.last_arg_idx], self.last_arg_idx
            else:
                return self.X[self.last_arg_idx]


    def draw_next(self, batch_size, return_idx = False):
        if batch_size > self.num_elems:
            raise NameError("Illegal batch size")

        if batch_size + self.curr_idx > self.num_elems:
            # shuffle

            if self.shuffle:
                self.random_idx = np.random.permutation(self.num_elems)
            else:
                self.random_idx = np.arange(self.num_elems)
            self.curr_idx = 0

        arg_idx = self.random_idx[self.curr_idx: self.curr_idx + batch_size]
        self.last_arg_idx = arg_idx

        self.curr_idx += batch_size

        if self.y is not None:
            if return_idx:
                return self.X[arg_idx], self.y[arg_idx], arg_idx
            else:
                return self.X[arg_idx], self.y[arg_idx]
        else:
            if return_idx:
                return self.X[arg_idx], arg_idx
            else:
                return self.X[arg_idx]

def get_initializer( name, args = None):
    valid_names = ['he_normal', 'glorot_normal']
    if name not in valid_names:
        raise NameError('invalid initilizer: %s' % name)

    if name == 'he_normal':
        return keras.initializers.he_normal( )

    if name == 'glorot_normal':
        return keras.initializers.glorot_normal( )

def init_log_file( log_file_path, data_name,  model_config, mode = 'a'):
    log_file = open( log_file_path, mode = mode)

    date = time.asctime()

    log_file.write('\n\n' + date + '\n')
    log_file.write("data set = %s\n" %data_name)
    log_file.write('model config:\n%s\n' %( str( model_config)))

    return log_file