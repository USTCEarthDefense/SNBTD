import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf
import common_funcs
from common_funcs import FLOAT_TYPE
import data_loader
from sklearn.cluster import KMeans

import time
import joblib as jb

# Streaming Sparse Gaussian Tensor Decomposition
# By ZMP

import sys
#run as
print("usage : python *.py rank=3 batch_size=256 dataset=dblp")

print('start')
print( sys.argv)
#parse args
py_name = sys.argv[0]

args = sys.argv[1:]
args_dict  = {}
for arg_pair in args:
    arg, val_str = arg_pair.split( '=')
    args_dict[ arg] = val_str.strip()

arg_rank = int( args_dict['rank'])
arg_data_name = args_dict['dataset']
arg_batch_size = int( args_dict['batch_size'])



class SSGP_CLF:

    def __init__(self, init_config):

        #Model configuration parameters
        self.num_pseudo_points = init_config['num_pseudo_points']
        self.rank = init_config['rank']
        self.init_method = init_config['init_method']
        self.elem_sizes = init_config['elem_sizes']  # list, number of elements( users, items, ...)
        self.learning_rate = init_config['learning_rate']
        self.N_data_points = init_config['N_data_points']

        if 'saved_model' in init_config:
            saved_model = init_config['saved_model']
            self.init_mu_U = saved_model['mu_U']
            self.init_std_vec_U = saved_model['std_vec_U']
            self.fix_U = True
        else:
            self.fix_U = False

        self.num_mods = len( self.elem_sizes)
        self.num_factors = np.sum( self.elem_sizes)
        self.rank_psd_input = self.num_mods * self.rank # Will be different if use neural kernel
        self.tf_initializer = common_funcs.get_initializer(self.init_method, args = None)

        #Parameters
        self.PARAS_SCOPE_NAME = 'PARAS'
        with tf.variable_scope( self.PARAS_SCOPE_NAME):

            if self.fix_U:
                self.tf_mu_U = [ tf.constant( self.init_mu_U[i], dtype = FLOAT_TYPE) for i in range( self.num_mods)]
                self.tf_std_vec_U = [ tf.constant( self.init_std_vec_U[i], dtype=FLOAT_TYPE) for i in range( self.num_mods)]
            else:
                #Embeddings initialized by default initlizer
                self.tf_mu_U = [tf.Variable(np.random.randn( num_elem, self.rank) * 1.0, dtype=FLOAT_TYPE) for num_elem in self.elem_sizes]
                self.tf_std_vec_U = [ tf.Variable( np.ones( shape = [ num_elem, self.rank]) * 0.1, dtype=FLOAT_TYPE) for num_elem in self.elem_sizes] #var = diag( std * std )

            self.B_init_holder = tf.placeholder( dtype=FLOAT_TYPE, shape=[ self.num_pseudo_points, self.rank_psd_input])
            self.tf_B = tf.Variable( initial_value=self.B_init_holder)

            self.tf_post_mu_b = tf.Variable(tf.random.normal( shape = [self.num_pseudo_points, 1], dtype=FLOAT_TYPE),  dtype=FLOAT_TYPE)
            self.tf_post_Ltrig_b = tf.linalg.band_part(tf.Variable(np.eye( self.num_pseudo_points), dtype=FLOAT_TYPE), -1, 0)

            #Kernel parameters. ARD
            self.tf_log_lengthscale = tf.Variable(np.zeros(shape = [self.rank_psd_input, 1]), dtype=FLOAT_TYPE)
            self.tf_log_amp = tf.Variable(0.0, dtype=FLOAT_TYPE)

        #Place holders
        self.batch_inds = tf.placeholder(dtype=tf.int32, shape=[None, self.num_mods])
        self.batch_rates = tf.placeholder(dtype=FLOAT_TYPE, shape=[None, ])
        self.batch_uniq_fac_inds = [tf.placeholder( dtype=tf.int32,shape= [None,] ) for _ in range( self.num_mods)]

        #Old values. Be aware, Complicated logic here. Becareful to modify.
        self.mu_b_old_ori =   tf.Variable( np.zeros( shape=[self.num_pseudo_points,1]), dtype=FLOAT_TYPE)
        self.mu_b_old = tf.stop_gradient(self.mu_b_old_ori )

        self.Ltrig_b_old_ori_init_holder = tf.placeholder( dtype=FLOAT_TYPE, shape=[ self.num_pseudo_points, self.num_pseudo_points])
        self.Ltrig_b_old_ori = tf.Variable(self.Ltrig_b_old_ori_init_holder , dtype=FLOAT_TYPE)
        self.Ltrig_b_old = tf.stop_gradient( self.Ltrig_b_old_ori)

        self.Kmm_old_ori = tf.Variable( np.zeros( shape = [ self.num_pseudo_points, self.num_pseudo_points]), dtype=FLOAT_TYPE)
        self.Kmm_old = tf.stop_gradient( self.Kmm_old_ori)

        self.mu_U_old_ori = [ tf.Variable( np.zeros( shape = [ num_elem, self.rank]), dtype=FLOAT_TYPE) for num_elem in self.elem_sizes]
        self.mu_U_old = [ tf.stop_gradient(  self.mu_U_old_ori[k]) for k in range( self.num_mods)]

        self.std_vec_U_old_ori = [tf.Variable(np.zeros(shape = [num_elem, self.rank]), dtype=FLOAT_TYPE) for num_elem in self.elem_sizes]
        self.std_vec_U_old = [tf.stop_gradient( self.std_vec_U_old_ori[k]) for k in range( self.num_mods)]

        self.var_normal_params_old_ori = tf.Variable(np.array([0, 1]), dtype=FLOAT_TYPE)
        self.var_normal_params_old = tf.stop_gradient(self.var_normal_params_old_ori)


        self.assign_old_values_op = [tf.assign( self.mu_b_old_ori, self.tf_post_mu_b), tf.assign( self.Ltrig_b_old_ori, self.tf_post_Ltrig_b)]

        self.assign_old_values_op = self.assign_old_values_op + [ tf.assign( self.mu_U_old_ori[k], self.tf_mu_U[k]) for k in range( self.num_mods)] + \
            [tf.assign(self.std_vec_U_old_ori[k], self.tf_std_vec_U[k]) for k in range( self.num_mods)]

        self.sub_batch_size = self.N_data_points

        #sample posterior embeddings
        sampled_embeddings, self.batch_mean, self.batch_std_vec = common_funcs.sample_embeddings( self.tf_mu_U, self.tf_std_vec_U, self.batch_inds, return_batch_info= True)
        self.sampled_X = tf.concat( sampled_embeddings, axis=1)
        '''
        Some neural kernel transform here if using neural kernel
        '''
        self.Kmm = common_funcs.kernel_cross_tf(self.tf_B, self.tf_B, self.tf_log_amp, self.tf_log_lengthscale)# + MATRIX_JITTER * tf.eye( self.num_pseudo_points)
        self.Knm = common_funcs.kernel_cross_tf(self.sampled_X, self.tf_B, self.tf_log_amp, self.tf_log_lengthscale)
        self.assign_old_values_op.append(  tf.assign( self.Kmm_old_ori, self.Kmm))

        post_sample_f, f_std = common_funcs.sample_sparse_f( self.tf_post_mu_b, self.tf_post_Ltrig_b, self.Kmm, self.Knm, self.tf_log_amp, return_std=True) #[batch_size, 1]
        self.post_sample_f = tf.reshape(post_sample_f, shape=[-1])  # [ batch_size,]

        #MLE sample of f. Used in prediction
        self.f_mle = tf.reshape( self.Knm @ tf.linalg.solve(  self.Kmm, self.tf_post_mu_b), shape=[-1])
        self.f_std = tf.reshape( f_std, shape = [-1])

        self.data_fidelity = self.sub_batch_size * tf.reduce_mean(- tf.nn.sigmoid_cross_entropy_with_logits( labels=self.batch_rates, logits=self.post_sample_f))

        # KL U
        # Note this is biased, because uniformly sampling from rating is not equivalent to uniformly sampling from factors
        uniq_mu_U = common_funcs.get_uniq_factors(self.tf_mu_U, self.batch_uniq_fac_inds)
        uniq_std_vec_U = common_funcs.get_uniq_factors(self.tf_std_vec_U, self.batch_uniq_fac_inds)

        uniq_mu_U_old = common_funcs.get_uniq_factors( self.mu_U_old, self.batch_uniq_fac_inds)
        uniq_std_vec_U_old = common_funcs.get_uniq_factors( self.std_vec_U_old, self.batch_uniq_fac_inds)

        self.batch_KL_U = common_funcs.KL_Gaussian_std_vec_tf(tf.concat(uniq_mu_U, axis=0),
                                                              tf.concat(uniq_std_vec_U, axis=0),
                                                              tf.concat(uniq_mu_U_old, axis=0),
                                                              tf.concat(uniq_std_vec_U_old, axis=0), self.rank)
        self.KL_U = self.batch_KL_U

        # KL( q(b)|| p(b))
        self.KL_q_pb_new = common_funcs.KL_pseudo_output(self.Kmm, self.tf_post_Ltrig_b, self.tf_post_mu_b,
                                                  self.num_pseudo_points)
        # KL( q(b) || q(b)_old)
        self.KL_q_qb_old = common_funcs.KL_Gaussian_Ltrig_tf(  self.tf_post_mu_b, self.tf_post_Ltrig_b,  self.mu_b_old, self.Ltrig_b_old, self.num_pseudo_points)

        # KL ( q(b) || p(b)_old)
        self.KL_q_pb_old = common_funcs.KL_pseudo_output( self.Kmm_old, self.tf_post_Ltrig_b, self.tf_post_mu_b,self.num_pseudo_points)

        self.KL_b = self.KL_q_qb_old +  self.KL_q_pb_new  - self.KL_q_pb_old

        # Loss functions
        self.ELBO = self.data_fidelity - self.KL_b - self.KL_U

        #Session settings
        self.min_opt = tf.train.AdamOptimizer(self.learning_rate)
        self.min_step = self.min_opt.minimize(- self.ELBO)

        self.train_hist = []
        # GPU settings
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)

        #Pre-initialize pseudo input
        self.sess.run( tf.global_variables_initializer(), feed_dict={ self.B_init_holder : np.random.randn(self.num_pseudo_points, self.rank_psd_input),
                                                                      self.Ltrig_b_old_ori_init_holder : np.random.randn( self.num_pseudo_points, self.num_pseudo_points)} )
        self.is_B_initialized = False

    def _get_init_pseudo_input(self, inds):
        max_num_input_points = self.num_pseudo_points * 100
        if  len( inds) > max_num_input_points:
            arg_random = np.random.permutation( len( inds))
            inds = inds[ arg_random[ : max_num_input_points]]

        X = self.sess.run( self.sampled_X, feed_dict={ self.batch_inds : inds})

        kmeans = KMeans( n_clusters = self.num_pseudo_points, n_jobs=-1)
        _ = kmeans.fit(X)

        return kmeans.cluster_centers_

    def _fit(self, inds, rates, batch_size, num_iters_per_batch, print_every_by_iters):

        num_batches = int( len( inds / batch_size))
        self.batch_X_y_gnrt = common_funcs.DataGenerator(inds, rates, shuffle=True)

        for n_batch in range( 1, num_batches + 1):
            batch_inds, batch_rates = self.batch_X_y_gnrt.draw_next(batch_size)

            self.fit_batch( batch_inds, batch_rates, num_iters_per_batch, print_every = print_every_by_iters)

    def fit_batch(self, inds, rates, steps, print_every = 100, clean_hist = True, verbose = True ):
        start_time = time.time()
        if not self.is_B_initialized:
            # Initialized model using
            print('Re-initializing B using Kmeans')

            cluster_centers = self._get_init_pseudo_input( inds)
            self.sess.run( self.tf_B.initializer, feed_dict = { self.B_init_holder : cluster_centers})
            self.is_B_initialized = True

            # update old posteriors and hyper-parameters
            _ = self.sess.run(self.assign_old_values_op)

            init_Kmm = self.sess.run(self.Kmm)
            L = np.linalg.cholesky(init_Kmm)
            self.sess.run(self.Ltrig_b_old_ori, feed_dict={self.Ltrig_b_old_ori_init_holder: L})

            print("Re-initializing Done")


        if clean_hist:
            self.train_hist = []

        # Get unique inds
        uniq_inds = [np.unique(inds[:, k]) for k in range(self.num_mods)]

        for step in range( 1, steps + 1):
            # Get unique inds

            train_feed = {self.batch_inds: inds, self.batch_rates: rates}
            for k in range( self.num_mods):
                train_feed[ self.batch_uniq_fac_inds[k]] = uniq_inds[k]

            ELBO, sampled_f, data_fidelity,KL_U, KL_b, batch_U_mean, batch_U_std_vec, _  = self.sess.run( [
                self.ELBO, self.post_sample_f,  self.data_fidelity,self.KL_U,self.KL_b, self.batch_mean, self.batch_std_vec, self.min_step], feed_dict= train_feed, options= self.run_options)

            self.train_hist.append( ELBO)
            if step % print_every == 0 and verbose:
                print('true_rates: ', rates[:5])
                print('sampled logits: ', sampled_f[:5])
                print('sampled rates: ', (sampled_f[:5] >= 0).astype(np.float32))
                train_auc = common_funcs.metrics_auc( rates, sampled_f )
                print( '\nstep = %d, ELBO = %g, data_fidelity = %g, -KL_U = %g, -KL_b = %g, train_auc = %g' % ( step, ELBO,data_fidelity, -KL_U, -KL_b,train_auc))

        # update old posteriors and hyper-parameters
        _ = self.sess.run(self.assign_old_values_op)
        end_time = time.time()

        if verbose:
            print( 'secs_per_entry = %e' % (( end_time - start_time)/ len( inds)))

        return self

    def predict_log_llk(self, inds, y, batch_size):
        N = len( inds)

        test_llk = []
        start_idx = 0
        end_idx = start_idx + batch_size
        while( start_idx < N):
            end_idx = min( end_idx, N)
            batch_inds = inds[ start_idx : end_idx]
            batch_y = y[ start_idx : end_idx]
            test_feed = { self.batch_inds : batch_inds}

            batch_f, batch_f_std = self.sess.run( [ self.f_mle, self.f_std], feed_dict=test_feed)
            f_var = batch_f_std ** 2

            kappa = np.sqrt( 1 + np.pi * f_var/ 8)
            p = common_funcs.sigmoid( kappa * batch_f)
            llk = batch_y * np.log( p) + ( 1 - batch_y) * np.log( 1 - p)
            test_llk.append(llk)

            start_idx += batch_size
            end_idx = start_idx + batch_size

        test_llk = np.concatenate( test_llk)
        assert len( test_llk) == N, "Dims not match"
        return test_llk



    def _batch_wise_predict(self, inds, batch_size, return_logits):
        y_pred_logits = []

        N = len(inds)
        start_idx = 0
        end_idx = start_idx + batch_size
        while (start_idx < N):
            end_idx = min(end_idx, N)
            batch_inds = inds[start_idx:end_idx]
            test_feed = {self.batch_inds: batch_inds}

            batch_y = self.sess.run(self.f_mle, feed_dict=test_feed)
            y_pred_logits.append(batch_y)

            start_idx += batch_size
            end_idx = start_idx + batch_size

        y_pred_logits = np.concatenate(y_pred_logits)
        y_pred = ( y_pred_logits >= 0).astype( np.float32)

        assert len(y_pred_logits) == N, "prediction length not match"

        if return_logits:
            return y_pred, y_pred_logits
        else:
            return y_pred

    def predict(self, inds, batch_size=None, return_logits = False):

        if batch_size is not None:
            return self._batch_wise_predict(inds, batch_size, return_logits)

        else:
            test_feed = {self.batch_inds: inds}
            y_pred_logits = self.sess.run(self.f_mle, feed_dict=test_feed)
            y_pred = (y_pred_logits >= 0).astype( np.float32)

            if return_logits:
                return y_pred, y_pred_logits
            else:
                return y_pred


def main():

    assert arg_data_name in ['dblp','anime'], 'Wrong data name %s' % (arg_data_name)
    if arg_data_name == 'dblp':
        data = data_loader.load_dblp()
    elif arg_data_name == 'anime':
        data = data_loader.load_anime_binary( )
    else:
        raise NameError('wrong data set: %s' % arg_data_name)

    train_inds = data['train_X']
    train_rates = data['train_y']
    test_inds = data['test_X']
    test_rates = data['test_y']
    data_name = data['name']
    elem_sizes = data['elem_sizes']



    N_train = len( train_rates)
    N_test = len(test_rates)
    print('elem size:', elem_sizes)
    print('pseudo N train = %d, true N train = %d' % (N_train, len(train_rates)))

    print("N train = %d, N test = %d" % (N_train, N_test))
    print('mods = ', elem_sizes)

    # np.random.seed(47)
    # tf.random.set_random_seed( 47)

    #parameters settings--------------
    batch_size = arg_batch_size
    num_iters_per_batch = 100
    # init U
    init_config = {
        'elem_sizes': elem_sizes,
        'learning_rate': 1e-3,
        'init_method': 'he_normal',
        'rank': arg_rank,
        'num_pseudo_points': 128,
        'batch_size': batch_size,
        'num_iters_per_batch': num_iters_per_batch,
        'N_data_points': N_train,
        'init_batch_size' : 2048
    }
    #end parameters settings----------


    if 'USER' in os.environ:
        user_name = os.environ['USER']
    else:
        user_name = os.environ['USERNAME']

    log_file = common_funcs.init_log_file('ssgp_classifier_by_%s.txt' % user_name, data_name, init_config)
    init_config['log_file'] = log_file
    model = SSGP_CLF(init_config)


    num_batches = int(len(train_inds) / batch_size)
    print("num train = %d, num test = %d, batch_size = %d, num batches = %d" % (
    len(train_inds), len(test_inds), batch_size,  num_batches))
    batch_X_y_gnrt = common_funcs.DataGenerator(train_inds, train_rates, shuffle=True)
    batch_inds, batch_rates = batch_X_y_gnrt.draw_next(init_config['init_batch_size'])
    model.fit_batch(batch_inds, batch_rates, num_iters_per_batch, print_every=20, verbose=True)
    for n_batch in range(1, num_batches + 1):

        batch_inds, batch_rates = batch_X_y_gnrt.draw_next(batch_size)

        verbose = n_batch % int(num_batches / 20) == 0

        model.fit_batch(batch_inds, batch_rates, steps=num_iters_per_batch, verbose=verbose,print_every=50)

        if verbose:
            y_pred, logtis_pred = model.predict(test_inds, return_logits=True, batch_size = 1024)
            acc = common_funcs.metrics_accuracy(test_rates, y_pred)
            auc = common_funcs.metrics_auc(test_rates, logtis_pred)

            test_llk = model.predict_log_llk( test_inds, test_rates, batch_size=1024)
            ave_test_llk = np.average( test_llk)

            print("\nbatch = %d, progress = %4.3g, test_acc = %g, test_auc = %g, ave_llk = %g\n" % ( n_batch, n_batch / num_batches * 100, acc, auc, ave_test_llk))
            log_file.write("batch = %d, progress = %4.3g, test_acc = %g, test_auc = %g, ave_llk = %g\n" % ( n_batch, n_batch / num_batches * 100, acc, auc, ave_test_llk))
            log_file.flush()
            os.fsync(log_file.fileno())

    y_pred, logtis_pred = model.predict(test_inds, return_logits=True, batch_size = 1024)
    acc = common_funcs.metrics_accuracy(test_rates, y_pred)
    auc = common_funcs.metrics_auc(test_rates, logtis_pred)

    test_llk = model.predict_log_llk(test_inds, test_rates, batch_size=1024)
    ave_test_llk = np.average(test_llk)

    print("\nbatch = %d, progress = %4.3g, test_acc = %g, test_auc = %g, ave_llk = %g\n" % ( n_batch, n_batch / num_batches * 100, acc, auc, ave_test_llk))
    log_file.write("batch = %d, progress = %4.3g, test_acc = %g, test_auc = %g, ave_llk = %g\n" % ( n_batch, n_batch / num_batches * 100, acc, auc, ave_test_llk))
    log_file.close()


if __name__ == '__main__':
    main()

















