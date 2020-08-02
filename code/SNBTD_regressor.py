import numpy as np
import utils_np
from utils_np import NP_FLOAT_TYPE
from warnings import  warn
import time
import common_funcs
import data_loader
import os
from tensorflow.keras.utils import to_categorical

import sys
#run as
print("usage : python *.py rank=5 batch_size=256 dataset=mv_1m n_rff=128")

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
arg_n_rff = int( args_dict['n_rff'])
arg_batch_size = int( args_dict['batch_size'])

print('arg_rank = %d, arg_data_name = %s, arg_n_rff = %d, arg_batch_size = %d' % ( arg_rank, arg_data_name, arg_n_rff, arg_batch_size))

# Random Fourier Feature Tensor Decomposition using Combined( for frequencies) Batch CEP update scheme
class S_RFF_CEP_Regressor:
    def __init__(self, init_config):

        self.rank = init_config['rank']
        self.elem_sizes = init_config['elem_sizes']  # list, number of elements( users, items)
        self.N_data_points = init_config['N_data_points']
        self.N_rff = init_config['N_rff']  # Number of random features.
        self.N_GHQ_points = init_config['N_GHQ_points'] # Number of Gaussian-Hermite Quadrature points
        self.N_inner_iters =  init_config['N_inner_iters']
        self.damping_factor = init_config['damping_factor']
        self.y_scale = init_config['y_scale']

        self.nmod = len(self.elem_sizes)
        self.num_factors = sum(self.elem_sizes)

        ### Hyper-parameters
        self.PRCS_ZERO = 1e-6 #zero precission
        self.VAR_INF = 1.0 / self.PRCS_ZERO
        self.INIT_MEAN = 0.01
        self.INIT_STD = 0.1
        ###

        # Model Parameters
        self.mu_U = [np.random.randn(self.elem_sizes[i], self.rank,).astype( NP_FLOAT_TYPE) * self.INIT_MEAN for i in range( self.nmod)]
        self.std_vec_U = [ np.ones( shape= [ self.elem_sizes[i], self.rank], dtype=NP_FLOAT_TYPE) * self.INIT_STD for i in range( self.nmod)]
        self.std_vec_U_prior = [ np.ones( shape= [ self.elem_sizes[i], self.rank], dtype=NP_FLOAT_TYPE) * self.INIT_STD for i in range( self.nmod)]

        self.w_mu = np.random.randn( self.N_rff * 2, 1).astype(NP_FLOAT_TYPE) * self.INIT_MEAN
        self.w_prcs = np.eye( self.N_rff *2,dtype=NP_FLOAT_TYPE) * self.N_rff
        self.w_prcs_prior = np.eye( self.N_rff *2,dtype=NP_FLOAT_TYPE) * self.N_rff

        self.mu_S = np.random.randn( self.N_rff, self.nmod * self.rank).astype(NP_FLOAT_TYPE) * self.INIT_MEAN
        self.std_S = np.ones( shape = [ self.N_rff, self.nmod * self.rank], dtype=NP_FLOAT_TYPE)
        self.std_S_prior = np.ones( shape = [ self.N_rff, self.nmod * self.rank], dtype=NP_FLOAT_TYPE)

        self.tau_alpha = np.array(2e5,dtype=NP_FLOAT_TYPE)
        self.tau_beta = np.array( 1e5, dtype=NP_FLOAT_TYPE)
        ###

        self.GHQ_points, self.GHQ_weights = np.polynomial.hermite.hermgauss( self.N_GHQ_points)
        self.GHQ_points = self.GHQ_points.astype(NP_FLOAT_TYPE)
        self.GHQ_weights = self.GHQ_weights.astype(NP_FLOAT_TYPE)

        self.global_epoch = 1

    # Fit one batch of data
    def fit_batch(self, batch_inds, batch_y, n_batch = -1, global_epoch = 1, verbose = True):
        # Reset variance if multiple epoch
        if global_epoch != self.global_epoch:
            self.global_epoch = global_epoch
            self.w_prcs = self.w_prcs * 0.5
            self.std_S = self.std_S * 2.0
            for k in range( self.nmod):
                self.std_vec_U[k] *= 2.0

        batch_y = batch_y / self.y_scale
        start_time = time.time_ns()

        N = len( batch_inds) # Batch Size

        w_prior_mu = self.w_mu
        w_prior_prcs = self.w_prcs

        alpha_prior = self.tau_alpha
        beta_prior = self.tau_beta

        self.mu_S = np.fmod( self.mu_S, 2 * np.pi)
        S_prior_mu = self.mu_S
        S_prior_std = self.std_S

        U_prior_mu = self.mu_U
        U_prior_std = self.std_vec_U
        prior_cal_mu_X_n = utils_np.get_concat_embeddings(U_prior_mu, batch_inds)
        prior_cal_std_X_n =utils_np.get_concat_embeddings(U_prior_std, batch_inds)

        if verbose:
            train_pred = self.predict(batch_inds)
            train_rmse = common_funcs.metrics_rmse(batch_y, train_pred)
            print('\nn_batch = %d, n_iter = %d, train_rmse = %g' % (n_batch, -1, train_rmse))

        # N_inner_iters should be 1 in one-shot setting
        for n_iter in range( 1,self.N_inner_iters + 1):

            #mean values
            mu_tau = self.tau_alpha / self.tau_beta #[ 1,]
            mu_X = np.expand_dims( utils_np.get_concat_embeddings( self.mu_U, batch_inds), -1) #[ N, R ,1]
            mu_P =  self.mu_S @ mu_X #[N, M,1]

            cos_mu_P = np.cos( mu_P)
            sin_mu_P = np.sin( mu_P)
            mu_phi = np.concatenate( [ cos_mu_P, sin_mu_P], axis=1) #[ N, 2M, 1]

            # Update of w
            # Post dist of w of each data point
            # Paras to update
            w_prcs_new_star = w_prior_prcs + np.sum(mu_tau * mu_phi @ np.transpose(mu_phi, [0, 2, 1]), axis=0)
            w_mu_new_star = np.linalg.inv(w_prcs_new_star) @ (w_prior_prcs @w_prior_mu + np.sum(mu_tau * batch_y.reshape((-1, 1, 1)) * mu_phi, axis=0))

            # Damping update for stability
            w_prcs_new = self.damping_factor * self.w_prcs + ( 1.0 - self.damping_factor) * w_prcs_new_star
            w_mu_new = np.linalg.inv( w_prcs_new) @( self.damping_factor * self.w_prcs @ self.w_mu +
                                                     ( 1.0 - self.damping_factor) * w_prcs_new_star @ w_mu_new_star )
            # Update of tau
            tau_alpha_new_star = alpha_prior + 0.5 * N
            tau_beta_new_star = beta_prior + 0.5 * np.sum( ( batch_y - np.squeeze( np.squeeze( mu_phi, -1) @ self.w_mu, -1)) ** 2 )

            # Damping update
            tau_alpha_new = self.damping_factor * self.tau_alpha + ( 1.0 - self.damping_factor) * tau_alpha_new_star
            tau_beta_new = self.damping_factor * self.tau_beta + ( 1.0 - self.damping_factor) * tau_beta_new_star


            # Update of S.
            # Complicated logic here to fully parallelize operations
            QS = self.GHQ_points * np.sqrt(2) *np.expand_dims(S_prior_std, -1) + np.expand_dims(S_prior_mu,-1) # [M,R,J], Quadrature nodes of Smr
            DS = np.expand_dims( QS - np.expand_dims(S_prior_mu,-1),0)  # [N,M,R,J]
            DP = DS * np.expand_dims(mu_X, 1)  # [N,M,R,J]
            P_NMRJ = np.expand_dims(mu_P, -1)  # [ N,M,1,1], mu_P : [N,M,1], mu_phi: [N, 2M,1]
            QP = P_NMRJ + DP  # [ N,M,R,J]

            mu_phi_w = np.expand_dims(  np.expand_dims(  np.squeeze( mu_phi,-1) @ self.w_mu, -1), -1) #[N, 1,1,1]
            w_cos = np.expand_dims(self.w_mu[: self.N_rff], -1)  # [M,1,1]
            w_sin = np.expand_dims(self.w_mu[self.N_rff:], -1)  # [M,1,1]

            D_phi_w = np.cos(QP) * w_cos -np.expand_dims(cos_mu_P, -1) * w_cos + np.sin(QP) * w_sin \
                      - np.expand_dims(sin_mu_P, -1) * w_sin  # [N,M,R,J]
            assert D_phi_w.shape == (N, self.N_rff, self.nmod * self.rank, self.N_GHQ_points), 'Wrong shape'

            Q_phi_w = mu_phi_w + D_phi_w  # [N, M, R, J]

            log_f_S_mr = -mu_tau/ 2.0  * (  batch_y.reshape((-1, 1, 1, 1))-Q_phi_w )** 2 #[N,M,R,J]
            log_f_S_mr = np.sum( log_f_S_mr, axis=0, keepdims=False) #[M,R,J]
            max_log_f_S = np.max( log_f_S_mr,axis=-1, keepdims=True)
            f_S_mr = np.exp( log_f_S_mr - max_log_f_S)  # [M,R,J]

            Z_mr = np.sum(f_S_mr * self.GHQ_weights, axis=-1)
            first_order_S = f_S_mr * QS * self.GHQ_weights
            mu_S_star = np.sum(first_order_S, axis=-1) / Z_mr #[M,R]
            mu_2_S_star = np.sum(first_order_S * QS, axis=-1) / Z_mr #[M,R]
            var_S_star = mu_2_S_star - mu_S_star ** 2  # [M, R]

            # Set failed update to prior
            idx_failed = np.where(  var_S_star < self.PRCS_ZERO)
            var_S_star[ idx_failed] = S_prior_std[idx_failed]**2
            mu_S_star[ idx_failed] = S_prior_mu[ idx_failed]

            # Damping update
            prcs_S_new = self.damping_factor / S_prior_std**2 + ( 1 - self.damping_factor)  / var_S_star
            mu_S_new =  1.0 / prcs_S_new * ( self.damping_factor * S_prior_mu / S_prior_std **2 + ( 1 - self.damping_factor) * mu_S_star / var_S_star)
            std_S_new = np.sqrt( 1.0 / prcs_S_new)

            # Update of U
            cal_mu_X_n = prior_cal_mu_X_n
            cal_std_X_n = prior_cal_std_X_n

            QX = self.GHQ_points * np.sqrt(2) * np.expand_dims(cal_std_X_n, -1) + np.expand_dims(cal_mu_X_n,-1)  # [N,R,J]
            DX = QX - mu_X  # [N,R,J]
            QP = np.expand_dims(np.transpose(mu_P, [0, 2, 1]), 1) + np.expand_dims(np.expand_dims(self.mu_S.T, 0), 2) * np.expand_dims(DX, -1)  # [N,R,J,M]
            assert QP.shape == (N, self.nmod * self.rank, self.N_GHQ_points, self.N_rff), 'Wrong shape'

            Q_phi = np.concatenate([np.cos(QP), np.sin(QP)], axis=-1)  # [N,R,J,2M]
            log_f_U_rj = np.squeeze(mu_tau * batch_y.reshape((-1, 1, 1, 1)) * Q_phi @ self.w_mu - mu_tau / 2.0 * (Q_phi @ self.w_mu) ** 2, -1)  # [N, R,J]
            log_f_U_rj = np.transpose( log_f_U_rj, [2,0,1]) #[ J, N, R]
            
            
            std_U_new = []
            mu_U_new = []
            for m in range(self.nmod):

                idx = batch_inds[:, m]
                one_hot_idx = to_categorical(idx, self.elem_sizes[m], dtype=NP_FLOAT_TYPE).T #[L,N]
                one_hot_count = np.sum(one_hot_idx, -1)
                assert np.sum(one_hot_count) == N, "Incorrect one hot encoding"
                uniq_mask = one_hot_count != 0
                one_hot_count = one_hot_count[uniq_mask]

                uniq_one_hot = one_hot_idx[uniq_mask]
                prior_prcs = (1.0 / U_prior_std[m] ** 2)[uniq_mask]
                prior_mu = U_prior_mu[m][uniq_mask]

                sum_log_f_U_rj = np.expand_dims(uniq_one_hot, 0) @ log_f_U_rj[:, :, m * self.rank: (m + 1) * self.rank]  # [ J, L, r]
                sum_log_f_U_rj = np.transpose(sum_log_f_U_rj, [1, 2, 0])  # [Dmq,r,J]

                QX_m = np.expand_dims(uniq_one_hot, 0) @ np.transpose(QX, [2, 0, 1])[:, :, m * self.rank: (m + 1) * self.rank]  # [ J, L,r]
                QX_m = np.transpose(QX_m, [1, 2, 0])  # [Dmq,r,J]
                QX_m = QX_m / one_hot_count.reshape((-1, 1, 1))  # [Dmq,r,J]

                max_sum_log = np.max(sum_log_f_U_rj, axis=-1, keepdims=True)
                f_U_rj = np.exp(sum_log_f_U_rj - max_sum_log)
                Z_lr = np.sum(f_U_rj * self.GHQ_weights, -1)  # [ L, r]

                first_order_X = f_U_rj * QX_m * self.GHQ_weights
                mu_X_star = np.sum(first_order_X, -1) / Z_lr
                mu_2_X_star = np.sum(first_order_X * QX_m, -1) / Z_lr
                var_X_star = mu_2_X_star - mu_X_star ** 2

                var_X_star_failed = np.where(var_X_star < self.PRCS_ZERO)
                var_X_star = np.maximum(self.PRCS_ZERO, var_X_star)
                prcs_X_star = 1.0 / var_X_star

                prcs_X_star[var_X_star_failed] = prior_prcs[var_X_star_failed] #1.0  # self.PRCS_ZERO
                mu_X_star[var_X_star_failed] = prior_mu[var_X_star_failed]

                # Damping update
                X_prcs_new = self.damping_factor * prior_prcs + (1 - self.damping_factor) * prcs_X_star
                X_mu_new_uniq = 1.0 / X_prcs_new * (self.damping_factor * prior_mu * prior_prcs + ( 1.0 - self.damping_factor) * mu_X_star * prcs_X_star)
                X_std_new_uniq = np.sqrt(1.0 / X_prcs_new)

                X_mu_new = U_prior_mu[m].copy()
                X_mu_new[ uniq_mask] = X_mu_new_uniq

                X_std_new = U_prior_std[m].copy()
                X_std_new[ uniq_mask] = X_std_new_uniq

                mu_U_new.append(X_mu_new)
                std_U_new.append(X_std_new)

            # Update all paramters
            self.w_mu = w_mu_new
            self.w_prcs = w_prcs_new

            self.tau_alpha = tau_alpha_new
            self.tau_beta = tau_beta_new

            self.mu_S = mu_S_new
            self.std_S = std_S_new


            self.mu_U = mu_U_new
            self.std_vec_U = std_U_new

            # End of Iteration
            if verbose:
                train_pred = self.predict(batch_inds)
                train_rmse = common_funcs.metrics_rmse(batch_y, train_pred)
                print('\nn_batch = %d, n_iter = %d' % (n_batch,n_iter))
                print('max_w = %g, min = %g, w = ' % (np.max(w_mu_new),np.min( w_mu_new)), w_mu_new.reshape(-1)[:5])
                print('S_max = %g, min = %g, s = ' % (np.max(mu_S_new), np.min(mu_S_new)), mu_S_new[0,:5])
                X = utils_np.get_concat_embeddings( self.mu_U, batch_inds)
                print('max_X_n new = %g, min = %g' % (np.max(X), np.min(X)))
                print('mean_tau = %g, train_rmse = %g' % (self.tau_alpha / self.tau_beta, train_rmse))

        # End of Batch
        end_time = time.time_ns()
        batch_time = (end_time - start_time) / 1e9
        if verbose:
            train_pred = self.predict( batch_inds)
            train_rmse = common_funcs.metrics_rmse( batch_y, train_pred)
            print("\nend_train_batch = %d, time_per_entry = %e, end_rmse = %g" % (n_batch, batch_time / N, train_rmse))

    def predict(self, batch_inds):
        N, NMOD = batch_inds.shape
        assert  N >=1 and NMOD == self.nmod, "Wrong inds shape: %s" % str( batch_inds.shape)

        X = utils_np.get_concat_embeddings( self.mu_U, batch_inds)
        f = utils_np.cal_f_rff( X, self.mu_S, self.w_mu)
        f = f * self.y_scale

        return f

# Modify this function here to account for other data sets
def test_regressor():

    assert  arg_data_name in [ 'mv_1m', 'acc'],'Wrong data name %s' % ( arg_data_name)
    if arg_data_name =='mv_1m':
        data = data_loader.load_movielens_small_1m_random_split()
    elif arg_data_name =='acc':
        data = data_loader.load_ibm_acc()
    else:
        raise NameError('wrong data set: %s' % arg_data_name)

    train_inds = data['train_X']
    train_rates = data['train_y'].astype(NP_FLOAT_TYPE)
    test_inds = data['test_X']
    test_rates = data['test_y'].astype(NP_FLOAT_TYPE)
    data_name = data['name']
    elem_sizes = data['elem_sizes']

    #init U
    batch_size =  arg_batch_size
    N_data_points = len( train_rates)

    # Should be 1
    NUM_EPOCHS = 1

    # Model configuration.
    init_config = {
        'num_epochs': NUM_EPOCHS, # Should be 1
        'elem_sizes' :elem_sizes ,
        'rank' : arg_rank,
        'batch_size' : arg_batch_size,
        'N_data_points': N_data_points,
        'N_rff' : arg_n_rff,
        'N_GHQ_points': 5,
        'N_inner_iters': 1,
        'damping_factor' :0.10, # Larger values could stabilize update process
        'y_scale':1.0,
        'np_float_type':NP_FLOAT_TYPE
    }

    if 'USER' in os.environ:
        user_name = os.environ['USER']
    else:
        user_name = os.environ['USERNAME']

    # Logging
    log_file = common_funcs.init_log_file( 'SNBTD_regressor_by_%s.txt' % user_name,data_name, init_config)
    init_config['log_file'] = log_file

    #np.random.seed(47)
    model = S_RFF_CEP_Regressor(init_config)

    num_batches = int(len(train_inds) / batch_size)
    print("num train = %d, num test = %d, batch_size = %d, num batches = %d" % ( len(train_inds), len(test_inds), batch_size, num_batches))
    batch_X_y_gnrt = common_funcs.DataGenerator(train_inds, train_rates, shuffle=True)

    for n_epoch in range( 1 , NUM_EPOCHS + 1):
        print('epoch = %d\n' %  n_epoch)
        for n_batch in range(1, num_batches + 1):

            batch_inds, batch_rates = batch_X_y_gnrt.draw_next(batch_size)

            verbose = n_batch % int(num_batches / 20) == 0
            if verbose:
                print('\nbatch = %d' % n_batch)
            model.fit_batch(batch_inds, batch_rates,n_batch, verbose=verbose)

            if verbose:
                y_pred = model.predict(test_inds)
                rmse = common_funcs.metrics_rmse(test_rates, y_pred)

                print("batch = %8d, progress = %4.3g, rmse = %8.6g, mse = %8.6g\n" % ( n_batch, n_batch / num_batches * 100, rmse, rmse ** 2))
                log_file.write("batch = %8d, progress = %4.3g, rmse = %8.6g, mse = %8.6g\n" % ( n_batch, n_batch / num_batches * 100, rmse, rmse ** 2,))
                log_file.flush()
                os.fsync(log_file.fileno())

    y_pred = model.predict(test_inds)
    rmse = common_funcs.metrics_rmse(test_rates, y_pred)
    print("batch = %8d, progress = %4.3g, rmse = %8.6g, mse = %8.6g\n" % ( n_batch, n_batch / num_batches * 100, rmse, rmse ** 2))
    log_file.write("batch = %8d, progress = %4.3g, rmse = %8.6g, mse = %8.6g\n" % ( n_batch, n_batch / num_batches * 100, rmse, rmse ** 2,))
    log_file.close()
    ###

if __name__ == '__main__':
    test_regressor()