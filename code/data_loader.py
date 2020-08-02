import numpy as np
import joblib as jb
import os
import scipy.io as sio
from utils_np import NP_FLOAT_TYPE


def load_movielens_small_1m_random_split(include_timestamps = False, data_folder ='../data/preprocessing'):
    '''
    load movie lens 1m data set up 9:1 train test split
    return dict
    '''
    data_folder = os.path.abspath( data_folder)

    train_df_name = "ml_1m_train_df_6040_user_3706_movie.jb"
    test_df_name = "ml_1m_test_df_6040_user_3706_movie.jb"

    train_df_path = os.path.join( data_folder, train_df_name)
    test_df_path = os.path.join( data_folder, test_df_name)

    train_df = jb.load( train_df_path )
    test_df = jb.load( test_df_path )

    if include_timestamps:
        train_inds = train_df[ ['user_id', 'movie_id', 'time_stamp']].values.astpye( np.int32)
    else:
        train_inds = train_df[ ['user_id', 'movie_id']].values.astype( np.int32)
    train_rates = train_df['rate'].values.astype( np.float32)

    test_inds = test_df[['user_id', 'movie_id']].values.astype( np.int32)
    test_rates = test_df['rate'].values.astype( np.float32)

    ret = { "train_X": train_inds, "train_y" : train_rates, "test_X" : test_inds, "test_y" : test_rates}
    ret['num_users'] = 6040
    ret['num_items'] = 3706
    ret['name'] = 'movie_lens_1M'
    ret['elem_sizes'] = [6040, 3706]
    return ret

def load_dblp( data_folder = "../data/preprocessing"):
    data_folder = os.path.abspath( data_folder)
    data_file_name = 'dblp.jb'
    data_file_path = os.path.join( data_folder, data_file_name)
    data = jb.load( data_file_path)
    return data

def load_anime_binary( data_folder = '../data/preprocessing'):
    data_folder = os.path.abspath( data_folder)
    data_file_name = 'anime_binary_1m.jb'
    data_file_path = os.path.join( data_folder, data_file_name)
    data = jb.load( data_file_path)
    return data

def load_ibm_acc( data_folder = "../data/preprocessing"):
    data_folder = os.path.abspath( data_folder)
    data_file_name = 'ibm_acc.jb'
    data_file_path = os.path.join( data_folder, data_file_name)

    data = jb.load( data_file_path)
    return data


# Test loading functionality
if __name__ == '__main__':
    data = load_dblp()
    train_inds = data['train_X']
    train_rates = data['train_y']
    test_inds = data['test_X']
    test_rates = data['test_y']
    elem_sizes = data['elem_sizes']

    print( train_inds.shape, train_rates.shape, test_inds.shape, test_rates.shape)
    print( elem_sizes)
    print( len( train_inds) + len( test_inds))




