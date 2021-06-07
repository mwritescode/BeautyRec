import argparse
from argparse import RawTextHelpFormatter

import pandas as pd
import numpy as np
from autoencoder import Autoencoder

from utils import train_test_split, plot_learning_rates, plot_losses
from matrix_factorization import MatrixFactorization
from optimizers import Adam, RMSProp

COLORS = {
    'green': '\033[32m',
    'endc': '\033[m'
}

def run_matrix_factorization():
    """ Executes the Matrix Factorization model on the ratings dataset. """
    ratings = pd.read_csv('../data/ratings2.csv', sep='\t')
    num_users = len(ratings.buyer_id.unique())
    num_items = ratings.product_id.max() + 1
    train, val = train_test_split(ratings)
    model = MatrixFactorization(train, num_users, num_items, num_latent_factors=20)
    model.train(max_iter=20, learning_rate=0.01, regularize=0.5, val=val, lr_scheduler=True)

def get_training_and_val_data():
    """ Loads the user-item matrix and splits it into train and val sets. 
    
    Returns
    -------
    train_matrix : numpy.array
        Training user-item matrix
    val_matrix : numpy.array
        Validation user-item matrix

    """
    user_matrix = pd.read_csv('../data/user_item_matrix.csv', sep='\t', header=None)
    train_matrix, val_matrix = train_test_split(user_matrix)
    train_matrix, val_matrix = train_matrix.values, val_matrix.values
    return train_matrix, val_matrix

def run_autoencoder(optimizer):
    """ Runs the autoencoder model using the specified optimizer.

    Parameters
    ----------
    optimizer : RMSProp/Adam
        Optimization algorithm to be used for parameter learning

    """
    optimizer = Adam(learning_rate=0.03) if optimizer == 'adam' else RMSProp(learning_rate=0.05)
    train_matrix, val_matrix = get_training_and_val_data()
    model = Autoencoder(input_dim=train_matrix.shape[1])
    model.print_summary()
    model.compile(optimizer)
    errors = model.fit(train_matrix, train_matrix, num_epochs=60, val_set=(val_matrix, val_matrix), early_stopping=True)
    plot_losses(errors['training'], errors['validation'])
    neuron_num = model.model.layers[0].optimizer.reference_index
    learning_rates = model.model.layers[0].optimizer.learning_rates
    plot_learning_rates(learning_rates['weights'], learning_rates['bias'], neuron_num)

def main():
    np.random.seed(42)
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--model', 
                        choices=('mf', 'autoenc'), 
                        default='mf', 
                        help='\nModel to be used for the recommender system')
    parser.add_argument('--optimizer', 
                        choices=('adam', 'rmsprop'),
                        help='Optimizer should only be specified when using --model autoenc.' +\
                        '\nSince the matrix factorization method uses sgd by default,' +\
                        '\nif it is specified along with --model mf it will be ignored.')
    args = parser.parse_args()
    if args.model == 'mf':
        run_matrix_factorization()
    else:
        if args.optimizer is None:
                print(COLORS['green'], 
                     'No optimizer was chosen for autoencoder model, using the default RMSProp.',
                     COLORS['endc'])
                args.optimizer = 'rmsprop'
        run_autoencoder(args.optimizer)
        
if __name__ == "__main__":
    main()