import argparse

def process_args():
    parser = argparse.ArgumentParser()

    # actions
    parser.add_argument('--train', action='store_true', help='Train a new or restored model.')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate a model.')
    parser.add_argument('--generate', action='store_true', help='Generate samples from a model.')
    parser.add_argument('--vis_styles', action='store_true', help='Visualize styles manifold.')
    parser.add_argument('--cuda', type=int, help='Which cuda device to use')
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')

    # file paths
    parser.add_argument('--restore_file', type=str, help='Path to model to restore.')
    parser.add_argument('--data_dir', default='./data/', help='Location of dataset.')
    parser.add_argument('--output_dir', default='./results/')
    parser.add_argument('--input_file', default='../DataSource/yeast_train')
    parser.add_argument('--results_file', default='results.txt',
                        help='Filename where to store settings and test results.')

    # model parameters
    parser.add_argument('--image_dims', type=tuple, default=(1, 28, 28),
                        help='Dimensions of a single datapoint (e.g. (1,28,28) for MNIST).')
    parser.add_argument('--z_dim', type=int, default=256, help='Size of the latent representation.')
    parser.add_argument('--y_dim', type=int, default=14, help='Size of the labels / output.')
    parser.add_argument('--h_dim', type=int, default=128, help='Size of the hidden layer.')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--x_dim', type=int, default=103)

    # training params
    parser.add_argument('--n_labeled', type=int, default=3000,
                        help='Number of labeled training examples in the dataset')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--n_epochs', type=int, default=1, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Classifier loss multiplier controlling generative vs. discriminative learning.')
    args =parser.parse_args()
    return args