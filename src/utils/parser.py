
import argparse

def args_prs_train():
    # Initialize parser
    parser = argparse.ArgumentParser(description='Input samples for the training process.')

    parser.add_argument('--npz_file', type=str, required=True,
                        help='Compressed input numpy file containing: images, poses, and focal info')
    parser.add_argument('--N_samples', type=int, required=False, default=64,
                        help='Number of samples in the 3D space (default: 64)')
    parser.add_argument('--N_iter', type=int, required=False, default=1000,
                        help='Number of training iterations (default: 1000)')
    parser.add_argument('--save_pts', type=int, required=False, default=100,
                        help='Save model every N iterations (default: 100)')
    parser.add_argument('--depth', type=int, required=False, default=8,
                        help='Model depth (default: 8)')
    parser.add_argument('--width', type=int, required=False, default=256,
                        help='Model width (default: 256)')
    parser.add_argument('--pos_enc', type=int, required=False, default=6,
                        help='Positional encodings dimension (default: 6)')

    # Parse arguments
    args = parser.parse_args()

    # Accessing argument values
    depth = args.depth
    width = args.width
    pos_enc_l = args.pos_enc
    N_samples = args.N_samples
    N_iters = args.N_iter
    save_i = args.save_pts
    data_path = args.npz_file
    i_plot = 2

    return depth, width, pos_enc_l, N_samples, N_iters, save_i, data_path, i_plot

def args_prs_load():

    parser = argparse.ArgumentParser(description='Input samples for the training process.')
    parser.add_argument('--npz_file', type=str, required=True,
                            help='compressed numpy where you have: images, poses and focal info')
    parser.add_argument('--model_path', type=str, required=True,
                            help='compressed numpy where you have: images, poses and focal info')

    args = parser.parse_args()
    # data prep 
    data_path = args.npz_file
    model_path = args.model_path

    return data_path, model_path