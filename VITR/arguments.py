import argparse

def get_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='/home/datasets',
                        help='path to datasets')
    parser.add_argument('--dataset', default='F30K',
                        help='dataset')
    parser.add_argument('--logger_name', default='./runs/F30K/log',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--model_name', default='./runs/F30K/model',
                        help='Path to save the model.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--margin', default=0.185, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--learning_rate', default=.0004, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=5, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--thre_cat', default=3, type=int,
                        help='Number of epochs to cat sim_clip for training.')
    parser.add_argument('--optim', default='adam', type=str,
                        help='the optimizer')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to logger.info and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')

    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--bert_size', default=512, type=int,
                        help='B16, L14 768')
    parser.add_argument('--embed_size', default=2048, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--sim_dim', default=128, type=int,
                        help='Dimensionality of the sim embedding.')

    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--max_epochs', default=0, type=int,
                        help='Number of epochs to use max of loss.')

    return parser

