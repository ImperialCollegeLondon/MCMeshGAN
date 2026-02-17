import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import model
import logging
import glob
import argparse
import sys
from utils import *
import torch

# Train command
# --mode train --epochs 50000
# --mode test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AgeProgression on PyTorch.', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--mode', choices=['train', 'test'], default='train')

    # train params
    parser.add_argument('--epochs', '-e', default=1, type=int)
    parser.add_argument(
        '--models-saving',
        '--ms',
        dest='models_saving',
        choices=('always', 'last', 'tail', 'never'),
        default='always',
        type=str,
        help='Model saving preference.{br}'
             '\talways: Save trained model at the end of every epoch (default){br}'
             '\tUse this option if you have a lot of free memory and you wish to experiment with the progress of your results.{br}'
             '\tlast: Save trained model only at the end of the last epoch{br}'
             '\tUse this option if you don\'t have a lot of free memory and removing large binary files is a costly operation.{br}'
             '\ttail: "Safe-last". Save trained model at the end of every epoch and remove the saved model of the previous epoch{br}'
             '\tUse this option if you don\'t have a lot of free memory and removing large binary files is a cheap operation.{br}'
             '\tnever: Don\'t save trained model{br}'
             '\tUse this option if you only wish to collect statistics and validation results.{br}'
             'All options except \'never\' will also save when interrupted by the user.'.format(br=os.linesep)
    )
    parser.add_argument('--batch-size', '--bs', dest='batch_size', default=1, type=int)
    parser.add_argument('--weight-decay', '--wd', dest='weight_decay', default=1e-5, type=float)
    parser.add_argument('--learning-rate', '--lr', dest='learning_rate', default=2e-4, type=float)
    parser.add_argument('--b1', '-b', dest='b1', default=0.5, type=float)
    parser.add_argument('--b2', '-B', dest='b2', default=0.999, type=float)
    parser.add_argument('--shouldplot', '--sp', dest='sp', default=False, type=bool)
    parser.add_argument('--watermark', '-w', action='store_true')

    # shared params
    parser.add_argument('--cpu', '-c', action='store_true', help='Run on CPU even if CUDA is available.')
    parser.add_argument('--load', '-l', required=False, default=None, help='Trained models path for pre-training or for testing')
    parser.add_argument('--input', '-i', default='/mnt/storage/home/lchen6/lchen6/data/TAAMesh/train/MeshALL_10000/', help='Training dataset path (default is {}) or testing image path'.format(default_train_results_dir()))
    parser.add_argument('--output', '-o', default='')
    parser.add_argument('-z', dest='z_channels', default=50, type=int, help='Length of Z vector')
    args = parser.parse_args()

    data_path = args.input
    data_list = glob.glob(data_path + '*.obj')
    label_path = data_path
    label_list = glob.glob(label_path + '*.obj')
    consts.NUM_Z_CHANNELS = args.z_channels
    net = model.MeshKCNGCNNet()
    print(net)

    if not args.cpu and torch.cuda.is_available():
        net.cuda()

    if args.mode == 'train':
        betas = (args.b1, args.b2) if args.load is None else None
        weight_decay = args.weight_decay if args.load is None else None
        lr = args.learning_rate if args.load is None else None
        if args.load is not None:
            net.load(args.load)
            print("Loading pre-trained models from {}".format(args.load))

        print("Data folder is {}".format(args.input))
        results_dest = args.output or default_train_results_dir()
        os.makedirs(results_dest, exist_ok=True)
        print("Results folder is {}".format(results_dest))

        with open(os.path.join(results_dest, 'session_arguments.txt'), 'w') as info_file:
            info_file.write(' '.join(sys.argv))

        log_path = os.path.join(results_dest, 'log_results.log')
        if os.path.exists(log_path):
            os.remove(log_path)
        logging.basicConfig(filename=log_path, level=logging.DEBUG)

        net.teachMesh(
            data_list=data_list,
            label_list=label_list,
            batch_size=args.batch_size,
            betas=betas,
            epochs=args.epochs,
            weight_decay=weight_decay,
            lr=lr,
            should_plot=args.sp,
            where_to_save=results_dest,
            models_saving=args.models_saving
        )

    elif args.mode == 'test':
        epoches = ['5']
        for epoch_id in epoches:
            args.load = '/mnt/storage/home/lchen6/lchen6/Remote/AgeAortaGCN/trained_models/2026_02_17/19_57/epoch'+epoch_id+'/'
            test_data_path='/mnt/storage/home/lchen6/lchen6/data/TAAMesh/test/MeshALL_10000/'
            if args.load is None:
                raise RuntimeError("Must provide path of trained models")
            net.load(path=args.load, slim=True)
            results_dest = args.output or default_test_results_dir()
            if not os.path.isdir(results_dest):
                os.makedirs(results_dest)
            data_list = glob.glob(test_data_path + '*.obj')
            label_path = data_path
            label_list = glob.glob(test_data_path + '*.obj')
            net.test_single(
                data_list=data_list,
                target=results_dest,
                epoch_id=args.load,
                watermark=args.watermark
            )
