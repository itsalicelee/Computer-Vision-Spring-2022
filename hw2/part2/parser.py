import argparse

## You could add some configs to perform other training experiments...
def arg_parse():
    parser = argparse.ArgumentParser(description='CV_hw2_part2_b07303024')
    parser.add_argument('--model_type', type=str, default='result')
    # load from ckpt
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--split_ratio', type=float, default=0.9)

    # training parameters
    parser.add_argument('--data_root', type=str, default='./p2_data/annotations/train_annos.json')
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--milestones', default=[10, 20, 25, 30]) # [15, 30]
    parser.add_argument('--num_out', type=int, default=10)

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', default=4, type=int, help="number of data loading workers")
    parser.add_argument('--lr_scheduler', type=str, default='')


    args = parser.parse_args()
    return args
