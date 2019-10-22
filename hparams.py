import argparse

class Hparams:
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--no-cuda', action='store_true', default=False, 
        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=72, 
        help='Random seed.')
    parser.add_argument('--epochs', type=int, default=10000, 
        help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=10, 
        help='Batch size for training / testing')
    parser.add_argument('--lr', type=float, default=0.005, 
        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, 
        help='Weight decay (L2 loss on parameters).')    
    parser.add_argument('--train_test_split', type=float, default=0.05,
        help='Train-test split ratio')
    parser.add_argument('--dropout', type=float, default=0.3, 
        help='Dropout rate (1 - keep probability).')
