import argparse
from sae import *

parser = argparse.ArgumentParser()
parser.add_argument("--k",         help="Sparsity level",
                      type=int,      default=None,  required=False)
parser.add_argument("--exp",         help="Expansion factor",
                      type=int,      default=None,  required=False)

parser.add_argument("--lr",          help="Learning rate",
                    type=float,      default=1e-6,  required=False)
parser.add_argument("--epochs", help="Number of epochs to train",
                    type=int,        default=10,  required=False)
parser.add_argument("--log_interval",   help="How often to log losses",
                    type=int,        default=10,    required=False)
parser.add_argument("--val_interval",help="Validation interval",
                    type=int,        default=1,    required=False)

args = parser.parse_args()

train(args)