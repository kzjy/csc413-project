import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    parser.add_argument("--model", choices=['ssd', 'efficientdet'], help="Detection model")
    parser.add_argument("--lr", default=0.01, help="Learning rate")
    parser.add_argument("--momentum", default=0.8, help="Learning momentum")
    return args

def main(args):
    pass

if __name__ == "__main__":
    args = parse_args()
    main(args)