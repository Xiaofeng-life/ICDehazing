# -*- coding: utf-8 -*-
import argparse


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("--model", type=str)
        self.parser.add_argument("--ndf", type=int, default=64)
        self.parser.add_argument("--num_d_layers", type=int, default=4)

        self.parser.add_argument("--beta1", type=float, default=0.9)
        self.parser.add_argument("--beta2", type=float, default=0.999)
        self.parser.add_argument("--total_epoches", type=int, default=50)

        self.parser.add_argument("--dataset", type=str)
        self.parser.add_argument('--num_workers', type=int, default=8)
        self.parser.add_argument("--img_w", type=int)
        self.parser.add_argument("--img_h", type=int)
        self.parser.add_argument("--g_lr", type=float, default=0.0001)
        self.parser.add_argument("--d_lr", type=float, default=0.0001)

        self.parser.add_argument("--train_batch_size", type=int)
        self.parser.add_argument("--val_batch_size", type=int, default=1)
        self.parser.add_argument("--results_dir", type=str)

        self.parser.add_argument("--rec_loss", type=str)
        self.parser.add_argument("--prior_per", type=str)
        self.parser.add_argument("--prior_per_weight", type=float)
        self.parser.add_argument("--prior_decay", type=float)
        self.parser.add_argument("--save_w", type=str, default="True")

        self.parser.add_argument("--cyc_w", type=float, default=10)
        self.parser.add_argument("--idt_w", type=float, default=5)

    def parse(self):
        parser = self.parser.parse_args()
        return parser


if __name__ == "__main__":
    parser = Options()
    parser = parser.parse()
    print(parser)
