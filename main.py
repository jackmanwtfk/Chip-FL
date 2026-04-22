import time
import json
import sys
import torch.multiprocessing as mp

from architecture import *

def main(config_path="/root/autodl-tmp/Fed-Learning/Fed-Learning/config_prerout.json"):
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    with open(config_path) as f:
        config = json.load(f)

    arch = config['basic']['arch']
    train_params = config['train']

    if arch == 'cl':
        trainer = CLTrainer(**train_params)
    elif arch == 'fl':
        trainer = FedTrainer(**train_params)
    elif arch == 'sl':
        trainer = SplitTrainer(**train_params)
    elif arch == 'ampere':
        trainer = AmpereTrainer(**train_params)
    elif arch == 'prerout_fl':
        trainer = PreRoutFedTrainer(**train_params)
    elif arch == 'fededa':
        trainer = FedEDATrainer(**train_params)
    else:  # no arch
        print("Who?")
        exit()

    print("========== pretrain ==========" )
    trainer.pretrain()

    tik = time.time()
    print("========== train ==========" )
    trainer.train()

    tok = time.time()
    print("========== posttrain ==========" )
    trainer.posttrain(tok-tik)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
