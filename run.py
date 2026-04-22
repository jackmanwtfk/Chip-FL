import sys
import time
import json

from common import HOST_IP, HOST_PORT

from architecture import (
    FedClientTrainer, FedServerTrainer,
    SplitClientTrainer, SplitServerTrainer,
    AmpereClientTrainer, AmpereServerTrainer
)


def find_trainer(arch='fl'):
    if arch == 'fl':
        base_ctrainer = FedClientTrainer
        base_strainer = FedServerTrainer
    elif arch == 'sl':
        base_ctrainer = SplitClientTrainer
        base_strainer = SplitServerTrainer
    elif arch == 'ampere':
        base_ctrainer = AmpereClientTrainer
        base_strainer = AmpereServerTrainer
    else:
        raise NotImplementedError

    return base_ctrainer, base_strainer


def run_client(base_ctrainer, config):
    train_params = config['train']
    host_ip = config['basic']['host_ip']
    host_port = config['basic']['host_port']

    ctrainers = []
    print('========== pretrain ==========')
    for i in range(train_params['num_clients']):
        ctrainer = base_ctrainer(host_ip, host_port, **train_params)
        ctrainer.pretrain()
        ctrainers.append(ctrainer)

    print('========== train ==========')
    epochs = train_params['epochs']
    for epoch in range(epochs):
        print(f'===== epoch {epoch+1}/{epochs} =====')
        print('train:')
        for ctrainer in ctrainers:
            ctrainer.train_epoch(epoch)

        print('update:')
        for ctrainer in ctrainers:    
            ctrainer.update_state()

    print('========== posttrain ==========')
    for ctrainer in ctrainers:
        ctrainer.posttrain()


def run_server(base_strainer, config):
    train_params = config['train']
    host_ip = config['basic']['host_ip']
    host_port = config['basic']['host_port']

    strainer = base_strainer(host_ip, host_port, **train_params)
    print('========== pretrain ==========')
    strainer.pretrain()

    tik = time.time()
    print('========== train ==========')
    strainer.train()

    tok = time.time()
    print('========== posttrain ==========')
    strainer.posttrain(tok-tik)


if __name__ == '__main__':
    arch = 'ampere'
    choice = sys.argv[1]

    with open('config.json') as f:
        config = json.load(f)

    arch = config['basic']['arch']
    base_ctrainer, base_strainer = find_trainer(arch)

    if choice == 'client':
        run_client(base_ctrainer, config)
    elif choice == 'server':
        run_server(base_strainer, config)
    else:
        exit()