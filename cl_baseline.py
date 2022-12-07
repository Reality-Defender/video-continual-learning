import os
from utils import architectures
from utils.data import stable_diffusion_scenario
# from avalanche.training.supervised import EWC
from utils.strategy import EWCPretrained
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
import albumentations as A
import albumentations.pytorch as Ap
import torchvision.transforms as transforms
from torch.optim import Adam, SGD
from collections import OrderedDict
import torch


def main():
    checkpoint_path = 'weigths/method_A.pth'
    device = 'cuda'

    network_class = getattr(architectures, 'EfficientNetB4')
    net = network_class(n_classes=2, pretrained=False).eval().to(device)
    print(f'Loading model A...')
    state_tmp = torch.load(checkpoint_path, map_location='cpu')

    if 'net' not in state_tmp.keys():
        state = OrderedDict({'net': OrderedDict()})
        [state['net'].update({'model.{}'.format(k): v}) for k, v in state_tmp.items()]
    else:
        state = state_tmp
    incomp_keys = net.load_state_dict(state['net'], strict=True)
    print(incomp_keys)
    print('Model loaded!\n')

    net_normalizer = net.get_normalizer()  # pick normalizer from last network
    transform = [
        transforms.transforms.ToTensor(),
        transforms.Normalize(mean=net_normalizer.mean, std=net_normalizer.std),

    ]

    cropper = transforms.RandomCrop(128)
    train_trans = transforms.Compose(transform + [cropper])
    # optimizer = Adam(net.parameters(), lr=0.0001)
    optimizer = SGD(net.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()  # TODO: do we need this for EWC?

    scenario = stable_diffusion_scenario(train_trans)
    train_stream = scenario.train_stream
    test_stream = scenario.test_stream

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()
    tensorboard_logger = TensorboardLogger()
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=False, epoch=True, experience=True, stream=True
        ),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True),
        loggers=[interactive_logger, tensorboard_logger],
    )

    # Choose a CL strategy
    strategy = EWCPretrained(
        model=net,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=128,  # batch size
        train_epochs=15,
        eval_mb_size=128,  # eval batch size
        device=device,
        evaluator=eval_plugin,  # TODO: do we need this?
        ewc_lambda=1,
    )

    # initialize lmbda
    strategy.experience = test_stream[1]
    strategy.manual_importance(strategy)

    # train and test loop
    # for train_task in train_stream:
    strategy.eval(test_stream)
    strategy.train(train_stream)
    strategy.eval(test_stream)


if __name__ == '__main__':
    main()
