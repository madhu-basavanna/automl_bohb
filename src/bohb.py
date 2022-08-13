"""
===========================
Optimization using BOHB
===========================
"""

import logging
import os
import json
import time

import ConfigSpace as CS
import numpy as np
import argparse
from functools import partial
import matplotlib.pyplot as plt

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant, OrderedDict
from sklearn.model_selection import StratifiedKFold

from smac.configspace import ConfigurationSpace
from smac.facade.smac_bohb_facade import BOHB4HPO
from smac.scenario.scenario import Scenario

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
from torchvision.datasets import ImageFolder

from cnn import torchModel

max_acc = 0


def plot(file):
    fig = plt.figure(figsize=(8, 4))
    acc, prec, loss, data = [], [], [], []

    with open(file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            for d in data:
                acc.append(d["top3_accuracy"])
                prec.append(d["precision"])
                loss.append(d['loss'])

    res = len([a for a in data if isinstance(a, dict)])

    plt.plot(acc, label='Accuracy')
    plt.plot(prec, label='Precision')
    plt.plot(loss, label='Loss')
    plt.xlabel('Trials over architectures')
    plt.ylabel('Performance of architecture')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('performance_plot_0.png', bbox_inches='tight')


def get_optimizer_and_crit(cfg):
    sgd_momentum = 0
    if cfg['optimizer'] == 'SGD':
        model_optimizer = torch.optim.SGD
        if 'sgd_momentum' in cfg:
            sgd_momentum = cfg['sgd_momentum']
        else:
            sgd_momentum = 0.9
    elif cfg['optimizer'] == 'AdamW':
        model_optimizer = torch.optim.AdamW
    else:
        model_optimizer = torch.optim.Adam

    if cfg['train_criterion'] == 'mse':
        train_criterion = torch.nn.MSELoss
    else:
        train_criterion = torch.nn.CrossEntropyLoss
    return model_optimizer, train_criterion, sgd_momentum


# Target Algorithm
# The signature of the function determines what arguments are passed to it
# i.e., budget is passed to the target algorithm if it is present in the signature
def cnn_from_cfg(cfg, seed, instance, budget, run='1', args=None, use_teset_data=False,
                 save_model_str='/home/madhu/automl-ss21-final-project-madhu-basavanna/src/models',
                 data_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'micro17flower')):
    """
        Creates an instance of the torch_model and fits the given data on it.
        This is the function-call we try to optimize. Chosen values are stored in
        the configuration (cfg).

        Parameters
        ----------
        cfg: Configuration (basically a dictionary)
            configuration chosen by smac
        seed: int or RandomState
            used to initialize the rf's random generator
        instance: str
            used to represent the instance to use (just a placeholder for this example)
        budget: float
            used to set max iterations for the MLP

        Returns
        -------
        float
    """
    global max_acc
    lr = cfg['learning_rate_init'] if cfg['learning_rate_init'] else 0.001
    batch_size = cfg['batch_size'] if cfg['batch_size'] else 200

    if (args.constraint_min_precision and args.constraint_max_model_size) is None:
        constraints = OrderedDict([('model_size', 2e7), ('precision', 0.39)])
    else:
        constraints = OrderedDict(
            [('model_size', args.constraint_max_model_size), ('precision', args.constraint_min_precision)])

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img_width = 16
    img_height = 16
    data_augmentations = transforms.ToTensor()

    data = ImageFolder(os.path.join(data_dir, "train"), transform=data_augmentations)
    test_data = ImageFolder(os.path.join(data_dir, "test"), transform=data_augmentations)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    targets = data.targets

    # image size
    input_shape = (3, img_width, img_height)

    model = torchModel(cfg,
                       input_shape=input_shape,
                       num_classes=len(data.classes)).to(device)
    total_model_params = np.sum(p.numel() for p in model.parameters())
    # instantiate optimizer
    model_optimizer, train_criterion, sgd_momentum = get_optimizer_and_crit(cfg)
    if model_optimizer.__name__ == 'SGD':
        optimizer = model_optimizer(model.parameters(),
                                    lr=lr, momentum=sgd_momentum, weight_decay=0.1)
    else:
        optimizer = model_optimizer(model.parameters(),
                                    lr=lr)
    # instantiate training criterion
    train_criterion = train_criterion().to(device)

    logging.info('Generated Network:')
    summary(model, input_shape,
            device='cuda' if torch.cuda.is_available() else 'cpu')

    # returns the cross validation accuracy
    cv = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)  # to make CV splits consistent
    train_sets = []
    val_sets = []
    if use_teset_data:
        train_sets.append(data)
        val_sets.append(test_data)
    else:
        for train_idx, valid_idx in cv.split(data, data.targets):
            train_sets.append(Subset(data, train_idx))
            val_sets.append(Subset(data, valid_idx))

    num_epochs = int(np.ceil(budget))
    score_accuracy = []
    score_precision = []
    c = 0
    training_loss = []
    train_loss = 0
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, num_epochs, eta_min=lr)
    for train_set, val_set in zip(train_sets, val_sets):
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)

        if total_model_params <= constraints['model_size']:
            # Train the model
            for epoch in range(num_epochs):
                # scheduler.step()
                # lr = scheduler.get_lr()[0]
                train_loss = 0
                logging.info('#' * 50)
                logging.info('Epoch [{}/{}]'.format(epoch + 1, num_epochs))

                train_acc, train_loss = model.train_fn(optimizer, train_criterion, train_loader, device)
                val_acc, val_acc_top5, precision = model.eval_fn(validation_loader, device)

                logging.info('Train accuracy of epochs: %f', train_acc)
                logging.info('Validation accuracy of epochs: %f', val_acc)

            training_loss.append(train_loss)
            # Accuracy and precision of each subset
            val_acc, val_acc_top5, precision = model.eval_fn(validation_loader, device)
            score_accuracy.append(val_acc)
            score_precision.append(np.mean(precision))
            logging.info('Validation accuracy of subset of data %d:%f ', c + 1, val_acc)
            logging.info('Precision %d:%f', c + 1, np.mean(precision))

            if val_acc >= 0.8 and np.mean(precision) >= args.constraint_min_precision:
                max_acc = val_acc
                if save_model_str:
                    # Save the model checkpoint can be restored via "model = torch.load(save_model_str)"
                    if os.path.exists(save_model_str):
                        save_model_str += '_'.join(time.ctime()) + '_' + str(max_acc) + '.pth'
                    torch.save(model.state_dict(), save_model_str)
        else:
            score_accuracy.append(0)
            score_precision.append(0)

    # Mean accuracy of all subsets of data
    acc = np.mean(score_accuracy)
    prec = np.mean(score_precision)
    loss = np.mean(training_loss)

    logging.info('Validation accuracy : %f', acc)
    logging.info('Validation precision : %f', prec)
    logging.info('Loss : %f', loss)

    optimized_metrics = {"model_size": total_model_params,
                         "config": dict(cfg),
                         "precision": prec,
                         "top3_accuracy": acc,
                         "loss": loss}

    with open('output_all.json', 'a+') as f:
        json.dump(optimized_metrics, f)
        f.write("\n")

    # if acc >= 0.8 and prec >= args.constraint_min_precision:
    accuracy, _, precision = model.eval_fn(test_loader, device)

    test_data_metrics = {"model_size": total_model_params,
                         "config": dict(cfg),
                         "precision": np.mean(precision),
                         "top3_accuracy": accuracy}

    with open('test_acc_prec.json', 'a+') as f:
        json.dump(test_data_metrics, f)
        f.write("\n")
    logging.info('Test accuracy : %f', accuracy)
    logging.info('Test precision : %f', np.mean(precision))

    return loss, {"val_acc": acc, "val_precision": prec}


if __name__ == '__main__':
    """
    Applying NAS and HPO to the given data set using BOHB to obtain best accuracy
    """

    cmdline_parser = argparse.ArgumentParser('AutoML SS21 final project Madhu Basavanna')

    # cmdline_parser.add_argument('-e', '--epochs',
    #                             default=50,
    #                             help='Number of epochs',
    #                             type=int)
    cmdline_parser.add_argument('-m', '--model_path',
                                default='/home/madhu/automl-ss21-final-project-madhu-basavanna/src/models/',
                                help='Path to store model',
                                type=str)
    cmdline_parser.add_argument('-v', '--verbose',
                                default='INFO',
                                choices=['INFO', 'DEBUG'],
                                help='verbosity')
    cmdline_parser.add_argument('-s', "--constraint_max_model_size",
                                default=2e7,
                                help="maximal model size constraint",
                                type=int)
    cmdline_parser.add_argument('-p', "--constraint_min_precision",
                                default=0.39,
                                help='minimal constraint constraint',
                                type=float)
    cmdline_parser.add_argument('-r', "--run_id",
                                default='0',
                                help='run id ',
                                type=str)
    cmdline_parser.add_argument('-t', '--use_test_data',
                                default='False',
                                choices=['False', 'True'],
                                help='use a separate test sets instead of cross validation sets')
    cmdline_parser.add_argument('-b', '--max_budget', type=float,
                                help='Maximum budget used during the optimization.',
                                default=50)

    args, unknowns = cmdline_parser.parse_known_args()
    # HERE ARE THE CONSTRAINTS!
    # HERE ARE THE CONSTRAINTS!
    # HERE ARE THE CONSTRAINTS!
    constraint_model_size = args.constraint_max_model_size
    constraint_precision = args.constraint_min_precision

    run_id = args.run_id

    logger = logging.getLogger("MLP-example")
    logging.basicConfig(level=logging.INFO)

    # Build Configuration Space which defines all parameters and their ranges.
    # To illustrate different parameter types,
    # we use continuous, integer and categorical parameters.
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'micro17flower')
    cs = ConfigurationSpace()
    # Adding learning rate and optimizer
    learning_rate = UniformFloatHyperparameter('learning_rate', lower=1e-6, upper=1.0,
                                               default_value=2.244958736283895e-05, log=True)
    optimizer = CategoricalHyperparameter('optimizer', ['Adam', 'SGD', 'AdamW'])
    sgd_momentum = UniformFloatHyperparameter('sgd_momentum', lower=0.0, upper=0.99, default_value=0.9, log=False)

    cs.add_hyperparameters([learning_rate, optimizer, sgd_momentum])
    condition = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
    cs.add_condition(condition)

    # We can add multiple hyperparameters at once:
    num_conv_layers = UniformIntegerHyperparameter('n_conv_layers', lower=1, upper=3, default_value=3)
    conv_layer_1 = UniformIntegerHyperparameter('n_channels_conv_0', lower=4, upper=1024, default_value=457, log=True)
    conv_layer_2 = UniformIntegerHyperparameter('n_channels_conv_1', lower=4, upper=1024, default_value=511, log=True)
    conv_layer_3 = UniformIntegerHyperparameter('n_channels_conv_2', lower=4, upper=1024, default_value=38, log=True)
    kernal_size = UniformIntegerHyperparameter('kernal_size', lower=2, upper=5, default_value=5, log=True)
    dropout_rate = UniformFloatHyperparameter('dropout_rate', lower=0.0, upper=0.9, default_value=0.2, log=False)

    cs.add_hyperparameters([num_conv_layers, conv_layer_1, conv_layer_2, conv_layer_3, kernal_size, dropout_rate])
    # Add conditions to restrict the hyperparameter space
    use_con_layer_2 = CS.conditions.InCondition(conv_layer_2, num_conv_layers, [3])
    use_con_layer_1 = CS.conditions.InCondition(conv_layer_1, num_conv_layers, [2, 3])
    # Add  multiple conditions on hyperparameters at once:
    cs.add_conditions([use_con_layer_2, use_con_layer_1])

    global_avg_pooling = CategoricalHyperparameter('global_avg_pooling', [True, False], default_value=False)
    num_fc_units = UniformIntegerHyperparameter('n_fc_layers', lower=1, upper=3, default_value=2, log=True)
    fc_units_1 = UniformIntegerHyperparameter('n_channels_fc_0', lower=4, upper=2048, default_value=27, log=True)
    fc_units_2 = UniformIntegerHyperparameter('n_channels_fc_1', lower=4, upper=2048, default_value=17, log=True)
    fc_units_3 = UniformIntegerHyperparameter('n_channels_fc_2', lower=4, upper=2048, default_value=273, log=True)

    cs.add_hyperparameters([global_avg_pooling, num_fc_units, fc_units_1, fc_units_2, fc_units_3])

    use_fc_layer_2 = CS.conditions.InCondition(fc_units_2, num_fc_units, [3])
    use_fc_layer_1 = CS.conditions.InCondition(fc_units_1, num_fc_units, [2, 3])
    cs.add_conditions([use_fc_layer_2, use_fc_layer_1])

    use_BN = CategoricalHyperparameter('use_BN', [True, False], default_value=False)
    batch_size = CategoricalHyperparameter('batch_size', [128, 282, 512], default_value=282)
    cs.add_hyperparameters([batch_size, use_BN])

    # SMAC scenario object
    scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternative to runtime)
                         "wallclock-limit": 60,  # max duration to run the optimization (in seconds)
                         "cs": cs,  # configuration space
                         "deterministic": "False",
                         "limit_resources": True,  # Uses pynisher to limit memory and runtime
                         # Alternatively, you can also disable this.
                         # Then you should handle runtime and memory yourself in the TA
                         "memory_limit": 3048,  # adapt this to reasonable value for your hardware
                         })

    # max budget for hyperband can be anything. Here, we set it to maximum no. of epochs to train the MLP for
    max_iters = args.max_budget
    # intensifier parameters (Budget parameters for BOHB)
    intensifier_kwargs = {'initial_budget': 3, 'max_budget': max_iters, 'eta': 3}
    # To optimize, we pass the function to the SMAC-object
    smac = BOHB4HPO(scenario=scenario, rng=np.random.RandomState(42),
                    tae_runner=partial(cnn_from_cfg, data_dir=data_dir, run=run_id, args=args),
                    intensifier_kwargs=intensifier_kwargs,
                    # all arguments related to intensifier can be passed like this
                    initial_design_kwargs={'n_configs_x_params': 1,  # how many initial configs to sample per parameter
                                           'max_config_fracs': .2})
    def_value = smac.get_tae_runner().run(config=cs.get_default_configuration(),
                                          instance='1', budget=max_iters, seed=0)[1]
    # Start optimization
    try:  # try finally used to catch any interrupt
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    inc_value = smac.get_tae_runner().run(config=incumbent, instance='1',
                                          budget=max_iters, seed=0)[1]
    print("Optimized Value: %.4f" % inc_value)

    # store your optimal configuration to disk
    opt_config = incumbent.get_dictionary()
    with open('opt_cfg.json', 'w') as f:
        json.dump(opt_config, f)

    plot('output_acc_prec.json')
