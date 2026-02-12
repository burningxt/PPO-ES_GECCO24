from src.analysis.comparing_algorithms import comparing_algorithms
import argparse
import os
import torch
device = torch.device("cuda")  # Use the first CUDA device


def run():
    parser = argparse.ArgumentParser(description="Compare algorithms based on the provided flags.")
    parser.add_argument('--train', action='store_true', help='Include training in the comparison.')
    parser.add_argument('--type', type=str, default='bbob',
                        help='Specify the problem type to test (e.g., bbob, bbob-largescale)')
    parser.add_argument('--instance', type=int, default='1', help="Instance of the test problem.")
    parser.add_argument('--dim', type=int, default='40', help="Dimensionality of the test problem.")
    parser.add_argument('--test_models', action='store_true', help='Include model testing in the comparison.')
    parser.add_argument('--test_cma_es', action='store_true', help='Include pure CMA-ES testing in the comparison.')
    parser.add_argument('--test_one_fifth_es', action='store_true', help='Include pure CMA-ES testing in the comparison.')

    parser.add_argument('--experiment_name', type=str, default="", help='Include the name of your experiment.')
    parser.add_argument('--use_space', type=int, default=1, help='Weather to train with SPACE curriculum.')
    # parser.add_argument('--use_default', type=int, default=1, help='Weather to train with default behavior (just cycling through instances).')
    parser.add_argument('--num_training_instances', type=int, default=12, help='How many instances to train models on.')
    
    # parser.add_argument('--cuda_device', type=str, default='cuda',
    #                     help='Specify the CUDA device to use (e.g., cuda:0, cuda:1)')
    args = parser.parse_args()


    comparing_algorithms(need_train=args.train,
                         test_problem_type=args.type,
                         test_instance=args.instance,
                         test_dimension=args.dim,
                         need_test_models=args.test_models,
                         need_test_cma_es=args.test_cma_es,
                         need_test_one_fifth_es=args.test_one_fifth_es,
                         cuda_device=device, 
                         experiment_name=args.experiment_name,
                         use_space=args.use_space,
                        #  use_default=args.use_default,
                         num_training_instances=args.num_training_instances
                         )


if __name__ == '__main__':
    run()


