from dotenv import load_dotenv
load_dotenv()

import json
import sys
import warnings
import argparse
import itertools
from types import SimpleNamespace

warnings.filterwarnings("ignore")

from lpnets.pipeline.exp_build import ExperimentRunner
from lpnets.pipeline.exp_train import ExperimentTrainSingle, ExperimentTrainGrid
from lpnets.datasets.pn_dataset import PNDatasetFromSSN
from lpnets.datasets.data_utils import *
from lpnets.pipeline.run_utils import *

BUILDPARAMS = ["cohort", "fold", "time_strategy", "edge_method", "agg_method", "bin", "subset"]

# ARG PARSING (for single runs)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default="build")

    # dataset parameters
    parser.add_argument("--cohort", type=str)
    parser.add_argument("--fold", type=str)
    parser.add_argument("--bin", type=str, default=None)
    parser.add_argument("--time_strategy", type=str)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--subsample", type=int, default=None)
    
    # edge building parameters
    parser.add_argument("--edge_method", type=str)
    parser.add_argument("--agg_method", default="napyPCC")
    parser.add_argument("--agg_fast", action="store_true")
    parser.add_argument("--stats", nargs="+", default=["mean","median","max","min"])

    parser.add_argument("--pipeline", default="SSNs")
    parser.add_argument("--exp_name", default=None)


    # filetring parameters
    parser.add_argument("--significant", action="store_true")
    parser.add_argument("--zscores", action="store_true")
    parser.add_argument("--zmode", default="graphwise")
    parser.add_argument("--threshold", default="p_val")

    parser.add_argument("--model", default="catboost_under")


    # model training parameters
    parser.add_argument("--features", nargs="+", default=["edge","node"])
    parser.add_argument("--hptune", action="store_true")

    # grid execution
    parser.add_argument("--build_all", type=str)
    parser.add_argument("--train_all", type=str)

    return parser.parse_args()


# EXPERIMENT RUNNER
       

class Experiment:

    def __init__(self, base_args):
        self.base_args = base_args
        self.add_defaults()

    def add_defaults(self):
        defaults = dict(
            logger=sys.stdout,
            p_value=0.05,
        )
        for k, v in defaults.items():
            setattr(self.base_args, k, v)

    def run_experiment(self, args):
        dataset = PNDatasetFromSSN(args) # check if graphs already computed
        runner = ExperimentRunner(args, dataset)
        runner.save_original_data()
        outs = runner.run()
        return outs


    def check_totune(self, args):
        totune = False

        # to tune in single run
        if args.hptune:
            totune = True

        # to tune in crossvalidation
        if args.train_all:
            args_grid = load_grid(args.train_all)

            if True in args_grid.get("hptune", []):
                totune = True

        return totune

    
    def run_single(self, config: dict):

        args = SimpleNamespace(**config)
        add_input_paths(args)

        totune = self.check_totune(args)

        stored_outputs = {}

        # FINAL
        args.stage = "final"
        stored_outputs[args.stage] = self.run_experiment(args)

        # DEV
        if totune:
            args.stage = "dev"
            stored_outputs[args.stage] = self.run_experiment(args)

        # TRAIN
        if args.mode == "train":
            if args.train_all:
                exp_trainer = ExperimentTrainGrid(args, stored_outputs)
            else:
                exp_trainer = ExperimentTrainSingle(args, stored_outputs)
            exp_trainer.run()



class ExperimentBuildSingle(Experiment):

    def __init__(self, base_args):
        super().__init__(base_args)
        self.check_complete()

    def check_complete(self):
        required = ["cohort", "fold", "time_strategy", "edge_method"]

        missing = [r for r in required if getattr(self.base_args, r) is None]

        if missing:
            raise ValueError(f"Missing required arguments: {missing}")

    def run(self):
        print_config(vars(self.base_args), params=BUILDPARAMS)
        self.run_single(vars(self.base_args))


class ExperimentBuildGrid(Experiment):

    def __init__(self, base_args):
        super().__init__(base_args)
        self.keys, self.combos = parse_grid(self.base_args.build_all)

    def run(self):
        for combo in self.combos:
            grid_config = dict(zip(self.keys, combo))

            config = vars(self.base_args).copy()
            config.update(grid_config)

            print_config(config, params=BUILDPARAMS)
            self.run_single(config)



##### MAIN ##### 

if __name__ == "__main__":

    base_args = parse_args()

    if base_args.build_all:
        exp = ExperimentBuildGrid(base_args)    
    else:
        exp = ExperimentBuildSingle(base_args)
    
    exp.run()


    




#TODO - add TS1 concat and TS2 concat
#TODO add a logger (log.txt generated for each experiment)
#now it is a mix of log and print statements
#add time so that we can track what operations are time consuming 
#we also need to understand what operations are memory-consuming and optimise (we want to run on 40k admissions)

