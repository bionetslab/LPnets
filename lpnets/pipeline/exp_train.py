from types import SimpleNamespace
from lpnets.ml_training.ml_trainer import MLTrainer
from lpnets.pipeline.run_utils import parse_grid
from lpnets.pipeline.run_utils import print_config

TRAINPARAMS = ["significant", "zscores", "zmode", "threshold", "features", "hptune", "model"]


class ExperimentTrain:

    def __init__(self, args, stored_outputs):
        self.args = args
        self.stored_outputs = stored_outputs

    def train_single(self, args_run):

        ml_trainer = MLTrainer(args_run, self.stored_outputs)
        ml_trainer.train_comb()



class ExperimentTrainSingle(ExperimentTrain):

    def run(self):
        required = ["significant", "zscores", "zmode", "threshold"]
        missing = [r for r in required if getattr(self.args, r) is None]

        if missing:
            raise ValueError(f"Missing required arguments: {missing}")

        print_config(vars(self.args), params=TRAINPARAMS)
        self.train_single(self.args)



class ExperimentTrainGrid(ExperimentTrain):

    def __init__(self, args, stored_outputs):
        super().__init__(args, stored_outputs)
        self.keys, self.combos = parse_grid(args.train_all)

    def run(self):

        for combo in self.combos:

            train_config = dict(zip(self.keys, combo))

            args_run = vars(self.args).copy()
            args_run.update(train_config)

            # constraints
            if not args_run.get("significant", False):
                args_run["threshold"] = "NA"

            if not args_run.get("significant", False) and not args_run.get("zscores", False):
                args_run["zmode"] = "NA"

            args_run = SimpleNamespace(**args_run)

            # IMPORTANT: pass args
            print_config(vars(args_run), params=TRAINPARAMS)
            self.train_single(args_run)