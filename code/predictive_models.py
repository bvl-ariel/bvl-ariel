import xgboost as xgb
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import os
import subprocess
import shutil
import re
from sys import stdout, stderr
from typing import List, Union
import time
try:
    from lightgbm import LGBMRegressor
except ImportError:
    print("let there not be ligth ...")
import random


class PredictiveModel:
    def __init__(self, model):
        self.model = model

    def fit(self, features: np.ndarray, target: np.ndarray, *args):
        raise NotImplementedError("Implement in a subclass")

    def predict(self, features: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Implement in a subclass")

    def get_params(self, deep=True):
        raise NotImplementedError("Implement in a subclass")

    def set_params(self, **params):
        raise NotImplementedError("Implement in a subclass")


class ScikitPredictiveModel(PredictiveModel):
    def __init__(self, model):
        super().__init__(model)

    def fit(self, features: np.ndarray, target: np.ndarray, *args):
        self.model.fit(features, target)
        return self

    def predict(self, features: np.ndarray):
        return self.model.predict(features)

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
            setattr(self.model, key, value)
        return self

    def get_params(self, deep=True):
        raise NotImplementedError("Implement in a subclass.")


class KNN(ScikitPredictiveModel):
    def __init__(self, n_neighbors=5, p=2):
        super().__init__(KNeighborsRegressor(n_neighbors=n_neighbors, p=p))
        self.n_neighbors = n_neighbors
        self.p = p

    def get_params(self, deep=True):
        return {"n_neighbors": self.n_neighbors, "p": self.p}


class RandomForest(ScikitPredictiveModel):
    def __init__(self, max_features=1.0, min_samples_leaf=1, n_estimators=100,
                 criterion="mse"):
        # bagging turned out to be the best for now
        super().__init__(RandomForestRegressor(n_estimators=n_estimators, criterion=criterion,
                                               max_features=max_features,
                                               min_samples_leaf=min_samples_leaf))
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.n_estimators = n_estimators

    def get_params(self, deep=True):
        return {"max_features": self.max_features, "min_samples_leaf": self.min_samples_leaf,
                "n_estimators": self.n_estimators, "criterion": self.criterion}


class GradientBoosting(PredictiveModel):
    def __init__(self, eta=0.01, max_depth=5, subsample=0.8, colsample_bytree=1.0,
                 min_child_weight=1.0, n_trees=150, num_parallel_tree=1):
        super().__init__(None)
        parameters = {
            'objective': 'reg:linear',
            'booster': 'gbtree',
            'eta': eta,
            'nthread': 1,
            'max_depth': max_depth,
            'eval_metric': 'rmse',
            'silent': 1,
            'seed': 505,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "num_parallel_tree": num_parallel_tree
        }
        self.parameters = parameters
        self.eta = eta
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.n_trees = n_trees
        self.num_parallel_tree = num_parallel_tree

    def fit(self, features: np.ndarray, target: np.ndarray, *args):
        data = xgb.DMatrix(features, label=target)
        self.model = xgb.train(self.parameters, data, self.n_trees)
        return self

    def predict(self, features: np.ndarray):
        return self.model.predict(xgb.DMatrix(features))

    def get_params(self, deep=True):
        return {'eta': self.eta, 'max_depth': self.max_depth, 'subsample': self.subsample,
                'colsample_bytree': self.colsample_bytree, 'min_child_weight': self.min_child_weight,
                "n_trees": self.n_trees, "num_parallel_tree": self.num_parallel_tree}

    def set_params(self, **params):
        for key, value in params.items():
            if key != "n_trees":
                setattr(self, key, value)
            self.parameters[key] = value
        return self


class LightGradientBoosting(ScikitPredictiveModel):
    def __init__(self, boosting_type='gbdt', num_leaves=100, max_depth=20, learning_rate=0.04, n_estimators=300,
                 min_child_samples=1, subsample=0.8, subsample_freq=0,
                 colsample_bytree=1.0):
        super().__init__(LGBMRegressor(boosting_type=boosting_type, num_leaves=num_leaves, max_depth=max_depth,
                                       learning_rate=learning_rate, n_estimators=n_estimators,
                                       min_child_samples=min_child_samples, subsample=subsample,
                                       subsample_freq=subsample_freq, colsample_bytree=colsample_bytree))

        self.boosting_type = boosting_type
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree

    def get_params(self, deep=True):
        return {"boosting_type": self.boosting_type, "num_leaves": self.num_leaves, "max_depth": self.max_depth,
                "learning_rate": self.learning_rate, "n_estimators": self.n_estimators,
                "min_child_samples": self.min_child_samples, "subsample": self.subsample,
                "subsample_freq": self.subsample_freq, "colsample_bytree": self.colsample_bytree}


class BMachine(ScikitPredictiveModel):
    def __init__(self, C=0.01, epsilon=0.1, gamma=50):  # 'scale'):
        super().__init__(SVR(C=C, epsilon=epsilon, gamma=gamma))
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma

    def get_params(self, deep=True):
        return {"C": self.C, "epsilon": self.epsilon, "gamma": self.gamma}


class BaggingSVM(ScikitPredictiveModel):
    def __init__(self, C=0.01, epsilon=0.1, gamma=50):
        super().__init__(BaggingRegressor(base_estimator=BMachine(C=C, epsilon=epsilon, gamma=gamma),
                                          n_estimators=100,
                                          max_features=1.0))
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma

    def get_params(self, deep=True):
        return {"C": self.C, "epsilon": self.epsilon, "gamma": self.gamma}


class BaggingBoosting(GradientBoosting):
    def __init__(self, n_bags=1, **boosting_params):
        super().__init__(**boosting_params)
        self.n_bags = n_bags
        self.random_seed = 123
        self.models = []

    @staticmethod
    def _generate_bag_indices(n):
        return [int(random.random() * n) for _ in range(n)]

    def fit(self, features: np.ndarray, target: np.ndarray, *args):
        self.models = []
        n, _ = features.shape
        random.seed(self.random_seed)
        for _ in range(self.n_bags):
            bag = BaggingBoosting._generate_bag_indices(n)
            features_bag = features[bag, :]
            target_bag = target[bag]
            data_bag = xgb.DMatrix(features_bag, label=target_bag)
            model = xgb.train(self.parameters, data_bag, self.n_trees)
            self.models.append(model)
        return self

    def predict(self, features: np.ndarray):
        predictions = [model.predict(xgb.DMatrix(features)) for model in self.models]
        return np.mean(predictions, axis=0)


class ClusForest(PredictiveModel):
    ENSEMBLE_METHOD = "ensemble_method"
    FEATURE_SUBSET = "feature_subset"
    MIN_LEAF_SIZE = "min_leaf_size"
    N_TREES = "n_trees"
    SPLIT_HEURISTIC = "split_heuristic"
    BAG_SELECTION = "bag_selection"
    ROSTargetSubspaceSize = "ROSTargetSubspaceSize"
    ROSAlgorithmType = "ROSAlgorithmType"

    def __init__(self, ensemble_method="RForest", feature_subset=1.0,
                 min_leaf_size=1, n_trees: Union[int, List[int]] = 250, split_heuristic="VarianceReduction",
                 bag_selection="-1", ROSTargetSubspaceSize=0.75, ROSAlgorithmType="Disabled"):
        super().__init__(None)
        self.models = []
        self.predictions = None  # type: np.ndarray
        self.features_test_id = -1  # id of the array for which predictions have been made
        self.parameters = {"General": {"Verbose": 1, "RandomSeed": 1234},
                           "Data": {},  # filled up later
                           "Attributes": {},  # filled up later
                           "Ensemble": {"EnsembleMethod": ensemble_method,
                                        "SelectRandomSubspaces": feature_subset,
                                        "Iterations": n_trees,
                                        "BagSelection": bag_selection,
                                        "Optimize": "No", "OOBestimate": "No", "EnsembleBootstrapping": "Yes",
                                        # "FeatureRanking": "None",  # "Genie3",
                                        "ROSAlgorithmType": ROSAlgorithmType,  # FixedSubspaces
                                        "ROSTargetSubspaceSize": ROSTargetSubspaceSize,
                                        "ROSVotingType": "SubspaceAveraging"},
                           "Model": {"MinimalWeight": min_leaf_size},
                           "Tree": {"SplitPosition": "Middle", "Heuristic": split_heuristic},
                           "Output": {"TrainErrors": "No", "TestErrors": "Yes", "WritePredictions": "Test"}}
        self.temp_dir_pattern = "temp_clus_forest_run{}"
        self.target_shape = tuple([])
        self.feature_ranking = []  # lines of fimp file, one after another ;)

    def _find_temp_dir(self):
        i = 0
        while os.path.exists(self.temp_dir_pattern.format(i)):
            i += 1
        return self.temp_dir_pattern.format(i)

    @staticmethod
    def _convert_to_slash(path):
        return re.sub("\\\\", "/", path)

    @staticmethod
    def _create_arff(file_name, features, targets):
        if len(targets.shape) == 1:
            expanded = [[t] for t in targets]
        else:
            expanded = targets
        with open(file_name, "w", newline='') as f:
            print("@relation professionalSlashFriends", file=f)
            for i in range(ClusForest.size(features, 1)):
                print("@attribute feature{} numeric".format(i + 1), file=f)
            for i in range(ClusForest.size(targets, 1)):
                print("@attribute target{} numeric".format(i + 1), file=f)
            print("@data", file=f)
            assert features.shape[0] == targets.shape[0]
            for xs_ys in zip(features, expanded):
                line = [str(x_y) for space in xs_ys for x_y in space]
                print(",".join(line), file=f)

    @staticmethod
    def size(array: np.ndarray, direction):
        shape = array.shape
        if direction >= len(shape):
            return 1
        else:
            return shape[direction]

    @staticmethod
    def _read_predictions(pred_file):
        """
        Loads the predictions from the predictions arff file, which is of the following form:

        @RELATION <some relations>

        @ATTRIBUTE <key attribute>           key
        @ATTRIBUTE <target 1>                numeric
        ...
        @ATTRIBUTE <target N>                numeric
        @ATTRIBUTE <model 1>-p-<target 1>    numeric
        ...
        @ATTRIBUTE <model 1>-p-<target N>    numeric
        @ATTRIBUTE <model 1>-models          string
        @ATTRIBUTE <model 2>-p-<target 1>    numeric
        ...
        ...
        @ATTRIBUTE <model K>-p-<target N>    numeric
        @ATTRIBUTE <model K>-p-models        string

        @DATA
        <comma separated table of values>

        The key attribute may not be present.

        Each row of the table corresponds to an example from the set.
        The first component is the key value.
        The next N components are the true values of the target attributes.
        The next K groups of N + 1 components consist of
        - predicted values of the i-th model (N)
        - some string that is not important (1)

        :param pred_file:
        :return: {"true_values": true_values, <model 1>: predictions1, ... , <model K>: predictionsK},
        where true_values and predictions
        are 2D arrays, whose [i, j]-th value belongs to the j-th target of the i-th instance.
        """

        attributes = []  # all lines that start with attribute ...
        nb_targets = None
        has_key = False
        true_values = "true_values"
        models = [true_values]
        results = {true_values: []}
        # instances = []
        with open(pred_file) as f:
            for _ in range(2):  # @relation, empty line
                f.readline()
            for l in f:
                line = l.strip()
                if line.lower().endswith("key"):
                    has_key = True
                elif line.lower().endswith("string"):
                    if nb_targets is None:
                        nb_targets = (len(attributes) - int(has_key)) // 2
                        attributes = attributes[1: 1 + nb_targets]
                    models.append(re.search("@ATTRIBUTE (.+)-models", line).group(1))
                    results[models[-1]] = []
                elif nb_targets is None:
                    attributes.append(re.search("@ATTRIBUTE ([^ ]+) ", line).group(1))
                elif not line:
                    break
            nb_models = len(models) - 1
            data_line = f.readline().lower()
            if not data_line.startswith("@data"):
                print(pred_file, data_line, sep="\n")
                exit(-1234)
            ok_indices = [list(range(int(has_key), int(has_key) + nb_targets))]
            ok_indices += [[(1 + m) * (1 + nb_targets) + t - int(not has_key) for t in range(nb_targets)]
                           for m in range(nb_models)]
            for l in f:
                line = l.strip().split(",")
                if not line:
                    # ignore empty lines
                    continue
                line = [[float(line[i]) for i in package] for package in ok_indices]
                assert len(line) == len(models)
                for model, values in zip(models, line):
                    results[model].append(values)
                    # if has_key:
                    #     instances.append(line[0])
        return models, {x: np.array(y) for x, y in results.items()}

    def fit(self, features: np.ndarray, target: np.ndarray, *args):
        """
        In addition to standard arguments, one must also provide the descriptive part (features) of the test set.
        :param features: np.ndarray
        :param target: np.ndarray
        :param args: 1-tuple, its first element is np.ndarray with the same number of columns as features.
        :return:
        """
        self.target_shape = target.shape
        features_test = args[0]
        temp_dir = os.path.abspath(self._find_temp_dir())
        os.makedirs(temp_dir)  # must not exist yet anyway

        n_instances_test = features_test.shape[0]
        n_features = features.shape[1]
        n_targets = ClusForest.size(target, 1)

        # temporary files
        s_file = os.path.join(temp_dir, "experiment.s")
        arff_train = os.path.join(temp_dir, "train.arff")
        arff_test = os.path.join(temp_dir, "test.arff")

        self.parameters["Data"]["File"] = arff_train
        self.parameters["Data"]["TestSet"] = arff_test

        # convert arrays to arff, create settings file

        targets_test = np.zeros((n_instances_test, n_targets))  # fake target values
        ClusForest._create_arff(arff_train, features, target)
        ClusForest._create_arff(arff_test, features_test, targets_test)

        self.parameters["Attributes"]["Descriptive"] = "1-{}".format(n_features)
        self.parameters["Attributes"]["Target"] = "{}-{}".format(n_features + 1, n_features + n_targets)
        self._params_to_s_file(s_file)
        # run the experiment
        clus = os.path.abspath("clus.jar")
        commands = "java -Xmx10G -jar {} -forest {}".format(clus, s_file).split(" ")
        print("clus start", time.asctime(), "calling", commands)
        p = subprocess.Popen(commands,
                             stdout=stdout, stderr=stderr)  # shell=True,
        p.communicate()
        print("clus end", time.asctime())
        # , stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        # save predictions (ignore everything but the last model)
        prediction_file = s_file[:s_file.rfind(".")] + ".test.pred.arff"
        models, predictions = ClusForest._read_predictions(prediction_file)
        self.models = models
        self.predictions = predictions[models[-1]]
        # we want to keep the same shape as the target has
        self._try_reshape_predictions()

        self.features_test_id = id(features_test)

        # ranking

        shutil.rmtree(temp_dir)
        return self

    def _try_reshape_predictions(self):
        if len(self.predictions.shape) == 2 and self.predictions.shape[1] == 1 and len(self.target_shape) == 1:
            print("predictions and target shapes:", self.predictions.shape, self.target_shape)
            self.predictions = np.reshape(self.predictions, self.predictions.shape[0])

    def predict(self, features: np.ndarray):
        if id(features) != self.features_test_id:
            raise ValueError("Predictions were made for different array!")
        return self.predictions

    def get_params(self, deep=True):
        return self.parameters

    def _set_ensemble_method(self, v):
        self.parameters["Ensemble"]["EnsembleMethod"] = v

    def _set_feature_subset(self, feature_subset):
        self.parameters["Ensemble"]["SelectRandomSubspaces"] = feature_subset

    def _set_min_leaf_size(self, min_leaf_size):
        self.parameters["Model"]["MinimalWeight"] = min_leaf_size

    def _set_n_trees(self, n_trees):
        self.parameters["Ensemble"]["Iterations"] = n_trees

    def _set_split_heuristic(self, split_heuristic):
        self.parameters["Tree"]["Heuristic"] = split_heuristic

    def _set_bag_selection(self, bag_selection):
        self.parameters["Ensemble"]["BagSelection"] = bag_selection

    def _set_ros_subspace_size(self, size):
        self.parameters["Ensemble"]["ROSTargetSubspaceSize"] = size

    def _set_ros_algorithm_type(self, t):
        self.parameters["Ensemble"]["ROSAlgorithmType"] = t

    def _set_param(self, key, value):
        if key == ClusForest.ENSEMBLE_METHOD:
            self._set_ensemble_method(value)
        elif key == ClusForest.FEATURE_SUBSET:
            self._set_feature_subset(value)
        elif key == ClusForest.MIN_LEAF_SIZE:
            self._set_min_leaf_size(value)
        elif key == ClusForest.N_TREES:
            self._set_n_trees(value)
        elif key == ClusForest.SPLIT_HEURISTIC:
            self._set_split_heuristic(value)
        elif key == ClusForest.BAG_SELECTION:
            self._set_bag_selection(value)
        elif key == ClusForest.ROSTargetSubspaceSize:
            self._set_ros_subspace_size(value)
        elif key == ClusForest.ROSAlgorithmType:
            self._set_ros_algorithm_type(value)
        else:
            raise KeyError("Wrong key: {}".format(key))

    def set_params(self, **params):
        for key, value in params.items():
            self._set_param(key, value)
        return self

    def _params_to_s_file(self, temp_s):
        with open(temp_s, "w") as f:
            for section in self.parameters:
                print("[{}]".format(section), file=f)
                for k, v in self.parameters[section].items():
                    print("{} = {}".format(k, v), file=f)
                print("", file=f)


class SemiMTR(ClusForest):
    TARGET_GROUPS = "target_groups"

    def __init__(self,
                 target_groups: Union[None, List[List[int]]] = None,
                 target_group: Union[None, List[int]] = None,
                 **clus_forest_parameters):
        """
        SemiMTR predictive model.
        :param target_groups: If None, STR will be performed. Example: If the groups are [[0, 2], [4], [0, 3]],
        three models are built: one for the targets with indices 0 and 2, ... At the end, the predictions
        are joined to the array, such that its [i, j]-th position gives the prediction of the target j for instance i.
        :param target_group: Subgroup of targets that has to be evaluated
        :param clus_forest_parameters: Parameters for ClusForest model.
        """
        super().__init__()
        self.target_groups = target_groups
        self.target_group = target_group
        self.children_parameters = clus_forest_parameters
        self.predictions = []
        self.features_test_id = -1

    def fit(self, features: np.ndarray, target: np.ndarray, *args):
        n_examples_test = args[0].shape[0]
        n_targets = ClusForest.size(target, 1)
        self.predictions = [[-1] * n_targets for _ in range(n_examples_test)]
        conditions = int(self.target_groups is None) + int(self.target_group is None)
        if conditions == 0:
            self.target_groups = [[i] for i in range(n_targets)]
        elif conditions == 2:
            raise ValueError("What do you want: group or groups?")
        elif self.target_group is not None:
            self.target_groups = [self.target_group]
        else:
            assert sorted([t for package in self.target_groups for t in package]) == list(range(n_targets))
        assert isinstance(self.target_groups, list)
        for group in self.target_groups:
            model = ClusForest(**self.children_parameters)
            model.fit(features, target[:, group], args[0])
            for i, predictions in enumerate(model.predict(args[0])):
                for j, y in zip(group, predictions):
                    self.predictions[i][j] = y
        self.predictions = np.array(self.predictions)
        self.features_test_id = id(args[0])
        return self

    def predict(self, features: np.ndarray):
        if id(features) != self.features_test_id:
            raise ValueError("Predictions were made for different array!")
        return self.predictions

    def get_params(self, deep=True):
        s = {k: v for k, v in self.children_parameters.items()}
        s["target_groups"] = self.target_groups
        return s

    def set_params(self, **params):
        for key, value in params.items():
            if key == SemiMTR.TARGET_GROUPS:
                self.target_groups = value
            else:
                self._set_param(key, value)  # for consistency reasons and raising the exceptions
                self.children_parameters[key] = value
        return self


class RandomForestSTRviaMTR(SemiMTR):
    def __init__(self,
                 target: int = 0,
                 target_group: Union[None, List[int]] = None,
                 **clus_forest_parameters):
        super().__init__(target_group=target_group, **clus_forest_parameters)
        self.target = target

    def fit(self, features: np.ndarray, target: np.ndarray, *args):
        super().fit(features, target, *args)
        if len(self.predictions.shape) == 2:
            self.predictions = self.predictions[:, [self.target]]
        return self
