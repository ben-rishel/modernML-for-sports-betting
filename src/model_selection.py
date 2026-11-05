import os 
import logging
import pandas as pd 
import numpy as np
import json
from typing import List, Dict, Tuple, Union

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from calibra.errors import classwise_ece
from calibra.utils import bin_probabilities, get_classwise_bin_weights


logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)


class ModelSelector:
    """Select the optimal model under the given metric.

    Args:
        X_train_extended (pd.DataFrame):
            Feature data for the extended training set.
        X_test (pd.DataFrame):
            Feature data for the test set.
        y_train_extended (List[int]):
            Target data for the extended training set.
        y_test (lList[int]):
            Target data for the test set.
        optimal_features (List[str]):
            List of optimal features.
        optimal_hyperparameters (Dict[str, Dict[str, Union[str, float, int, Tuple[int]]]]):
            Dictionary containing the optimal hyperparameter values for each model.
        metric (str):
            Model evaluation metric.
        num_scoring_runs (int):
            Number of times the model is fit (using different random seeds) and scored to evaluate model performance.                 
    
    Attributes:
        X_train_extended (pd.DataFrame):
            Feature data for the extended training set.
        X_test (pd.DataFrame):
            Feature data for the test set.
        y_train_extended (List[int]):
            Target data for the extended training set.
        y_test (lList[int]):
            Target data for the test set.
        optimal_features (List[str]):
            List of optimal features (under the given metric).
        optimal_hyperparameters (Dict[str, Dict[str, Union[str, float, int, Tuple[int]]]]):
            Dictionary containing the optimal hyperparameter values for each model.
        metric (str):
            Model evaluation metric.
        num_scoring_runs (int):
            Number of times the model is fit (using different random seeds) and scored to evaluate model performance.                 
        model_scores (Dict[str, float]):
            Dictionary containing the score achieved by each model, under the given metric.
    """
    def __init__(
        self,
        X_train_extended: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train_extended: List[int],
        y_test: List[int],
        optimal_features: List[str],
        optimal_hyperparameters: Dict[str, Dict[str, Union[str, float, int, Tuple[int]]]],
        metric: str,
        num_scoring_runs: int = 1,                 
        ):
    
        self.X_train_extended = X_train_extended[optimal_features]
        self.X_test = X_test[optimal_features]
        self.y_train_extended = y_train_extended.copy()
        self.y_test = y_test.copy()
        self.optimal_hyperparameters = optimal_hyperparameters
        self.metric = metric
        self.num_scoring_runs = num_scoring_runs
        self.model_scores = {}

    def get_score(self, y_pred_proba: np.ndarray) -> float:
        """Calculate the score from averaged predicted probabilities evaluated on the test data.

        Per the paper's methodology, predictions are averaged across random seeds before scoring,
        rather than averaging per-seed scores. This matches the approach used in BettingSimulation.

        Args:
            y_pred_proba (np.ndarray):
                Averaged predicted probabilities across all scoring runs, shape (n_samples, 2).

        Returns:
            float:
                Score under the given metric.
        """
        if self.metric == "calibration":
            num_bins = 20
            binned_probabilities = bin_probabilities(y_pred_proba, self.y_test, num_bins=num_bins)
            bin_weights = get_classwise_bin_weights(binned_probabilities)[0]
            non_empty_bins = [1 for weight in bin_weights if weight > 0]
            if sum(non_empty_bins) >= 0.8 * num_bins:
                return classwise_ece(y_pred_proba, self.y_test, num_bins)
            return 1
        predicted_labels = (y_pred_proba[:, 1] >= 0.5).astype(int)
        return accuracy_score(self.y_test, predicted_labels)
        

    def score_lr(self):
        """
        Over several random seeds, fit the logistic regression model on the extended training data, with optimal features and hyperpameters.

        Average the predicted probabilities across seeds and score the averaged predictions.
        """
        prediction_array = np.zeros((len(self.y_test), 2))

        for r in range(self.num_scoring_runs):
            model = LogisticRegression(
                C=self.optimal_hyperparameters["lr"]["C"],
                solver=self.optimal_hyperparameters["lr"]["solver"],
                random_state=r,
            )
            model.fit(self.X_train_extended, self.y_train_extended)
            prediction_array += model.predict_proba(self.X_test)

        prediction_array /= self.num_scoring_runs
        self.model_scores['lr'] = self.get_score(prediction_array)

    def score_rf(self):
        """
        Over several random seeds, fit the random forest model on the extended training data, with optimal features and hyperpameters.

        Average the predicted probabilities across seeds and score the averaged predictions.
        """
        prediction_array = np.zeros((len(self.y_test), 2))

        for r in range(self.num_scoring_runs):
            model = RandomForestClassifier(
                criterion=self.optimal_hyperparameters["rf"]["criterion"],
                max_depth=self.optimal_hyperparameters["rf"]["max_depth"],
                max_features=self.optimal_hyperparameters["rf"]["max_features"],
                min_samples_leaf=self.optimal_hyperparameters["rf"]["min_samples_leaf"],
                min_samples_split=self.optimal_hyperparameters["rf"]["min_samples_split"],
                n_estimators=self.optimal_hyperparameters["rf"]["n_estimators"],
                random_state=r,
            )
            model.fit(self.X_train_extended, self.y_train_extended)
            prediction_array += model.predict_proba(self.X_test)

        prediction_array /= self.num_scoring_runs
        self.model_scores['rf'] = self.get_score(prediction_array)

    def score_svm(self):
        """
        Over several random seeds, fit the support vector machine model on the extended training data, with optimal features and hyperpameters.

        Average the predicted probabilities across seeds and score the averaged predictions.
        """
        prediction_array = np.zeros((len(self.y_test), 2))

        for r in range(self.num_scoring_runs):
            model = SVC(
                C=self.optimal_hyperparameters["svm"]["C"],
                kernel=self.optimal_hyperparameters["svm"]["kernel"],
                degree=self.optimal_hyperparameters["svm"]["degree"],
                probability=True,
                random_state=r,
            )
            model.fit(self.X_train_extended, self.y_train_extended)
            prediction_array += model.predict_proba(self.X_test)

        prediction_array /= self.num_scoring_runs
        self.model_scores['svm'] = self.get_score(prediction_array)


    def score_mlp(self):
        """
        Over several random seeds, fit the multi-layer perceptron model on the extended training data, with optimal features and hyperpameters.

        Average the predicted probabilities across seeds and score the averaged predictions.
        """
        prediction_array = np.zeros((len(self.y_test), 2))

        for r in range(self.num_scoring_runs):
            model = MLPClassifier(
                hidden_layer_sizes=self.optimal_hyperparameters["mlp"]["hidden_layer_sizes"],
                activation=self.optimal_hyperparameters["mlp"]["activation"],
                solver=self.optimal_hyperparameters["mlp"]["solver"],
                alpha=self.optimal_hyperparameters["mlp"]["alpha"],
                batch_size=self.optimal_hyperparameters["mlp"]["batch_size"],
                learning_rate=self.optimal_hyperparameters["mlp"]["learning_rate"],
                learning_rate_init=self.optimal_hyperparameters["mlp"]["learning_rate_init"],
                random_state=r,
            )
            model.fit(self.X_train_extended, self.y_train_extended)
            prediction_array += model.predict_proba(self.X_test)

        prediction_array /= self.num_scoring_runs
        self.model_scores['mlp'] = self.get_score(prediction_array)

    def extract_best_model(self) -> str:
        """Extract the best-performing model from the scores dictionary.

        Returns (str): Name of best-performing model.
        """
        scores_list = list(self.model_scores.values())
        best_score = np.max(scores_list) if self.metric == 'accuracy' else np.min(scores_list)

        for model, score in self.model_scores.items():
            if score == best_score:
                return model

    def save_model_selection_results(self, best_model: str):
        """Save the scores of each model to file.

        Args:
            best_model (str): best model under the given model evaluation metric.
        """
        best_model_dict = {'best_model': best_model}

        with open(os.path.join(".", "data", "output", "model_selection", f"{self.metric}_driven_model_scores.json"), "w") as file:
            json.dump(self.model_scores, file, indent=4)

        with open(os.path.join(".", "data", "output", "model_selection", f"{self.metric}_driven_best_model.json"), "w") as file:
            json.dump(best_model_dict, file, indent=4)


    def run_model_selection(self) -> str:
        """Score each model and select the best-performing model under the given metric.

        Returns (str): Name of best-performing model.        
        """
        self.score_lr()
        self.score_rf()
        self.score_svm()
        self.score_mlp()
        
        logging.info(f'Logistic Regression Score: {round(100 * self.model_scores["lr"], 2)}%')        
        logging.info(f'Random Forest Score: {round(100 * self.model_scores["rf"], 2)}%')        
        logging.info(f'Support Vector Machine Score: {round(100 * self.model_scores["svm"], 2)}%')                
        logging.info(f'Multi-layer Perceptron Score: {round(100 * self.model_scores["mlp"], 2)}%')
        
        best_model = self.extract_best_model()        
        self.save_model_selection_results(best_model)
        logging.info(f'Best {self.metric}-driven model: {best_model}')
        
        return best_model
