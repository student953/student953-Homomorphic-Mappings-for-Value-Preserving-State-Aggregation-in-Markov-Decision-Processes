import os
from mdptoolbox.mdp import RelativeValueIteration, ValueIteration, PolicyIteration
from typing import List, Tuple, Optional
from utils.generic_model import GenericModel
from utils.generic_solver import GenericSolver
from utils.pickle_my import pickle_save, pickle_load
from utils.data_management import build_solver_list
from utils.data_management import build_model_list
from utils.calculus import norminf
from utils.exact_value_function import get_exact_value
import numpy as np
import pandas as pd
import time


class Experience:
    def __init__(
            self,
            discount: float = 0.999,  # Discount factor
            n_exp: int = 10,  # Number of experience with same parameters to measure solving time
            solver_path: str = "solvers_ori",
            model_path: str = "models",
            result_path: str = "results",
            model_saved_path: str = "saved_models",
            exact_value_functions_path: str = "saved_value_functions",
            method: str = "discounted",
            epsilon_exact_value: float = 1e-5,
            verbose: bool = True,
            time_limit: Optional[float] = None,
    ) -> None:
        # Parsing arguments
        self.discount = discount
        self.n_exp = n_exp
        self.solver_path = os.path.join(os.getcwd(), solver_path)
        self.model_path = os.path.join(os.getcwd(), model_path)
        self.result_path = os.path.join(os.getcwd(), result_path)
        self.model_saved_path = os.path.join(os.getcwd(), model_saved_path)
        self.exact_value_functions_path = os.path.join(
            os.getcwd(), exact_value_functions_path
        )
        self.method = method
        self.epsilon_exact_value = epsilon_exact_value
        self.verbose = verbose

        # Creating variables
        self.model_list = []
        self.solver_list = []
        self.current_experience_index: int = 0
        self.avg_runtimes_dictionary = {}
        self.avg_runtimes_dataframe: pd.DataFrame
        self.std_runtimes_dictionary = {}
        self.std_runtimes_dataframe: pd.DataFrame
        self.avg_precision_dictionary = {}
        self.avg_precision_dataframe: pd.DataFrame
        self.std_precision_dictionary = {}
        self.std_precision_dataframe: pd.DataFrame

        # Try folder creation
        self._try_folder_creation()

        # Current experience index
        self.def_rng = np.random.default_rng(int(time.time()))

        self.current_experience_index = 0  # self.def_rng.integers(
        # 0, 1000
        # )  # self._get_current_experience_index()

        # Build model list
        self._get_model_list()
        assert self.model_list, "There should be at least one model selected."

        # Build solvers list
        self._get_solver_list()
        assert self.solver_list, "There should be at least one solvers selected."

        if self.verbose:
            self.print_model_solver_list()

        # self.exp_name = input("Enter experience name \n")
        self.exp_name = "ICAPS_agg"

    def run(self):
        """
        Run the full experiment.
        """
        for model in self.model_list:
            model: GenericModel
            model.create_model()
            model.name += "_{}".format(self.discount)
            for solver in self.solver_list:
                # Solver initialization
                solver: GenericSolver = solver(model, self.discount)
                solver.__init__(model, self.discount)

                # Defining names
                if self.verbose:
                    print("{} {}".format(model.name, solver.name))

                # Run experiment and save
                (
                    avg_runtime,
                    std_runtime,
                    avg_precision,
                    std_precision,
                ) = self._model_specific_solver_specific_experience(model, solver)
                self.avg_runtimes_dictionary[(model.name, solver.name)] = avg_runtime
                self.std_runtimes_dictionary[(model.name, solver.name)] = std_runtime
                self.avg_precision_dictionary[(model.name, solver.name)] = avg_precision
                self.std_precision_dictionary[(model.name, solver.name)] = std_precision
                self._save_current_excel_files()

            model.lighten_model()

    def _get_current_experience_index(self):
        """
        Return the current experience index
        to save the result into a specific file.
        """
        if not os.listdir(self.result_path):
            return 1
        else:
            return (
                    max(
                        int(strings.split("_")[-1][:-5])
                        for strings in os.listdir(self.result_path)
                    )
                    + 1
            )

    def _try_folder_creation(self):
        """
        Folder creation to save results,
        models and value functions.
        """
        for folder_path in [
            self.result_path,
            self.model_saved_path,
            self.exact_value_functions_path,
        ]:
            try:
                os.mkdir(folder_path)
            except FileExistsError:
                pass

    def _compute_exact_value_function_or_save_it(
            self, model: GenericModel
    ) -> np.ndarray:
        """
        For a given model, compute the exact value function
        if it has not been saved yet.
        """
        return get_exact_value(model, self.method, self.discount)

    def _get_model_list(self):
        """
        Build the list of models used in the benchmark.
        """
        self.model_list = build_model_list(self.model_path)

    def _get_solver_list(self):
        """
        Build the list of solvers_ori used in the benchmark.
        """
        self.solver_list = build_solver_list()

    def _model_specific_solver_specific_experience(
            self, model: GenericModel, solver: GenericSolver
    ) -> Tuple[float]:
        """
        For the given model and solvers, returns :
        - the average precision
        - the average runtime
        - the standard deviation of precision
        - the standard deviation of runtime
        """
        assert hasattr(model, "transition_matrix") and hasattr(
            model, "reward_matrix"
        ), "Model should have been created using model.create_model()."

        runtimes_experience, precision_experience = [], []

        # As transition and reward have been deleted, we load it back.
        # exact_value_function = self._compute_exact_value_function_or_save_it(model)

        for _ in range(self.n_exp):
            solver.__init__(model, self.discount)
            solver.run()
            runtimes_experience.append(solver.runtime)
            value_final = solver.value * 1.0

        return (np.mean(runtimes_experience),
                np.std(runtimes_experience),
                np.mean(precision_experience),
                np.std(precision_experience),)

    @staticmethod
    def _convert_dictionary_to_pandas_dataframe(dictionary: dict):
        """
        Convert a dictionary like {(row, col) : value}
        into a Pandas Dataframe.
        """
        rows, columns = zip(*dictionary.keys())
        result_dataframe = pd.DataFrame(
            None, index=list(set(rows)), columns=list(set(columns)), dtype=float
        )
        for row, col in dictionary.keys():
            result_dataframe.at[row, col] = dictionary[(row, col)]

        result_dataframe["state_dim"] = [
            int(model_name.split("_")[0])
            for model_name in result_dataframe.index.to_list()
        ]

        # Sort the DataFrame based on the 'IntPart' column
        result_dataframe = result_dataframe.sort_values(by="state_dim")

        # Drop the 'IntPart' column if you don't need it in the final DataFrame
        result_dataframe = result_dataframe.drop(columns="state_dim")

        return result_dataframe

    def _save_current_excel_files(self):
        """
        Given the current result in the dictionaries,
        update the current excel files.
        """
        # Make and save average runtime DataFrame
        self.avg_runtimes_dataframe = self._convert_dictionary_to_pandas_dataframe(
            self.avg_runtimes_dictionary
        )
        avg_runtime_result_excel_file_name = "{}_avg_runtime_{}_{}.xlsx".format(
            self.method, self.exp_name, self.current_experience_index
        )
        avg_runtime_result_full_path = os.path.join(
            self.result_path, avg_runtime_result_excel_file_name
        )
        self.avg_runtimes_dataframe.to_excel(avg_runtime_result_full_path)

        # # Make and save standard deviation runtime DataFrame
        # self.std_runtimes_dataframe = self._convert_dictionary_to_pandas_dataframe(
        #     self.std_runtimes_dictionary
        # )
        # std_runtime_result_excel_file_name = "{}_std_runtime_{}_{}.xlsx".format(
        #     self.method, self.exp_name, self.current_experience_index
        # )
        # std_runtime_result_full_path = os.path.join(
        #     self.result_path, std_runtime_result_excel_file_name
        # )
        # self.std_runtimes_dataframe.to_excel(std_runtime_result_full_path)

        # Make and save average precision DataFrame
        # self.avg_precision_dataframe = self._convert_dictionary_to_pandas_dataframe(
        #     self.avg_precision_dictionary
        # )
        # avg_precision_result_excel_file_name = "{}_avg_precision_{}_{}.xlsx".format(
        #     self.method, self.exp_name, self.current_experience_index
        # )
        # avg_precision_result_full_path = os.path.join(
        #     self.result_path, avg_precision_result_excel_file_name
        # )
        # self.avg_precision_dataframe.to_excel(avg_precision_result_full_path)

        # # Make and save standard deviation precision DataFrame
        # self.std_precision_dataframe = self._convert_dictionary_to_pandas_dataframe(
        #     self.std_precision_dictionary
        # )
        # std_precision_result_excel_file_name = "{}_std_precision_{}_{}.xlsx".format(
        #     self.method, self.exp_name, self.current_experience_index
        # )
        # std_precision_result_full_path = os.path.join(
        #     self.result_path, std_precision_result_excel_file_name
        # )
        # self.std_precision_dataframe.to_excel(std_precision_result_full_path)

    def print_model_solver_list(self):
        """
        Show the available solvers_ori and models.
        """
        base_model: GenericModel = self.model_list[0]
        base_model.create_model()

        print()
        print("Model list :")
        for model in self.model_list:
            model: GenericModel
            print(model.name)

        print()
        print("Solver list :")
        for solver in self.solver_list:
            base_model.create_model()
            solver: GenericSolver = solver(base_model, self.discount)
            print(solver.name)
        print()

        base_model.lighten_model()
