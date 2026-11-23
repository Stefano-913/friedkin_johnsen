import sys
from pathlib import Path
import numpy as np
import pytest
from pytest import fixture
from itertools import combinations
from scipy.sparse import csr_array, dia_array
from ..simulation import run_simulation, run_simulations_parallel
from .test_get_matrices import simulate_config_file


class mock_DeGrootModel:
    def __init__(self, adjacency, prejudice):
        pass

    def simulate(self, *args, **kwargs):
        return "degroot"


class mock_FJModel:
    def __init__(self, adjacency, prejudice, susceptibility):
        pass

    def simulate(self, *args, **kwargs):
        return "friedkin-johnsen"


@fixture
def matrix_dict():
    return {"adjacency": ("[[0, 0.5, 0.5],"
                          "[0.25, 0, 0.75],"
                          "[0.6, 0.4, 0]]"),
            "prejudice": "[0, 0.5, 1]",
            "susceptibility": ("[[0.2, 0, 0],"
                               "[0, 0.4, 0],"
                               "[0, 0, 0.3]]")}

# -----


class TestRunSimulation:
    @pytest.mark.parametrize("susceptibility, tolerance, expected_model", [
        ([1, 1], 1e-10, "degroot"),
        ([0.5, 1], 1e-10, "friedkin-johnsen"),

        ([1, 1, 1-1e-6], 1e-5, "degroot"),
        ([1, 1, 1-1e-4], 1e-5, "friedkin-johnsen")
    ])
    def test_uses_correct_model_based_on_tolerance(self,
                                                   monkeypatch,
                                                   susceptibility,
                                                   tolerance,
                                                   expected_model):

        def mock_get_matrices(config_file, directory):
            return {"adjacency": csr_array([[0, 1], [1, 0]]),
                    "prejudice": np.array([1, 0.4]),
                    "susceptibility": dia_array(np.diag(susceptibility))}

        monkeypatch.setattr("friedkin_johnsen.simulation.get_matrices",
                            mock_get_matrices)
        monkeypatch.setattr("friedkin_johnsen.simulation.DeGrootModel",
                            mock_DeGrootModel)
        monkeypatch.setattr("friedkin_johnsen.simulation.FJModel",
                            mock_FJModel)

        model = run_simulation("config.json", Path("."), tolerance)

        assert model == expected_model

    def test_correct_simulation_result(self, tmp_path, matrix_dict):
        simulate_config_file(tmp_path=tmp_path,
                             matrix_dict=matrix_dict,
                             no_of_nodes=3,
                             file_name="file.json")

        config_directory = tmp_path / "file"

        assert np.allclose(
            run_simulation(steps=1,
                           config_file="file.json",
                           directory=config_directory)[1].flatten(),
            [0.15, 0.60, 0.76]
            )

        sim_opinions = list(
            run_simulation(steps=10_000,
                           config_file="file.json",
                           directory=config_directory).values()
                           )

        assert np.allclose(sim_opinions[-1].flatten(),
                           [0.134065, 0.550463, 0.790187])

    def test_correct_degroot_simulation(self, tmp_path, matrix_dict):
        matrix_dict["susceptibility"] = "[[1,0,0],[0,1,0],[0,0,1]]"
        simulate_config_file(tmp_path=tmp_path,
                             matrix_dict=matrix_dict,
                             no_of_nodes=3,
                             file_name="file.json")

        config_directory = tmp_path / "file"

        assert np.allclose(
            run_simulation(steps=1,
                           config_file="file.json",
                           directory=config_directory)[1].flatten(),
            [0.75, 0.75, 0.2]
            )

        last_opinion = run_simulation(config_file="file.json",
                              directory=config_directory)
        assert np.allclose(last_opinion[max(last_opinion)].flatten(),
                           [0.53846]*3)

    def test_snapshot_frequency(self, tmp_path, matrix_dict):
        simulate_config_file(tmp_path=tmp_path,
                             matrix_dict=matrix_dict,
                             no_of_nodes=3,
                             file_name="file.json")

        config_directory = tmp_path / "file"
        sim = run_simulation(steps=4,
                             snapshot_freq=2,
                             config_file="file.json",
                             directory=config_directory)

        assert list(sim) == [0, 2, 4]

    @pytest.mark.parametrize("wrong_type", [object(), [1], {3}, "1e-10"])
    def test_raise_if_tolerance_wrong_type(self, wrong_type):

        with pytest.raises(TypeError,
                           match="Invalid type of tolerance"):
            run_simulation(tol=wrong_type)

    @pytest.mark.parametrize("negative_value", [-10, -1, -0.001])
    def test_raise_if_tolerance_negative(self, negative_value):

        with pytest.raises(ValueError,
                           match="Invalid value of tolerance"):
            run_simulation(tol=negative_value)


class TestRunSimulationsParallel:

    @pytest.mark.filterwarnings(
        r"ignore:This process .* is multi-threaded, use of fork.* "
        "may lead to deadlocks in the child.:DeprecationWarning"
    )
    def test_independent_rng_from_parallel_runs(self, tmp_path):

        simulate_config_file(
            tmp_path=tmp_path,
            matrix_dict={"adjacency": "[[0, 1, 0], [1, 0, 1], [0, 1, 0]]"},
            no_of_nodes=3,
            file_name="file.json"
            )

        config_directory = tmp_path / "file"
        simulations = run_simulations_parallel(no_simulations=5,
                                               config_file="file.json",
                                               directory=config_directory)

        for first_sim, second_sim in combinations(simulations, 2):
            assert all(not np.allclose(first_sim[step], second_sim[step])
                       for step in first_sim if step in second_sim)

    def test_correct_simulation(self, tmp_path, matrix_dict, subtests):
        simulate_config_file(tmp_path=tmp_path,
                             matrix_dict=matrix_dict,
                             no_of_nodes=3, file_name="file.json")
        config_directory = tmp_path / "file"

        with subtests.test(msg="single step in single run"):
            sim_result = run_simulations_parallel(
                steps=1,
                config_file="file.json",
                directory=config_directory
                )

            first_sim_first_step = sim_result[0][1].flatten()
            assert np.allclose(first_sim_first_step, [0.15, 0.60, 0.76])

        with subtests.test(msg="single step in multiple runs"):
            sim_result = run_simulations_parallel(
                no_simulations=10,
                steps=1,
                config_file="file.json",
                directory=config_directory
                )

            first_step_multi_runs = [sim[1].flatten() for sim in sim_result]
            assert np.allclose(
                first_step_multi_runs,
                [[0.15, 0.60, 0.76] for i in range(10)]
                )

        with subtests.test(msg="multiple steps in single run"):
            sim_result = run_simulations_parallel(
                steps=10_000,
                config_file="file.json",
                directory=config_directory
                )

            first_sim = sim_result[0]
            first_sim_last_step = first_sim[max(first_sim)].flatten()

            assert np.allclose(first_sim_last_step,
                               [0.134065, 0.550463, 0.790187])

        with subtests.test(msg="multiple steps in multiple runs"):
            sim_result = run_simulations_parallel(
                no_simulations=10,
                steps=10_000,
                config_file="file.json",
                directory=config_directory
                )

            last_step_multi_runs = [sim[max(sim)].flatten()
                                    for sim in sim_result]
            assert np.allclose(
                last_step_multi_runs,
                [[0.134065, 0.550463, 0.790187] for i in range(10)]
            )

    def test_snapshot_frequency(self, tmp_path, matrix_dict):
        simulate_config_file(tmp_path=tmp_path,
                             matrix_dict=matrix_dict,
                             no_of_nodes=3,
                             file_name="file.json")

        config_directory = tmp_path / "file"
        sim = run_simulations_parallel(steps=4,
                                       snapshot_freq=2,
                                       config_file="file.json",
                                       directory=config_directory)

        assert list(sim[0]) == [0, 2, 4]

    @pytest.mark.parametrize("wrong_type", [object(), [1], {3}, 2.5, "3"])
    def test_raise_if_no_simulations_wrong_type(self, wrong_type):
        with pytest.raises(TypeError,
                           match="Invalid type of number of simulations"):
            run_simulations_parallel(no_simulations=wrong_type)

    @pytest.mark.parametrize("non_positive_value", [0, -1, -10])
    def test_raise_if_no_simulations_non_positive(self,
                                                  non_positive_value):
        with pytest.raises(ValueError,
                           match="Invalid value of number of simulations"):
            run_simulations_parallel(no_simulations=non_positive_value)

    @pytest.mark.parametrize("wrong_type", [object(), [1], {3}, 2.5, "3"])
    def test_raise_if_num_processes_wrong_type(self, wrong_type):
        with pytest.raises(TypeError,
                           match="Invalid type of number of processes"):
            run_simulations_parallel(num_processes=wrong_type)

    @pytest.mark.parametrize("non_positive_value", [0, -1, -10])
    def test_raise_if_num_processes_non_positive(self,
                                                 non_positive_value):
        with pytest.raises(ValueError,
                           match="Invalid value of number of processes"):
            run_simulations_parallel(num_processes=non_positive_value)


if __name__ == "__main__":
    sys.exit(pytest.main(
        ["-s", "-v",  "--cov=your_package", "--cov-report=term-missing"
         ]))
