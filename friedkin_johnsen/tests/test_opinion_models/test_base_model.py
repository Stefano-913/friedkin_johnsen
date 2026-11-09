import pytest
from pytest import fixture
import sys
import numpy as np
from scipy.sparse import csr_array
from ...opinion_models.base_model import OpinionModel


class ModelTest(OpinionModel):
    """ Simple stand-in subclass for OpinionModel """
    def _evolve(self, X):
        return X

    def calculate_limit_solution(self, tolerance=0):
        return None

    def convergence_rate(self):
        return 1

    def is_convergent(self):
        return False


class ModelSimulateTest(ModelTest):
    """ All purpose class for testing the simulate() method of
        the OpinionModel class.
        It focuses less on the actual matrix values and more on
        the general behaviour, as the numerical result will be later tested
        for specific models according to their respective _evolve().

        Parameters
        ----------
        max_steps:  simulate() is expected to stop once either one of the
                    criteria is fulfilled, the max_steps argument dictates
                    when one of these criteria forcefully sets off and the
                    simulation is expected to end.

        error:  the criteria both involve a vector comparison and, in order to
                stop the simulation at max_steps, the comparison is trivially
                satisfied (e.g. X==X): "error" introduces a variable in the
                comparison as to simulate and control the difference in the
                values ( np.allclose(X, X+error, tol=tolerance) ).
                This value is thought to check for the simulate()'s
                "tolerance" argument, that controls the margin of error in the
                comparison.

        limit_solution: the value of the limit solution doesn't matter per se,
                        as the only real difference is whether it's "None" or
                        a valid np.ndarray:
                            - in the "None" case, the criteria that's set off
                            to stop the simulation is the comparison among
                            adjacent values X(k) and X(k+1));
                            - if it's a valid value, the limit_solution
                            criterion is used instead and X(k) is set close
                            to that.
    """
    def __init__(self,
                 adjacency: csr_array = csr_array([[0, 1], [1, 0]]),
                 prejudice: np.ndarray = np.array([1, 0.5]),
                 max_steps: int = 100,
                 error: float = 0,
                 limit_solution: None | np.ndarray = None,
                 seed: int = 1):

        super().__init__(adjacency, prejudice)
        self.steps = 0
        self.max_steps = max_steps
        self.error = error
        self.limit_solution = limit_solution
        self.rng = np.random.default_rng(seed)

    def _evolve(self, X):
        self.steps += 1

        if self.steps == self.max_steps:
            if self.limit_solution is not None:
                return self.limit_solution+self.error
            return X+self.error

        return self.rng.random(self.no_users)

    def calculate_limit_solution(self, tolerance=0):
        return self.limit_solution

# -----


@fixture
def model():
    return ModelTest(adjacency=csr_array([[0, 1], [1, 0]]),
                     prejudice=np.array([1, 0.5]))


class TestOpinionModel:

    @pytest.mark.parametrize("wrong_type", [np.asarray, list, tuple])
    def test_raise_if_adjacency_wrong_type(self, wrong_type):
        with pytest.raises(TypeError,
                           match=r"Adjacency matrix is of.*"
                           "type, expected 'csr_array'."):
            ModelTest(adjacency=wrong_type([[0, 1], [0, 1]]),
                      prejudice=np.array([0.1, 0.5]))

    @pytest.mark.parametrize("wrong_type", [csr_array, list, tuple])
    def test_raise_if_prejudice_wrong_type(self, wrong_type):
        with pytest.raises(TypeError,
                           match=r"Prejudice matrix is of.*"
                           "type, expected 'np.ndarray'."):
            ModelTest(adjacency=csr_array([[0, 1], [0, 1]]),
                      prejudice=wrong_type([0.1, 0.5]))


class TestSimulate:
    def test_simulate(self, model, subtests):
        model = ModelSimulateTest(max_steps=301)
        steps = 300
        input_matrices = {"W": model.adjacency.copy(),
                          "u": model.prejudice.copy()}

        sim = model.simulate(steps=steps, snapshot_freq=10)

        with subtests.test(msg="returns a dictionary"):
            assert isinstance(sim, dict)

        with subtests.test(msg="input matrices are unchanged"):
            # Since there is no built-in '==' operator for sparse arrays
            assert (input_matrices["W"] != model.adjacency).nnz == 0
            assert (input_matrices["u"] == model.prejudice).all()

    @pytest.mark.parametrize("steps, snapshot_freq, max_steps, is_stopped", [
        (50, 50, 100, False),
        (100, 1, 101, False),
        (100, 3, 50, True),
        (205, 11, 93, True),

        (200, None, 87, True),
        (30, None, 33, False)
    ])
    def test_correct_snapshot_recording(self,
                                        steps,
                                        snapshot_freq,
                                        max_steps,
                                        is_stopped,
                                        subtests):

        model = ModelSimulateTest(max_steps=max_steps)
        sim = model.simulate(steps=steps, snapshot_freq=snapshot_freq)

        with subtests.test(msg="first state is always recorded"):
            assert list(sim.keys())[0] == 0
            assert np.all(sim[0] == model.prejudice)

        with subtests.test(msg="simulation not stopped sees all the steps"):
            if not is_stopped:
                assert list(sim.keys())[-1] == steps

        with subtests.test(msg="correct snapshot spacing"):
            if snapshot_freq is not None:
                expected_spacing = list(range(0,
                                              min(steps, max_steps),
                                              snapshot_freq)
                                        )
                assert list(sim.keys())[:-1] == expected_spacing

        with subtests.test(msg="only first/last snapshots if freq is None"):
            if snapshot_freq is None:
                assert list(sim.keys()) == [0, min(steps, max_steps)]

    @pytest.mark.parametrize("steps, max_steps", [(500, 100), (100, 92)])
    @pytest.mark.parametrize("limit_solution", [None, np.array([0.5, 0.5])])
    def test_correct_stop_criteria(self, steps, max_steps, limit_solution):
        model = ModelSimulateTest(max_steps=max_steps,
                                  limit_solution=limit_solution)
        sim = model.simulate(steps=steps)

        assert list(sim.keys())[-1] == max_steps

    @pytest.mark.parametrize("limit_solution", [None,
                                                np.array([0.5, 0.5]),
                                                np.array([0.1, 0.2])])
    def test_sim_saves_last_status_if_stopped(self, limit_solution):
        limit_solution = np.array([0.5, 0.5])

        model = ModelSimulateTest(max_steps=20,
                                  limit_solution=limit_solution)
        sim = model.simulate(steps=30)

        if limit_solution is None:
            assert np.all(list(sim.values())[-1] == list(sim.values())[-2])
        else:
            assert np.all(list(sim.values())[-1] == limit_solution)

    @pytest.mark.parametrize("sim_tolerance, model_error, is_sim_stopped", [
        (1e-10, 1e-12, True),
        (1e-10, 1e-8, False),
        (1e-5, 9*1e-6, True),
        (1e-5, 1e-5+1e-10, False)
    ])
    @pytest.mark.parametrize("limit_solution", [None, np.array([0.5, 0.5])])
    def test_stop_check_tolerance(self,
                                  sim_tolerance,
                                  model_error,
                                  is_sim_stopped,
                                  limit_solution):
        max_steps = 100
        steps = 200

        model = ModelSimulateTest(max_steps=max_steps,
                                  error=model_error,
                                  limit_solution=limit_solution)
        sim = model.simulate(steps=steps, tolerance=sim_tolerance)

        if is_sim_stopped:
            assert list(sim.keys())[-1] == max_steps
        else:
            assert list(sim.keys())[-1] == steps

    # ----

    @pytest.mark.parametrize("wrong_type", [1.5, object(), [1], {3}, None])
    def test_raise_if_steps_wrong_type(self, wrong_type, model):
        with pytest.raises(TypeError,
                           match="Invalid type of steps for evolution"):
            model.simulate(steps=wrong_type, snapshot_freq=100)

    @pytest.mark.parametrize("negative_value", [-10, -1])
    def test_raise_if_steps_negative(self, negative_value, model):
        with pytest.raises(ValueError,
                           match="Invalid value of steps for evolution"):
            model.simulate(steps=negative_value, snapshot_freq=100)

    # ----

    def test_snapshot_freq_accepts_none(self, model):
        model.simulate(steps=10, snapshot_freq=None)

    @pytest.mark.parametrize("wrong_type", [1.5, object(), [1], {3}])
    def test_raise_if_snapshot_freq_wrong_type(self, wrong_type, model):
        with pytest.raises(TypeError,
                           match="Invalid type of snapshot frequency"):
            model.simulate(steps=10, snapshot_freq=wrong_type)

    @pytest.mark.parametrize("non_positive_value", [-10, -1, 0])
    def test_raise_if_snapshot_freq_non_positive(self,
                                                 non_positive_value,
                                                 model):
        with pytest.raises(ValueError,
                           match="Invalid value of snapshot frequency"):
            model.simulate(steps=10, snapshot_freq=non_positive_value)

    # ----

    @pytest.mark.parametrize("wrong_type", [object(), [1], {3}])
    def test_raise_if_tolerance_wrong_type(self, wrong_type, model):
        with pytest.raises(TypeError,
                           match="Invalid type of tolerance"):
            model.simulate(steps=10, snapshot_freq=None, tolerance=wrong_type)

    @pytest.mark.parametrize("negative_value", [-10, -1, -0.001])
    def test_raise_if_tolerance_negative(self, negative_value, model):
        with pytest.raises(ValueError,
                           match="Invalid value of tolerance"):
            model.simulate(steps=10,
                           snapshot_freq=None,
                           tolerance=negative_value)


if __name__ == "__main__":
    sys.exit(pytest.main(
         ["-v", "--cov=your_package", "--cov-report=term-missing"]))
