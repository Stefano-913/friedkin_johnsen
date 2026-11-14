import pytest
from pytest import fixture
import sys
import numpy as np
from scipy.sparse import csr_array
from ...opinion_models.de_groot import DeGrootModel


def degroot_model_from_lists(adjacency, prejudice):

    adjacency = csr_array(adjacency)
    prejudice = np.asarray(prejudice)

    return DeGrootModel(adjacency=adjacency,
                        prejudice=prejudice)


@fixture
def model():
    return degroot_model_from_lists(adjacency=[[0, 0.5, 0.5],
                                               [0.25, 0, 0.75],
                                               [0.6, 0.4, 0]],
                                    prejudice=[0, 0.5, 1])

# The testing in this file will be partial, as the class has already been
# tested in the 'test_models_shared.py' file across that compared the results
# of its methods with various simulations with a very high degree of
# accuracy (up to 1e-10).


class TestSimulateDeGrootModel:
    def test_convergent_without_consensus_model(self, subtests):
        # The graph is weakly connected, as the nodes are divided in two
        # groups (1, 2, 3) and (5, 6, 7) with the 4th node only being
        # influenced and not influencing either of them.

        model = degroot_model_from_lists(

            adjacency=[[0, 1/2, 1/2, 0, 0, 0, 0],
                       [1, 0, 0, 0, 0, 0, 0],
                       [1/2, 1/2, 0, 0, 0, 0, 0],
                       [0, 0, 1/2, 0, 0, 1/2, 0],
                       [0, 0, 0, 0, 0, 1/2, 1/2],
                       [0, 0, 0, 0, 1/2, 0, 1/2],
                       [0, 0, 0, 0, 1, 0, 0]],

            prejudice=[1, 0.3, 0.6, 0.5, 0.2, 0, 0.4]
            )

        assert not model.is_there_consensus()
        assert model.is_convergent()

        steps = 3000
        sim = model.simulate(steps=steps, snapshot_freq=1)

        with subtests.test(msg="correct convergence"):
            assert np.allclose(sim[max(sim)],
                               np.array([61/90, 61/90, 61/90,
                                         9/20,
                                         2/9, 2/9, 2/9]), atol=1e-10)

        with subtests.test(msg="convergence but no limit solution"):
            # for lack of a closed formula in cases of non consensus
            assert model.calculate_limit_solution() is None

        with subtests.test(msg="opinions evolve within bounds"):
            assert all(np.all((opinion >= 0) & (opinion <= 1))
                       for opinion in sim.values())


class TestDeGrootModelEvolveMethod:

    def test_correct_evolve(self, model):

        X_step = model._evolve(model.prejudice)

        assert isinstance(X_step, np.ndarray)
        assert X_step.shape == model.prejudice.shape

        assert np.allclose(X_step, np.array([0.75, 0.75, 0.2]))

    def test_correct_evolve_multitopic(self):
        model = degroot_model_from_lists(adjacency=[[0, 0.2, 0.5, 0.3],
                                                    [0.2, 0, 0.1, 0.7],
                                                    [0.6, 0.4, 0, 0],
                                                    [0.4, 0.1, 0.5, 0]],

                                         prejudice=[[0, 1, 0.9],
                                                    [0.5, 1, 0.1],
                                                    [1, 0.4, 0.2],
                                                    [0.3, 0.2, 0.5]])

        X_step = model._evolve(model.prejudice)

        assert isinstance(X_step, np.ndarray)
        assert X_step.shape == model.prejudice.shape

        assert np.allclose(X_step, np.array([[0.69, 0.46, 0.27],
                                            [0.31, 0.38, 0.55],
                                            [0.2, 1, 0.58],
                                            [0.55, 0.7, 0.47]]))

    def test_correct_evolve_oscillating_multitopic(self, subtests):
        model = degroot_model_from_lists(adjacency=[[0, 0, 0, 0.5, 0, 0.5],
                                                    [0.5, 0, 0, 0, 0.5, 0],
                                                    [0, 0.5, 0, 0, 0, 0.5],
                                                    [0.5, 0, 0.5, 0, 0, 0],
                                                    [0, 0.5, 0, 0.5, 0, 0],
                                                    [0, 0, 0.5, 0, 0.5, 0]],

                                         prejudice=[[0, 1, 0.9, 1],
                                                    [0.5, 1, 0.1, 0],
                                                    [1, 0.4, 0.2, 0],
                                                    [0.3, 0.2, 0.6, 0],
                                                    [0.1, 1, 0.5, 0],
                                                    [0.2, 0.2, 0.2, 0]])

        assert not model.is_convergent()
        with subtests.test(msg="correct next step"):
            X_step = model._evolve(model.prejudice)
            assert np.allclose(X_step,
                               np.array([[0.25, 0.2, 0.4, 0],
                                         [0.05, 1, 0.7, 0.5],
                                         [0.35, 0.6, 0.15, 0],
                                         [0.5, 0.7, 0.55, 0.5],
                                         [0.4, 0.6, 0.35, 0],
                                         [0.55, 0.7, 0.35, 0]]))

        with subtests.test(msg="correct oscillating states"):
            # Calculated using the N-th power of the adjacency matrix
            # to jump ahead after the system already stabilized
            # (used N and N+1 to obtain both states, N=200),
            # then applied it to the prejudice matrix.

            state_1 = np.array([[11/30, 4/5, 8/15, 1/3],
                                [1/3, 7/15, 3/10, 0],
                                [11/30, 4/5, 8/15, 1/3],
                                [1/3, 7/15, 3/10, 0],
                                [11/30, 4/5, 8/15, 1/3],
                                [1/3, 7/15, 3/10, 0]])

            state_2 = np.array([[1/3, 7/15, 3/10, 0],
                                [11/30, 4/5, 8/15, 1/3],
                                [1/3, 7/15, 3/10, 0],
                                [11/30, 4/5, 8/15, 1/3],
                                [1/3, 7/15, 3/10, 0],
                                [11/30, 4/5, 8/15, 1/3]])

            sim = model.simulate(steps=250, snapshot_freq=1)

            # Keep the states after the 200th step, to ensure stabilization
            states = [state for param, state in sim.items()][200:]

            for i in range(len(states) - 1):
                assert not np.allclose(states[i], states[i + 1])

                assert (np.allclose(states[i], state_1) or
                        np.allclose(states[i], state_2))


class TestDeGrootConvergence:

    def test_double_one_eigenvalue(self):
        # Eigenvalues are 1, 1 and -2.
        # The matrix is clearly invalid for the model, but the
        # convergence test should work regardless.
        model = degroot_model_from_lists(adjacency=[[0, -1, -1],
                                                    [-1, 0, -1],
                                                    [-1, -1, 0]],
                                         prejudice=[0, 0.5, 1])
        assert model.is_convergent()


class TestDeGrootConsensus:
    def test_consensus(self, model):
        assert model.is_convergent()
        assert model.is_there_consensus()

    def test_no_consensus_if_no_convergece(self, monkeypatch):
        # Oscillating models can satisfy the requirement of
        # consensus (lambda = 1 is a simple eigenvalue) without
        # satisfying those of convergence.

        model = degroot_model_from_lists(adjacency=[[0, 1], [1, 0]],
                                         prejudice=[1, 0])

        assert not model.is_convergent()
        assert not model.is_there_consensus()

        def mock_convergence_test(*args, **kwargs):
            return True
        monkeypatch.setattr("friedkin_johnsen.opinion_models.de_groot."
                            "DeGrootModel.is_convergent",
                            mock_convergence_test)

        assert model.is_convergent()
        assert model.is_there_consensus()


class TestDegrootLimitSolution:

    def test_evolve_in_fixed_point_already(self, model):
        X = model.calculate_limit_solution()
        assert np.allclose(model._evolve(X), X)

    def test_correct_limit_solution(self, model):
        assert np.allclose(model.calculate_limit_solution(),
                           [0.538, 0.538, 0.538],
                           atol=1e-3)

    @pytest.mark.parametrize("prejudice", [
        [2, 5, 10], [-6, 0.4, 0.5]
    ])
    def test_raise_if_outside_bounds(self, prejudice):
        model = degroot_model_from_lists(adjacency=[[0, 1/2, 1/2],
                                                    [1/3, 0, 2/3],
                                                    [1/2, 1/2, 0]],
                                         prejudice=prejudice)
        with pytest.raises(ValueError,
                           match="Consensus calculation yielded an "
                           "unexpected result"):
            model.calculate_limit_solution()


class TestDeGrootModelConvergenceRate:

    def test_correct(self, model):
        assert np.allclose(model.convergence_rate, 0.524, atol=1e-3)

    def test_oscillating_model(self):
        model = degroot_model_from_lists(adjacency=[[0, 1, 0],
                                                    [1, 0, 0],
                                                    [0.5, 0.5, 0]],
                                         prejudice=[0, 1, 0.5])
        # Oscillating, it never converges
        assert np.allclose(model.convergence_rate, 1)

    def test_raise_if_outside_bounds(self):
        # This adjacency matrix yields conv_rate > 1
        model = degroot_model_from_lists(adjacency=[[3, 1], [1, 3]],
                                         prejudice=[0, 1])

        with pytest.raises(ValueError,
                           match="Convergence rate calculation yielded an "
                                 "unexpected result"):
            model.convergence_rate


class TestDeGrootToleranceValidation:

    @pytest.mark.parametrize("wrong_type", [object(), [1], {3}])
    def test_raise_if_tolerance_wrong_type(self, wrong_type, model, subtests):

        with subtests.test(msg="convergence"):
            with pytest.raises(TypeError,
                               match="Invalid type of tolerance"):
                model.is_convergent(tol=wrong_type)

        with subtests.test(msg="consensus"):
            with pytest.raises(TypeError,
                               match="Invalid type of tolerance"):
                model.is_there_consensus(tol=wrong_type)

        with subtests.test(msg="limit solution"):
            with pytest.raises(TypeError,
                               match="Invalid type of tolerance"):
                model.calculate_limit_solution(tol=wrong_type)

    @pytest.mark.parametrize("negative_value", [-10, -1, -0.001])
    def test_raise_if_tolerance_negative(self,
                                         negative_value,
                                         model,
                                         subtests):

        with subtests.test(msg="convergence"):
            with pytest.raises(ValueError,
                               match="Invalid value of tolerance"):
                model.is_convergent(tol=negative_value)

        with subtests.test(msg="consensus"):
            with pytest.raises(ValueError,
                               match="Invalid value of tolerance"):
                model.is_there_consensus(tol=negative_value)

        with subtests.test(msg="limit solution"):
            with pytest.raises(ValueError,
                               match="Invalid value of tolerance"):
                model.calculate_limit_solution(tol=negative_value)


if __name__ == "__main__":
    sys.exit(pytest.main(
         ["-v", "--cov=your_package", "--cov-report=term-missing"]))
