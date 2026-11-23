import pytest
from pytest import fixture
import sys
import numpy as np
from scipy.sparse import csr_array, dia_array
from ...opinion_models.de_groot import DeGrootModel
from ...opinion_models.friedkin_johnsen import FJModel

from .helper_test_funcs import fj_model_from_lists


class TestSimulateFJModel:

    def test_convergent_not_asympt_stable_model(self, subtests, monkeypatch):
        # Nodes (1, 2, 3, 4) are P-indipendent, nodes (5, 6, 7, 8) are
        # P-dependent (more details in the calculate_limit_solution test).
        model = fj_model_from_lists(
            adjacency=[[0, 0, 1, 0, 0, 0, 0, 0],
                       [1/2, 0, 0, 1/2, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0],
                       [0, 1/2, 1/2, 0, 0, 0, 0, 0],
                       [0, 0, 1/3, 0, 0, 1/3, 0, 1/3],
                       [1/4, 0, 1/4, 0, 1/4, 0, 1/4, 0],
                       [0, 0, 0, 0, 0, 1/2, 0, 1/2],
                       [0, 1/4, 1/4, 0, 0, 1/4, 1/4, 0]],
            prejudice=[0.24, 0.6, 0.1, 0.5, 1, 0.25, 0.5, 0.4],
            susceptibility=[1, 1, 1, 1, 0.1, 0.2, 0.4, 0.3]
            )

        assert not model.is_asympt_stable()
        assert model.is_convergent()

        def mock_simplified_formula(*args, **kwargs):
            raise AssertionError("The simplified formula should not "
                                 "be used now.")
        monkeypatch.setattr("friedkin_johnsen.opinion_models.friedkin_johnsen."
                            "FJModel._simplified_limit_solution",
                            mock_simplified_formula)

        steps = 3000
        sim = model.simulate(steps=steps, snapshot_freq=100)

        with subtests.test(msg="correct convergence"):
            assert np.allclose(sim[max(sim)],
                               model.calculate_limit_solution(),
                               atol=1e-10)

        with subtests.test(msg="opinions evolve within bounds"):
            assert all(np.all((opinion >= 0) & (opinion <= 1))
                       for opinion in sim.values())

    def test_oscillating_model(self, subtests):

        model = fj_model_from_lists(
            adjacency=[[0, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0],
                       [1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0.5, 0, 0.5, 0],
                       [0, 0, 0, 0, 0, 0.5, 0, 0.5],
                       [1, 0, 0, 0, 0, 0, 0, 0]],
            prejudice=[1, 0, 0, 0.4, 0.1, 0.5, 0.8, 0.3],
            susceptibility=[1, 1, 1, 0.2, 0.5, 0.8, 0.7, 0.3])

        assert not model.is_asympt_stable()

        with subtests.test(msg="correct next step"):
            X_step = model._evolve(model.prejudice)

            assert np.allclose(
                X_step,
                np.array([0, 0, 1, 0.32, 0.25, 0.46, 0.52, 0.51])
                )

        with subtests.test(msg="correct oscillating states"):
            sim = model.simulate(steps=250, snapshot_freq=1)

            # Keep the states after the 200th step, to ensure stabilization
            states = [state for param, state in sim.items()][200:]

            state_1, state_2, state_3 = states[0], states[1], states[2]

            # I use the P-indipendent nodes as tags, as they are the only
            # ones easily predictable, for a quick sanity check (period of 3)
            marker_nodes = [state_1[:3], state_2[:3], state_3[:3]]

            assert any(np.allclose(np.array([1, 0, 0]), state)
                       for state in marker_nodes)
            assert any(np.allclose(np.array([0, 1, 0]), state)
                       for state in marker_nodes)
            assert any(np.allclose(np.array([0, 0, 1]), state)
                       for state in marker_nodes)

            for i in range(len(states) - 2):
                assert not np.allclose(states[i], states[i + 1])
                assert not np.allclose(states[i], states[i + 2])

                assert (np.allclose(states[i], state_1) or
                        np.allclose(states[i], state_2) or
                        np.allclose(states[i], state_3))


class TestFJModelEvolveMethod:
    @fixture
    def model(self):
        return fj_model_from_lists(adjacency=[[0, 0.5, 0.5],
                                              [0.25, 0, 0.75],
                                              [0.6, 0.4, 0]],
                                   prejudice=[0, 0.5, 1],
                                   susceptibility=[0.2, 0.4, 0.3])

    # ------

    def test_correct_evolve(self, model):
        A = [[0, 0.1, 0.1], [0.1, 0, 0.3], [0.18, 0.12, 0]]
        B = [0, 0.3, 0.7]
        X_step = model._evolve(np.array([1, 0.2, 0.5]))

        assert isinstance(X_step, np.ndarray)
        assert X_step.shape == model.prejudice.shape

        assert np.allclose(A, model._influence_matrix.toarray())
        assert np.allclose(B, model._prejudice_term)
        assert np.allclose(X_step, np.array([0.07, 0.55, 0.904]))

    def test_zero_susceptibility(self, model):
        model.susceptibility = dia_array(np.diag([0, 0, 0]))

        assert np.allclose(model._evolve(np.array([1, 0.2, 0.5])),
                           model.prejudice)

    def test_full_susceptibility(self, model):
        model.susceptibility = dia_array(np.diag([1, 1, 1]))

        assert np.allclose(model._evolve(np.array([1, 0.2, 0.5])),
                           [0.35, 0.625, 0.68])

    def test_evolve_in_fixed_point_already(self, model):
        X = model.calculate_limit_solution()
        assert np.allclose(model._evolve(X), X)

    def test_correct_caching(self, model):
        model._evolve(np.array([0, 0, 0]))
        A = model._influence_matrix
        B = model._prejudice_term

        model._evolve(np.array([1, 1, 1]))

        assert A is model._influence_matrix
        assert B is model._prejudice_term


class TestFJModelDependentUsers:

    @pytest.mark.parametrize("adjacency, susceptibility, P_dependency", [
        ([[0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 1, 0],
         [1/3, 0, 1/3, 0, 1/3],
         [0, 0, 0, 1, 0]],

         [1, 1, 1, 0.4, 0.2],
         [False, True, True, True, True]),

        ([[0, 0, 0, 0, 0],
          [1, 0, 0, 0, 0],
          [1, 1, 0, 0, 0],
          [0, 1, 1, 0, 0],
          [0, 0, 0, 1, 0]],

         [1, 1, 1, 0.4, 1],
         [False, False, False, True, True])
        ])
    def test_dependency_calculation(self,
                                    adjacency,
                                    susceptibility,
                                    P_dependency):
        prejudice = np.full((len(susceptibility), ), 0.5)

        model = fj_model_from_lists(adjacency=adjacency,
                                    prejudice=prejudice,
                                    susceptibility=susceptibility)

        assert all(model.dependent_users == P_dependency)


class TestFJModelLimitSolution:
    # The testing here will be partial, as the limit_solution has already been
    # used in the test 'TestFJModel:test_rng_model()' that showed its
    # compatibility with the results of various simulations to a very high
    # degree of accuracy (1e-10).

    @fixture
    def model(self):
        return fj_model_from_lists(adjacency=[[0, 0.5, 0.5],
                                              [0.25, 0, 0.75],
                                              [0.6, 0.4, 0]],
                                   prejudice=[0, 0.5, 1],
                                   susceptibility=[0.2, 0.4, 0.3])

    @fixture
    def not_asympt_stable_model(self):
        # The graph can be divided in two strongly connected components,
        # built so that the first three nodes (0, 1, 2) are all influenced by
        # each other (and themselves only); while the three nodes in the last
        # component (3, 4, 5) are influenced by the first component also.

        # (0, 1, 2) also have highest susceptibility, making it a De Groot
        # subgraph of P-indipendent nodes, while (3, 4, 5) are all P-dependent.
        # This makes the model convergent but non asymptotically stable.

        model = fj_model_from_lists(
            adjacency=[[0, 1, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0],
                       [1/2, 1/2, 0, 0, 0, 0],
                       [1/3, 0, 0, 0, 1/3, 1/3],
                       [0, 0, 0, 1, 0, 0],
                       [0, 1/3, 0, 1/3, 1/3, 0]],
            prejudice=[0.24, 0.6, 0.1, 0.5, 1, 0.25],
            susceptibility=[1, 1, 1, 0.2, 0.4, 0.3])

        assert not model.is_asympt_stable()
        assert model.is_convergent()

        return model

    # ------

    def test_correct_limit_solution(self, model):
        assert np.allclose(model.calculate_limit_solution(),
                           [0.134, 0.55, 0.79],
                           atol=1e-3)

    def test_limit_of_max_stubborness_is_prejudice(self, model):
        model.susceptibility = dia_array(np.diag([0, 0, 0]))

        assert np.allclose(model.calculate_limit_solution(), model.prejudice)

    def test_not_sympt_stable(self, not_asympt_stable_model, monkeypatch):
        degroot_submodel = DeGrootModel(adjacency=csr_array([[0, 1, 0],
                                                             [0, 0, 1],
                                                             [1/2, 1/2, 0]]),
                                        prejudice=np.array([0.24, 0.6, 0.1]))
        sub_limit = degroot_submodel.calculate_limit_solution()

        def mock_simplified_formula(*args, **kwargs):
            raise AssertionError("The simplified formula should not "
                                 "be used now.")
        monkeypatch.setattr("friedkin_johnsen.opinion_models.friedkin_johnsen."
                            "FJModel._simplified_limit_solution",
                            mock_simplified_formula)

        # The formula being quite complex, its manual check on WolframAlpha has
        # been done in tranches after splitting it into two matrices (A, B).
        #
        #   X_d(inf) = (I - Λ_dd W_dd)⁻¹ (Λ_dd W_di X_i(inf) + (I - Λ_dd)u_d ),
        #           with consensus in X_i(inf) being 0.328 (De Groot subgraph).
        #
        # 1) A⁻¹ = (I - Λ_dd W_dd)⁻¹
        #    Inverse[
        #       IdentityMatrix[3] -  DiagonalMatrix[{0.2,0.4,0.3}] .
        #           {{0,1/3,1/3},{1,0,0},{1/3,1/3,0}}
        #           ]
        #
        #    Result:   {{1.03734, 0.07607, 0.06916},
        #               {0.41494, 1.03043, 0.02766},
        #               {0.14523, 0.11065, 1.00968}}
        #
        # 2) B = Λ_dd W_di x_i(inf) + (I - Λ_dd) u_d
        #    DiagonalMatrix[{0.2,0.4,0.3}] .
        #       {{1/3,0,0},{0,0,0},{0,1/3,0}} . {0.328,0.328,0.328}
        #      +
        #       (IdentityMatrix[3] - DiagonalMatrix[{0.2,0.4,0.3}]) .
        #           {0.5,1,0.25}
        #
        #    Result: {0.421867, 0.6, 0.2078}
        #
        # 3) X_d(inf) = A⁻¹ . B
        #    Result: {0.49763485, 0.79905394, 0.33746888}

        expected_result = np.append(sub_limit,
                                    [0.49763485, 0.79905394, 0.33746888])

        assert np.allclose(not_asympt_stable_model.calculate_limit_solution(),
                           np.array(expected_result))


class TestFJModelConvergenceRate:
    # The testing here will be partial, as the convergence_rate has already
    # been compared to the evolution of the model in various simulations in
    # the test (the 'TestFJModel:test_speed_of_convergence()' and yielded
    # positive results two were compatible, according to theory).

    @fixture
    def model(self):
        return fj_model_from_lists(adjacency=[[0, 0.5, 0.5],
                                              [0.25, 0, 0.75],
                                              [0.6, 0.4, 0]],
                                   prejudice=[0, 0.5, 1],
                                   susceptibility=[0.2, 0.4, 0.3])

    def test_correct(self, model):
        assert np.allclose(model.convergence_rate, 0.294, atol=1e-3)

    def test_when_uniform_susceptibility(self, model):
        model.susceptibility = dia_array(np.diag([0.75, 0.75, 0.75]))

        # W being stochastic, its highest eigenvalue is 1
        # -> if lambda is const*I, the highest eigenvalue of lambda*W is const
        assert np.allclose(model.convergence_rate, 0.75)

    def test_highest_stubborness(self, model):
        model.susceptibility = dia_array(np.diag([0, 0, 0]))

        # No one ever changes their mind, the model reaches x* limit solution
        # immediately -> already converged, rate is 0
        assert np.allclose(model.convergence_rate, 0)

    def test_highest_susceptibility(self, model):
        model.susceptibility = dia_array(np.diag([1, 1, 1]))

        # The model degenerates to De Groot and loses the guarantee
        # of convergence, this method isn't useful anymore but
        # mathematically should return 1 anyways
        assert np.allclose(model.convergence_rate, 1)

    def test_oscillating_model(self):
        model = fj_model_from_lists(adjacency=[[0, 1, 0],
                                               [1, 0, 0],
                                               [0.5, 0.5, 0]],
                                    prejudice=[0, 1, 0.5],
                                    susceptibility=[1, 1, 0.5])
        # No convergence
        assert np.allclose(model.convergence_rate, 1)


@pytest.mark.parametrize("type", [np.array, csr_array, list, str])
def test_raise_if_wrong_susceptibility_type(type):
    with pytest.raises(TypeError,
                       match="Susceptibility matrix is of .* type"):
        FJModel(adjacency=csr_array([0, 1], [1, 0]),
                prejudice=np.array([1, 0.4]),
                susceptibility=type([0.2, 0.5]))


if __name__ == "__main__":
    sys.exit(pytest.main(
         ["-v", "--cov=your_package", "--cov-report=term-missing"]))
