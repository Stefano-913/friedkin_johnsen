import pytest
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from ...opinion_models.friedkin_johnsen import FJModel

from .helper_test_funcs import (rng_degroot_model,
                                rng_fj_model,
                                degroot_model_from_lists,
                                fj_model_from_lists,
                                plot_opinions)


@pytest.fixture(params=[rng_degroot_model, rng_fj_model],
                ids=["de_groot", "friedkin_johnsen"])
def rng_model(request):
    return request.param


@pytest.fixture(params=[degroot_model_from_lists, fj_model_from_lists],
                ids=["de_groot", "friedkin_johnsen"])
def model_from_lists(request):
    return request.param


# -----

@pytest.mark.parametrize("no_users", [10, 50, 1000])
@pytest.mark.parametrize("no_of_topics", [1, 5, 10])
def test_rng_model(rng_model, no_users, no_of_topics, subtests):

    model = rng_model(no_users=no_users,
                      no_of_topics=no_of_topics,
                      seed=2)

    steps = 3000
    input_matrices = {"W": model.adjacency.copy(),
                      "u": model.prejudice.copy()}

    if isinstance(model, FJModel):
        input_matrices = {**input_matrices,
                          "lambda": model.susceptibility.copy()}

    sim = model.simulate(steps=steps, snapshot_freq=100)

    with subtests.test(msg="correct convergence"):
        assert np.allclose(sim[max(sim)],
                           model.calculate_limit_solution(),
                           atol=1e-10)

    with subtests.test(msg="opinions evolve within bounds"):
        assert all(np.all((opinion >= 0) & (opinion <= 1))
                   for opinion in sim.values())

    with subtests.test(msg="input matrices are unchanged"):

        # Since there is no built-in '==' operator for sparse arrays
        assert (input_matrices["W"] != model.adjacency).nnz == 0
        assert (input_matrices["u"] == model.prejudice).all()

        if isinstance(model, FJModel):
            assert (input_matrices["lambda"] != model.susceptibility).nnz == 0


@pytest.mark.parametrize("adjacency, prejudice, period", [
    # alternates [1, 0] and [0, 1]
    ([[0, 1], [1, 0]],
     [1, 0],
     2),

    # alternates [1, 0, 0], [0, 1, 0] and [0, 0, 1]
    ([[0, 0, 1], [1, 0, 0], [0, 1, 0]],
     [1, 0, 0],
     3),

    # alternates between [0, 0, 0.5, 0.5] and [0.5, 0.5, 0, 0]
    ([[0, 0, 0.5, 0.5], [0, 0, 0.5, 0.5], [0.5, 0.5, 0, 0], [0.5, 0.5, 0, 0]],
     [1, 0, 0, 0],
     2)])
def test_oscillating_model(model_from_lists,
                           adjacency,
                           prejudice,
                           period,
                           subtests):

    # The simulation is expected to have already stabilized after the
    # 100th cycle and have a period lower than 200 for appreciable results
    k_stabilized = 100
    steps = 300

    if steps-k_stabilized < period:
        raise ValueError("The test isn't significant, as the simulation is "
                         "stopped before the periodic behaviour is reached.")

    model = model_from_lists(adjacency, prejudice)

    # Snapshot frequency is lowest possible so the alternating of states
    # is visible in the simulation
    sim = model.simulate(steps=steps, snapshot_freq=1)

    with subtests.test(msg="expected oscillating behaviour"):
        assert not model.is_convergent()
        states = [opinion for k_param, opinion in sim.items()][k_stabilized:]

        for i in range(len(states) - 1):
            assert not np.allclose(states[i], states[i + 1])

        for i in range(len(states) - period):
            assert np.allclose(states[i], states[i + period])

    with subtests.test(msg="opinions evolve within bounds"):
        assert all(np.all((opinion >= 0) & (opinion <= 1))
                   for opinion in sim.values())


@pytest.mark.filterwarnings(r"ignore:Gtk.* is deprecated:DeprecationWarning")
@pytest.mark.filterwarnings(r"ignore:GLib.* is deprecated:DeprecationWarning")
@pytest.mark.parametrize("no_users", [50, 100, 200, 500])
def test_graph_opinion_evolution(rng_model, no_users):

    # This test has no assert and passes regardless.
    # Used to discover the correlation between λ and u in case of
    # wrong seeding (susceptibility and prejudice sharing the same
    # seed, see 'rng_model_matrices()' in the 'helper_test_funcs file'
    # for more info)
    model = rng_model(no_users=no_users,
                      no_of_topics=1,
                      seed=1,
                      mean_node_degree=10)
    steps = 10_000

    # High tolerance so that the stabilization period (i.e. "tails"
    # of the graph) do not take the majority of the plot
    sim = model.simulate(steps=steps, snapshot_freq=1, tolerance=1e-2)
    plot_opinions(model, sim)


@pytest.mark.parametrize("save_graphs", [True])
@pytest.mark.parametrize("no_users", [500, 1000, 2000])
@pytest.mark.parametrize("no_of_topics", [1, 2, 4])
def test_speed_of_convergence(rng_model,
                              no_users,
                              no_of_topics,
                              save_graphs,
                              subtests):

    # The test compares the theoretical rate of convergence to the rate from
    # extracted from the simulation (via fit).

    # lowering mean node degree to slow down the convergence
    model = rng_model(no_users=no_users,
                      no_of_topics=no_of_topics,
                      seed=2,
                      mean_node_degree=5)

    steps = 10_000
    assert model.is_convergent()

    sim = model.simulate(steps=steps, snapshot_freq=5, tolerance=1e-12)
    limit_solution = model.calculate_limit_solution()

    distance = {step: np.max(np.abs(value-limit_solution))
                for step, value in sim.items()}

    def convergence_func_fit(k, C, rho):
        return C * np.power(rho, k)

    def convergence_func_theoretical(k, C):
        return C * np.power(model.convergence_rate, k)

    k_param = np.array(list(distance.keys()))
    errors = np.array(list(distance.values()))

    # Excludes early 'errors' to avoid overfitting on those, as they tend
    # to weight disproportionately much in the fit
    mask = (k_param >= 0.5 * k_param.max()) & (errors > 1e-14)

    # Fit the data as a line, later reinterpreted as the convergence rate
    slope, intercept = np.polyfit(k_param[mask], np.log(errors[mask]), 1)

    rho_fit = np.exp(slope)
    C_fit = np.exp(intercept)

    # Fit the data having the theoretical convergence rate fixed
    popt, _ = sp.optimize.curve_fit(convergence_func_theoretical,
                                    k_param[mask],
                                    errors[mask],
                                    p0=[1.0])
    C_fixed_rate = popt[0]

    with subtests.test(msg="Fixed constant is positive"):
        assert C_fit > 0
        assert C_fixed_rate > 0

    with subtests.test(msg="Fitted rate is in [0, 1] interval"):
        assert 0 <= rho_fit <= 1

    with subtests.test(msg="Fitted rate is lower or equal to the maximum "
                       "theoretical value"):
        assert (rho_fit < model.convergence_rate or
                np.isclose(rho_fit, model.convergence_rate, rtol=1e-2))

    if save_graphs:
        fig, ax = plt.subplots()

        ax.semilogy(k_param, errors, 'o', label='Observed error', markersize=6)

        ax.semilogy(k_param,
                    convergence_func_fit(k_param, C_fit, rho_fit),
                    '-',
                    label=f'Fit: C·rho^k (rho={rho_fit:.3f})')

        ax.semilogy(k_param,
                    convergence_func_theoretical(k_param, C_fixed_rate),
                    '-',
                    label='Upper convergence limit')

        ax.set_xlabel('Iteration k')
        ax.set_ylabel('|x(k) - x*|')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if isinstance(model, FJModel):
            model_name = "fj_model"
        else:
            model_name = "degroot_model"

        plot_dir = f"friedkin_johnsen/tests/plots/{model_name}/convergence"
        os.makedirs(plot_dir, exist_ok=True)
        fig.savefig(plot_dir + f"/users_{no_users}_topics_{no_of_topics}.png",
                    dpi=120)
        plt.close(fig)
