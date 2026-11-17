import os
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.sparse import csr_array, dia_array
from ...opinion_models.de_groot import DeGrootModel
from ...opinion_models.friedkin_johnsen import FJModel
from ...build_rng_matrices import (build_adjacency_matrix,
                                   build_prejudice_matrix,
                                   build_susceptibility_matrix)


def rng_model_matrices(no_users, no_of_topics, seed=1, mean_node_degree=None):
    if mean_node_degree is None:
        mean_node_degree = random.Random(seed).randint(
            3, max(4, int(np.sqrt(no_users))))

    # Seeds need to be different, since prejudice and susceptibility use
    # the same generation method. If they used the same, the FJ model would
    # degenerate in a state where a node's prejudice is equal to its
    # susceptibility (λ = u for each node)
    adjacency = build_adjacency_matrix(
         nodes=no_users,
         mean_node_degree=mean_node_degree,
         seed=seed
        )
    prejudice = build_prejudice_matrix(
        nodes=no_users,
        no_of_topics=no_of_topics,
        seed=seed+1
        )
    susceptibility = build_susceptibility_matrix(
        nodes=no_users,
        seed=seed+2
        )
    return {"adjacency": adjacency,
            "prejudice": prejudice,
            "susceptibility": susceptibility}


def rng_degroot_model(no_users,
                      no_of_topics,
                      seed=1,
                      mean_node_degree=None,
                      max_tries=100):
    """ Returns a De Groot model with random adjacency and prejudice matrices
        for testing purposes; the model is resampled until it converge and
        reaches consensus.

        Parameters
        ----------
        no_users : int
            Number of users in the model.

        no_of_topics : int
            Number of topics considered in the simulation.

        seed : int, default=1
            Seed for the random number generator that drives the samples.

        mean_node_degree : float or None, default=None
            Mean degree of the nodes in the adjacency matrix.

        max_tries : int, default=100
            The maximum number of model resampling allowed before raising
            error.

    Returns
    -------
    DeGrootModel
        A DeGroot model whose both convergent and with consensus.

    Raises
    ------
    ValueError
        If no model with consensus is found within the given attempts.

    Notes
    -----
    In case of repeated failure in generating models with consensus, it's
    advised to gradually heighten the 'mean_node_degree' argument until the
    conditions are satisfied.
    """

    # I want to avoid changing randomness by simply using "seed+no_attemps"
    # because then consecutive inputs would yield the exact same matrices
    # (e.g. block of seeds 1-2-3 all reach seed = 4 where there is consensus)
    # -> changing the seeds between these would be irrelevant and confusing.
    rng = np.random.default_rng(seed)
    seed_list = rng.integers(low=0, high=max_tries*100, size=max_tries)

    for attempt in range(max_tries):
        matrix_dict = rng_model_matrices(no_users,
                                         no_of_topics,
                                         seed=int(seed_list[attempt]),
                                         mean_node_degree=mean_node_degree)

        model = DeGrootModel(adjacency=matrix_dict["adjacency"],
                             prejudice=matrix_dict["prejudice"])

        if model.is_there_consensus():
            return model

    raise ValueError("Couldn't generate a De Groot model with consensus "
                     f"in the given number of tries ({max_tries}).")


def rng_fj_model(no_users, no_of_topics, seed=1, mean_node_degree=None):
    """ Returns a Friedkin-Johnsen model with random adjacency, prejudice
        and susceptibility matrices for testing purposes;
        the model is convergent (see "Notes" section).

        Parameters
        ----------
        no_users : int
            Number of users in the model.

        no_of_topics : int
            Number of topics considered in the simulation.

        seed : int, default=1
            Seed for the random number generator that drives the samples.

        mean_node_degree : float or None, default=None
            Mean degree of the nodes in the adjacency matrix.

    Returns
    -------
    FJModel
        A convergent Friedkin_Johnsen model.

    Notes
    -----
    As long as the model does not fall in the DeGroot special case, the
    convergence is guaranteed thanks to the connectedness of the graph obtained
    by 'build_adjacency_matrix', which is sufficient condition on its own.
    The risk of accidentally falling in the De Groot case (by randomly sampling
    all susceptibilities as close to 1 as the tolerance requires) is considered
    to be negligible.
    """
    matrix_dict = rng_model_matrices(no_users,
                                     no_of_topics,
                                     seed=seed,
                                     mean_node_degree=mean_node_degree)
    return FJModel(adjacency=matrix_dict["adjacency"],
                   prejudice=matrix_dict["prejudice"],
                   susceptibility=matrix_dict["susceptibility"])


def degroot_model_from_lists(adjacency, prejudice):

    adjacency = csr_array(adjacency)
    prejudice = np.asarray(prejudice)

    return DeGrootModel(adjacency=adjacency,
                        prejudice=prejudice)


def fj_model_from_lists(adjacency, prejudice, susceptibility=None):

    adjacency = csr_array(adjacency)
    prejudice = np.asarray(prejudice)

    # When there are tests in common between Degroot and FJ,
    # usually FJ is under the degenerate case of
    # susceptibility = [1, 1, ... 1] anyways and this allows
    # for an easier parametrization.

    if susceptibility is None:
        susceptibility = [1]*len(prejudice)

    susceptibility = dia_array(np.diag(susceptibility))

    return FJModel(adjacency=adjacency,
                   prejudice=prejudice,
                   susceptibility=susceptibility)


def plot_opinions(model, sim, title=""):
    """ Saves a plot of the opinion evolution in a model simulation.
        The destination directory of file will depend on the model and
        its name will depend on the input title (optional) and the number
        of users.
    """

    fig, ax = plt.subplots()
    steps = sim.keys()
    values = sim.values()
    users_opinion_hist = [[value[i] for value in values]
                          for i in range(model.no_users)]

    for i, values in enumerate(users_opinion_hist):
        ax.plot(steps, values, linewidth=1, alpha=0.7)

    ax.set_xlabel('cycles')
    ax.set_ylabel('Opinion values')
    ax.grid(True, alpha=0.3)

    if isinstance(model, FJModel):
        model_name = "fj_model"
    else:
        model_name = "degroot_model"

    if title != "":
        title = title + "_"

    plot_directory = f"friedkin_johnsen/tests/plots/{model_name}/evolve_opinion"
    os.makedirs(plot_directory, exist_ok=True)
    fig.savefig(plot_directory + f"/{title}users_{model.no_users}.png",
                dpi=120)
    plt.close(fig)
