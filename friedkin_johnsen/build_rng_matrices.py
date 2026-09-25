import networkx as nx
import numpy as np
from scipy.sparse import diags_array
import warnings
from .error_messages import item_summary_for_error


def check_nodes(nodes, mean_node_degree=None):
    # In the adjacency matrix, the 'mean node degree' parameter
    # adds a requirement.
    # Susceptibility and prejudice matrices instead
    # keep the plain 'nodes >= 3' condition.

    if type(nodes) is not int:
        raise TypeError("Invalid type for number of nodes: "
                        f"{type(nodes)}. Expected integer number.")

    if mean_node_degree is not None:
        if nodes <= mean_node_degree:
            raise ValueError("Invalid number of nodes: "
                             f"{nodes}. "
                             "Expected integer number higher than mean node "
                             f"degree (current value: {mean_node_degree}).")
    if nodes < 3:
        raise ValueError("Invalid number of nodes: "
                         f"{nodes}. Expected positive integer number "
                         f"equal or higher than 3.")


def check_seed(seed):
    if seed is not None and type(seed) is not int:
        raise TypeError(f"Invalid type for input seed: '{type(seed)}'. "
                        "Expected either 'None' or integer number.")
    if type(seed) is int and seed < 0:
        raise ValueError(f"Invalid input seed: '{seed}'. "
                         "Expected either 'None' or non-negative integer.")

# ------


def check_adjacency_input(nodes,
                          mean_node_degree,
                          prob_rewire_edge,
                          weights_variability,
                          min_edge_weight,
                          tries,
                          seed,
                          lowest_edge_weight):

    if type(mean_node_degree) is not int:
        raise TypeError("Invalid type for mean node degree: "
                        f"{item_summary_for_error(mean_node_degree)}. "
                        "Expected integer number.")
    if mean_node_degree < 2:
        raise ValueError("Invalid mean node degree value: "
                         f"{mean_node_degree}. "
                         "Expected integer number higher or equal than 2.")

    check_nodes(nodes, mean_node_degree=mean_node_degree)

    if type(prob_rewire_edge) not in (float, int):
        raise TypeError("Invalid type for probability of edge rewiring: "
                        f"{item_summary_for_error(prob_rewire_edge)}. "
                        "Expected floating-point value.")
    if prob_rewire_edge < 0 or prob_rewire_edge > 1:
        raise ValueError("Invalid probability of edge rewiring: "
                         f"{prob_rewire_edge}. "
                         "Expected floating-point comprised between "
                         "0 and 1.")

    if type(weights_variability) not in (int, float):
        raise TypeError("Invalid type for variability of edge weight: "
                        f"{item_summary_for_error(weights_variability)}. "
                        "Expected a floating point.")
    if weights_variability < 0:
        warnings.warn("Invalid value for variability of edge weight: "
                      f"{weights_variability}. "
                      "During process, the value will be assumed to be 0.",
                      UserWarning)

    if type(min_edge_weight) not in (int, float):
        raise TypeError("Invalid type for minimum edge weight: "
                        f"{item_summary_for_error(min_edge_weight)}. "
                        "Expected a floating point.")
    if min_edge_weight < 0:
        warnings.warn("Invalid value for minimum edge weight: "
                      f"{min_edge_weight}. During process, the "
                      "value will be assumed to be the lowest "
                      f"possible ({lowest_edge_weight}).", UserWarning)
    if min_edge_weight >= 1/mean_node_degree:
        warnings.warn(f"'min_edge_weight' value is {min_edge_weight}, "
                      "while the mean degree of every node is "
                      f"{mean_node_degree}. "
                      "These values will result in a significant portion "
                      "of the edges having the same weight.", UserWarning)
    if min_edge_weight >= 0.5:
        warnings.warn(f"'min_edge_weight' value is {min_edge_weight}, "
                      "which will result in all edges having the same weight.",
                      UserWarning)

    if seed is not None and type(seed) is not int:
        raise TypeError("Invalid type for input seed: "
                        f"{item_summary_for_error(seed)}. "
                        "Expected either 'None' or integer number.")
    if type(seed) is int and seed < 0:
        raise ValueError("Invalid input seed: "
                         f"{seed}. "
                         "Expected either 'None' or non-negative integer.")

    if type(tries) is not int:
        raise TypeError("Invalid type for input 'number of tries': "
                        f"{item_summary_for_error(tries)}. "
                        "Expected integer number.")
    if tries <= 0:
        raise ValueError("Invalid input 'number of tries': "
                         f"{tries}. "
                         "Expected positive integer number.")


def build_adjacency_matrix(nodes: int = 1000,
                           mean_node_degree: int = 6,
                           prob_rewire_edge: float = 0.1,
                           weights_variability: float = 0.5,
                           min_edge_weight: float = 0.01,
                           tries: int = 100,
                           seed: int | None = None):
    """ Build a sparse directed adjacency matrix for a Friedkin-Johnsen model,
        using the Watts-Strogatz algorithm to generate a small-world graph
        with a connected graph of users to which weights are later added.

        The graph starts from a ring lattice, where each node is
        connected to a fixed number of its neighbours and its edges randomly
        rewired until the resulting graph is connected.
        The values of the edge weights for each node are drawn from a
        Dirichlet distribution, which naturally produces a stochastic matrix,
        with a set minimum edge weight so that influence remains relevant and
        no edge gets functionally erased by an especially low sampling.

        Parameters
        ----------
        nodes : int, default=1000
            Number of nodes in the graph, which must be greater than
            `mean_node_degree`.

        mean_node_degree : int, default=6
            Mean degree of each node (or `k` in the Watts-Strogatz notation).
            It's required to be an integer equal or greater than 2 and it's
            important to also node that the actual value used in the
            algorithm is k//2 (so, k=4 yields the same result as k=5 ).

        prob_rewire_edge : float, default=0.1
            Probability of rewiring each edge, its value required to be
            in [0,1].

        weights_variability : float, default=0.5
            Parameter of the Dirichlet distribution used to sample each node's
            outgoing edge weights, required to be non-negative.
            Low values produce near-uniform weights while high values produce
            more extreme, unevenly distributed weights.
            A minimum value of 0.001 is enforced regardless of input to
            guarantee connectivity, but the suggested range is
            roughly (0.01, 5).

        min_edge_weight : float, default=0.01
            Minimum possible weight for an edge. If the minimum edge weight
            is set equal or higher than the degree of a node, all its edges
            will have the same value 1/degree due to normalization constraints.

        tries : int, default=100
            Number of attempts to generate a connected graph before
            raising an error, must be greater than 0.

        seed : int, optional
            The seed for the random number generator, required to be
            non-negative if given (None by default, which generates a
            random seed).

        Returns
        -------
        scipy.sparse.csr_array
            A square floating-point sparse array which is connected and
             stochastic; has shape (nodes, nodes) and weighted edges.

        Raises
        ------
        TypeError
            If any of the inputs do not align with the type prescribed above.

        ValueError
            If the constraints on the input values are not respected, namely:
            - 'nodes' must be higher than 'mean_node_degree';
            - `mean_node_degree` must be equal or greater than 2;
            - `prob_rewire_edge` must be a valid probability ([0,1]);
            - `std_edge_weight` must be non-negative;
            - 'seed' must be non-negative;
            - 'tries' must be higher than 0.

        RuntimeError
            If a connected graph could not be generated within `tries`
            attempts.

        Notes
        -----
        Modifying the weights, one can functionally isolate users if formally
        present edges are set to have zero weight. To prevent this, a minimum
        edge value is chosen; conscious of the fact that in cases of very high
        degree (or very high minimum) this de-facto nullifies the Dirichlet
        distribution and defaults to a uniform weight distribution of 1/degree
        across the majority of the edges.
    """
    lowest_edge_weight = 0.001

    check_adjacency_input(nodes=nodes,
                          mean_node_degree=mean_node_degree,
                          prob_rewire_edge=prob_rewire_edge,
                          weights_variability=weights_variability,
                          min_edge_weight=min_edge_weight,
                          tries=tries,
                          seed=seed,
                          lowest_edge_weight=lowest_edge_weight)

    rng = np.random.default_rng(seed)

    try:
        G = nx.connected_watts_strogatz_graph(n=nodes,
                                              k=mean_node_degree,
                                              tries=tries,
                                              p=prob_rewire_edge,
                                              seed=rng)
    except nx.NetworkXError as e:
        raise RuntimeError("Could not generate a connected graph "
                           f"after {tries} tries: {e}")

    directed_graph = G.to_directed()

    if min_edge_weight is None or min_edge_weight < lowest_edge_weight:
        min_edge_weight = lowest_edge_weight

    alpha = 1/np.clip(weights_variability, a_min=0.01, a_max=100)
    node_degrees = G.degree()
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        row_weights = rng.dirichlet([alpha]*node_degrees[node])

        if min(row_weights) < min_edge_weight:
            row_weights = np.clip(row_weights, min_edge_weight, None)
            row_weights /= row_weights.sum()

        for neighbor, weight in zip(neighbors, row_weights):
            directed_graph[node][neighbor]['weight'] = weight

    return nx.adjacency_matrix(directed_graph, dtype=float, weight='weight')


def check_prejudice_input(nodes,
                          no_of_topics,
                          mean_prejudice,
                          std_prejudice,
                          seed):

    check_nodes(nodes)

    if type(no_of_topics) is not int:
        raise TypeError("Invalid type for number of topics: "
                        f"{item_summary_for_error(no_of_topics)}. "
                        "Expected integer number.")
    if no_of_topics <= 0:
        raise ValueError("Invalid number of topics: "
                         f"{no_of_topics}. "
                         "Expected positive integer number.")

    if type(mean_prejudice) not in (int, float):
        raise TypeError("Invalid mean prejudice value: "
                        f"{item_summary_for_error(mean_prejudice)}. "
                        "Expected a floating point.")

    if type(std_prejudice) not in (int, float):
        raise TypeError("Invalid type for standard deviation of prejudice "
                        f"value: {item_summary_for_error(std_prejudice)}. "
                        "Expected a floating point.")
    if std_prejudice < 0:
        raise ValueError("Invalid standard deviation of prejudice value: "
                         f"{std_prejudice}. "
                         "Expected a non-negative floating point.")

    check_seed(seed)


def build_prejudice_matrix(nodes: int = 1000,
                           no_of_topics: int = 1,
                           mean_prejudice: float = 0.5,
                           std_prejudice: float = 0.25,
                           seed: int = None):

    """ Build a dense matrix representing prejudice values in a
        Friedkin-Johnsen model (with custom number of topics).

        The prejudice values are drawn from a normal distribution and
        clipped to [0, 1].

        Parameters
        ----------
        nodes : int, default=1000
            Number of nodes in the graph, required to be equal or
            higher than 3.

        no_of_topics : int, default=1
            Number of topics considered in the model. The output matrix shape
            will vary according to this parameter, as (nodes, no_of_topics),
            notably with (nodes, 1) as default case.

        mean_prejudice : float, default=0.5
            Mean of the normal distribution used to sample prejudice values.
            Albeit the weight is capped at 1, it's possible for
            mean_prejudice value to be set higher than that if the
            clipping is deemed desirable.

        std_prejudice : float, default=0.25
            Standard deviation of the normal distribution used to sample
            prejudice values, required to be non-negative.

        seed : int, optional
            The seed for the random number generator (None by default,
            which generates a random seed).

        Returns
        -------
        numpy.ndarray
            Floating-point array of shape (nodes, no_of_prejudices).

        Raises
        ------
        TypeError
            If any of the inputs do not align with the type prescribed above.

        ValueError
            If the constraints on the input values are not respected, namely:
            - 'nodes' must be equal or higher than 3;
            - 'no_of_topics' must be higher than 0;
            - `std_prejudice` must be non-negative;
            - 'seed' must be non-negative.
    """

    check_prejudice_input(nodes=nodes,
                          no_of_topics=no_of_topics,
                          mean_prejudice=mean_prejudice,
                          std_prejudice=std_prejudice,
                          seed=seed)

    rng = np.random.default_rng(seed)

    return np.clip(rng.normal(loc=mean_prejudice,
                              scale=std_prejudice,
                              size=(nodes, no_of_topics)),
                   a_min=0, a_max=1)


def check_susceptibility_input(nodes, mean_susceptibility,
                               std_susceptibility, seed):
    check_nodes(nodes)

    if type(mean_susceptibility) not in (int, float):
        raise TypeError(f"Invalid mean susceptibility value: "
                        f"'{item_summary_for_error(mean_susceptibility)}'. "
                        "Expected a floating point.")

    if type(std_susceptibility) not in (int, float):
        raise TypeError(f"Invalid type for standard deviation of "
                        "susceptibility value: "
                        f"'{item_summary_for_error(std_susceptibility)}'. "
                        "Expected a floating point.")
    if std_susceptibility < 0:
        raise ValueError(f"Invalid standard deviation of susceptibility "
                         f"value: '{std_susceptibility}'. "
                         "Expected a non-negative floating point.")

    check_seed(seed)


def build_susceptibility_matrix(nodes: int = 1000,
                                mean_susceptibility: float = 0.5,
                                std_susceptibility: float = 0.25,
                                seed: int | None = None):
    """ Build a sparse matrix representing susceptibility values in a
        Friedkin-Johnsen model (with custom number of topics).

        The susceptibility values are drawn from a normal distribution and
        clipped to [0, 1].

        Parameters
        ----------
        nodes : int, default=1000
            Number of nodes in the graph, required to be equal or higher than
            3.

        mean_susceptibility : float, default=0.5
            Mean of the normal distribution used to sample
            susceptibility values.
            Albeit the weight is capped at 1, it's possible for
            mean_susceptibility value to be set higher than that if the
            clipping is deemed desirable.

        std_susceptibility : float, default=0.25
            Standard deviation of the normal distribution used to sample
            susceptibility values, required to be non-negative.

        seed : int, optional
            The seed for the random number generator (None by default,
            which generates a random seed).

        Returns
        -------
        scipy.sparse.dia_array
            Floating-point diagonal matrix of shape (nodes, nodes).

        Raises
        ------
        TypeError
            If any of the inputs do not align with the type prescribed above.

        ValueError
            If the constraints on the input values are not respected, namely:
            - 'nodes' must be equal or higher than 3;
            - `std_susceptibility` must be non-negative;
            - 'seed' must be non-negative.
    """

    check_susceptibility_input(nodes=nodes,
                               mean_susceptibility=mean_susceptibility,
                               std_susceptibility=std_susceptibility,
                               seed=seed)

    rng = np.random.default_rng(seed)

    susceptibilities = np.clip(
        rng.normal(loc=mean_susceptibility,
                   scale=std_susceptibility,
                   size=(nodes, 1)),
        a_min=0, a_max=1)
    return diags_array(susceptibilities.flatten())
