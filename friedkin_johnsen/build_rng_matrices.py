import networkx as nx
import numpy as np
from scipy.sparse import diags_array
import warnings
from .error_messages import item_summary_for_error


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
            Probability of rewiring each edge, its value in [0,1].

        weights_variability : float, default=0.5
            Parameter of the Dirichlet distribution used to sample each node's
            outgoing edge weights, required to be non-negative.
            Low values produce near-uniform weights while high values produce
            more extreme, unevenly distributed weights.
            A minimum value of 0.001 is enforced regardless of input to
            guarantee connectivity, but the suggested range is
            roughly (0.01, 5).

        min_edge_weight : float, default=0.01
            Minimum possible weight for an edge.

        tries : int, default=100
            Number of attempts to generate a connected graph before
            raising an error.

        seed : int, optional
            The seed for the random number generator (None by default,
            which generates a random seed).

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

        Warns
        -----
        UserWarning
            If the number of `nodes` exceeds 100_000, the maximum
            capped value for memory and performance purposes.

        Notes
        -----
        Modifying the weights, one can functionally isolate users if formally
        present edges are set to have zero weight. To prevent this, a minimum
        edge value is chosen; conscious of the fact that in cases of very
        degree(~900) this de-facto nullifies the Dirichlet distribution and
        defaults to a uniform (and very low) weight distribution across
        all edges.
    """
    lowest_edge_weight = 0.001

    if type(mean_node_degree) is not int:
        raise TypeError("Invalid type for mean node degree: "
                        f"{item_summary_for_error(mean_node_degree)}. "
                        "Expected integer number.")
    if mean_node_degree < 2:
        raise ValueError("Invalid mean node degree value: "
                         f"{item_summary_for_error(mean_node_degree)}. "
                         "Expected integer number higher or equal than 2.")

    if type(nodes) is not int:
        raise TypeError("Invalid type for number of nodes: "
                        f"{item_summary_for_error(nodes)}. "
                        "Expected integer number.")
    if nodes <= mean_node_degree:
        raise ValueError("Invalid number of nodes: "
                         f"{item_summary_for_error(nodes)}. "
                         "Expected integer number higher than mean node "
                         f"degree (current value: {mean_node_degree}.)")

    if type(prob_rewire_edge) not in (float, int):
        raise TypeError("Invalid type for probability of edge rewiring: "
                        f"{item_summary_for_error(prob_rewire_edge)}. "
                        "Expected floating-point value.")
    if prob_rewire_edge < 0 or prob_rewire_edge > 1:
        raise ValueError("Invalid probability of edge rewiring: "
                         f"{item_summary_for_error(prob_rewire_edge)}. "
                         "Expected floating-point comprised between "
                         "0 and 1.")

    if type(weights_variability) not in (int, float):
        raise TypeError("Invalid type for variability of edge weight: "
                        f"{item_summary_for_error(weights_variability)}. "
                        "Expected a floating point.")
    if weights_variability < 0:
        warnings.warn("Invalid value for variability of edge weight: "
                      f"{weights_variability}. "
                      "During process, the value will be assumed to be 0.")

    if type(min_edge_weight) not in (int, float):
        raise TypeError("Invalid type for minimum edge weight: "
                        f"{item_summary_for_error(min_edge_weight)}. "
                        "Expected a floating point.")
    if min_edge_weight < 0:
        warnings.warn("Invalid value for minimum edge weight: "
                      f"{min_edge_weight}. During process, the "
                      "value will be assumed to be the lowest "
                      f"possible ({lowest_edge_weight}).")
    if min_edge_weight >= 0.5:
        warnings.warn(f"'min_edge_weight' value is {min_edge_weight}, "
                      "which will result in all edges having the same weight."
                      "\nSee --help 'Notes' for further informations, "
                      "otherwise launch with --no_warn command to "
                      "disable warnings.")

    if seed is not None and type(seed) is not int:
        raise TypeError("Invalid type for input seed: "
                        f"{item_summary_for_error(seed)}. "
                        "Expected either 'None' or integer number.")
    if type(seed) is int and seed < 0:
        raise ValueError("Invalid input seed: "
                         f"{item_summary_for_error(seed)}. "
                         "Expected either 'None' or non-negative integer.")

    if type(tries) is not int:
        raise TypeError("Invalid type for input 'number of tries': "
                        f"{item_summary_for_error(tries)}. "
                        "Expected integer number.")
    if tries <= 0:
        raise ValueError("Invalid input 'number of tries': "
                         f"{item_summary_for_error(tries)}. "
                         "Expected positive integer number.")

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


def build_prejudice_matrix(nodes: int = 1000,
                           no_of_topics: int = 1,
                           mean_prejudice: float = 0.5,
                           std_prejudice: float = 0.25,
                           seed: int = None):

    """ Build a sparse matrix representing prejudice values in a
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
            - 'seed' must be non-negative;

        Warns
        -----
        UserWarning
            If the number of `nodes` exceeds 100_000, the maximum
            capped value for memory and performance purposes.
    """

    if type(nodes) is not int:
        raise TypeError("Invalid type for number of nodes: "
                        f"{item_summary_for_error(nodes)}. "
                        "Expected integer number.")
    if nodes < 3:
        raise ValueError("Invalid number of nodes: "
                         f"{item_summary_for_error(nodes)}. "
                         "Expected positive integer number equal "
                         "or higher than 3.")

    if type(no_of_topics) is not int:
        raise TypeError("Invalid type for number of topics: "
                        f"{item_summary_for_error(nodes)}. "
                        "Expected integer number.")
    if no_of_topics <= 0:
        raise ValueError("Invalid number of topics: "
                         f"{item_summary_for_error(nodes)}. "
                         "Expected positive integer number.")

    if (type(mean_prejudice) not in (int, float)):
        raise TypeError("Invalid mean prejudice value: "
                        f"{item_summary_for_error(mean_prejudice)}. "
                        "Expected a floating point.")

    if type(std_prejudice) not in (int, float):
        raise TypeError("Invalid type for standard deviation of prejudice "
                        f"value: {item_summary_for_error(std_prejudice)}. "
                        "Expected a floating point.")
    if std_prejudice < 0:
        raise ValueError("Invalid standard deviation of prejudice value: "
                         f"{item_summary_for_error(std_prejudice)}. "
                         "Expected a non-negative floating point.")

    if seed is not None and type(seed) is not int:
        raise TypeError(f"Invalid type for input seed: "
                        f"{item_summary_for_error(seed)}. "
                        "Expected either 'None' or integer number.")
    if type(seed) is int and seed < 0:
        raise ValueError(f"Invalid input seed: "
                         f"{item_summary_for_error(seed)}. "
                         "Expected either 'None' or non-negative integer.")

    rng = np.random.default_rng(seed)

    return np.clip(rng.normal(loc=mean_prejudice,
                              scale=std_prejudice,
                              size=(nodes, no_of_topics)),
                   a_min=0, a_max=1)


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
        scipy.sparse.csr_array
            Floating-point square matrix of shape (nodes, nodes).

        Raises
        ------
        TypeError
            If any of the inputs do not align with the type prescribed above.

        ValueError
            If the constraints on the input values are not respected, namely:
            - 'nodes' must be equal or higher than 3;
            - `std_susceptibility` must be non-negative;
            - 'seed' must be non-negative;

        Warns
        -----
        UserWarning
            If the number of `nodes` exceeds 100,000, the maximum
            capped value for memory and performance purposes.
    """

    if type(nodes) is not int:
        raise TypeError("Invalid type for number of nodes: "
                        f"{item_summary_for_error(nodes)}. "
                        "Expected integer number.")
    if nodes < 3:
        raise ValueError("Invalid number of nodes: "
                         f"{item_summary_for_error(nodes)}. "
                         "Expected positive integer number equal or "
                         "higher than 3.")

    if (type(mean_susceptibility) not in (int, float)):
        raise TypeError("Invalid mean susceptibility value: "
                        f"'{type(mean_susceptibility)}'. "
                        "Expected a floating point.")

    if type(std_susceptibility) not in (int, float):
        raise TypeError("Invalid type for standard deviation of "
                        f"susceptibility value: '{type(std_susceptibility)}'. "
                        "Expected a floating point.")

    if std_susceptibility < 0:
        raise ValueError("Invalid standard deviation of susceptibility value: "
                         f"'{std_susceptibility}'. "
                         "Expected a non-negative floating point.")

    if seed is not None and type(seed) is not int:
        raise TypeError(f"Invalid type for input seed: '{type(seed)}'. "
                        "Expected either 'None' or integer number.")

    if type(seed) is int and seed < 0:
        raise ValueError(f"Invalid input seed: '{seed}'. "
                         "Expected either 'None' or non-negative integer.")

    rng = np.random.default_rng(seed)

    susceptibilities = np.clip(
        rng.normal(loc=mean_susceptibility,
                   scale=std_susceptibility,
                   size=(nodes, 1)),
        a_min=0, a_max=1)
    return diags_array(susceptibilities.flatten())
