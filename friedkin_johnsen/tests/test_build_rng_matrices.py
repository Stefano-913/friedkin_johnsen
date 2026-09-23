import pytest
import sys
import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.stats import kstest

from ..build_rng_matrices import (build_adjacency_matrix,
                                  build_prejudice_matrix,
                                  build_susceptibility_matrix)


def csr_array_equal(array_a: sp.csr_array, array_b: sp.csr_array) -> bool:
    """ Compares two floating csr sparse matrices """
    return (array_a.shape == array_b.shape
            and np.all(array_a.indices == array_b.indices)
            and np.all(array_a.indptr == array_b.indptr)
            and np.allclose(array_a.data, array_b.data))


def dia_array_equal(array_a: sp.dia_array, array_b: sp.dia_array) -> bool:
    """ Compares two floating diagonal sparse arrays """
    return (array_a.shape == array_b.shape
            and np.array_equal(array_a.offsets, array_b.offsets)
            and np.allclose(array_a.data, array_b.data))


def mean_row_half_difference(matrix: sp.csr_array):
    half_differences = []

    for i in range(matrix.shape[0]):
        row_values = matrix.data
        half_differences.append((row_values.max() - row_values.min()) / 2)
    return sum(half_differences) / len(half_differences)


class TestAdjacencyMatrix:

    def test_default_adjacency_matrix(self, subtests):
        default_matrix = build_adjacency_matrix()

        with subtests.test(msg="has correct shape"):
            assert default_matrix.shape == (1000, 1000)

        with subtests.test(msg="has correct dimension"):
            assert default_matrix.ndim == 2

        with subtests.test(msg="is correct type"):
            assert np.issubdtype(default_matrix.dtype, float)

        with subtests.test(msg="is connected"):
            no_components = connected_components(csgraph=default_matrix,
                                                 directed=False)[0]
            assert no_components == 1

    @pytest.mark.parametrize("nodes, mean_node_degree", [
        (4, 3),
        (50, 4),
        (100, 6),
        (10_000, 40)
    ])
    @pytest.mark.parametrize("prob_rewire_edge", [0., 0.01, 0.5, 0.99, 1.])
    def test_adjacency_matrix(self, subtests,
                              nodes, mean_node_degree, prob_rewire_edge):
        matrix = build_adjacency_matrix(
            nodes=nodes,
            mean_node_degree=mean_node_degree,
            prob_rewire_edge=prob_rewire_edge,
            seed=None
        )
        with subtests.test(msg="is sparse"):
            assert sp.issparse(matrix)

        with subtests.test(msg="is csr format"):
            assert isinstance(matrix, sp.csr_array)

        with subtests.test(msg="has correct shape"):
            assert matrix.shape == (nodes, nodes)

        with subtests.test(msg="has correct dimension"):
            assert matrix.ndim == 2

        with subtests.test(msg="entries are correct type"):
            assert np.issubdtype(matrix.dtype, np.floating)

        # ----

        with subtests.test(msg="graph is undirected"):
            assert csr_array_equal(matrix, matrix.T)

        with subtests.test(msg="graph is connected"):
            n_components = connected_components(matrix, directed=False)[0]
            assert n_components == 1

        with subtests.test(msg="no self loops"):
            assert np.all(matrix.diagonal() == 0)

        # ----

        with subtests.test(msg="correct mean node degree"):
            # Edges are preserved by rewiring,
            # hence the mean degree is maintained
            no_edges = nodes * (mean_node_degree // 2)
            assert matrix.nnz == 2 * no_edges

    @pytest.mark.parametrize("weights_variability", [
        0,
        1,
        10,
        100,
        1000
    ])
    def test_weight_values_clamped(self, weights_variability):

        matrix = build_adjacency_matrix(nodes=10_000,
                                        weights_variability=weights_variability,
                                        seed=1)
        assert (matrix.max() <= 1 and matrix.min() >= 0)

    def test_low_weights_variability(self):
        matrix = build_adjacency_matrix(
                    nodes=10000,
                    weights_variability=0,
                    seed=1
                )
        assert mean_row_half_difference(matrix) < 0.2

    def test_stuff(self):
        variabilities = [0.01, 0.1, 1, 10, 100]

        semi_differences = [
            mean_row_half_difference(
                build_adjacency_matrix(nodes=200,
                                       weights_variability=variability,
                                       seed=1))
            for variability in variabilities
        ]

        assert np.all(np.diff(semi_differences) > 0)

    def test_is_every_node_influenced(self):
        matrix = build_adjacency_matrix(nodes=10_000,
                                        weights_variability=0,
                                        min_edge_weight=0.001,
                                        seed=1)
        # "effective_graph" verifies connectivity via actual influence
        # rather than edges alone, which is more relevant to the model
        matrix = nx.DiGraph(matrix)
        effective_graph = nx.DiGraph()
        effective_graph.add_nodes_from(matrix.nodes())
        effective_graph.add_edges_from(
            (u, v) for u, v, weight in matrix.edges(data='weight')
            if weight > 0.001)

        assert nx.is_strongly_connected(effective_graph)

    @pytest.mark.parametrize("even_k, odd_k", [
        (4, 5),
        (6, 7),
        (10, 11),
    ])
    def test_mean_node_degree_uses_floored_quotient(self, even_k, odd_k):
        # Test that k and k+1 build the same graph
        # (equivalent to mean_node_degree//2)

        k_even = build_adjacency_matrix(nodes=60,
                                        mean_node_degree=even_k,
                                        seed=1)

        k_odd = build_adjacency_matrix(nodes=60,
                                       mean_node_degree=odd_k,
                                       seed=1)

        assert k_even.nnz == k_odd.nnz

    @pytest.mark.parametrize("nodes", [100, 200, 500, 1000])
    @pytest.mark.parametrize("seed", [1, 2, 3, 4])
    def test_same_seed_same_output(self, nodes, seed):

        matrix_a = build_adjacency_matrix(nodes=nodes, seed=seed)
        matrix_b = build_adjacency_matrix(nodes=nodes, seed=seed)

        assert csr_array_equal(matrix_a, matrix_b)

    @pytest.mark.parametrize("nodes", [100, 200, 500, 1000])
    @pytest.mark.parametrize("seed_1, seed_2", [
        (None, None),
        (None, 2),
        (2, 3),
        (100, 101)])
    def test_different_seeds_different_output(self, nodes, seed_1, seed_2):

        matrix_a = build_adjacency_matrix(nodes=nodes, seed=seed_1)
        matrix_b = build_adjacency_matrix(nodes=nodes, seed=seed_2)

        assert not csr_array_equal(matrix_a, matrix_b)

    @pytest.mark.parametrize("name, value", [
        ("nodes", "100"),
        ("mean_node_degree", "5"),
        ("prob_rewire_edge", "0.5"),
        ("weights_variability", "0.8"),
        ("min_edge_weight", "4"),
        ("tries", "100"),
        ("seed", "1"),

        ("nodes", 53.5),
        ("mean_node_degree", 4.75),
        ("tries", 4.7),
        ("seed", 3.9),
    ])
    def test_input_typeerror(self, name, value):
        with pytest.raises(TypeError):
            build_adjacency_matrix(**{name: value})

    @pytest.mark.parametrize("nodes, mean_node_degree", [
        (2, 2),
        (4, 5),
        (-12, 11)
    ])
    def test_nodes_valueerror(self, nodes, mean_node_degree):
        # 'nodes' must be higher than 'mean_node_degree'
        with pytest.raises(ValueError, match="Invalid number of nodes"):
            build_adjacency_matrix(nodes=nodes,
                                   mean_node_degree=mean_node_degree)

    @pytest.mark.parametrize("mean_node_degree", [
        1,
        0,
        -1,
        -5
    ])
    def test_mean_node_degree_valueerror(self, mean_node_degree):
        # `mean_node_degree` must be equal or greater than 2
        with pytest.raises(ValueError, match="Invalid mean node degree value"):
            build_adjacency_matrix(mean_node_degree=mean_node_degree)

    @pytest.mark.parametrize("prob_rewire_edge", [
        -1,
        -0.5,
        1.1,
        10
    ])
    def test_prob_rewire_edge_valueerror(self, prob_rewire_edge):
        # `prob_rewire_edge` must be a valid probability ([0,1])
        with pytest.raises(ValueError,
                           match="Invalid probability of edge rewiring"):
            build_adjacency_matrix(prob_rewire_edge=prob_rewire_edge)

    @pytest.mark.parametrize("seed", [
        -10,
        -5,
        -1,
    ])
    def test_seed_valueerror(self, seed):
        # 'seed' must be non-negative
        with pytest.raises(ValueError, match="Invalid input seed"):
            build_adjacency_matrix(seed=seed)

    @pytest.mark.parametrize("tries", [
        -10,
        -1,
        0,
    ])
    def test_tries_valueerror(self, tries):
        # 'tries' must be higher than 0
        with pytest.raises(ValueError,
                           match="Invalid input 'number of tries'"):
            build_adjacency_matrix(tries=tries)

    @pytest.mark.parametrize("weights_variability", [
        -1,
        -0.5,
        -0.1,
    ])
    def test_negative_weights_variability_warn(self, weights_variability):
        # 'weights_variability' must be non-negative
        with pytest.warns(UserWarning, match="Invalid value for variability "
                          r"of edge weight:.* During process, the value "
                          "will be assumed to be 0."):
            build_adjacency_matrix(weights_variability=weights_variability)

    @pytest.mark.parametrize("min_edge_weight", [
        -1,
        -0.5,
        -0.1,
    ])
    def test_negative_min_edge_weight_warn(self, min_edge_weight):
        # 'min_edge_weight' must be non-negative
        with pytest.warns(UserWarning, match="Invalid value for minimum "
                          r"edge weight:.*During process, the value"
                          r"will be assumed to be the lowest possible.*"):
            build_adjacency_matrix(min_edge_weight=min_edge_weight)

    @pytest.mark.parametrize("min_edge_weight", [
        0.5,
        0.7,
        1,
    ])
    def test_high_min_edge_weight_warn(self, min_edge_weight):
        # 'min_edge_weight' must be non-negative
        with pytest.warns(UserWarning, match=r"'min_edge_weight' value is.*"
                          "which will result in all edges having the "
                          "same weight."):
            build_adjacency_matrix(min_edge_weight=min_edge_weight)

    def test_runtimeerror(self, monkeypatch):
        # A patch of the 'connected_watts_strogatz_graph', so that we simulate
        # the scenario in which it fails to generate a connected graph in the
        # given tries without having to rely on rng-fishing for testing.

        def raise_networkx_error(*args, **kwargs):
            raise nx.NetworkXError

        monkeypatch.setattr(
            "friedkin_johnsen.build_rng_matrices."
            "nx.connected_watts_strogatz_graph",
            raise_networkx_error)

        with pytest.raises(RuntimeError,
                           match="Could not generate a connected graph "
                           "after 1 tries"):
            build_adjacency_matrix(mean_node_degree=2,
                                   prob_rewire_edge=1,
                                   tries=1,
                                   seed=1)


class TestPrejudiceMatrix:

    def test_default_prejudice_matrix(self, subtests):
        default_matrix = build_prejudice_matrix()
        with subtests.test(msg="is correct shape"):
            assert default_matrix.shape == (1000, 1)

        with subtests.test(msg="is correct dimension"):
            assert default_matrix.ndim == 2

        with subtests.test(msg="is correct type"):
            assert np.issubdtype(default_matrix.dtype, float)

    @pytest.mark.parametrize("nodes, mean_prejudice, std_prejudice", [
        (4, 0.3, 0.4),
        (50, 0.54, 0.25),
        (100, 0.1, 1),
        (200, 0.3, 0),
        (10_000, 0.01, 0.001)
    ])
    def test_prejudice_matrix(self,
                              subtests,
                              nodes,
                              mean_prejudice,
                              std_prejudice):
        matrix = build_prejudice_matrix(
            nodes=nodes,
            mean_prejudice=mean_prejudice,
            std_prejudice=std_prejudice,
            seed=None
        )

        with subtests.test(msg="is np.ndarray type"):
            assert isinstance(matrix, np.ndarray)

        with subtests.test(msg="is correct shape"):
            assert matrix.shape == (nodes, 1)

        with subtests.test(msg="entries are correct type"):
            assert np.issubdtype(matrix.dtype, np.floating)

    @pytest.mark.parametrize("mean_prejudice, std_prejudice", [
        (0.5, 0.25),
        (0.75, 1),
        (0.01, 2),
        (5, 0.01)
    ])
    def test_prejudice_values_clamped(self, mean_prejudice, std_prejudice):
        matrix = build_prejudice_matrix(
            nodes=10000, mean_prejudice=mean_prejudice,
            std_prejudice=std_prejudice, seed=1
        )
        assert (matrix.max() <= 1 and matrix.min() >= 0)

    @pytest.mark.parametrize("mean_prejudice, std_prejudice", [
        (0.3, 0.1),
        (0.4, 0.133),
        (0.5, 0.166),
        (0.6, 0.133),
        (0.7, 0.1)
    ])
    def test_prejudice_values_normally_distributed(self,
                                                   mean_prejudice,
                                                   std_prejudice):
        # The standard deviations were chosen so that the clamping of the
        # values in (0, 1) started at a 3σ distance and so the truncation
        # statistically irrelevant.
        matrix = build_prejudice_matrix(
            nodes=10000, mean_prejudice=mean_prejudice,
            std_prejudice=std_prejudice, seed=1
        )
        values = matrix.ravel()
        z_values = (values - mean_prejudice) / std_prejudice
        statistic, p_value = kstest(z_values, 'norm')
        assert p_value > 0.05

    @pytest.mark.parametrize("nodes", [100, 200, 500, 1000])
    @pytest.mark.parametrize("seed", [1, 2, 3, 4])
    def test_same_seed_same_output(self, nodes, seed):

        matrix_a = build_prejudice_matrix(nodes=nodes, seed=seed)
        matrix_b = build_prejudice_matrix(nodes=nodes, seed=seed)

        assert np.allclose(matrix_a, matrix_b)

    @pytest.mark.parametrize("nodes", [100, 200, 500, 1000])
    @pytest.mark.parametrize("seed_1, seed_2", [
        (None, None),
        (None, 2),
        (2, 3),
        (100, 101)])
    def test_different_seeds_different_output(self, nodes, seed_1, seed_2):

        matrix_a = build_prejudice_matrix(nodes=nodes, seed=seed_1)
        matrix_b = build_prejudice_matrix(nodes=nodes, seed=seed_2)

        assert not np.allclose(matrix_a, matrix_b)

    @pytest.mark.parametrize("name, value", [
        ("nodes", "100"),
        ("no_of_topics", "5"),
        ("mean_prejudice", "0.8"),
        ("std_prejudice", "0.2"),
        ("seed", "1"),

        ("nodes", 53.5),
        ("no_of_topics", 7.4),
        ("seed", 3.9),
    ])
    def test_input_typeerror(self, name, value):
        with pytest.raises(TypeError):
            build_prejudice_matrix(**{name: value})

    @pytest.mark.parametrize("nodes", [
        -5,
        -1,
        0,
        2
    ])
    def test_nodes_valueerror(self, nodes):
        # 'nodes' must be equal or higher than 3
        with pytest.raises(ValueError, match="Invalid number of nodes"):
            build_prejudice_matrix(nodes=nodes)

    @pytest.mark.parametrize("no_of_topics", [
        -5,
        -1,
        0
    ])
    def test_no_of_topics_valueerror(self, no_of_topics):
        # 'no_of_topics' must be higher than 0
        with pytest.raises(ValueError, match="Invalid number of topics"):
            build_prejudice_matrix(no_of_topics=no_of_topics)

    @pytest.mark.parametrize("std_prejudice", [
        -10,
        -5,
        -1,
        -0.5
    ])
    def test_std_prejudice_valueerror(self, std_prejudice):
        # `std_prejudice` must be non-negative
        with pytest.raises(ValueError,
                           match="Invalid standard deviation of "
                           "prejudice value"):
            build_prejudice_matrix(std_prejudice=std_prejudice)

    @pytest.mark.parametrize("seed", [
        -10,
        -5,
        -1,
    ])
    def test_seed_valueerror(self, seed):
        # 'seed' must be non-negative
        with pytest.raises(ValueError, match="Invalid input seed"):
            build_prejudice_matrix(seed=seed)


class TestSusceptibilityMatrix:

    def test_default_susceptibility_matrix(self, subtests):
        default_matrix = build_susceptibility_matrix()
        with subtests.test(msg="is correct shape"):
            assert default_matrix.shape == (1000, 1000)

        with subtests.test(msg="is correct dimension"):
            assert default_matrix.ndim == 2

        with subtests.test(msg="is correct type"):
            assert np.issubdtype(default_matrix.dtype, float)

    @pytest.mark.parametrize(
            "nodes, mean_susceptibility, std_susceptibility",
            [
                (4, 0.3, 0.4),
                (50, 0.54, 0.25),
                (100, 0.1, 1),
                (200, 0.3, 0),
                (10_000, 0.01, 0.001)
            ])
    def test_susceptibility_matrix(self,
                                   subtests,
                                   nodes,
                                   mean_susceptibility,
                                   std_susceptibility):
        matrix = build_susceptibility_matrix(
            nodes=nodes,
            mean_susceptibility=mean_susceptibility,
            std_susceptibility=std_susceptibility,
            seed=None
        )

        with subtests.test(msg="is dia_array type"):
            assert isinstance(matrix, sp.dia_array)

        with subtests.test(msg="is correct shape"):
            assert matrix.shape == (nodes, nodes)

        with subtests.test(msg="entries are correct type"):
            assert np.issubdtype(matrix.dtype, np.floating)

    @pytest.mark.parametrize("mean_susceptibility, std_susceptibility", [
        (0.5, 0.25),
        (0.75, 1),
        (0.01, 2),
        (5, 0.01)
    ])
    def test_susceptibility_values_clamped(self,
                                           mean_susceptibility,
                                           std_susceptibility):
        matrix = build_susceptibility_matrix(
            nodes=10000, mean_susceptibility=mean_susceptibility,
            std_susceptibility=std_susceptibility, seed=1
        )
        assert (matrix.tocsr().max() <= 1 and matrix.tocsr().min() >= 0)

    @pytest.mark.parametrize("mean_susceptibility, std_susceptibility", [
        (0.3, 0.1),
        (0.4, 0.133),
        (0.5, 0.166),
        (0.6, 0.133),
        (0.7, 0.1)
    ])
    def test_susceptibility_values_normally_distributed(self,
                                                        mean_susceptibility,
                                                        std_susceptibility):
        # The standard deviations were chosen so that the clamping of the
        # values in (0, 1) started at a 3σ distance and so the truncation
        # statistically irrelevant.
        matrix = build_susceptibility_matrix(
            nodes=10000, mean_susceptibility=mean_susceptibility,
            std_susceptibility=std_susceptibility, seed=1
        )
        values = matrix.data
        z_values = (values - mean_susceptibility) / std_susceptibility
        statistic, p_value = kstest(z_values, 'norm', axis=None)
        assert p_value > 0.05

    @pytest.mark.parametrize("nodes", [100, 200, 500, 1000])
    @pytest.mark.parametrize("seed", [1, 2, 3, 4])
    def test_same_seed_same_output(self, nodes, seed):

        matrix_a = build_susceptibility_matrix(nodes=nodes, seed=seed)
        matrix_b = build_susceptibility_matrix(nodes=nodes, seed=seed)

        assert dia_array_equal(matrix_a, matrix_b)

    @pytest.mark.parametrize("nodes", [100, 200, 500, 1000])
    @pytest.mark.parametrize("seed_1, seed_2", [
        (None, None),
        (None, 2),
        (2, 3),
        (100, 101)
    ])
    def test_different_seeds_different_output(self, nodes, seed_1, seed_2):

        matrix_a = build_susceptibility_matrix(nodes=nodes, seed=seed_1)
        matrix_b = build_susceptibility_matrix(nodes=nodes, seed=seed_2)

        assert not dia_array_equal(matrix_a, matrix_b)

    @pytest.mark.parametrize("name, value", [
        ("nodes", "100"),
        ("mean_susceptibility", "0.8"),
        ("std_susceptibility", "0.2"),
        ("seed", "1"),

        ("nodes", 53.5),
        ("seed", 3.9),
    ])
    def test_input_typeerror(self, name, value):
        with pytest.raises(TypeError):
            build_susceptibility_matrix(**{name: value})

    @pytest.mark.parametrize("nodes", [
        -5,
        -1,
        0,
        2
    ])
    def test_nodes_valueerror(self, nodes):
        # 'nodes' must be equal or higher than 3
        with pytest.raises(ValueError, match="Invalid number of nodes"):
            build_susceptibility_matrix(nodes=nodes)

    @pytest.mark.parametrize("std_susceptibility", [
        -10,
        -5,
        -1,
        -0.5
    ])
    def test_std_susceptibility_valueerror(self, std_susceptibility):
        # `std_susceptibility` must be non-negative
        with pytest.raises(ValueError,
                           match="Invalid standard deviation of "
                           "susceptibility value"):
            build_susceptibility_matrix(std_susceptibility=std_susceptibility)

    @pytest.mark.parametrize("seed", [
        -10,
        -5,
        -1,
    ])
    def test_seed_valueerror(self, seed):
        # 'seed' must be non-negative
        with pytest.raises(ValueError, match="Invalid input seed"):
            build_susceptibility_matrix(seed=seed)


if __name__ == "__main__":
    sys.exit(pytest.main(
         ["-v", "--cov=your_package", "--cov-report=term-missing"]))
