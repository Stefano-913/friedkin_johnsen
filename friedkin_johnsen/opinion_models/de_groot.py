from .base_model import OpinionModel
import numpy as np
from scipy.sparse import csr_array
import scipy as sp


class DeGrootModel(OpinionModel):
    """ Class that represents a De Groot continuous opinion model, that
        tries to represent the evolution and spreading of opinions
        inside a population of agents/users.

        At its core, the model is comprised of two elements:
        -   an adjacency matrix W, that describes the relationships
            amongst the users;
        -   a prejudice (or opinion) vector X, that describes the
            opinions of the individual users as numerical values on a
            continuous slider ([0,1]).

        The evolution of the opinions of the users is hence calculated
        as a simple multiplication between vector and matrix:
            X(k+1) = W X(k)     with k evolution parameter of the system

        The adjacency matrix is required to have non-negative entries,
        be squared and be stochastic.

        While usually the users' opinions don't have mathematical
        boundaries, here they will instead be limited to a [0,1] interval.
        Moreover, to simulate the indipendent evolution of multiple topics
        at once, the prejudice vector simply becomes a matrix of size
        (no_users, no_topics) with unchanged evolution formula.

        Parameters
        ----------
        adjacency : csr_array
            Adjacency matrix used by the model, assumed to have non-negative
            values, be squared and be stochastic.
            Input will not be validated thoroughly, as it's expected to already
            be validated in get_matrices().

        prejudice : np.ndarray
            Starting opinions used by the model, expected to be confined in the
            [0,1] numeric interval. To simulate the indipendent evolution of
            multiple topics at once, the prejudice vector can become a
            matrix of size (no_users, no_topics).
            Input will not be validated thoroughly, as it's expected to already
            be validated in get_matrices().

        Raises
        ------
        TypeError
            If either adjacency or prejudice aren't of their respective
            expected type (csr_array or np.ndarray respectively).

        Attributes
        ----------
        adjacency : csr_array
            Adjacency matrix used by the model, with non-negative values and
            stochastic.

        prejudice : np.ndarray
            Starting opinions used by the model, confined in the
            [0,1] numeric interval.

        eigenvalues : np.ndarray[float]
            Eigenvalues of the adjacency matrix.

        convergence_rate : float
            Rate of convergence of the model to the limit solution, comprised
            between 0 and 1 (if the convergence rate is =1, the model
            never converges).
    """

    def __init__(self,
                 adjacency: csr_array,
                 prejudice: np.ndarray):

        super().__init__(adjacency, prejudice)

    def _evolve(self, X: np.ndarray) -> np.ndarray:
        """ Evolution step of the De Groot model:
                X(k+1) = W X(k),    with W adjacency matrix
                                    and X(k) user opinion.

            Params
            ------
            X : np.ndarray
                Current opinion.

            Returns
            -------
            X(k+1) : np.ndarray
                The next step in the simulation

            References
            ----------
            Mohammad Shirzadi, Emilio Cruciani, Ahad N. Zehmakan,
            "Opinion Dynamics: A Comprehensive Overview",
            section 2.3.1 at page 14.
        """
        return self.adjacency @ X

    def is_convergent(self, tol=1e-10) -> bool:
        """ Calculates if the De Groot model is convergent;
            characteristic achieved if and only if λ = 1, guaranteed
            by the stochasticity, is the only one of the adjacency
            matrix on the unit circle: {λ ∈ C : |λ| = 1}.

            Parameters
            ----------
            tol : float, default = 1e-10
                Error tolerance to decide whether or not an eigenvalue
                belongs on the circle or not (required to be non-negative).

            Returns
            -------
            bool

            Raises
            ------
            TypeError
                If 'tol' input isn't of float type.
            ValueError
                If 'tol' input is negative.

            References
            ----------
            Mohammad Shirzadi, Emilio Cruciani, Ahad N. Zehmakan,
            "Opinion Dynamics: A Comprehensive Overview", page 23.
            """
        if not isinstance(tol, (int, float)):
            raise TypeError("Invalid type of tolerance for convergence/"
                            "consensus test of De Groot model: "
                            f"'{type(tol)}'. Expected a floating-point.")
        if tol < 0:
            raise ValueError("Invalid value of tolerance for convergence/"
                             f"consensus test of De Groot model: '{tol}'. "
                             "Expected a non-negative floating-point.")

        eigenvalues = self.eigenvalues
        eigens_on_circle = np.isclose(np.abs(eigenvalues), 1, atol=tol, rtol=0)
        only_eigen_one_on_circle = np.all(
            np.isclose(eigenvalues[eigens_on_circle], 1, atol=tol, rtol=0)
            )
        if not only_eigen_one_on_circle:
            return False

        return True

    def is_there_consensus(self, tol=10e-10) -> bool:
        """ Calculates if the De Groot model's final equilibrium is in a
            state of consensus.
            The corresponding mathematical condition is the simplicity
            (algebraic multeplicity of one) of the eigenvalue λ=1, whose
            existence is guaranteed by the stochasticity of the
            adjacency matrix.

            [ A convergence test is performed first, as without it consensus
            is ill-defined. ]

            Parameters
            ----------
            tol : float, default = 1e-10
                Error tolerance to decide whether or not an eigenvalue
                is close to "1" (required to be non-negative).

            Returns
            -------
            bool

            Raises
            ------
            TypeError
                If 'tol' input isn't of float type.
            ValueError
                If 'tol' input is negative.

            References
            ----------
            Mohammad Shirzadi, Emilio Cruciani, Ahad N. Zehmakan,
            "Opinion Dynamics: A Comprehensive Overview", page 23.
        """

        # The 'tol' checks are done in is_convergent
        if not self.is_convergent(tol=tol):
            return False

        no_eigens_equal_to_one = np.sum(np.isclose(self.eigenvalues,
                                                   1,
                                                   atol=tol,
                                                   rtol=0))
        return no_eigens_equal_to_one == 1

    def calculate_limit_solution(self, tol=1e-10) -> np.ndarray | None:
        """ Calculates the limit solution of a De Groot model, after checking
            in advance for both its convergence and potential of reaching
            consensus.

            If the model is either non convergent or unable to reach consensus,
            the function simply returns 'None'.

            The formula used to calculate the limit solution, applicable only
            in case of consensus (uniform final opinion, 'c') is the following:

                c = <w, x_0>,   with 'x_0' starting opinions
                                and 'w' the normalized left eigenvector of
                                the adjacency matrix associated with
                                the λ=1 eigenvalue.

            Parameters
            ----------
            tol : float, default = 1e-10
                Error tolerance to decide the closeness of a value to "1" for
                both the test of convergence (compares eigenvalue magnitude
                to the unit circle's radiud) and the test of consensus
                (compares eigenvalues to one).
                Required to be non-negative.

            Returns
            -------
            limit solution : np.ndarray | None
                If there is no convergence or consensus, None is returned.

            Raises
            ------
            TypeError
                If 'tol' input isn't of float type.
            ValueError
                If 'tol' input is negative or if the limit solution indicates
                a wrong input in the model (the user opinions in the limit
                aren't in the [0,1] interval).

            See Also
            --------
            is_convergent

            References
            ----------
            Mohammad Shirzadi, Emilio Cruciani, Ahad N. Zehmakan;
            "Opinion Dynamics: A Comprehensive Overview", page 23.

            Yucheng Dong, Zhaogang Ding, Gang Kou;
            "Social Network DeGroot Model", theorem 1.3 at page 7.
        """

        # The 'tol' checks are done with the convergence test
        if not self.is_there_consensus(tol=tol):
            return None

        W = self.adjacency

        eigenvalues, left_eigenvecs = sp.sparse.linalg.eigs(W.T,
                                                            k=1,
                                                            which="LM")
        w_vect = np.real(left_eigenvecs[:, 0])
        w_vect /= w_vect.sum()
        consensus = w_vect @ self.prejudice

        if not np.allclose(consensus, np.clip(consensus, 0, 1)):
            raise ValueError("Consensus calculation yielded an unexpected "
                             "result, possibly due to inadequate "
                             "input matrices of the model.")

        return np.full(self.prejudice.shape, consensus)

    @property
    def convergence_rate(self) -> float:
        """ Calculates the convergence rate of a De Groot model,
            that corresponds to the absolute value of the second
            highest eigenvalue λ_2 (in module).

            The convergence speed can then be expressed as:
                |x* - x(k)| <= D*|λ_2|^k    for k->inf

            where x* is the limit_solution, x(k) is the current
            iteration of the opinion vector and D is a constant >=0.

            Returns
            -------
            convergence rate (|λ_2|): float within 0 and 1
                Note that a convergence rate of '1' is to be interpreted
                as the model never converging.

            Raises
            ------
            ValueError
                If the calculated convergence rate isn't in the [0,1] interval.

            References
            ----------
            Yucheng Dong, Zhaogang Ding, Gang Kou;
            "Social Network DeGroot Model", theorem 1.4 at page 7.
        """
        if not hasattr(self, "_convergence_rate"):

            # self.eigenvalues are already in decreasing order of magnitude
            self._convergence_rate = np.abs(self.eigenvalues[1])

            # Surely not negative due to absolute value, so only <=1 check
            if not (self._convergence_rate < 1 or
                    np.allclose(self._convergence_rate, 1)):
                raise ValueError("Convergence rate calculation yielded an "
                                 "unexpected result, possibly due to "
                                 "inadequate input matrices of the model.")

        return self._convergence_rate
