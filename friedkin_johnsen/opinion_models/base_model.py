import numpy as np
from scipy.sparse import csr_array
from abc import ABC, abstractmethod
from ..matrix_check import highest_eigenvalues, get_number_of_users


class OpinionModel(ABC):
    """ Abstract class for opinion models.
        The derived models are assumed to make use of both an adjacency square
        matrix (that details the agents/users interactions) and a prejudice
        matrix, that states the users' starting positions.
        Subclasses are expected to implement their own '_evolve()' method
        according to the model they represent, as well as provide methods to
        calculate their limit solution, convergence test and convergence rate.

        The built-in 'simulate()' method assumes that the model does *not* have
        a time-varying adjacency matrix and demands that any variable in the
        evolution formula other than the current opinion to be stored as
        attributes of the subclass and accessed throught 'self.' inside the
        '_evolve()' method.
        (e.g. if the update rule were WX,   with W=adjacency_matrix,
                                            and X=current opinion
            def _evolve(self, X):
                return self.adjacency@X
        ).

        Parameters
        ----------
        adjacency : csr_array
            Adjacency matrix used by the model, assumed to have non-negative
            values and be stochastic.
            Input will not be validated thoroughly, as it's expected to already
            be validated in get_matrices().

        prejudice : np.ndarray
            Starting opinions used by the model, expected to be confined in the
            [0,1] numeric interval.
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

        if not isinstance(adjacency, csr_array):
            raise TypeError(f"Adjacency matrix is of {type(adjacency)} "
                            "type, expected 'csr_array'.")
        if not isinstance(prejudice, np.ndarray):
            raise TypeError(f"Prejudice matrix is of {type(prejudice)} "
                            "type, expected 'np.ndarray'.")

        self.adjacency = adjacency
        self.prejudice = prejudice

        self.no_users = get_number_of_users(adjacency)

    @abstractmethod
    def _evolve(self, X):
        """ Method that defines the evolution step of the model.

            The only external parameter allowed is the current status
            of opinion. If the evolution of the models in the subclasses
            depends on any other variable (adjacency matrix,
            prejudice etc.), it will have to be categorized as an
            attribute and accessed through 'self.'.
            (e.g. if the update rule were WX,   with W=adjacency_matrix,
                                                        and X=current opinion
                        def _evolve(self, X):
                            return self.adjacency@X
            ).

            Params
            ------
            X : np.ndarray
                Current opinion

            Returns
            -------
                The next step of the simulation.
        """
        pass

    def simulate(self,
                 steps: int = 10_000,
                 snapshot_freq: int | None = None,
                 tolerance: float = 1e-10
                 ) -> dict[int, np.ndarray]:

        """ Forwards the evolution of the opinion model of an
            arbitrary number of steps according to its evolution rule.
            The method assumes the model not to be time-varying (adjacency
            matrix or prejudice are never updated).
            There is also a distinction between the 'prejudice' (u) and
            current model opinions (X), as they coincide for k=0 but,
            while the simulated opinions vary (X(k)), the prejudice is fixed.

            The formula used in the forwarding is defined by the
            '_evolve(X)' method, whose only argument is the current opinion
            vector 'X'.

            The simulation can get prematurely halted in two cases,
            according to the convergence and the consensus of the model:
            -   in a general case, the evolution can get stopped using the
                solution limit of the system as a comparison: after the model
                approches that within a certain accuracy, the simulation is
                halted;
            -   if the model either doesn't converge or there is no final
                consensus, the final limit is impossible to estimate and
                hence the previous iteration is used instead as a comparison
                (the simulation is stopped if change between iterations
                becomes infinitesimal).

            The function returns a dictionary that connects the parameter
            numbers (k) to the respective opinion vector (x(k)),
            snapshots of the evolution taken according to the specified
            intervals. The first and the last iterations
            (k=0 and highest k reached) will be collected regardless.

            Params
            ------
            steps : int, default = 10_000
                Number of steps that the model gets forwarded to, bar
                preemptive stop due to fulfillment of either the criteria.
                Required to be non-negative.

            snapshot_freq : int | None, default = None
                Number of iterations between consecutive 'extra' snapshots
                other than the first (k=0) and the last reached (k=steps,
                if not interrupted).
                Required to be either a positive int or None: in the latter
                case, no 'extra' snapshots will be ever taken.

            tolerance : float, default = 1e-10
                Tolerance used in the comparison of different opinion vectors
                for a premature exit from the simulation
                (e.g. compare the values of consecutive opinion vectors).

            Returns
            -------
            evolution of the model : dict[int, np.ndarray]
                A dictionary of the snapshots of the system taken at
                various values of the parameter k.
                The first (k=0) and last k values are always included.

            Raises
            ------
            TypeError
                If steps, snapshot_freq or tolerance aren't of the correct type:
                - steps: integer;
                - snapshot_freq: either integer or None;
                - tolerance: floating-point.
            ValueError
                If steps, snapshot_freq or tolerance assume values outside of
                the prescribed boundaries:
                - steps must be non-negative;
                - snapshot_freq must be either None or a positive integer;
                - tolerance must be a non-negative floating-point.
        """
        if type(steps) is not int:
            raise TypeError("Invalid type of steps for evolution of "
                            f"the model: '{type(steps)}'. "
                            "Expected an integer.")
        if steps < 0:
            raise ValueError("Invalid value of steps for evolution of "
                             f"the model: '{steps}'. "
                             "Expected a non-negative integer.")

        if (type(snapshot_freq) is not int and
                snapshot_freq is not None):
            raise TypeError("Invalid type of snapshot frequency of the "
                            "evolution of the model: "
                            f"'{type(snapshot_freq)}'. Expected either "
                            "'None' or an integer.")
        if (type(snapshot_freq) is int and
                snapshot_freq < 1):
            raise ValueError("Invalid value of snapshot frequency of the "
                             "evolution of the model: "
                             f"'{snapshot_freq}'. "
                             "Expected a positive integer.")

        if not isinstance(tolerance, (int, float)):
            raise TypeError("Invalid type of tolerance for convergence "
                            "test for evolution of the model: "
                            f"'{type(tolerance)}'. "
                            "Expected a floating-point.")
        if tolerance < 0:
            raise ValueError("Invalid value of tolerance for convergence "
                             f"test for evolution of the model: "
                             f"'{tolerance}'. "
                             "Expected a non-negative floating-point.")

        X = self.prejudice.copy()
        X_sol = self.calculate_limit_solution()

        take_snapshot = False
        if snapshot_freq is not None:
            take_snapshot = True

        snapshots = {0: X}
        final_snapshot = steps

        for t in range(steps):
            if take_snapshot and t % snapshot_freq == 0:
                snapshots[t] = X

            X_next = self._evolve(X)

            if X_sol is not None:
                if np.abs(X_next - X_sol).max() < tolerance:
                    final_snapshot = t+1
                    X = X_next
                    break
            else:
                if np.abs(X_next - X).max() < tolerance:
                    final_snapshot = t+1
                    X = X_next
                    break

            X = X_next

        snapshots[final_snapshot] = X
        return snapshots

    @abstractmethod
    def calculate_limit_solution(self) -> np.ndarray | None:
        """ Returns the limit solution of the model.
            If the model does not converge or there is no closed
            form formula of the limit solution, this method
            is expected to return 'None'.
        """
        pass

    @abstractmethod
    def is_convergent(self, tol):
        """ Returns the result of the model-specific convergence test.
            The model is convergent if, for every initial condition x(0),
            x(inf) exists such that:
                x(inf) = lim f^k(x(0))  with f(x) evolution function
                                        and k-> inf
        """
        pass

    @property
    @abstractmethod
    def convergence_rate(self) -> float:
        """ Returns the theoretical convergence rate of the model rho,
            such that:
                |x(k) - x*| <= O(rho^k) with rho in [0, 1]
                                        and x* limit solution.

            Note: if rho is equal to one, the model is lacking convergence.
        """
        pass

    @property
    def eigenvalues(self) -> np.ndarray:
        """ Returns the eigenvalues of the adjacency matrix of the model
            in order of decreasing magnitude, computed once and then cached.
        """
        if not hasattr(self, "_eigenvalues"):
            self._eigenvalues = highest_eigenvalues(self.adjacency)
        return self._eigenvalues
