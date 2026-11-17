import numpy as np
from scipy.sparse import csr_array, dia_array, eye_array, linalg
from .base_model import OpinionModel
from .de_groot import DeGrootModel
from ..matrix_check import highest_eigenvalues


class FJModel(OpinionModel):
    """ Class that represents a Friedkin-Johnsen continuous opinion model,
        that tries to represent the evolution and spreading of opinions
        inside a population of agents/users.

        At its core, the model is comprised of three elements:
        -   an adjacency matrix W, that describes the relationships
            amongst the users;
        -   a prejudice (or opinion) vector X, that describes the
            opinions of the individual users as numerical values on a
            continuous slider ([0,1]);
        -   a susceptibility matrix lambda (Λ), that for each user weights
            the opinion of the neighbourhood with its own prejudice using
            values comprised in the [0, 1] interval (total stubborness and
            no opinion inertia respectively).
            Sometimes the opposite concept is used, the "stubborness" of a
            node, calculated as 1-Λ.

        The evolution of the opinions of the users is hence calculated
        using the following formula:
            X(k+1) = ΛW X(k) + (1-Λ)u   with k evolution parameter,
                                        Λ susceptibility matrix,
                                        W adjacency matrix,
                                        X current user opinion vector,
                                        u prejudice vector.

        The adjacency matrix is required to have non-negative entries
        and be stochastic and, while the users' opinions don't usually have
        mathematical boundaries, they will instead be limited to a [0,1]
        interval.

        Parameters
        ----------
        adjacency : csr_array
            Adjacency matrix used by the model, assumed to have non-negative
            values comprised within the [0, 1] interval and be stochastic.
            Input will not be validated thoroughly, as it's expected to already
            be validated in get_matrices().

        prejudice : np.ndarray
            Starting opinions used by the model, expected to be confined in the
            [0,1] numeric interval.
            Input will not be validated thoroughly, as it's expected to already
            be validated in get_matrices().

        susceptibility : dia_array
            Diagonal matrix that for each user indicates how much weight their
            heigbours' opinions have in respect to their own prejudice.
            The values are expected to be in the [0, 1] range.
            Input will not be validated thoroughly, as it's expected to already
            be validated in get_matrices().

        Raises
        ------
        TypeError
            If either adjacency, prejudice or susceptibility aren't of their
            respective expected type (csr_array or np.ndarray respectively).

        Attributes
        ----------
        adjacency : csr_array
            Adjacency matrix used by the model, with non-negative values and
            stochastic.

        prejudice : np.ndarray
            Starting opinions used by the model, confined in the
            [0,1] numeric interval.

        susceptibility : dia_array
            Diagonal matrix that for each user indicates how much weight their
            heigbours' opinions have in respect to their own prejudice.
            The values are expected to be in the [0, 1] range.

        eigenvalues : np.ndarray[float]
            Eigenvalues of the adjacency matrix.

        convergence_rate : float
            Rate of convergence of the model to the limit solution, comprised
            between 0 and 1 (if the convergence rate is =1, the model
            never converges).
    """

    def __init__(self,
                 adjacency: csr_array,
                 prejudice: np.ndarray,
                 susceptibility: dia_array):

        super().__init__(adjacency, prejudice)

        if not isinstance(susceptibility, dia_array):
            raise TypeError("Susceptibility matrix is of "
                            f"{type(susceptibility)} type, "
                            "expected 'dia_array'.")

        self.susceptibility = susceptibility

    def _evolve(self, X):
        """ Evolution step of the Friedkin-Johnsen model:
                X(k+1) = Λ W X(k) + (I - Λ) u   with Λ susceptibility matrix,
                                                W adjacency matrix,
                                                X(k) user opinion vector
                                                and u prejudice vector.

            The formula is rewritten to favour optimization as:
                X(k+1) = A X(k) + B     with A = Λ W and B = (I - Λ) u,
                                        cached under the attributes
                                        '_influence matrix' and
                                        '_prejudice_term' respectively.

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
            section 2.3.2 at page 14.
        """
        if (not hasattr(self, "_influence_matrix") or
                not hasattr(self, "_prejudice_term")):
            self._influence_matrix = (self.susceptibility @
                                      self.adjacency).tocsr()
            self._prejudice_term = (self.prejudice -
                                    self.susceptibility @ self.prejudice)

        A = self._influence_matrix
        B = self._prejudice_term

        return A @ X + B

    @property
    def dependent_users(self):
        """ Returns a mask that indicates "P-dependent" users of the model, the
            set of users comprised of both the agents that have
            susceptibility <1 and those who are influenced by them.

            The calculation is performed once (then cached) by starting from
            the susceptible nodes and gradually spreading the property along
            the edges until it no longer adds any new user.

            Returns
            -------
            P-dependent users : np.ndarray[bool]

            Notes
            -----
            If the P-dependent users mask is comprised exclusively of zeros,
            the model falls back into the De Groot case.
            """

        if not hasattr(self, "_dependent_users"):
            dep_users = self.susceptibility.diagonal() < 1

            new_dep_users = dep_users.copy()
            while (new_dep_users).any():

                frontier_dep_users = (self.adjacency @ new_dep_users) > 0
                new_dep_users = frontier_dep_users & ~dep_users
                dep_users = dep_users | new_dep_users

            self._dependent_users = dep_users

        return self._dependent_users

    @property
    def _lambda_W_spectral_radius(self):
        """ Returns the spectral radius of the matrix multiplication between
            the susceptibility (Λ) and the adjacency (W) matrices.
            (calculated once, then cached)

            Returns
            -------
            float
        """
        if not hasattr(self, "_spectral_radius"):
            Λ_W = (self.susceptibility @ self.adjacency).tocsc()
            self._spectral_radius = abs(
                highest_eigenvalues(Λ_W, no_eigens=1)[0]
                )

        return self._spectral_radius

    def is_asympt_stable(self):
        """ Checks for the asymptotic stability of the model.
            The check consists in the evaluation of the spectral
            radius of the matrix product between the susceptibility
            matrix and the adjacency matrix (ΛW), required to be
            <1 for convergence.

            Returns
            -------
            bool

            References
            ----------
            Anton V.Proskurnikova, Roberto Tempo;
            "A Tutorial on Modeling and Analysis of Dynamic Social
            Networks. Part I", page 14 section 6.1.
            """
        # For this, to check the non-singularity of the matrix, the
        # spectral radius of ΛW is evaluated (sufficient condition).
        return (self._lambda_W_spectral_radius < 1 and
                not np.allclose(self._lambda_W_spectral_radius, 1))

    def is_convergent(self, tol=1e-10):
        """ Checks for the convergence of the model.
            Condition sufficient for convergence is the P-dependency
            of all users; but were this not the case the convergence
            of the De Groot graph comprised of the remaining P-indipendent
            users is evaluated.

            Returns
            -------
            bool

            References
            ----------
            Anton V.Proskurnikova, Roberto Tempo;
            "A Tutorial on Modeling and Analysis of Dynamic Social
            Networks. Part I", page 14 section 6.1 (theorem 21).
            """
        if (self.dependent_users).sum() == self.no_users:
            return True

        ind_users = ~self.dependent_users
        sub_degroot_adjacency = self.adjacency[ind_users][:, ind_users]
        sub_degroot_prejudice = self.prejudice[ind_users]

        independent_model = DeGrootModel(sub_degroot_adjacency,
                                         sub_degroot_prejudice)
        return independent_model.is_convergent(tol=tol)

    def _simplified_limit_solution(self):
        """ Calculates the simplified limit solution of a Friedkin-Johnsen
            model in the case each one of its users is P-dependent.
            This alone is a sufficient condition for asymptotic stability,
            as it guarantees spectral radius of ΛW to be lower than 1.

            The theoretical formula used to estimate the equilibrium of the
            system is hence described by:
                X(k->inf) = (I - Λ W)⁻¹ * (I - Λ)u  with Λ susceptibility,
                                                    W adjacency
                                                    and u prejudice matrices

            Returns
            -------
            limit solution : np.ndarray | None

            Raises
            ------
            ValueError
                If (I - Λ W) is singular and hence not invertible, which
                happens if there are P-indipendent nodes and the formula
                would not be applicable.

            References
            ----------
            Anton V.Proskurnikov, Roberto Tempo, Ming Cao, Noah E. Friedkin;
            "Opinion evolution in time-varying social influence networks
            with prejudiced agents", page 3 section 3.

            Anton V.Proskurnikova, Roberto Tempo;
            "A Tutorial on Modeling and Analysis of Dynamic Social
            Networks. Part I", page 14 section 6.1 (theorem 21).
        """
        if not (self.dependent_users).sum() == self.no_users:
            raise RuntimeError("Applied simplified formula for Friedkin-"
                               "Johnsen model limit solution in the wrong "
                               "context: not all nodes are P-dependent.")

        W = self.adjacency
        u = self.prejudice
        Λ = self.susceptibility

        I_matrix = eye_array(self.no_users)
        Λ_W = Λ @ W.tocsc()
        A = I_matrix - Λ_W

        # In efficiency's interest, the problem gets changed
        # from X = (I - Λ W)⁻¹ * (I - Λ)u to the
        # solution of the lin.eq. (I - Λ W)X =(I - Λ)u,
        # as matrix inversion is particularly costly.

        limit_solution = linalg.spsolve(A, u - Λ @ u)
        limit_solution = np.asarray(limit_solution).reshape(u.shape)

        assert np.allclose(limit_solution, np.clip(limit_solution, 0, 1))
        return limit_solution

    def calculate_limit_solution(self):
        """ Calculates the limit solution of a Friedkin-Johnsen model.

            Given a generic Friedkin-Johnsen model, it is convergent if
            and only if the De Groot subgraph of its P-indipendent nodes is
            convergent or if it's empty.
            While in the second case it's possible to use a simplified
            formula, the theoretical formula used to estimate the equilibrium
            of the system in the most generic case calculates separately the
            evolution for the P-dependent (X_d) and the P-indipendent
            nodes (X_i).

            The P-indipendent nodes contitute a de-facto De Groot submodel:
                X_i(k+1) = W_ii X_i(k),     with W_ii subsection of the
                                            adjacency matrix that only keeps
                                            edges among P-indipendent nodes.

                X_i(k->inf) = X_i(inf),     the De Groot limit solution already
                                            defined in the DeGrootModel class.

            The P-dependent nodes:
                X_d(k+1) = Λ_dd [ W_dd X_d(k) + W_di X_i(k) ] + (I - Λ_dd) u_d

                with W_dd, u_d and Λ_dd subsections of the adjacency matrix,
                prejudice and susceptibility matrix that concern property or
                interactions between P-dependent nodes and them only;
                while W_di is the subsection of the adjacency matrix that
                describes the influence of the P-indipendent nodes on the
                P-dependent ones.

                A = I - Λ_dd W_dd
                B = Λ_dd W_di X_i(inf) + (I - Λ_dd) u_d

                X_d(k->inf) = A⁻¹ B

            Returns
            -------
            limit solution : np.ndarray | None

            References
            ----------
            Anton V.Proskurnikov, Roberto Tempo, Ming Cao, Noah E. Friedkin;
            "Opinion evolution in time-varying social influence networks
            with prejudiced agents", page 3 section 3.

            Anton V.Proskurnikova, Roberto Tempo;
            "A Tutorial on Modeling and Analysis of Dynamic Social
            Networks. Part I", page 14 section 6.1 (theorem 21).
        """
        dep_users = self.dependent_users
        ind_users = ~dep_users

        if dep_users.sum() == self.no_users:
            return self._simplified_limit_solution()

        W = self.adjacency
        u = self.prejudice
        Λ = self.susceptibility

        ind_model = DeGrootModel(W[ind_users][:, ind_users], u[ind_users])
        ind_solution = ind_model.calculate_limit_solution()

        if ind_users.sum() == self.no_users:
            return ind_solution

        # Condition for convergence would be already lost
        if ind_solution is None:
            return None

        I_matrix = eye_array(dep_users.sum())
        W_dep = W[dep_users][:, dep_users]
        u_dep = u[dep_users]
        Λ_dep = Λ.tocsr()[dep_users][:, dep_users]

        A = I_matrix - Λ_dep @ W_dep.tocsc()

        W_dep_ind = W[dep_users][:, ind_users]
        B = (u_dep - Λ_dep @ u_dep + Λ_dep @ (W_dep_ind @ ind_solution))

        limit_solution = np.full(u.shape, 0.)
        limit_solution[ind_users] = ind_solution
        limit_solution[dep_users] = linalg.spsolve(A, B).reshape(u_dep.shape)

        return limit_solution

    @property
    def convergence_rate(self):
        """ Returns the convergence rate of a Friedkin-Johnsen model,
            that corresponds to the spectral radius (rho) of the product
            between the susceptibility matrix and the adjacency matrix (ΛW).

            The convergence speed can then be expressed as:
                |x* - x(k)| <= D*rho(ΛW)^k    for k->inf

            where x* is the limit_solution, x(k) is the current
            iteration of the opinion vector and D is a constant >=0.

            The value gets computed the first time it's called and then cached.

            Returns
            -------
            convergence rate (rho(ΛW)): float within 0 and 1
                Note that a convergence rate of '1' can be a sign of
                the model never converging.

            Raises
            ------
            ValueError
                If the calculated convergence rate isn't in the [0,1] interval.

            References
            ----------
            Anton V.Proskurnikov, Roberto Tempo, Ming Cao, Noah E. Friedkin;
            "Opinion evolution in time-varying social influence networks
            with prejudiced agents", page 3 section 4.2.
        """
        return self._lambda_W_spectral_radius
