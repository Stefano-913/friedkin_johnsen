from pathos.multiprocessing import ProcessingPool as Pool
from pathlib import Path
from .get_matrices import get_matrices
import numpy as np
from functools import partial
from .opinion_models.de_groot import DeGrootModel
from .opinion_models.friedkin_johnsen import FJModel


def run_simulation(config_file="matrix_config.json",
                   directory=Path.cwd(),
                   tol=1e-10,
                   steps=10_000,
                   snapshot_freq=None):
    """ Runs an opinion dynamics simulation.

        Loads the matrices from the configuration file and uses either De Groot
        or the Friedkin-Johnsen model to calculate their opinion evolution.
        The model is chosen according to the susceptibility matrix:
        if every node has susceptibility=1, a DeGroot model is used;
        otherwise it defaults to Friedkin-Johnsen.

        Returns a dictionary that connects the parameter numbers (k) to the
        respective opinion vector (x(k)), snapshots of the evolution taken
        according to the intervals specified in 'snapshot_freq'.
        The first and the last iterations (k=0 and highest k reached) will be
        collected regardless. The highest k reachable is given by 'steps', but
        the simulation can halt sooner if certain conditions are verified
        (class OpinionModel:simulate for more details).

    Parameters
    ----------
    config_file : str, default="matrix_config.json"
        Name of the JSON configuration file used to obtain the matrices.

    directory : Path, default=Path.cwd()
        Directory of the configuration file.

    tol : float, default=1e-10
        Tolerance in the comparison between floating values
        (e.g. comparison between the susceptibility values and '1'
        to discern the correct module or comparisons during the simulation).

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

    Returns
    -------
    dict
        The snapshots recorded by the model's `simulate` method.

    Raises
    ------
    TypeError
        If 'tol' input isn't of float type.
    ValueError
        If 'tol' input is negative.

    SeeAlso
    -------
    OpinionModel
    """
    if not isinstance(tol, (int, float)):
        raise TypeError("Invalid type of tolerance for the "
                        "comparison of floating values: "
                        f"'{type(tol)}'. Expected a floating-point.")
    if tol < 0:
        raise ValueError("Invalid value of tolerance for the "
                         f"comparison of floating values: '{tol}'. "
                         "Expected a non-negative floating-point.")

    matrices = get_matrices(config_file, directory)

    susceptibility = matrices["susceptibility"]
    if np.allclose(susceptibility.diagonal(), 1, rtol=0, atol=tol):
        model = DeGrootModel(adjacency=matrices["adjacency"],
                             prejudice=matrices["prejudice"])
    else:
        model = FJModel(**matrices)

    return model.simulate(steps=steps,
                          snapshot_freq=snapshot_freq,
                          tolerance=tol)


def run_simulations_parallel(no_simulations=1,
                             num_processes=4,
                             config_file="matrix_config.json",
                             directory=Path.cwd(),
                             tol=1e-10,
                             steps=10_000,
                             snapshot_freq=None):
    """ Runs several opinion dynamics simulations in parallel.

        Every simulation is based on the same configuration file but has an
        independent seed, so the deterministic elements fixed by the users will
        be common to all runs but the rng sampling will be different.

        Parameters
        ----------
        no_simulations : int, default=1
            Number of simulations to run.

        num_processes : int, default=4
            Number of worker processes in the pool.

        config_file : str, default="matrix_config.json"
            Name of the JSON configuration file passed to each simulation.

        directory : Path, default=Path.cwd()
            Directory of the configuration file.

        Returns
        -------
        list[dict]
            A list containing all the simulations' results.

        Raises
        ------
        TypeError
            If `no_simulations` or `num_processes` is not an integer.
        ValueError
            If `no_simulations` or `num_processes` is less than 1.
    """
    if not isinstance(no_simulations, int):
        raise TypeError("Invalid type of number of simulations: "
                        f"'{type(no_simulations)}', expected an integer.")
    if no_simulations < 1:
        raise ValueError("Invalid value of number of simulations: "
                         f"'{no_simulations}', expected a positive integer.")

    if not isinstance(num_processes, int):
        raise TypeError("Invalid type of number of processes: "
                        f"'{type(num_processes)}', expected an integer.")
    if num_processes < 1:
        raise ValueError("Invalid value of number of processes: "
                         f"'{num_processes}', expected a positive integer.")

    with Pool(nodes=num_processes) as pool:
        # An argument of run_simulation needs to be provided various times,
        # so that its gives 'Pool' the number of simulations to run.
        results = pool.map(partial(run_simulation,
                                   directory=directory,
                                   tol=tol,
                                   steps=steps,
                                   snapshot_freq=snapshot_freq),
                           [config_file] * no_simulations)

    return results
