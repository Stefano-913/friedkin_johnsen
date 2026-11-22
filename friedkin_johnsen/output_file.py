import json


def write_output_file(output_path, sim_results):
    """ Write simulation results to a JSON file.

        In case of repeated file creation with the same name, to avoid
        overwriting previous results, the file name gets changed to
        <stem>_<i>.<suffix>, with "i" first index available up to 999.
        (e.g. "name.json" -> "name_1.json" -> "name_2.json").

        If needed, the parent directory of `output_path` is created.

        Parameters
        ----------
        output_path : Path
            Path of the output file.

        sim_results : list[dict[int, np.ndarray]]
            List of the results of the various simulations.
            Each simulation gives a dictionary that links the
            step to its opinion state, an np.ndarray of shape
            (no_users, no_opinions).

        Returns
        -------
        Path
            Path of the file written.

        Raises
        ------
        FileExistsError
            If every version of <stem>_<i>.<suffix> up to i=999
            already exist.
    """
    max_file_index = 999
    output_path.parent.mkdir(parents=True, exist_ok=True)

    path = output_path
    i = 0
    while path.exists():
        i += 1

        if i > max_file_index:
            file_stem = output_path.stem
            suffix = output_path.suffix
            raise FileExistsError("All output file names up to "
                                  f"'{file_stem}_{max_file_index}{suffix}' "
                                  f"already exist in '{output_path.parent}'")

        path = output_path.with_name(
            f"{output_path.stem}_{i}{output_path.suffix}"
            )

    output_path = path

    simulations = {}
    for sim_index, simulation in enumerate(sim_results):
        simulations[sim_index] = {step: opinions.squeeze().tolist()
                                  for step, opinions in simulation.items()}

    output_path.write_text(json.dumps(simulations, indent=2))

    return output_path
