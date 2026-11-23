import argparse
import os.path
from pathlib import Path
from .handle_config_file import build_config_file, fill_out_config_file
from .simulation import run_simulations_parallel
from .output_file import write_output_file


def parse_args():
    parser = argparse.ArgumentParser(
        prog="friedkin-johnsen",
        description="Simulation of a Friedkin-Johnsen opinion model."
        )

    parser.add_argument("--no-simulations",
                        "-n",
                        type=int,
                        default=1,
                        help="Number of simulations to be executed "
                        "with the given configuration file (default=1).")

    parser.add_argument("--max_steps",
                        "-s",
                        type=int,
                        default=10_000,
                        help="Maximum number of cycles before the simulation "
                        "is stopped regardless of convergence "
                        "(default=10_000).")

    parser.add_argument("--tolerance",
                        "-tol",
                        type=float,
                        default=1e-10,
                        help="Tolerance used in the comparison of "
                        "floating-point numbers (default=1e-10).")

    parser.add_argument("--snapshot_freq",
                        "-f",
                        type=int,
                        default=None,
                        help="Number of iterations between consecutive "
                        "'extra' snapshots other than the first (k=0) and the "
                        "last reached (k=steps, if not interrupted)."
                        "Default entails no extra snapshots.")

    parser.add_argument("--config-file",
                        "-c",
                        type=str,
                        default="matrix_config.json",
                        help="Custom name of the configuration file "
                        "(default='matrix_config.json').")

    parser.add_argument("--reset-config",
                        "-rc",
                        action="store_true",
                        help="Add this flag to reset the preexisting "
                        "configuration file to the default values.")

    parser.add_argument("--config-dir",
                        "-cd",
                        type=str,
                        default=Path.cwd(),
                        help="Directory where the configuration file is "
                        "located/created.\nThe directory will be created if "
                        "it didn't exist beforehand (default is current "
                        "directory).")

    parser.add_argument("--output-dir",
                        "-od",
                        type=str,
                        default=Path.cwd(),
                        help="Directory where the file containing the output "
                        "of the simulation(s) will be located/created."
                        "\nThe directory will be created if it didn't exist "
                        "beforehand (default is current directory).")

    parser.add_argument("--output-file",
                        "-o",
                        type=str,
                        default="simulation_result.json",
                        help="Custom name for the simulation(s) output "
                        "file (default='simulation_result.json').")

    parser.add_argument("--no-workers",
                        "-w",
                        type=int,
                        default=4,
                        help="The number of worker threads to use for "
                        "parallel computation (default=4).")

    return parser


def main():
    parser = parse_args()
    args = parser.parse_args()

    build_config_file(file_name=args.config_file,
                      relative_path=args.config_dir,
                      overwrite=args.reset_config)

    dir = os.path.relpath(args.config_dir, Path.cwd())
    if dir == ".":
        dir = "current"

    input("The user may now modify the configuration file "
          f"'{args.config_file}' in the {dir} directory "
          "according to their preferences."
          "\nPress the Enter key when finished to proceed with "
          "the simulation(s): ")

    fill_out_config_file(file_path=args.config_dir/args.config_file)

    simulations = run_simulations_parallel(
        no_simulations=int(args.no_simulations),
        num_processes=args.no_workers,
        config_file=args.config_file,
        directory=args.config_dir,
        tol=args.tolerance,
        steps=args.max_steps,
        snapshot_freq=args.snapshot_freq
        )

    output_path = Path(args.output_dir) / args.output_file
    write_output_file(output_path, simulations)


if __name__ == '__main__':
    main()
