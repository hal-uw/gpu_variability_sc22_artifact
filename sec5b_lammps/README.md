# Section 5.B. LAMMPS

Public development project for the LAMMPS MD simulation package is hosted on GitHub at https://github.com/lammps/lammps. We used the LAMMPS tarball provided by the Coral-2 suite (https://asc.llnl.gov/coral-2-benchmarks)
We added the following build and running scripts:
| Added File                                  | Purpose                                             |
|:--------------------------------------------|:----------------------------------------------------|
| lammps_17Jan18/src/build-lammps.sh          | Runs `make` for required packages (KOKKOS, REAXC)   |
| lammps_17Jan18/reax_benchmark/src-lammps.sh | Sets up environment variables for run               |
| lammps_17Jan18/reax_benchmark/run-lammps.sh | Runs lammps with nvprof for profiling data          |

The scripts `build-all.sh` and `run-all.sh` internally call the above shell scripts to compile relevant lammps binaries and launch them as single-GPU jobs.
Profiling output can be observed in out/lammps-<UUID>_run<run_num>_<timestamp>.csv 
To modify the number of runs, device to run on etc, only run-all.sh needs to be updated. 
