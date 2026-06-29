import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from tqdm import tqdm
import pickle
import argparse
import os
import sys
from joblib import Parallel, delayed
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from external.md_calcs import md_calcs_par as mdcp
import subprocess
from time import time
from scipy.stats import linregress
import pandas as pd


A2m = 1e-10
fs2s = 1e-15

def bash_parser(filepath, N_time, N_chains, stride=1):
    try:
        subprocess.run(
            f"awk 'NF==4 && $1+0==$1 {{ print $2, $3, $4 }}' {os.path.join(filepath, 'com_mol.dat')} > {os.path.join(filepath, 'com_cleaned.txt')}",
            shell=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error: awk failed for {filepath}: {e}\n")

    out = np.loadtxt(os.path.join(filepath, 'com_cleaned.txt'))

    if out.shape[0] != N_time * N_chains:
        #raise ValueError(f"Expected {N_time*N_chains} lines, got {out.shape[0]}")
        N_time = out.shape[0] // N_chains # adjust N_time if it doesn't match
        print(f"Warning: Expected {N_time*N_chains} lines, got {out.shape[0]}. Adjusting N_time to {N_time}.\n")

    result = out.reshape(N_time, N_chains, 3)

    return result[::stride], N_time 

def compute_msd(pth, N_time, N_chains):

    trajectory, N_time = bash_parser(pth, N_time, N_chains)
    traj_rev = trajectory[::-1].copy()

    msd_fwd = mdcp.msd_calc(trajectory, '3d')
    msd_rev = mdcp.msd_calc(traj_rev, '3d')

    return [msd_fwd, msd_rev], N_time


### BELOW IS LINEAR REGRESSION METHOD
def process_one_run(path, N_time, N_chains, times, cutoff, dt=None, \
                    nfreq=None, stride=1, bootstrap=False):
    if bootstrap:
        traj, N_time = bash_parser(path, N_time, N_chains, stride=stride)
        traj_rev = traj[::-1].copy()
        return [mdcp.msd_calc(traj, '3d'), mdcp.msd_calc(traj_rev, '3d')], N_time
    else:
        times_ns = times*1e-15/1e-9
        traj, N_time = bash_parser(path, N_time, N_chains, stride=stride)
        traj_rev = traj[::-1].copy()
        msd = mdcp.msd_calc(traj, '3d')
        msd_rev = mdcp.msd_calc(traj_rev, '3d')

        if times.shape[0] != N_time:
            #N_time      = args.nsteps//args.nfreq+1
            nsteps =(N_time - 1)* nfreq 
            times = np.arange(0, nsteps+nfreq, nfreq)*dt
        # Take samples from 25 to 70 ns. 
        start_idx = np.sum(times < 2.5e7)
        end_idx = np.sum(times < 7e7)
        model = linregress(times[start_idx:end_idx], msd[start_idx:end_idx])
        model_rev = linregress(times[start_idx:end_idx], msd_rev[start_idx:end_idx])
        # D = d MSD / dt / 2 / dimensions 
        return [model.slope / 6.0, msd, model_rev.slope / 6.0, msd_rev]
    
def calculate_diffusivity(msd, times):
    """
    Calculate diffusivity from MSD data.
    """
    # Assuming msd is in Angstrom^2 and times are in fs
    # Convert times to seconds for diffusivity calculation
    times_sec = times * 1e-15 / 1e-9  # fs to seconds
    start_idx = np.sum(times < 2.5e7)
    end_idx = np.sum(times > 7e7)

    model = linregress(times[start_idx:end_idx], msd[start_idx:end_idx])
    slope = model.slope
    #diffusivity = msd[start_idx:end_idx] / (6 * times[start_idx:end_idx])
    return slope/6.0

def compute_diff_for_sequence(seq_idx, seq_dir, N_time, cutoff, N_chains, times, nsims, inner_jobs, stride=1, bootstrap=False, nstrap=None, nfreq=None, dt=None):
    #os.environ["OMP_NUM_THREADS"] = str(omp_threads)
    filepaths = [os.path.join(seq_dir, str(i)) for i in range(nsims)]

    t = time()
    print(f"Starting compute for seq {seq_idx}.\n", flush=True)

    if bootstrap:
        results = Parallel(n_jobs=inner_jobs)(
            delayed(process_one_run)(p, N_time, N_chains, times, cutoff, stride=stride, bootstrap=True) for p in filepaths
        )

        msds_fwd, msds_bwd = zip(*results)
        msds = msds_fwd+msds_bwd

        with open(os.path.join(seq_dir, 'msd_results.pkl'), 'wb') as f:
        # dump as pickle
            pickle.dump(
                [
                    times,
                    msds
                ],
                f,
            )  

        if nstrap is not None and type(nstrap) == int:

            msd_array = np.array(msds)
            n_runs = msd_array.shape[0]
            idxs = np.random.choice(n_runs, nstrap, replace=True)
            bootstrap_samples = msd_array[idxs]

            diffs = Parallel(n_jobs=inner_jobs)(
                delayed(calculate_diffusivity)(msd, times) for msd in bootstrap_samples
            )

            D_vals = [D * A2m**2 / fs2s / 1e-9 for D in diffs]
            print(f"Computed D for sequence {seq_idx} in {time()-t} seconds. D is {np.mean(D_vals)}, std is {np.std(D_vals)}\n", flush=True)

            return (np.mean(D_vals), np.std(D_vals), seq_idx)


        else:
            raise ValueError("To bootstrap, need to provide integer value for 'nstrap'.")


    try:
        results = Parallel(n_jobs=inner_jobs)(
            delayed(process_one_run)(p, N_time, N_chains, times, cutoff, stride=stride, nfreq=nfreq, dt=dt) for p in filepaths
        )
    except Exception as e:
        print(f"Error in getting diff for seq {seq_idx}: {e}\n", flush=True)

    res_flat = []

    D_vals_fwd, msds_fwd, D_vals_bwd, msds_bwd = zip(*results)

    msds = msds_fwd + msds_bwd
    D_vals = D_vals_fwd + D_vals_bwd

    with open(os.path.join(seq_dir, 'msd_results.pkl'), 'wb') as f:
        # dump as pickle
        pickle.dump(
            [
                times,
                msds
            ],
            f,
        )    

    D_vals = [D * A2m**2 / fs2s / 1e-9 for D in D_vals] # Return in units of (10^-9) m^2/s
    print(f"Computed D for sequence {seq_idx} in {time()-t} seconds. D is {np.mean(D_vals)}, std is {np.std(D_vals)}\n", flush=True)

    return (np.mean(D_vals), np.std(D_vals), seq_idx)



def main():
    parser = argparse.ArgumentParser(description='Computes diffusivities for a few trajectories.')

    parser.add_argument("--parent_dir", type=str, help="Parent directory for all simulations")
    parser.add_argument("--output_dir", type=str, help="Output directory for storing results")
    parser.add_argument("--sequence_file", type=str, help="File containing sequences to be analyzed")
    parser.add_argument('--nruns', required=True, type=int, help='Number of independent runs.')
    parser.add_argument('--nsteps', required=True, type=int, help='Number of simulation timesteps')
    parser.add_argument('--dt', required=True, type=float, help='Timestep of simulations.')
    parser.add_argument('--nchains', required=True, type=int, help='Number of polymers simulated.')
    parser.add_argument('--nfreq', required=True, type=int, help='Frame output frequency.')
    parser.add_argument("--inner_jobs", type=int, default=4)
    parser.add_argument("--omp_threads", type=int, default=4)
    parser.add_argument("--nseq_jobs", type=int, default=6)
    parser.add_argument("--cutoff", type=float)
    parser.add_argument("--stride", type=int, default=1, help="Stride over which to thin COM position data.")
    parser.add_argument("--bootstrap", action=argparse.BooleanOptionalAction, default=False, help='Flag for bootstrapping.')

    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.omp_threads)


    N_time      = args.nsteps//args.nfreq+1 # number of frames

    times = np.arange(0, args.nsteps+args.nfreq, args.nfreq)*args.dt
    times = times[::args.stride]

    with open(args.sequence_file, 'r') as f:
        sequences = [line.strip() for line in f.readlines()]

    seq_dirs = [os.path.join(args.parent_dir, f'poly{i}') for i in range(len(sequences))]

    print(f'Computing MSDs...\n', flush = True)

    #if args.bootstrap == True:
    nstrap = 100 if args.bootstrap else None
    try:
        results = Parallel(n_jobs=args.nseq_jobs)(
            delayed(compute_diff_for_sequence)(
                i, seq_dirs[i], N_time, args.cutoff, args.nchains, times,
                args.nruns, args.inner_jobs, stride=args.stride, bootstrap=args.bootstrap, nstrap=nstrap, dt=args.dt, nfreq=args.nfreq) for i in range(len(sequences))
        )
    except Exception as e:
        print(f"Error in processing results: {e}\n.", flush=True)

    D_means, D_stds, idxs = zip(*results)

    df = pd.DataFrame({
        "sequence": sequences,
        "seq_ID": idxs,
        "diff": D_means,
        "diff_std": D_stds
    })

    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, "diffusivities.csv"), index=False)

    print("Finished computing and saving diffusivities.", flush=True)

if __name__ == '__main__':
    main()



    

