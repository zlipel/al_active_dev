import numpy as np
import pandas as pd
import os
import sys
from joblib import Parallel, delayed
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from external.core import calculate_mass
#from scipy.integrate import quad
import scipy
from scipy.interpolate import CubicSpline
from scipy.optimize import root_scalar
import pickle as pkl


def main():
    parser = argparse.ArgumentParser(description='Calculate EOS, determine phase separation, and compute expenditure density')
    parser.add_argument("-parent_dir", type=str, help="Parent directory for all simulations")
    parser.add_argument("-output_dir", type=str, help="Output directory for storing results")
    parser.add_argument("-sequence_file", type=str, help="File containing sequences to be analyzed")
    parser.add_argument("-num_bootstrap", type=int, default=1000, help="Number of bootstrap resamples")

    args = parser.parse_args()

    parent_dir = args.parent_dir
    output_dir = args.output_dir
    num_bootstrap = args.num_bootstrap

    all_eos = []

    # Read sequences
    with open(args.sequence_file, "r") as f:
        sequences = [line.strip() for line in f.readlines()]

    idxs = np.arange(len(sequences))
    #f"{parent_dir}/poly{i}" 
    seq_dirs = [os.path.join(parent_dir, f"poly{i}") for i in idxs]

    # Prepare output dataframe
    df = pd.DataFrame(columns=['sequence', 'seq_ID', 'mass', 'density', 'density_std', 'psp', 'exp_density', 'exp_density_std'])
    

    # Calculate the masses of all sequences
    masses = [calculate_mass(seq) for seq in sequences]

    commands = []
    for i, (mass, idx) in enumerate(zip(masses, idxs)):
        commands.append((sequences[i], mass, idx, seq_dirs[i]))

    # Run the commands in parallel
    results = Parallel(n_jobs=-1)(delayed(wrapper)(cmd, num_bootstrap) for cmd in commands)

    for result in results:
        df = pd.concat([df,pd.DataFrame([{
            'sequence': result[0],
            'seq_ID': result[1],
            'mass': result[2],
            'density': result[3],
            'density_std': result[4],
            'psp': 1 if result[3] > 0 else 0,
            'exp_density': result[5],
            'exp_density_std': result[6]
        }])], ignore_index=True)
        P, err, rho = result[7:]
        all_eos.append([P, err, rho])

    with open(os.path.join(output_dir, f"eos_vals.pkl"), 'wb') as f:
        pkl.dump(all_eos, f)

    # Sort dataframe by seq_id
    df = df.sort_values(by='seq_ID')

    # save the results
    os.makedirs(output_dir, exist_ok=True)
    #f'{output_dir}/eos_results.csv'
    df.to_csv(os.path.join(output_dir, "eos_results.csv"), index=False)


def wrapper(cmd, num_bootstrap):
    seq, mass, idx, seq_dir = cmd

    try:
        P, err, rho = get_EOS(seq_dir, frac=0.5, bootstrap=True)

        # print(seq, P, err)

        rho_star_mean, rho_star_std, exp_density_mean, exp_density_std  =  bootstrap_eos_analysis(rho, P, nboot=num_bootstrap)
        
        return [seq, idx, mass, rho_star_mean, rho_star_std, exp_density_mean, exp_density_std, P, err, rho]
    except Exception as e:
        print(f"Error processing sequence {seq}: {e}")
        
        return [seq, idx, mass, np.nan, np.nan, np.nan, np.nan, P, err, rho] # Return NaNs if an error occurs

    #return [seq, idx, mass, rho_star_mean, rho_star_std, exp_density_mean, exp_density_std]


def find_highest_root(P, rho, error):
    """Find the highest density root where pressure crosses zero."""

    cs = CubicSpline(rho, P, bc_type=((1, 0), (2, 0)))

    def f(x):
        return cs(x)

    roots = []
    for i in range(len(rho) - 1):
        if P[i] * P[i + 1] < 0:  # Sign change detected
            try:
                root = root_scalar(f, bracket=[rho[i], rho[i + 1]], method='brentq').root
                roots.append(root)
            except:
                pass

    return max(roots) if roots else 0


def get_EOS(pth, frac=0.5, bootstrap=False):
    '''
    Get the EOS data from a simulation file containing different densities
    
    Parameters
    ----------
    path : str
        Path to file containing simulations at different densities
    frac : float
        Fraction of pressure data to use for EOS calculation
    bootstrap : bool
        Whether to use bootstrap error estimation

    Returns
    -------
    (P, err, rho) : tuple
        Tuple containing pressure, error, and density data
    '''

    labels = ['TimeStep', 'temp', 'etot', 'pe', 'ke', 'ent', 'P', 'rho']
    P = []
    rho = []
    err = []
    
    for density in sorted(os.listdir(pth)):
        subpath = os.path.join(pth,f"{density}")
        try:

            file = os.path.join(subpath, f"thermo.avg")
            if os.path.exists(file):
                data = pd.read_csv(file,delimiter=' ',header=None,names=labels,skiprows=2)
                data.dropna(inplace=True)

                N = int((1-frac)*len(data)) # get frac*100 % of the data (eg frac 0.8 => 80%)

                # Fixed 5-block split-error. Previously we tried a
                # correlation-time-based n_samples via statsmodels.acf, but
                # the zero-crossing estimator was empirically unreliable
                # (frequently returned -1) and we always fell back to n=5.
                # Skip the machinery entirely.
                n_samples = 5

                if bootstrap:
                    P.append(split_error(data['P'][N:].values, n_samples)[1])
                else:
                    P.append(data['P'][N:].mean())

                err.append(split_error(data['P'][N:].values, n_samples)[0])
                rho.append(float(data['rho'].values[-1]))
            else:
                continue

        except Exception as e:
            print(f'Error reading from {subpath}: {e}')

    #P = np.asarray(P) # shape: n_states x m_independent_pressures
    #err = np.asarray(err) # shape: n_states

    return (P, err, rho)


def split_error(a, n):
    '''Calculates error by splitting up array into n blocks
    
    Parameters
    ----------
    a : array_like
        Array for which to calculate error
    n : int
        Number of blocks to split into
        
    Returns
    -------
    std : float
        standard error, not standard deviation
    '''

    k, m = divmod(len(a), n)
    means = [np.mean(a[i*k+min(i, m):(i+1)*k+min(i+1, m)]) for i in range(n)]
    std = np.std(means)/np.sqrt(n)
    return std, means



def bootstrap_eos_analysis(rho_values, P_matrix, nboot=500, work=15, threshold_fraction=0.5):
    """
    Bootstraps the EOS to compute both condensed-phase density (ρ*) and expenditure density (W/m),
    ensuring statistical significance of negative pressures is checked **before** bootstrapping.

    Parameters:
    -----------
    rho_values : array-like
        Array of density values from simulations.
    P_matrix : 2D array (shape: n_rho × n_samples)
        Independent pressure values at each density.
    nboot : int, optional
        Number of bootstrap resamples (default is 1000).
    work : float, optional
        The target integral value for expenditure density (default is 15).
    threshold_fraction : float, optional
        Fraction of bootstrap samples that must show significant phase separation for ρ* to be computed (default is 0.5).

    Returns:
    --------
    rho_star_mean : float
        Mean of bootstrapped condensed-phase density (ρ*). Returns 0 if threshold not met.
    rho_star_std : float
        Standard deviation of ρ* (uncertainty estimate). Returns 0 if threshold not met.
    exp_density_mean : float
        Mean of bootstrapped expenditure density (W/m).
    exp_density_std : float   """

    exp_density_values = []
    
    cond_rho_values    = []
    

    rhos = np.insert(rho_values, 0, 0)  # Include zero density
    xs = np.linspace(0.05, max(rhos)+0.2, 2000)

    Pb = []
    splines = []

    for _ in range(nboot):
   
        P_boot = np.array([np.mean(np.random.choice(P_vals, size=len(P_vals), replace=True)) for P_vals in P_matrix])
        P_boot = np.insert(P_boot, 0, 0)  # Include zero

        P_spline = CubicSpline(rhos, P_boot, bc_type=((1, 0.0), (2, 0.0)))

        splines.append(P_spline)
        Pb.append(P_spline(xs))
        exp_density_values.append(calc_exp_density(P_spline, np.min(rho_values), np.max(rhos), work=work))


        # next check if at least one pressure is negative 
    try:
        Pb = np.array(Pb, dtype=np.float64) # shape: nboot x len(xs)
        sd = Pb.std(axis=0)
        eps = 1.96*sd  # 95% confidence interval

        near_flags = (Pb <= eps).any(axis=1) # shape: nboot
    except Exception as e:
        print(f"Error in processing bootstrapped pressures: {e}")
        return 0, 0, -1, -1

    for b, S in enumerate(splines):
        vals = S(xs)
        roots = []

        for a, b2, fa, fb in zip(xs[:-1], xs[1:], vals[:-1], vals[1:]):
            if fa == 0:
                roots.append(a)
            elif fa*fb < 0:
                try:
                    root = root_scalar(S, bracket=[a, b2], method='brentq').root
                    roots.append(root)
                except:
                    pass
        if roots and near_flags[b]:
            rstar = max(r for r in roots if r > np.min(rho_values))
            if np.isfinite(rstar):
                cond_rho_values.append(rstar)
        
    sep_frac = float(near_flags.mean())
    print(f"Fraction of phase-separating curves: {sep_frac*100:.2f}%")

    if sep_frac >= threshold_fraction:
        rho_star_mean = np.mean(cond_rho_values)
        rho_star_std = np.std(cond_rho_values, ddof=1)
    else:
        rho_star_mean = 0
        rho_star_std = 0
    mask = np.array(exp_density_values) > 0
    exp_density_values = np.array(exp_density_values)[mask]
    exp_density_mean = np.mean(exp_density_values) if len(exp_density_values) > 0 else -1
    exp_density_std = np.std(exp_density_values) if len(exp_density_values) > 0 else -1

    # print(f"Condensed-phase density (ρ*): {rho_star_mean} ± {rho_star_std}")
    # print(f"Expenditure density: {exp_density_mean} ± {exp_density_std}")

    return rho_star_mean, rho_star_std, exp_density_mean, exp_density_std #, rho_star_samples, exp_density_samples, P_bootstrapped  






# def bootstrap_eos_analysis(rho_values, P_matrix, err_array, num_bootstrap=500, work=15, confidence=0.01, threshold_fraction=0.5):
#     """
#     Bootstraps the EOS to compute both condensed-phase density (ρ*) and expenditure density (W/m),
#     ensuring statistical significance of negative pressures is checked **before** bootstrapping.

#     Parameters:
#     -----------
#     rho_values : array-like
#         Array of density values from simulations.
#     P_matrix : 2D array (shape: n_rho × n_samples)
#         Independent pressure values at each density.
#     err_array : 1D array (shape: n_rho,)
#         Standard error associated with the independent samples at each density.
#     num_bootstrap : int, optional
#         Number of bootstrap resamples (default is 1000).
#     target_exp : float, optional
#         The target integral value for expenditure density (default is 15).
#     confidence : float, optional
#         Confidence level for testing statistically significant negative pressures (default is 0.01).
#     threshold_fraction : float, optional
#         Fraction of bootstrap samples that must show significant phase separation for ρ* to be computed (default is 0.5).

#     Returns:
#     --------
#     rho_star_mean : float
#         Mean of bootstrapped condensed-phase density (ρ*). Returns 0 if threshold not met.
#     rho_star_std : float
#         Standard deviation of ρ* (uncertainty estimate). Returns 0 if threshold not met.
#     exp_density_mean : float
#         Mean of bootstrapped expenditure density (W/m).
#     exp_density_std : float
#         Standard deviation of W/m.
#     rho_star_samples : array-like
#         Full set of bootstrapped ρ* values. Empty if threshold not met.
#     exp_density_samples : array-like
#         Full set of bootstrapped expenditure densities.
#     P_bootstrapped : 2D array (shape: num_bootstrap × len(rho_values))
#         Bootstrapped pressure values for each density.
#     """

#     rho_star_samples = []
#     exp_density_samples = []
#     P_bootstrapped = []

#     valid_curves = 0  # Count of EOS curves showing phase separation

#     rhos = np.insert(rho_values, 0, 0)  # Include zero density

#     # Perform Bootstrapping
#     for _ in range(num_bootstrap):
#         # Bootstrap: Resample pressures at each density and take the mean
#         P_boot = np.array([np.mean(np.random.choice(P_vals, size=len(P_vals), replace=True)) for P_vals in P_matrix])
#         #print(P_boot, flush=True)
#         P_boot = np.insert(P_boot, 0, 0)  # Include zero pressure
#         #P_bootstrapped.append(P_boot)

#         P_spline = CubicSpline(rhos, P_boot, bc_type=((1, 0.0), (2, 0.0)))

#         # Check if at least one pressure is statistically significantly negative
#         has_negative = False
#         for i in range(len(rho_values)):
#             P = P_boot[i]
#             err = err_array[i]

#             if P < 0:
#                 # Perform a t-test for statistical significance
#                 t_stat = P / err
#                 p_value = scipy.stats.t.cdf(t_stat, df=4)  # 4 degrees of freedom (assumed)
                
#                 if p_value <= confidence:
#                     has_negative = True
#                     break  # Stop checking if we already found one

#         if has_negative:
#             valid_curves += 1  # Count valid EOS curves

#             # --- Compute ρ* (Condensed-Phase Density) ---
            

#             def f(x):
#                 return P_spline(x)
            
#             xval = np.linspace(1e-6, max(rho_values) + 0.2, 1000)

#             roots = []
#             for i in range(len(xval) - 1):
#                 if f(xval[i])*f(xval[i+1]) < 0:  # Sign change detected
#                     try:
#                         root = root_scalar(f, bracket=[xval[i], xval[i + 1]], method='brentq').root
#                         roots.append(root)
#                     except:
#                         pass
            
#             if roots:
#                 rho_star_samples.append(max(roots))  # Store the highest density root

        

        

#         # --- Compute Expenditure Density (W/m) ---
#         #P_spline = CubicSpline(rhos, P_boot, bc_type=((1, 0.0), (2, 0.0)))
#         #exp_density_samples.append(calc_exp_density(P_boot, rhos, work=work))
#         exp_density_samples.append(calc_exp_density(P_spline, np.min(rho_values), np.max(rhos), work=work))



#     print(f"Valid phase-separating curves: {valid_curves}/{num_bootstrap} ({(valid_curves/num_bootstrap)*100:.2f}%)", flush=True)      

#     # Convert to NumPy arrays
#     rho_star_samples = np.array(rho_star_samples)
#     exp_density_samples = np.array(exp_density_samples)
#     #print(exp_density_samples)
#     exp_density_samples = exp_density_samples[exp_density_samples > 0]
#     #P_bootstrapped = np.array(P_bootstrapped)

#     # Compute statistics for W/m
#     exp_density_mean = np.mean(exp_density_samples) if len(exp_density_samples) > 0 else -1
#     exp_density_std = np.std(exp_density_samples) if len(exp_density_samples) > 0 else -1

#     # Step 3: Decide Whether to Compute ρ*
#     if valid_curves / num_bootstrap >= threshold_fraction:
#         # Compute statistics for ρ*
#         rho_star_mean = np.mean(rho_star_samples) if len(rho_star_samples) > 0 and np.mean(rho_star_samples) > 0.05 else 0
#         rho_star_std = np.std(rho_star_samples) if len(rho_star_samples) > 0 and np.mean(rho_star_samples) > 0.05 else 0
#     else:
#         # If threshold is not met, return zero for condensed-phase density
#         rho_star_mean = 0
#         rho_star_std = 0
#         rho_star_samples = np.array([])

#     return rho_star_mean, rho_star_std, exp_density_mean, exp_density_std #, rho_star_samples, exp_density_samples, P_bootstrapped

def bootstrap_exp_dens_from_path(pth, frac, num_bootstrap=200, work=15, confidence=0.01, threshold_fraction=0.5):
    """
    Bootstraps the EOS to compute both condensed-phase density (ρ*) and expenditure density (W/m),
    ensuring statistical significance of negative pressures is checked **before** bootstrapping.

    Parameters:
    -----------
    path : string
        Path to simulation results directory for a sequence.
    frac : float
        Fraction of pressure data to use for EOS calculation.
    num_bootstrap : int, optional
        Number of bootstrap resamples (default is 1000).
    target_exp : float, optional
        The target integral value for expenditure density (default is 15).
    confidence : float, optional
        Confidence level for testing statistically significant negative pressures (default is 0.01).
    threshold_fraction : float, optional
        Fraction of bootstrap samples that must show significant phase separation for ρ* to be computed (default is 0.5).

    Returns:
    --------
    exp_density_mean : float
        Mean of bootstrapped expenditure density (W/m).
    exp_density_std : float
        Standard deviation of W/m.
    """

    labels = ['TimeStep', 'temp', 'etot', 'pe', 'ke', 'ent', 'P', 'rho']
    P = []
    rho = []
    err = []
    
    for density in sorted(os.listdir(pth)):
        subpath = os.path.join(pth, f'{density}')
        try:

            #file = path + f'/{density}/thermo.avg'
            file = os.path.join(subpath, f"thermo.avg")
            if os.path.exists(file):
                data = pd.read_csv(file,delimiter=' ',header=None,names=labels,skiprows=2)
                data.dropna(inplace=True)

                N = int((1-frac)*len(data)) # get frac*100 % of the data (eg frac 0.8 => 80%)

                # Fixed 5-block split-error (see get_EOS for rationale).
                n_samples = 5

                P.append(split_error(data['P'][N:].values, n_samples)[1])
                err.append(split_error(data['P'][N:].values, n_samples)[0])
                rho.append(float(data['rho'].values[-1]))
            else:
                continue

        except Exception as e:
            print(f'Error reading from {subpath}: {e}')

    P_matrix = P # shape: n_states × m_independent_pressures
    # print(P_matrix, flush=True)
    err_array = err # shape: n_states

    exp_density_samples = []
    #P_bootstrapped = []

    valid_curves = 0  # Count of EOS curves showing phase separation

    rhos = np.insert(rho, 0, 0)  # Include zero density

    # Bootstrapping
    for _ in range(num_bootstrap):
        # Resample pressures at each density and take the mean
        P_boot = np.array([np.mean(np.random.choice(P_vals, size=len(P_vals), replace=True)) for P_vals in P_matrix])
        # print(P_boot, flush=True)
        P_boot = np.insert(P_boot, 0, 0)  # Include zero pressure
        #P_bootstrapped.append(P_boot)

        P_spline = CubicSpline(rhos, P_boot, bc_type=((1, 0.0), (2, 0.0)))

        # # Check if at least one pressure is statistically significantly negative
        # has_negative = False
        # for i in range(len(rho)):
        #     P = P_boot[i]
        #     err = err_array[i]

        #     if P < 0:
        #         # Perform a t-test for statistical significance
        #         t_stat = P / err
        #         p_value = scipy.stats.t.cdf(t_stat, df=4)  # 4 degrees of freedom (assumed)
                
        #         if p_value < confidence:
        #             has_negative = True
        #             break  # Stop checking if we already found one

        #P_spline = CubicSpline(rhos, P_boot, bc_type=((1, 0.0), (2, 0.0)))
        exp_density_samples.append(calc_exp_density(P_spline, np.min(rhos[1:]), np.max(rhos), work=work))

    #print(f"Valid phase-separating curves: {valid_curves}/{num_bootstrap} ({(valid_curves/num_bootstrap)*100:.2f}%)", flush=True)      

    # Convert to np arrays
    exp_density_samples = np.array(exp_density_samples)
    exp_density_samples = exp_density_samples[exp_density_samples > 0]
    # print(exp_density_samples)
    #P_bootstrapped = np.array(P_bootstrapped)

    # Compute statistics for W/m
    exp_density_mean = np.mean(exp_density_samples) if len(exp_density_samples) > 0 else -1
    #exp_density_std = np.std(exp_density_samples) if len(exp_density_samples) > 0 else 0

    return exp_density_mean #, exp_density_std #, rho_star_samples, exp_density_samples, P_bootstrapped


#def calc_exp_density(Pinp, rhoinp, work=15, verbose=False):
def calc_exp_density(spline, rhomin, rhomax, work=15, verbose=False):
    '''Calculates the "expenditure density" from P and an alloted work

    Parameters
    ----------
    P : array_like
        Array of pressure values
    work : float
        Work parameter in the calculation of expenditure density
        
    Returns
    -------
    root : float
        Density at which the integral of P-rho curve equals work
    '''

    # Cubic spline where first derivative at beginning is zero and second derivative at end is zero
  
    cs = spline

    # Calculate values at points x along the spline to get y:q:
    #if cond == True:
    if rhomax > 1.6:
        x = np.linspace(0.000001,rhomax+0.4,10000)
    else:
        x = np.linspace(0.000001, rhomax+0.2, 10000)
    y = np.asarray([cs(z)/z**2 for z in x])



    # Calculates work at each point along the spline and then calculates where that equals the set work.
    # np.trapezoid (numpy>=2.0) is the non-deprecated name; np.trapz still works
    # on 2.x as an alias but is removed on newer numpys — the cluster hits that.
    # hasattr guard (not `getattr(..., np.trapz)`) so we don't touch np.trapz at
    # all when it's removed — a plain reference is enough to crash.
    _trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    roots = [_trap(y[:i+1], x[:i+1]) for i in range(len(x))]
    for i, value in enumerate(roots):
        if value > work:
            root = x[i]
            return root

    return -1


if __name__ == "__main__":
    main()
