import numpy as np
import h5py
from pathlib import Path
from functools import partial
from multiprocessing import Pool
import random
from .ripple import find_axis, compute_epstot,calculate_plasma_params,set_extcur,compute_initial_gradpsi_nemov,set_trace_parameters
import csv


def objective_function(extcur_free,extcur_fixed,Generation,Individual, initial_rz, nturn, nphi,delt_r=0.05):

    extcur = np.concatenate((extcur_fixed, extcur_free))
    extcur = set_extcur(extcur)
    failure_flag = False
    fieldline_data = np.zeros((nturn*nphi, 20), dtype=np.float64, order='F')
    axis_rz, R0, axis_fieldline, success = find_axis(initial_rz, xtol=1e-5, max_iter=100,delta_r=0.01, verbose=False)
    if success:
        print(f"✓ Magnetic axis found at R={axis_rz[0]:.10f}, Z={axis_rz[1]:.10f}, R0={R0:.10f}")
        RZ = np.array([axis_rz[0]+delt_r, axis_rz[1]], dtype=np.float64, order='F')
        initial_gradpsi = compute_initial_gradpsi_nemov(extcur, RZ[0],RZ[1],verbose=True)
        set_trace_parameters(nturn, nphi, verbose=False)
        epsilon_eff, bboundary ,fieldline_data,trace_istate= compute_epstot(R0, extcur, RZ, initial_gradpsi, fieldline_data, return_fieldline=True)
        if trace_istate == 0:
            vol,Am,iota = calculate_plasma_params(fieldline_data, axis_fieldline, nturn, nphi, R0)
            print(f"✓ Plasma parameters calculated: Volume={vol:.3f}, Major radius={Am:.3f}, iota={iota:.3f}")
        else:
            failure_flag = True
            print("Fieldline tracing failed during compute_epstot")
            epsilon_eff = 1e4
            bboundary = np.nan
            vol,Am,iota = np.nan, np.nan, np.nan
    else:
        failure_flag = True
        print("Fieldline tracing failed during compute_epstot")
        epsilon_eff = 1e4
        bboundary = np.nan
        vol,Am,iota = np.nan, np.nan, np.nan

    Individual_info = {
        'Generation': Generation,
        'Individual': Individual,
        'extcur': extcur,
        'epsilon_eff': epsilon_eff,
        'iota': iota,
        'volume': vol,
        'major radius': Am,
        'average B': bboundary,
        'failure_flag': failure_flag
    }

    return epsilon_eff, Individual_info, fieldline_data

def save_hdf5(Individual_info, fieldline_data, output_dir=None, device_name=None):
    if output_dir is None:
        output_dir = Path(".")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if device_name is None:
        filename = output_dir / f"Gen{Individual_info['Generation']}_Ind{Individual_info['Individual']}.h5"
    else:
        filename = output_dir / f"{device_name}_Gen{Individual_info['Generation']}_Ind{Individual_info['Individual']}.h5"
    base_filename = filename
    suffix = 1
    while filename.exists():
        filename = base_filename.with_name(f"{base_filename.stem}_{suffix}{base_filename.suffix}")
        suffix += 1
    with h5py.File(filename, 'w') as f:
        for key, value in Individual_info.items():
            if isinstance(value, np.ndarray):
                f.create_dataset(key, data=value)
            elif value is None:
                f.attrs[key] = np.nan
            elif isinstance(value, (list, tuple)):
                f.create_dataset(key, data=np.asarray(value))
            else:
                f.attrs[key] = value
        f.create_dataset('fieldline_data', data=fieldline_data)
    print(f"Saved results to {filename}")

def save_Individual_info_list2csv(Individual_info_list, filename=None):
    if filename is None:
        filename = "Individual_info_list.csv"
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    def _csv_value(value):
        if isinstance(value, np.ndarray):
            return np.array2string(value, separator=', ')
        if value is None:
            return ""
        return value
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Generation', 'Individual', 'extcur', 'epsilon_eff', 'iota', 'volume', 'major radius', 'average B', 'failure_flag'])
        for Individual_info in Individual_info_list:
            writer.writerow([
                Individual_info['Generation'],
                Individual_info['Individual'],
                _csv_value(Individual_info['extcur']),
                Individual_info['epsilon_eff'],
                Individual_info['iota'],
                Individual_info['volume'],
                Individual_info['major radius'],
                Individual_info['average B'],
                Individual_info['failure_flag']
            ])
    print(f"Saved Individual info list to {filename}")


def init_individual(bounds, n_dim):
    return [random.uniform(bounds[i, 0], bounds[i, 1]) for i in range(n_dim)]


def init_population(n_pop, bounds, n_dim):
    return [init_individual(bounds, n_dim) for _ in range(n_pop)]


def mutate(individual, population, bounds, F=0.5):
    size = len(individual)
    idxs = [i for i in range(len(population)) if population[i] != individual]
    r1, r2, r3 = random.sample(idxs, 3)
    mutant = [0.0] * size
    for i in range(size):
        mutant[i] = population[r1][i] + F * (population[r2][i] - population[r3][i])
        mutant[i] = max(bounds[i, 0], min(bounds[i, 1], mutant[i]))
    return mutant


def crossover(individual, mutant, CR=0.7):
    size = len(individual)
    trial = individual.copy()
    j_rand = random.randint(0, size - 1)
    for i in range(size):
        if random.random() < CR or i == j_rand:
            trial[i] = mutant[i]
    return trial


def evaluate_individual(individual, gen, ind_idx, extcur_fixed, initial_rz, nturn, nphi, delt_r=0.05):
    return objective_function(individual, extcur_fixed, gen, ind_idx, initial_rz, nturn, nphi, delt_r)


def evaluate_population(population, evaluate_func, gen, processes=None):
    with Pool(processes=processes) as pool:
        args = [(ind, gen, i) for i, ind in enumerate(population)]
        results = pool.starmap(evaluate_func, args)
    return results


def differential_evolution(extcur_fixed, initial_rz, nturn, nphi, initial_bounds,
                           n_pop=100, max_gen=100, F=0.5, CR=0.7,
                           processes=8, delt_r=0.05, output_dir=None,
                           csv_filename="Individual_info_list.csv", device_name=None):
    output_dir = Path(".") if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_individual_infos = []
    pop_fitnesses = []
    n_dim = len(initial_bounds)
    evaluate_func = partial(
        evaluate_individual,
        extcur_fixed=np.asarray(extcur_fixed, dtype=np.float64),
        initial_rz=np.asarray(initial_rz, dtype=np.float64),
        nturn=nturn,
        nphi=nphi,
        delt_r=delt_r,
    )

    pop = init_population(n_pop, initial_bounds, n_dim)
    results = evaluate_population(pop, evaluate_func, gen=0, processes=processes)
    for fit, info, fieldline_data in results:
        pop_fitnesses.append(float(fit))
        all_individual_infos.append(info)
        save_hdf5(info, fieldline_data, output_dir=output_dir, device_name=device_name)

    invalid_count = {i: 0 for i in range(n_pop)}
    for gen in range(max_gen):
        trials = []
        for i in range(n_pop):
            mutant = mutate(pop[i], pop, initial_bounds, F=F)
            trial = crossover(pop[i], mutant, CR=CR)
            trials.append(trial)

        results = evaluate_population(trials, evaluate_func, gen=gen + 1, processes=processes)
        invalid_solutions = 0
        trial_fitnesses = []
        for fit, info, fieldline_data in results:
            trial_fitnesses.append(float(fit))
            all_individual_infos.append(info)
            save_hdf5(info, fieldline_data, output_dir=output_dir, device_name=device_name)
            if fit >= 1e3:
                invalid_solutions += 1

        for i in range(n_pop):
            trial_fitness = trial_fitnesses[i]
            current_fitness = pop_fitnesses[i]
            print(f"Gen {gen+1}, Ind {i}, Current fitness = {current_fitness}, Trial fitness = {trial_fitness}")
            if trial_fitness >= 1e3:
                invalid_count[i] += 1
            else:
                invalid_count[i] = 0
            if invalid_count[i] >= 3:
                pop[i] = init_individual(initial_bounds, n_dim)
                recheck_fitness, recheck_info, recheck_fieldline_data = objective_function(
                    np.asarray(pop[i], dtype=np.float64),
                    np.asarray(extcur_fixed, dtype=np.float64),
                    gen + 1,
                    i,
                    np.asarray(initial_rz, dtype=np.float64),
                    nturn,
                    nphi,
                    delt_r,
                )
                pop_fitnesses[i] = float(recheck_fitness)
                all_individual_infos.append(recheck_info)
                save_hdf5(recheck_info, recheck_fieldline_data, output_dir=output_dir, device_name=device_name)
                invalid_count[i] = 0
            elif trial_fitness < 1e3 or trial_fitness <= current_fitness:
                pop[i] = trials[i]
                pop_fitnesses[i] = trial_fitness

        print(f"Generation {gen+1}, Invalid solutions: {invalid_solutions}/{n_pop} ({invalid_solutions/n_pop*100:.2f}%)")
        best_index = int(np.argmin(pop_fitnesses))
        print(f"Generation {gen+1}, Best Fitness: {pop_fitnesses[best_index]}")

    best_index = int(np.argmin(pop_fitnesses))
    best_ind = pop[best_index]
    save_Individual_info_list2csv(all_individual_infos, output_dir / csv_filename)
    return best_ind, pop_fitnesses[best_index], all_individual_infos

