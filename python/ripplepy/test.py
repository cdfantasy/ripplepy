import ripplepy
from pathlib import Path
import numpy as np
from deap import base, creator, tools
from multiprocessing import Pool
import random
import os
import glob
import pandas as pd
from func_timeout import func_timeout, FunctionTimedOut


def find_axis_with_timeout():
    """包装 ripplepy.find_axis()，设置 0.2 秒超时"""
    try:
        func_timeout(0.2, ripplepy.find_axis)
        return ripplepy.globalvariables.istate
    except FunctionTimedOut:
        print(f"find_axis timed out after 0.2s", flush=True)
        ripplepy.globalvariables.istate = -1  # 设置无效状态
        return -1
    
def effective_ripple_with_timeout():
    """包装 ripplepy.effective_ripple()，设置 20 秒超时"""
    try:
        func_timeout(10, ripplepy.effective_ripple)
        return ripplepy.globalvariables.epsilon_eff, ripplepy.globalvariables.istate
    except FunctionTimedOut:
        print(f"effective_ripple timed out after 20s", flush=True)
        ripplepy.globalvariables.istate = -1  # 设置无效状态
        return np.nan, -1

# 定义目标函数
def objective_function(x, gen=0, ind_idx=0):

    print(f"Gen {gen}, Ind {ind_idx}")
    
    nfp = ripplepy.globalvariables.nfp
    extcur = ripplepy.globalvariables.extcur
    n_dim = 4
    x_values = np.array(x[:])

    extcur[1:5] = x_values


    info_dict = {
        'gen': gen,
        'ind_idx': ind_idx,
        'epsilon': np.nan,
        'iota': np.nan,
        'asp': np.nan,
        'rm': np.nan,
        'am': np.nan,
        'volume': np.nan,
        'Baxis' : np.nan,
        'Bboundary': np.nan
    }

    x_values_dict = {
        'gen': gen,
        'ind_idx': ind_idx,
        'x_values': str(x[:])  # 转换为字符串以保存到 CSV
    }
    ripplepy.sum_bfield()
    ripplepy.compute_bspline_coeffs()

    istate = find_axis_with_timeout()
    istate = ripplepy.globalvariables.istate
    if istate < 0:
        return 1e3,info_dict, x_values_dict
    axisline = ripplepy.globalvariables.axisline
    rzphi = axisline[0, 0:3]
    rzphi[0] = rzphi[0] + deltaR
    rzphi[1:3] = 0
    ripplepy.globalvariables.rzphi = rzphi
    outputfilename = f"grid_qa_gen{gen}_ind{ind_idx}.nc"
    ripplepy.globalvariables.output_filename = outputfiledir / outputfilename

    # print(f"Gen {gen}, Ind {ind_idx}","epsilon")
    effective_ripple, istate = effective_ripple_with_timeout()

    if istate < 0 or  np.isnan(effective_ripple):
        ripplepy.globalvariables.istate = -1
        ripplepy.write_output()
        return 1e3, info_dict, x_values_dict
    

    ripplepy.iota_clt()
    ripplepy.volume()
    ripplepy.write_output()

    iota = ripplepy.globalvariables.iota
    asp = ripplepy.globalvariables.asp
    rm = ripplepy.globalvariables.rm
    am = ripplepy.globalvariables.am
    volume = ripplepy.globalvariables.vol
    Baxis = ripplepy.globalvariables.baxis
    Bboundary =  ripplepy.globalvariables.bboundary

    info_dict.update({
        'epsilon': effective_ripple,
        'iota': iota,
        'asp': asp,
        'rm': rm,
        'am': am,
        'volume': volume,
        'Baxis' : Baxis,
        'Bboundary': Bboundary
    })
    return effective_ripple, info_dict, x_values_dict  

# DEAP 设置
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# 初始化个体和种群
def init_individual(bounds, n_dim):
    return [random.uniform(bounds[i, 0], bounds[i, 1]) for i in range(n_dim)]

def init_population(n_pop, bounds, n_dim):
    return [creator.Individual(init_individual(bounds, n_dim)) for _ in range(n_pop)]

# DE 变异操作
def mutate(individual, population, F=0.5):
    size = len(individual)
    idxs = [i for i in range(len(population)) if population[i] != individual]
    r1, r2, r3 = random.sample(idxs, 3)
    mutant = [0] * size
    for i in range(size):
        mutant[i] = population[r1][i] + F * (population[r2][i] - population[r3][i])
        mutant[i] = max(bounds[i, 0], min(bounds[i, 1], mutant[i]))
    return mutant

# DE 交叉操作
def crossover(individual, mutant, CR=0.7):
    size = len(individual)
    trial = individual.copy()
    j_rand = random.randint(0, size - 1)
    for i in range(size):
        if random.random() < CR or i == j_rand:
            trial[i] = mutant[i]
    return trial

# 并行评估
def init_process():
    global outputfiledir, deltaR



def evaluate_population(population, evaluate_func, gen, processes=None):
    with Pool(initializer=init_process, processes=processes) as pool:
        args = [(ind, gen, i) for i, ind in enumerate(population)]
        results = pool.starmap(evaluate_func, args)
    fitnesses = [result[0] for result in results]
    infos = [result[1] for result in results]
    x_values_infos = [result[2] for result in results]
    return fitnesses, infos, x_values_infos

# 日志保存（原物理量日志）
def save_log(infos):
    temp_log_file = outputfiledir / f"temp_log_{os.getpid()}.csv"
    df = pd.DataFrame(infos)
    with open(temp_log_file, 'a', newline='') as f:
        df.to_csv(f, index=False, header=not temp_log_file.exists())

# 新增 x_values 日志保存
def save_x_values_log(x_values_infos):
    temp_x_log_file = outputfiledir / f"temp_x_log_{os.getpid()}.csv"
    df = pd.DataFrame(x_values_infos)
    with open(temp_x_log_file, 'a', newline='') as f:
        df.to_csv(f, index=False, header=not temp_x_log_file.exists())

# DE 算法
def differential_evolution(n_dim, initial_bounds, n_pop=100, max_gen=100, F=0.5, CR=0.7, processes=8):
    toolbox = base.Toolbox()
    toolbox.register("individual", init_individual, bounds=initial_bounds, n_dim=n_dim)
    toolbox.register("population", init_population, n_pop=n_pop, bounds=initial_bounds, n_dim=n_dim)
    toolbox.register("evaluate", objective_function)
    toolbox.register("mutate", mutate, F=F)
    toolbox.register("crossover", crossover, CR=CR)
    
    pop = toolbox.population()
    fitnesses, infos, x_values_infos = evaluate_population(pop, toolbox.evaluate, gen=0, processes=processes)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = (float(fit),)
    save_log(infos)
    save_x_values_log(x_values_infos)
    
    invalid_count = {i: 0 for i in range(n_pop)}
    for gen in range(max_gen):
        best_ind = tools.selBest(pop, 1)[0]
        bounds = initial_bounds
               
        trials = []
        trial_inds = []
        for i in range(n_pop):
            mutant = toolbox.mutate(pop[i], pop)
            trial = toolbox.crossover(pop[i], mutant)
            trial_ind = creator.Individual(trial)
            trials.append(trial_ind)
            trial_inds.append(trial_ind)
            
        fitnesses, infos, x_values_infos = evaluate_population(trials, toolbox.evaluate, gen=gen+1, processes=processes)
        invalid_solutions = 0
        for trial_ind, fit in zip(trial_inds, fitnesses):
            trial_ind.fitness.values = (float(fit),)
            if fit >= 1e3:
                invalid_solutions += 1
        
        for i in range(n_pop):
            trial_fitness = trial_inds[i].fitness.values[0]
            current_fitness = pop[i].fitness.values[0]
            print(f"Gen {gen+1}, Ind {i}, Current fitness = {current_fitness}, Trial fitness = {trial_fitness}")
            if trial_fitness >= 1e3:
                invalid_count[i] += 1
            else:
                invalid_count[i] = 0
            if invalid_count[i] >= 3:
                pop[i] = creator.Individual(init_individual(bounds, n_dim))
                pop[i].fitness.values = (float(objective_function(pop[i], gen=gen+1, ind_idx=i)[0]),)
                invalid_count[i] = 0
            elif trial_fitness < 1e3 or trial_fitness <= current_fitness:
                pop[i] = trial_inds[i]
 
        save_log(infos)
        save_x_values_log(x_values_infos)
        print(f"Generation {gen+1}, Invalid solutions: {invalid_solutions}/{n_pop} ({invalid_solutions/n_pop*100:.2f}%)")
        best_ind = tools.selBest(pop, 1)[0]
        print(f"Generation {gen+1}, Best Fitness: {best_ind.fitness.values[0]}")
    
    best_ind = tools.selBest(pop, 1)[0]
    return best_ind, best_ind.fitness.values[0]

# 主程序
if __name__ == "__main__":
    inputfiledir = Path("./inputfiles")
    outputfiledir = Path("./outputfiles")
    inputfilename = "mgrid_h1_design.nc"
    ripplepy.globalvariables.mgrid_filename = inputfiledir / inputfilename

    ripplepy.read_grid()
    ripplepy.allocate_extcur()
    nfp = 3
    deltaR = 0.08
    ripplepy.globalvariables.mpol = 100
    ripplepy.globalvariables.axis0 = [1.23,0.0]  # 轴心位置(R,Z)

    ripplepy.globalvariables.nfp = nfp
    ripplepy.globalvariables.extcur = [50000, 5000, 0, -80000, -40000]
    extcur = ripplepy.globalvariables.extcur
    
    n_dim = 4
    x = extcur[1:]
    bounds = np.zeros((n_dim, 2))
    for i in range(n_dim):
        bounds[i, 0] = x[i] - 0.1 * abs(x[i])
        bounds[i, 1] = x[i] + 0.1 * abs(x[i])
    
    n_pop = 16
    processes = n_pop
    max_gen = 10
    F = 0.5
    CR = 0.7
    
    print("Running Differential Evolution...")
    best_ind, best_fitness = differential_evolution(n_dim, bounds, n_pop, max_gen, F, CR, processes)
    print(f"DE Best Solution: {best_ind}")
    print(f"DE Best Fitness: {best_fitness}")