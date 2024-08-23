# -*- coding: UTF-8 -*-

"""
@Author  ：Zijin Qiu
"""
import copy
import time
import numpy as np

from step5_modeling import initial_solution,decode_solution,model_file_create,multiarchfl_archsf
from utilize_qzj_SAP_sheets import *
from step7_penalty_terms import get_fitness1,get_fitness2,sum_dosage_and_walllength,counter_walltype,sum_dosage_Steel
from step3_get_variables import get_sf_parametric_walls
import math
import numpy

def population_initialization(tgl,dxf,population,parameters):
    (archsf_height_num, wall_t_s, beam_section_wl, colume_section, beam_section_hl,
     peoperty_material_column, length_step,_,_) = parameters
    initial_solution_population = []
    for i in range(population):
        initial_solution_this = initial_solution(tgl, dxf, archsf_height_num, wall_t_s, beam_section_wl, colume_section,beam_section_hl,
                                       peoperty_material_column,length_step)
        initial_solution_population.append(initial_solution_this)

    return initial_solution_population

def evolve(tgl_class,dxf_class,population_initial, parameters):
    population_evolve_initial = copy.deepcopy(population_initial)
    (archsf_height_num, wall_t_s, beam_section_wl, colume_section, beam_section_hl,
     peoperty_material_column, length_step, indicators_limit, dict_cg_factor) = parameters

    def Mutation(population_evolve, mutation_p=16):
        # 确保变异的个体数量不超过种群大小
        mutation_p = min(len(population_evolve), mutation_p)
        # 随机选择要变异的个体
        mutation_indices = random.sample(range(len(population_evolve)), mutation_p)
        # 随机选择变异的位置
        mutation_position = random.choice(range(7))

        # 执行变异过程
        for idx in mutation_indices:
            pop = population_evolve[idx]
            pop_mutation = initial_solution(tgl_class['1'], dxf_class['1'], archsf_height_num, wall_t_s, beam_section_wl, colume_section,
                                            beam_section_hl, peoperty_material_column, length_step)

            # 执行变异
            if mutation_position == 0:
                # 墙体变异 k=10 墙体变异阈值
                mutation_positions_wall = sorted(random.sample(range(len(pop[0])), k=26))
                for i in mutation_positions_wall:
                    pop[0][i] = pop_mutation[0][i]

            elif mutation_position == 1:
                # 墙厚变异
                # mutation_position_wallthick = random.choice(range(len(pop[data])))
                # pop[data][mutation_position_wallthick] = pop_mutation[data][mutation_position_wallthick]
                pop[1] = pop_mutation[1]

            elif mutation_position == 2:
                # 梁混凝土截面变异
                # mutation_position_beam_c = random.choice(range(len(pop[2])))
                # pop[2][mutation_position_beam_c] = pop_mutation[2][mutation_position_beam_c]
                pop[2] = pop_mutation[2]
            elif mutation_position == 3:
                # 梁钢筋截面变异
                # mutation_position_beam_steel = random.choice(range(len(pop[3])))
                # pop[3][mutation_position_beam_steel] = pop_mutation[3][mutation_position_beam_steel]
                pop[3] = pop_mutation[3]
            elif mutation_position == 4:
                # 柱混凝土截面变异
                # mutation_position_column = random.choice(range(len(pop[4])))
                # pop[4][mutation_position_column] = pop_mutation[4][mutation_position_column]
                pop[4] = pop_mutation[4]
            elif mutation_position == 5:
                # 材料属性变异
                # mutation_position_material = random.choice(range(len(pop[5])))
                # pop[5][mutation_position_material] = pop_mutation[5][mutation_position_material]
                pop[5] = pop_mutation[5]
            elif mutation_position == 6:
                # 墙体长度变化变异
                pop[6] = pop_mutation[6]


            # 更新变异后的个体
            population_evolve[idx] = pop

        return population_evolve

    def crossover(population_evolve,crossover_p = 12):
        # 确保交叉的个体数量不超过种群大小且为偶数
        crossover_p = min(len(population_evolve), crossover_p)
        if crossover_p % 2 != 0:
            crossover_p -= 1
        # 随机选择要交叉的个体
        crossindex = random.sample(range(len(population_evolve)), crossover_p)
        # 随机选择交叉的位置
        # crosspositon = sorted(random.sample(range(6), k=2))
        # 执行交叉过程
        for i in range(0, crossover_p, 2):
            idx1, idx2 = crossindex[i], crossindex[i + 1]
            pop1, pop2 = population_evolve[idx1], population_evolve[idx2]
            # 随机选择交叉的位置
            crosspositon = sorted(random.sample(range(6), k=2))
            # 如果选择的是墙体交叉
            if crosspositon[0] == 0:
                # 随机选择墙体交叉的位置
                crosspositon_wall = sorted(random.sample(range(len(pop1[0])), k=2))
                # 交换选定位置的基因
                DNA_wall1, DNA_wall2 = (pop1[0][crosspositon_wall[0]:crosspositon_wall[1]],
                                        pop2[0][crosspositon_wall[0]:crosspositon_wall[1]])
                pop1[0][crosspositon_wall[0]:crosspositon_wall[1]], pop2[0][crosspositon_wall[0]:crosspositon_wall[
                    1]] = DNA_wall2, DNA_wall1
                # 执行每个属性的交叉
                for position in range(1, crosspositon[1] + 1):
                    # 交换选定位置的基因
                    pop1[position], pop2[position] = pop2[position], pop1[position]
            else:
                # 执行每个属性的交叉
                for position in range(crosspositon[0], crosspositon[1] + 1):
                    # 交换选定位置的基因
                    pop1[position], pop2[position] = pop2[position], pop1[position]

            population_evolve[idx1], population_evolve[idx2] = pop1, pop2

        return population_evolve

    population_evolve_cross = crossover(population_evolve_initial)
    population_evolve = Mutation(population_evolve_cross)

    return population_evolve

def select(tgl_class,dxf_class,population,parameters,iter,threshold = 0):

    def calculate_fitness(solution,sdbpath):
        (list_strsfs_wall, list_strsfs_beam, list_strsfs_wt, list_strsfs_bw,
         beam_concrete_section_selection, beam_steel_section, column_section_selection,
         peoperty_material) = solution
        # 惩罚项
        solution_indicator = run_sap2000(sdbpath)
        penality = get_fitness1(solution_indicator, indicators_limit)
        # 目标函数
        # data）获取 混凝土用量 m3

        arr_strsfs_c_dosage, arr_strsfs_wl = sum_dosage_and_walllength(archsf_height_num, list_strsfs_wall,
                                                                       list_strsfs_wt, column_section_selection,
                                                                       len(tgl_class['1'].column_coord_all),
                                                                       list_strsfs_beam, list_strsfs_bw,
                                                                       beam_concrete_section_selection)

        # 2）获取 钢材用量 m3
        arr_strsfs_s_dosage = sum_dosage_Steel(archsf_height_num, list_strsfs_beam, beam_steel_section)

        # 3) 墙体类型-惩罚值
        coun_short_c, coun_short_sw, coun_toolong = counter_walltype(list_strsfs_wall, list_strsfs_wt,
                                                                     archsf_height_num)

        fitness_values = [arr_strsfs_c_dosage, arr_strsfs_wl, arr_strsfs_s_dosage, peoperty_material, coun_short_c,
                          coun_short_sw, coun_toolong]
        fx_1,fx_2 = get_fitness2(fitness_values, dict_cg_factor)

        return [fx_1,fx_2,penality]

    (archsf_height_num, wall_t_s, beam_section_wl, colume_section, beam_section_hl,
     peoperty_material_column, length_step,indicators_limit,dict_cg_factor) = parameters

    fitness_ = []
    fx1_list = []
    fx2_list = []
    for i,individual in enumerate(population):
        individual_decode = decode_solution(tgl_class,dxf_class, archsf_height_num, beam_section_wl,
                                            peoperty_material_column,length_step,individual)
        filenamelist = model_file_create(tgl_class,individual_decode,archsf_height_num,iter,i+1)
        sdbpath = start(filenamelist,iter)
        [fx_1,fx_2,penality] = calculate_fitness(individual_decode,sdbpath)
        fitness_.append([fx_1,fx_2,penality])
        fx1_list.append(fx_1)
        fx2_list.append(fx_2)

    # 支配选择
    if threshold != 0:
        population_select = population
        fx1_select = fx1_list
        fx2_select = fx2_list

    else:
        # 先计算支配等级
        pareto = []
        for index, (fx_1, fx_2, penality) in enumerate(fitness_):
            pareto_grade = 1
            for index_, (fx_1_, fx_2_, penality_) in enumerate(fitness_):
                # 检查是否被另一个解支配
                if index != index_ and (fx_1_ <= fx_1 and fx_2_ <= fx_2):
                    if fx_1_ == fx_1 and fx_2_ == fx_2_:  # 相同解
                        pareto_grade += 0
                    else:
                        pareto_grade += 1

            pareto.append(pareto_grade)

        # 计算适应度
        fitness = []
        fx1_fx2_count = {}  # 用于存储每个 Fitness 值的出现次数
        for (fx_1,fx_2,penality),pareto_g in zip(fitness_,pareto):
            if penality == 0:
                # 检查解是否已存在
                if (fx_1, fx_2) in fx1_fx2_count:
                    Fitness = 1 / (pareto_g+len(population))
                else:
                    Fitness = 1 / pareto_g
            else:
                if (fx_1, fx_2) in fx1_fx2_count:
                    Fitness = 1 / (pareto_g/len(population)+penality+2*len(population))
                else:
                    Fitness = 1 / (pareto_g/len(population)+penality+len(population))

            fx1_fx2_count[(fx_1,fx_2)] = fx1_fx2_count.get((fx_1,fx_2), 0) + 1
            fitness.append(Fitness)

        # 根据适应度选择 80%可行解 20%不可行解
        fitness_array = np.array(fitness)
        sorted_indices = np.argsort(fitness_array)[::-1] # 逆序
        possible_solution = [index for index in sorted_indices if fitness_array[index] >= 1/len(population)]
        impossible_solution = [index for index in sorted_indices if fitness_array[index] < 1/len(population)]

        num_solutions = len(population) // 2
        num_possible = int(num_solutions * 0.8)
        num_possible_actual = min(num_possible, len(possible_solution))
        selected_solutions_index = possible_solution[:num_possible_actual]
        num_impossible = num_solutions - len(selected_solutions_index)
        selected_solutions_index += impossible_solution[:num_impossible]
        population_select = [population[i] for i in selected_solutions_index]
        fx1_select = [fx1_list[i] for i in selected_solutions_index]
        fx2_select = [fx2_list[i] for i in selected_solutions_index]

    return population_select, fx1_select,fx2_select

def genetic(tgl_class,dxf_class,population_num,parameters,iter):
    starttime = time.time()
    # >>>step1 种群初始化
    population = population_initialization(tgl_class['1'], dxf_class['1'], population_num, parameters)
    list_fx1_ave = []
    list_fx2_ave = []
    list_fx1 = []
    list_fx2 = []
    # 增加初始数据的计算
    population_select, fx1, fx2 = select(tgl_class,dxf_class,population, parameters,0,0 )
    list_fx1_ave.append(sum(fx1) / len(fx1))
    list_fx2_ave.append(sum(fx2) / len(fx2))
    endtime = time.time()
    time_total = endtime-starttime
    print("---第0代初始化完成,用时{}---".format(time_total))

    for i in range(iter):
        time_start = time.time()
        # >>>step2 进化
        population_evolve = evolve(tgl_class,dxf_class,population, parameters)
        population_unselect = population + population_evolve

        # >>>step3 选择
        population_select,fx1,fx2 = select(tgl_class,dxf_class,population_unselect,parameters,iter = i+1,threshold=0)
        list_fx1_ave.append(sum(fx1) / len(fx1))
        list_fx2_ave.append(sum(fx2) / len(fx2))
        list_fx1.append(fx1)
        list_fx2.append(fx2)
        time_finish = time.time()
        print("---第{}代完成".format(i + 1,)+" 用时{}---".format(time_finish-time_start))
        time_total += (time_finish-time_start)
        if population_select == population:
            break
        else:
            population = population_select
        # 信息保存
        path = 'D:\.py\PythonProjrct\public_building\model\caulate_result\Genetic{}.pkl'.format(i)
        model_data = [list_fx1, list_fx2, list_fx1_ave, list_fx2_ave]
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

    print("---优化完成,总耗时{}---".format(time_total))

    # 信息保存
    path = 'D:\.py\PythonProjrct\public_building\model\caulate_result\Genetic.pkl'
    model_data = [list_fx1,list_fx2,list_fx1_ave,list_fx2_ave]
    with open(path, 'wb') as f:
        pickle.dump(model_data, f)

    print("---计算结束---")
    # 可视化
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.title('Pareto前沿')
    plt.plot(fx1, fx2, color='green')
    plt.show(block=True)


    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    list_x = [i for i in range(len(list_fx1_ave)+1)]
    list_y = [point for point in list_fx1_ave]
    plt.plot(list_x, list_y, color='green')
    plt.title('fx1')
    plt.show(block=True)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    list_x = [i for i in range(len(list_fx2_ave)+1)]
    list_y = [point for point in list_fx2_ave]
    plt.plot(list_x, list_y, color='green')
    plt.title('fx2')
    plt.show(block=True)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from step1_read_tgl_p1 import ArchElementInfor
    from step2_read_dxf import DxfInfor


    # 参数
    tgl_list = [
        # 'D:\.py\PythonProjrct\public_building_sheets\input_file\model1\SF_1.tgl',
        'D:\.py\PythonProjrct\public_building_sheets\input_file\model1\SF_2.tgl',
        'D:\.py\PythonProjrct\public_building_sheets\input_file\model1\SF_3_8.tgl',
        'D:\.py\PythonProjrct\public_building_sheets\input_file\model1\SF_9_26.tgl',
        'D:\.py\PythonProjrct\public_building_sheets\input_file\model1\SF_27.tgl']

    dxf_list = [
        # 'D:\.py\PythonProjrct\public_building_sheets\input_file\model1\SF_1.dxf',
        'D:\.py\PythonProjrct\public_building_sheets\input_file\model1\SF_2.dxf',
        'D:\.py\PythonProjrct\public_building_sheets\input_file\model1\SF_3_8.dxf',
        'D:\.py\PythonProjrct\public_building_sheets\input_file\model1\SF_9_26.dxf',
        'D:\.py\PythonProjrct\public_building_sheets\input_file\model1\SF_27.dxf']

    # TGL DXF实例化
    tgl_class, dxf_class = multiarchfl_archsf(tgl_list, dxf_list)
    # Part1 参数
    archsf_height_num = [[4000, 1], [3800, 1], [3500, 6], [4000, 1]]  # 各建筑标准层 [层高,层数]
    wall_t_s = [400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900]  # 可取值的墙厚
    # wall_t_s = [250, 300, 350, 400, 450, 500,550,600,650,700,750,800]  # 可取值的墙厚
    colume_section = [400, 800]  # 柱截面取值范围
    beam_section_hl = [500, 1000]  # 梁高截面取值范围
    beam_section_wl = [[150, 150, 7, 10], [200, 200, 8, 12], [250, 250, 9, 14], [300, 300, 10, 15], [350, 350, 12, 19],
                       [400, 400, 13, 21]]  # 常见H型钢 hxbxt1xt2
    peoperty_material_column = [40, 50]
    peoperty_material_steel = ['Q345']
    length_step = 50
    population = 4 # 种群个数
    iter = 20

    indicators_limit = [0.85, 1 / 800] # 适应度1限值
    dict_cg_factor = {30: 1,
                      35: 1.05,
                      40: 1.1,
                      45: 1.15,
                      50: 1.2,
                      "Q345": 1.5}

    parameters = (archsf_height_num, wall_t_s, beam_section_wl, colume_section, beam_section_hl,
                  peoperty_material_column, length_step,indicators_limit,dict_cg_factor)

    genetic(tgl_class, dxf_class,population,parameters,iter)


