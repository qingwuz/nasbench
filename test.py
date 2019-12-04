def random_spec():
    while True:
        matrix = np.random.choice(ALLOWED_EDGES, size=(NUM_VERTICES, NUM_VERTICES))
        matrix = np.triu(matrix, 1)
        ops = np.random.choice(ALLOWED_OPS, size=(NUM_VERTICES)).tolist()
        ops[0] = INPUT
        ops[-1] = OUTPUT
        spec = api.ModelSpec(matrix=matrix, ops=ops)
        if nasbench.is_valid(spec):
            return spec

def mutate_spec(old_spec, mutation_rate=1.0):
    while True:
        new_matrix = copy.deepcopy(old_spec.original_matrix)    # 复制cell的matrix
        new_ops = copy.deepcopy(old_spec.original_ops)          # 复制cell的ops

        # In expectation, V edges flipped (note that most end up being pruned).
        edge_mutation_prob = mutation_rate / NUM_VERTICES       # 边的变化阈值 1.0/7
        for src in range(0, NUM_VERTICES - 1):                  
            for dst in range(src + 1, NUM_VERTICES):
                if random.random() < edge_mutation_prob:
                      new_matrix[src, dst] = 1 - new_matrix[src, dst]  # 遍历整个cell的matrix，
                                                                       # 如果随机数小于1/7，该处的连接方式反转
        # In expectation, one op is resampled.
        op_mutation_prob = mutation_rate / OP_SPOTS          # 操作类型改变阈值 1.0/5
        for ind in range(1, NUM_VERTICES - 1):
            if random.random() < op_mutation_prob:
                available = [o for o in nasbench.config['available_ops'] if o != new_ops[ind]]
                new_ops[ind] = random.choice(available)      # 只改变ops中的一个元素
        new_spec = api.ModelSpec(new_matrix, new_ops)        # 重新构建一个cell
        if nasbench.is_valid(new_spec):
            return new_spec

def random_combination(iterable, sample_size):
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), sample_size))
    return tuple(pool[i] for i in indices)

def run_evolution_search(max_time_budget=5e6,
                         population_size=50,
                         tournament_size=10,
                         mutation_rate=1.0):
    # Run a single roll-out of regularized evolution to a fixed time budget.
    nasbench.reset_budget_counters()
    times, best_valids, best_tests = [0.0], [0.0], [0.0]
    population = []   # (validation, spec) tuples

    # For the first population_size individuals, seed the population with randomly generated cells.
    for _ in range(population_size):                          # 这是一个随机选种子的过程
        spec = random_spec()
        data = nasbench.query(spec)
        time_spent, _ = nasbench.get_budget_counters()
        times.append(time_spent)
        population.append((data['validation_accuracy'], spec))

        if data['validation_accuracy'] > best_valids[-1]:
            best_valids.append(data['validation_accuracy'])
            best_tests.append(data['test_accuracy'])
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])

        if time_spent > max_time_budget:
            break
    # After the population is seeded, proceed with evolving the population.
    while True:
        sample = random_combination(population, tournament_size) # 所有的是个随机结合的模型
        best_spec = sorted(sample, key=lambda i:i[0])[-1][1]     # 选出10个模型中表现最好的（比较的是验证精度）
        new_spec = mutate_spec(best_spec, mutation_rate)         # 生成一个进化的cell

        data = nasbench.query(new_spec)       # 查这个新cell的表
        time_spent, _ = nasbench.get_budget_counters()    # 训练时间
        times.append(time_spent)              # 坐标轴的横轴

        # In regularized evolution, we kill the oldest individual in the population.
        population.append((data['validation_accuracy'], new_spec))
        population.pop(0)
        
        if data['validation_accuracy'] > best_valids[-1]:    # 给每个时间点上都对应上 val_acc 和 test_acc
            best_valids.append(data['validation_accuracy'])
            best_tests.append(data['test_accuracy'])
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])
        if time_spent > max_time_budget:
            break

    return times, best_valids, best_tests
