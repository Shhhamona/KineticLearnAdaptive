import numpy as np
import random
import math
random.seed(101)

# 1. Morris Sampling - use the same fixed bounds for all inputs - part I
def MorrisSampler(k_real, p, r, k_range_type, k_range, indexes=None):

    k_size = len(indexes)

    # 1. define region of experimentation w
    w = []
    if(k_range_type == "log"):
        w_log = np.logspace(k_range[0], k_range[1] ,p, base=10)
        mean_point = 1
        print("w log:" , w_log)
        w = w_log

    elif((k_range_type == "lin")):
        w_linear = np.linspace(k_range[0], k_range[1] ,p)
        mean_point = (k_range[1]+k_range[0])/2
        print("w linear:" , w_linear)
        w = w_linear

    # 2. define delta 
    delta = p/(2*(p-1))

    # create starting nodes
    start_nodes = []
    for i in range(r):
        start_nodes.append(random.choices(w,k= k_size)) # maybe use random.sample instead to avoid duplicates
    # print("start_nodes: ", start_nodes)
    # print("start_nodes.shape: ", np.array(start_nodes).shape)
    # create trajectories
    trajectories = []

    for (traj_idx, start_node) in enumerate(start_nodes):
        trajectory= []
        trajectory.append(start_node)                  # add starting node
        order = random.sample(range(0,k_size), k_size) # generate updating order

        # add the remaining nodes
        current_node = start_node.copy()
        for i in order:
            new_node = current_node.copy()

            if(new_node[i]/mean_point>1): # if it lies in the second half of the range interval
                new_node[i] = new_node[i]-delta
            else:
                new_node[i] = new_node[i]+delta

            trajectory.append(new_node)
            current_node = new_node
        
        # save current trajectory
        trajectories.append(trajectory)

    reshaped_traj =  np.reshape(trajectories, (r*(k_size+1),-1))  # r * (k+1) sets of inputs
    print("trajectories shape: ", np.array(reshaped_traj).shape)
    
    n_inputs = r * (k_size+1)
    print("n_inputs: ", n_inputs)

    # Transform to the real distribution of k values
    k_set=[]
    for item in k_real:
        k_set.append(np.full(n_inputs, item))

    for (i, k_idx) in enumerate(indexes):
        k_set[k_idx] =  reshaped_traj[:,i]* k_real[k_idx]

    print(np.transpose(np.array(k_set))[:,:3])
    return np.transpose(k_set)



# 2. Morris Sampling - use different bounds for each input feature - part II
def MorrisSampler2(boundaries, p, r, k_range_type, n_inputs, input_ref, indexes):

    # 1. define region of experimentation w
    w, delta, mean_points = [], [], []
    
    if(k_range_type == "log"):
        w_log, mean_points = [], []
        for idx in range(n_inputs):
            w_log.append(np.logspace(boundaries[idx][0], boundaries[idx][1] ,p, base=10))
            mean_points.append((boundaries[idx][0] + boundaries[idx][1])/2)
        w = w_log

    elif((k_range_type == "lin")):
        w_linear, mean_points_lin, multiples, delta_lin = [], [], [], []
        for idx in range(n_inputs):
            w_linear.append(np.linspace(boundaries[idx][0], boundaries[idx][1] ,p))
            mean_points_lin.append((boundaries[idx][0] + boundaries[idx][1])/2)
            multiples.append((boundaries[idx][0] + boundaries[idx][1])/(p-1))
            delta_lin.append(math.ceil(mean_points_lin[idx] / multiples[idx]) * multiples[idx])
        w = w_linear
        delta = delta_lin
        mean_points = mean_points_lin
         
    #check
    print("\nw = " , w)                    # w           = [[...], [...], [...]]
    print("mean points = " , mean_points)  # mean_points = [4.95, 1.06, 0.44]
    print("delta = " , delta, "/n")        # delta       = [5.50, 5.30, 3.08]

    # create starting nodes
    start_nodes = np.zeros((r, r, n_inputs))
    for i in range(r):
        for j in range(r):
            for k in range(n_inputs):
                start_nodes[i][j][k] = np.random.uniform(w[k][0], w[k][p-1])
    
    # create trajectories
    trajectories = []
    for (traj_idx, start_node) in enumerate(start_nodes):
        trajectory= []
        trajectory.append(start_node)                  # add starting node
        order = random.sample(range(0,n_inputs), n_inputs) # generate updating order

        # add the remaining nodes
        for i in order:
            new_node = start_node.copy()
            if(new_node[traj_idx, i] > mean_points[i]): # if it lies in the second half of the range interval
                new_node[traj_idx, i] = new_node[traj_idx, i] - delta[i]
            else:
                new_node[traj_idx, i] = new_node[traj_idx, i] + delta[i]
            trajectory.append(new_node)
        trajectories.append(trajectory)
    
    print("trajectories = ", trajectories)
    reshaped_traj =  np.reshape(trajectories, (r*(n_inputs+1),-1))  # r * (k+1) sets of inputs
    print("trajectories shape: ", np.array(reshaped_traj).shape)
    
    N_inputs = r * (n_inputs+1)
    print("n_inputs: ", N_inputs)

    # Transform to the real distribution of k values
    result = [[0] * N_inputs for _ in range(n_inputs)]
    for (i, input_idx) in enumerate(indexes):
        result[input_idx] =  reshaped_traj[:,i]*input_ref[input_idx]

    return np.transpose(result)


# 3. Corrected Grid-Based Morris Sampling - stays strictly on discrete grid points
def CorrectedGridMorrisSampler(k_real, p, r, k_range_type, k_range, indexes):
    """
    Corrected Morris sampler that stays strictly on grid points.
    Fixes the delta calculation issue from Method 1.
    
    Parameters:
    -----------
    k_real : array
        Reference values for all parameters
    p : int
        Number of grid levels
    r : int
        Number of trajectories
    k_range_type : str
        'log' or 'lin'
    k_range : list
        [min, max] range (log10 for log type)
    indexes : list
        Parameter indices to vary
    
    Returns:
    --------
    np.array
        Morris samples that stay on grid points
    """
    
    k_size = len(indexes)
    
    # 1. Create the grid (same as Method 1)
    if k_range_type == "log":
        w = np.logspace(k_range[0], k_range[1], p, base=10)
        print(f"Corrected log grid: {w}")
    else:
        w = np.linspace(k_range[0], k_range[1], p)
        print(f"Corrected linear grid: {w}")
    
    # 2. CORRECTED DELTA: Move by grid indices, not values
    delta_index = max(1, p // 4)  # Move 1 or p/4 grid positions
    print(f"Corrected delta (grid positions): {delta_index}")
    
    # 3. Create starting nodes (grid indices)
    start_nodes_indices = []
    for i in range(r):
        start_indices = [random.randint(0, p-1) for _ in range(k_size)]
        start_nodes_indices.append(start_indices)
    
    # 4. Create trajectories using grid indices
    trajectories = []
    
    for traj_idx, start_indices in enumerate(start_nodes_indices):
        trajectory = []
        
        # Convert indices to values for starting node
        start_node = [w[idx] for idx in start_indices]
        trajectory.append(start_node)
        
        # Generate random order for parameter updates
        order = random.sample(range(k_size), k_size)
        
        # Build trajectory
        current_indices = start_indices.copy()
        for param_order in order:
            new_indices = current_indices.copy()
            
            # CORRECTED: Move by grid index, stay on grid
            current_idx = current_indices[param_order]
            
            # Decide direction (up or down the grid)
            if current_idx >= p // 2:  # Upper half of grid
                new_idx = max(0, current_idx - delta_index)
            else:  # Lower half of grid
                new_idx = min(p - 1, current_idx + delta_index)
            
            new_indices[param_order] = new_idx
            
            # Convert indices to values
            new_node = [w[idx] for idx in new_indices]
            trajectory.append(new_node)
            
            # Update for next step
            current_indices = new_indices
        
        trajectories.append(trajectory)
    
    # 5. Reshape trajectories
    reshaped_traj = np.reshape(trajectories, (r * (k_size + 1), -1))
    print(f"Corrected trajectories shape: {reshaped_traj.shape}")
    
    n_inputs = r * (k_size + 1)
    
    # 6. Transform to real parameter distribution
    k_set = []
    for item in k_real:
        k_set.append(np.full(n_inputs, item))
    
    for i, k_idx in enumerate(indexes):
        k_set[k_idx] = reshaped_traj[:, i] * k_real[k_idx]
    
    return np.transpose(k_set)


# 4. Continuous Morris Sampling - true to original Morris (1991) theory
def ContinuousMorrisSampler(k_real, p, r, k_range_type, k_range, indexes):
    """
    Continuous Morris sampler following original Morris (1991) theory.
    Parameters can take any value within bounds, not limited to discrete grid.
    
    Parameters:
    -----------
    k_real : array
        Reference values for all parameters
    p : int
        Number of levels (used for delta calculation)
    r : int
        Number of trajectories
    k_range_type : str
        'log' or 'lin'
    k_range : list
        [min, max] range (log10 for log type)
    indexes : list
        Parameter indices to vary
    
    Returns:
    --------
    np.array
        Morris samples with continuous values
    """
    
    k_size = len(indexes)
    
    # 1. Define the parameter space bounds (continuous)
    if k_range_type == "log":
        min_bound = 10**k_range[0]
        max_bound = 10**k_range[1]
        print(f"Continuous log bounds: [{min_bound:.6f}, {max_bound:.6f}]")
        
        # For log space, delta should be multiplicative
        # Use Morris's original delta scaled to the log range
        log_range = k_range[1] - k_range[0]
        delta_fraction = 1.0 / (2 * (p - 1))  # Morris's original delta
        log_delta = delta_fraction * log_range
        
        print(f"Continuous log delta: {log_delta}")
        
    else:  # linear
        min_bound = k_range[0]
        max_bound = k_range[1]
        print(f"Continuous linear bounds: [{min_bound:.6f}, {max_bound:.6f}]")
        
        # For linear space, delta is additive
        param_range = max_bound - min_bound
        delta = param_range / (2 * (p - 1))  # Morris's original delta
        
        print(f"Continuous linear delta: {delta}")
    
    # 2. Create starting nodes (continuous random sampling)
    start_nodes = []
    for i in range(r):
        start_node = []
        for j in range(k_size):
            if k_range_type == "log":
                # Log-uniform sampling
                log_val = np.random.uniform(k_range[0], k_range[1])
                start_node.append(10**log_val)
            else:
                # Uniform sampling
                start_node.append(np.random.uniform(min_bound, max_bound))
        start_nodes.append(start_node)
    
    # 3. Create trajectories with continuous delta steps
    trajectories = []
    
    for traj_idx, start_node in enumerate(start_nodes):
        trajectory = []
        trajectory.append(start_node.copy())
        
        # Generate random order for parameter updates
        order = random.sample(range(k_size), k_size)
        
        # Build trajectory with continuous steps
        current_node = start_node.copy()
        for param_idx in order:
            new_node = current_node.copy()
            
            if k_range_type == "log":
                # Log space: multiplicative delta
                current_log = np.log10(current_node[param_idx])
                mean_log = (k_range[0] + k_range[1]) / 2
                
                # Apply delta in log space
                if current_log > mean_log:
                    new_log = current_log - log_delta
                else:
                    new_log = current_log + log_delta
                
                # Convert back to linear space
                new_node[param_idx] = 10**new_log
                
                # Ensure bounds are respected
                new_node[param_idx] = np.clip(new_node[param_idx], min_bound, max_bound)
                
            else:  # linear space
                # Linear space: additive delta
                mean_val = (min_bound + max_bound) / 2
                
                # Apply delta
                if current_node[param_idx] > mean_val:
                    new_node[param_idx] = current_node[param_idx] - delta
                else:
                    new_node[param_idx] = current_node[param_idx] + delta
                
                # Ensure bounds are respected
                new_node[param_idx] = np.clip(new_node[param_idx], min_bound, max_bound)
            
            trajectory.append(new_node)
            current_node = new_node
        
        trajectories.append(trajectory)
    
    # 4. Reshape trajectories
    reshaped_traj = np.reshape(trajectories, (r * (k_size + 1), -1))
    print(f"Continuous trajectories shape: {reshaped_traj.shape}")
    
    n_inputs = r * (k_size + 1)
    
    # 5. Transform to real parameter distribution
    k_set = []
    for item in k_real:
        k_set.append(np.full(n_inputs, item))
    
    for i, k_idx in enumerate(indexes):
        k_set[k_idx] = reshaped_traj[:, i] * k_real[k_idx]
    
    return np.transpose(k_set)