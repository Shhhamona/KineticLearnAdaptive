import numpy as np
from baseline_training_methods import LoadMultiPressureDatasetNumpy

src_train = 'data/SampleEfficiency/O2_simple_uniform.txt'
src_test = 'data/SampleEfficiency/O2_simple_test.txt'
nspecies = 3
num_pressure_conditions = 2

print('\n== Loading training dataset and inspecting scalers ==')
dt = LoadMultiPressureDatasetNumpy(src_train, nspecies, num_pressure_conditions, react_idx=[0,1,2])

# Load test dataset using the training scalers (this is the correct pattern)
dataset_test = LoadMultiPressureDatasetNumpy(src_test, nspecies, num_pressure_conditions,
                                            react_idx=[0,1,2], scaler_input=dt.scaler_input,
                                            scaler_output=dt.scaler_output)

print('\n== Transformed shapes ==')
print(' train.x_data shape:', dt.x_data.shape, ' train.y_data shape:', dt.y_data.shape)
print(' test.x_data shape:', dataset_test.x_data.shape, ' test.y_data shape:', dataset_test.y_data.shape)

print('x_data shape (flattened):', dt.x_data.shape)
print('y_data shape (flattened):', dt.y_data.shape)

for i in range(num_pressure_conditions):
    s_in = dt.scaler_input[i]
    s_out = dt.scaler_output[i]
    print(f"\nPressure {i}: input scaler type: {type(s_in).__name__}")
    print('  input.max_abs_ =', getattr(s_in, 'max_abs_', None))
    print('  output.max_abs_ =', getattr(s_out, 'max_abs_', None))

# Recompute raw arrays the same way the loader does so we can show raw stats
all_data = dt.all_data
ncolumns = all_data.shape[1]
x_columns = np.arange(ncolumns - nspecies, ncolumns)
# Use the same react_idx as when creating the dataset (3 reactions)
y_columns = np.array([0, 1, 2])

raw_x = all_data[:, x_columns]
raw_y = all_data[:, y_columns] * 1e30

raw_x_reshaped = raw_x.reshape(num_pressure_conditions, -1, nspecies)
raw_y_reshaped = raw_y.reshape(num_pressure_conditions, -1, raw_y.shape[1])

print('\n== Raw density stats (per pressure, per species): min, max, median ==')
for i in range(num_pressure_conditions):
    arr = raw_x_reshaped[i]
    print(f'Pressure {i} raw min = {np.min(arr,axis=0)}')
    print(f'Pressure {i} raw max = {np.max(arr,axis=0)}')
    print(f'Pressure {i} raw median = {np.median(arr,axis=0)}')

    transformed = dt.scaler_input[i].transform(arr)
    print(f'Pressure {i} transformed min = {np.min(transformed,axis=0)}')
    print(f'Pressure {i} transformed max = {np.max(transformed,axis=0)}')
    print(f'Pressure {i} transformed median = {np.median(transformed,axis=0)}')

    frac_out = np.mean(np.any(np.abs(transformed) > 1.0, axis=1))
    print(f'Pressure {i} fraction samples with any feature abs>1 after transform: {frac_out:.4f}\n')

print('== k (output) stats ==')
for i in range(num_pressure_conditions):
    arr = raw_y_reshaped[i]
    print(f'Pressure {i} raw k min = {np.min(arr,axis=0)}')
    print(f'Pressure {i} raw k max = {np.max(arr,axis=0)}')
    print(f'Pressure {i} raw k median = {np.median(arr,axis=0)}')

    transformed = dt.scaler_output[i].transform(arr)
    print(f'Pressure {i} transformed k min = {np.min(transformed,axis=0)}')
    print(f'Pressure {i} transformed k max = {np.max(transformed,axis=0)}')
    print(f'Pressure {i} transformed k median = {np.median(transformed,axis=0)}\n')

# Show how a user-specified k_true would be scaled using the existing output scaler
k_true = np.array([6.00E-16, 1.30E-15, 9.60E-16])
print('k_true raw =', k_true)
k_scaled = k_true * 1e30
print('k_true * 1e30 =', k_scaled)
kt = dt.scaler_output[0].transform(k_scaled.reshape(1, -1))
print('k_true transformed by scaler_output[0] =', kt)

# Demonstrate that the same scalers applied to test data give consistent transformed ranges
print('\n== Transformed ranges from training-scaled test dataset ==')
# reconstruct raw test arrays like above but from dataset_test
all_data_test = dataset_test.all_data
ncolumns_test = all_data_test.shape[1]
x_columns_test = np.arange(ncolumns_test - nspecies, ncolumns_test)
y_columns_test = np.array([0, 1, 2])
raw_x_test = all_data_test[:, x_columns_test].reshape(num_pressure_conditions, -1, nspecies)
raw_y_test = (all_data_test[:, y_columns_test] * 1e30).reshape(num_pressure_conditions, -1, raw_y.shape[1])

for i in range(num_pressure_conditions):
    t_x = dataset_test.scaler_input[i].transform(raw_x_test[i])
    print(f' Test Pressure {i} transformed x min = {np.min(t_x,axis=0)}')
    print(f' Test Pressure {i} transformed x max = {np.max(t_x,axis=0)}')
    print(f' Test Pressure {i} transformed x median = {np.median(t_x,axis=0)}')

    t_y = dataset_test.scaler_output[i].transform(raw_y_test[i])
    print(f' Test Pressure {i} transformed y min = {np.min(t_y,axis=0)}')
    print(f' Test Pressure {i} transformed y max = {np.max(t_y,axis=0)}')
    print(f' Test Pressure {i} transformed y median = {np.median(t_y,axis=0)}\n')

# Show first 3 rows of flattened x and y (post-processed) for reference
print('\nFirst 3 rows of dt.x_data (flattened, post-transform):')
print(dt.x_data[:3])
print('\nFirst 3 rows of dt.y_data (post-transform):')
print(dt.y_data[:3])

print('\nTest complete.')
