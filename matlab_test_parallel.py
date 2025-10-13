import matlab.engine
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

def run_simulation(setup_file):
    """
    Run a single MATLAB simulation with the given setup file.
    
    Args:
        setup_file (str): Path to the setup file for the simulation
    
    Returns:
        dict: Result information including setup file and execution time
    """
    start_time = time.time()
    
    loki_path = "C:\\MyPrograms\\LoKI_v3.1.0-v2"
    os.chdir(loki_path + "\\Code")  # Change working directory for LoKI relative paths
    
    # Start MATLAB engine
    eng = matlab.engine.start_matlab()
    
    # Add LoKI code folder to MATLAB search path
    s = eng.genpath(loki_path)
    eng.addpath(s, nargout=0)
    
    # Run the simulation
    print(f"Starting simulation with setup file: {setup_file}")
    eng.loki(setup_file, nargout=0)  # Specify no output arguments expected
    
    # Clean up
    eng.quit()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    result = {
        'setup_file': setup_file,
        'execution_time': execution_time,
        'status': 'completed'
    }
    
    print(f"Completed simulation with {setup_file} in {execution_time:.2f} seconds")
    return result

def run_parallel_simulations():
    """
    Run multiple MATLAB simulations in parallel.
    """
    # Define the complex O2_novib setup files for parallel execution
    setup_files = [
        "oxygen_novib\\oxygen_chem_setup_novib.in",
        "oxygen_novib\\oxygen_chem_setup_novib.in",
        "oxygen_novib\\oxygen_chem_setup_novib.in",
        "oxygen_novib\\oxygen_chem_setup_novib.in",
        "oxygen_novib\\oxygen_chem_setup_novib.in",
        "oxygen_novib\\oxygen_chem_setup_novib.in"
    ]
    
    print(f"Starting {len(setup_files)} simulations in parallel...")
    start_total_time = time.time()
    
    # Run simulations in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=5) as executor:
        # Submit all tasks
        future_to_setup = {executor.submit(run_simulation, setup_file): setup_file 
                          for setup_file in setup_files}
        
        # Collect results as they complete
        results = []
        for future in as_completed(future_to_setup):
            setup_file = future_to_setup[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"Simulation with {setup_file} generated an exception: {exc}")
                results.append({
                    'setup_file': setup_file,
                    'status': 'failed',
                    'error': str(exc)
                })
    
    end_total_time = time.time()
    total_time = end_total_time - start_total_time
    
    # Print summary
    print("\n" + "="*50)
    print("PARALLEL SIMULATION SUMMARY")
    print("="*50)
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Number of simulations: {len(setup_files)}")
    
    for result in results:
        if result['status'] == 'completed':
            print(f"✓ {result['setup_file']}: {result['execution_time']:.2f}s")
        else:
            print(f"✗ {result['setup_file']}: FAILED - {result.get('error', 'Unknown error')}")
    
    return results

if __name__ == "__main__":
    results = run_parallel_simulations()
