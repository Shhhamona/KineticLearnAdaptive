import matlab.engine
import os


#loki_path = "C:\\MyPrograms\\Loki\\LoKI-Edmond"

loki_path = "C:\MyPrograms\LoKI_v3.1.0-v2"
os.chdir(loki_path+ "\\Code") # First change the working directory so that the relatives paths of loki work


eng = matlab.engine.start_matlab()



# Definition of reaction scheme and setup files
#chem_file = "O2_simple_1.chem" 
setup_file = "O2_simple_1\\Real_k\\setup_O2_simple_0.in"
#setup_file = "oxygen_novib\\oxygen_chem_setup_novib_0.in"
#setup_file = "default_lokib_setup.in"

os.chdir(loki_path+ "\\Code") # First change the working directory so that the relatives paths of loki work

s = eng.genpath(loki_path)
eng.addpath(s, nargout=0) # add loki code folder to search path of matlab
eng.loki(setup_file, nargout=0)  # run the matlab script with no output expected

eng.quit()