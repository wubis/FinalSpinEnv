# Neutron Scattering ML
Here we explore applications of ML to spinwaves, starting with reinforcement learning. The current RL model used is PPO, which combines ideas from A2C (Actor Critic model) and TRPO (using a trust region to improve the Actor in the A2C)

# SpinW MATLAB Compiling into Python: Workflow and Files
Workflow:
How to compile MATLAB/SpinW into Python
- Pull SpinW from the repository
- Install libpymcr: pip install libpymcr​
- Copy the file call.m into your SpinW folder
- You can find it under libpymcr src directory
- Compile Matlab code with: mcc -W ctf:PySpinW -U call.m -a swfiles -a external -a dat_files
- Directories and files might have slightly different spelling, so be aware in case an error shows up
- If needed (not sure if you do), move libpymcr directory into your SpinW directory
- In python, import libpymcr and initialize MCR/ctf:
  - import libpymcr
  - m = libpymcr.Matlab('/path/to/ctf')​
  - If not already installed onto IDE, use pip to install libpymcr
  - Run Matlab commands through the m​ object: m.eig([[1,2,3], [4,5,6], [7,8,9]])

Files:
- spin_env.py
  - main code that trains the reinforcement learning agent
  - modified from the old code to allow for parallel-processing/multi-processing during simulation and logging
  - can adjust manually the epochs, steps, batch size, etc
- main.py
  - Example to show capability of plotting both spin dispersion and neutron intensity
- glob_logs.py
  - Conglomerates logs together
  - Due to the nature of parallel processing, all data points logged are split into 8 (or however many environments/cores you're using) so this code combines them into one json file (albeit in different arrays, which will be solved with another code)
- json load.py
  - UNFINISHED CODE
  - Supposed to combine arrays (split due to the multiple envs/cores) for plotting
  - Functioning code was done in Jupyter notebooks (code was unfortunately lost due to deletion of NIST email account)

# How the Algorithm Works
### Reinforcement Learning:
- Given a state of the world, an RL algorithm follows the following actions:
  - Agent takes an action to interact with world
  - Owner gives feedback depending on how optimal the action is (positive or negative)
  - Through repeated actions, the agent learns what or what not to do, maximizing the reward and optimizing behavior
  
### RL Applied to Neutron Scattering Experiments:
- We train an RL agent (think automated scientist) to figure out how a given material behaves
- Goal is to figure out two hidden properties, J1 and J2, which describes how the magnetic particles in the material interact
- Learning agent function:
  - Make a guess: makea guess about J1 and J2 after every scan
  - Get feedback: get a positive or negative reward depending on how good the guess is
  - Learn: Over time, agent learns the best directions to scan to get feedback
