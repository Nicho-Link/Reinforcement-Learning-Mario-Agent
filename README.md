# Super-Mario-Bros Agent with Reinforcement Learning


## What is this?
This is a Reinforcement Learning Agent for the [gym-super-mario-bros Environment](https://github.com/Kautenja/gym-super-mario-bros/) using a DDQN (Double Deep Q-Network) and a decaying epsilon-greedy strategy.

This is a project for a Lecture at DHBW Mannheim. The goal is to develop an own Reinforcement Learning Agent.

## Directory Structure
- [models](models): contains the final models of our DQN and DDQN
- [notebooks](notebooks): contains both notebooks to train & evaluate a DQN-Agent or DDQN-Agent
- [references](references): contains ressources such as plots and videos of the training and evaluation of our agents
- [src](src): containts the sourcecode used by our jupyter-notebooks
- [setup.py](setup.py): contains code to create the initial folderstructure to train an agent

## Run Code
- install the required version of [gym-super-mario-bros Environment](https://github.com/Kautenja/gym-super-mario-bros/) and follow their installation guide.
  (You need to install the C++ build tools)
- For [PyTorch](https://pytorch.org/get-started/locally/) please use their installation guide. (Especially for GPU usage with Cuda -> [Nvidia Cuda Drivers](https://developer.nvidia.com/cuda-toolkit) required)
  To install the PyTorch Version of this project the command is:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
- install the remaining required [packages](requirements.txt)
```bash
pip install -r requirements.txt
```
- make sure there is at least one model file in the models directory
- to let an agent play the trained model, start the [Notebook](main.ipynb) and run all cells

## License

This Code is licensed under the [MIT-License](LICENSE).
