# Super-Mario-Bros Agent with Reinforcement Learning


## What is this?
This is a Reinforcement Learning Agent for the [gym-super-mario-bros Environment](https://github.com/Kautenja/gym-super-mario-bros/) using a DDQN (Double Deep Q-Network) and an (exponential) decaying epsilon-greedy strategy.

This is a project for a Lecture at DHBW Mannheim. The goal is to develop an own Reinforcement Learning Agent.

## Directory Structure
- [doc](doc): XXX
- [models](models): contains the checkpoints of the different training sessions and the final model
  - [final_model.pth](models/final_model.pth): The final model
- [res](res): contains ressources of the training sessions
  - [training_v1](res/training_v1): contains all ressources of session 1
    - [plots](res/training_v1/plots): contains the different plots generated in the training session
    - [videos](res/training_v1/videos): contains selected videos of some episodes of the training
  - [training_v2](res/training_v2): contains all ressources of session 1
    - [plots](res/training_v2/plots): contains the different plots generated in the training session
    - [videos](res/training_v2/videos): contains selected videos of some episodes of the training
- [src](src): containts all of the code
  - [helper_functions](src/helper_functions): contains all files that create some classes to build & train the RL-Agent
    - [create_Agent.py](src/helper_functions/create_Agent.py) class to create the mario-agent
    - [create_ExpRepBuf.py](src/helper_functions/create_ExpRepBuf.py) class to create a custom experience replay buffer
    - [create_NN.py](src/helper_functions/create_NN.py) class to create the DDQN-Neural-Network
    - [create_Plot](src/helper_functions/create_Plot.py) function to plot all necessary data
  - [train_v1.ipynb](src/train_v1.ipynb): notebook-script to train the agent with the hyperparameters of v1
  - [train_v2.ipynb](src/train_v2.ipynb): notebook-script to train the agent with the hyperparameters of v2
- [main.ipynb](main.ipynb): notebook-script to let the final model play the game

## Run Code
- install the required version of [gym-super-mario-bros Environment](https://github.com/Kautenja/gym-super-mario-bros/) and follow their installation guide.
- if you want to use your GPU for training please follow the installation guide of [PyTorch](https://pytorch.org/get-started/locally/). To install the PyTorch Version of this project the command is:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
- install the required [packages](requirements.txt)
```bash
pip install -r requirements.txt
```
- make sure there is at least one model file in the models directory
- to run the classification with the trained model, start the [Notebook](main.ipynb) and run all cells

## License

This Code is licensed under the [MIT-License](LICENSE).