# beam-pattern-rlearning
TODO

## Setup and Installation

To set up the project, follow the steps below:

1. **Clone the repository**:

   First, clone the repository to your local machine using:

   ```bash
   git clone https://github.com/zakaria-narjis/beam-pattern-rlearning.git
   cd beam-pattern-rlearning
   ```
2. **Install requirements**:
    ```bash
    pip install -r requirements.txt
    ```
3. **Modify experiments configuration**:
    - modify environement configuration ```experiments/config/env_config.yaml```
    - modify RL algo configuration ```experiments/config/algo_config.yaml``` (Note: in some previous experiments it was named sac_config.yaml)

4. **Run training**
    ```bash
    python src/train.py
    ```
5. **Run Tensorboard for monitoring (OPTIONAL)**
    - If you want to monitor all actual and previous runs ```bash
    tensorboard --logdir=experiments/runs --bind_all```
    - If you want to monitor just the runing experiment ```bash
    tensorboard --logdir=experiments/runs/<experiment-name>```
       
5. **Get the training results and models**
    ```experiments/run/<experiment-name>```