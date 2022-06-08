# T-MAAC
This is the official implementation of the paper Stabilizing Voltage in Power Distribution Networks via
Multi-Agent Reinforcement Learning with Transformer (KDD2022 Research Track).

The Transformer-based Multi-Agent Actor-Critic Framework (T-MAAC) is based on [MAPDN](https://github.com/Future-Power-Networks/MAPDN). Please refer to that repo for more documentation.

## Installation

We suggest you install dependencies with Dockerfile and run the code with Docker.
```bash
docker build . -t tmaac
```

## Downloading the Dataset
We use load profiles and PV profiles provided by [MAPDN](https://github.com/Future-Power-Networks/MAPDN).
1. Download the data from the [link](https://drive.google.com/file/d/1ry0-K5M-YMw7TcQQYeFb7U-pYvdWKm9A/view?usp=sharing).
2. Unzip the zip file and there are 3 folders as following:
    * case33_3min_final
    * case141_3min_final
    * case322_3min_final
3. Go to the directory ``[Your own parent path]/T-MAAC/environments/var_voltage_control/`` and create a folder called ``data``.
4. Move the 3 folders in step 2 to the directory ``[Your own parent path]/T-MAAC/environments/var_voltage_control/data``.

## Running experiments
### Training
You can train the model using the following command.
```bash
source activate mapdn

## running in case141
python train.py --alg icstransmaddpg --alias example_0 --mode distributed --scenario case141_3min_final --qweight 0.01 --voltage-barrier-type l2 --save-path trial

## running in case322
python train.py --alg icstransmatd3 --alias example_0 --mode distributed --scenario case141_3min_final --qweight 0.1 --voltage-barrier-type l2 --save-path trial
```
The meanings of the arguments:
* `--alg` is the MARL algorithm, e.g. `maddpg`, `matd3`, `icstransmaddpg`, `icstransmatd3`.
* `--alias` is the alias to distinguish different experiments.
* `--mode` is the mode of environment, e.g. `distributed`.
* `--scenario` is the power system on which you like to train, e.g. `case141_3min_final`, `case322_3min_final`.
* `--qweight` is the q_weight used in training. We recommend 0.01 for case141 and 0.1 for case322.
* `--voltage-barrier-type` is the voltage barrier function in training, e.g. `l1`, `l2`, `bowl`.
* `--save-path` is the path to save the model and configures.

### Testing
It is worth noting that the difficulty of voltage control problem varies during different months of a year. For example, during the midday summer, excessive active power from intense sunlight is injected into the grid, creating a more significant challenge for the voltage control task than in winter. Thus, a series of fixed scenarios must be chosen to evaluate algorithms fairly.

We randomly select 10 episodes per month, a total of 120 episodes, which constitute the test dataset `test_data.csv`. Each episode lasts for 480 time steps (i.e., a day). And fixed 10 episodes of 120 episodes are selected to evaluate performance in training phase. (see `def evaluation()` in models/model.py)

```bash
python test.py --save-path trial/model_save --alg icstransmaddpg --alias example_0 --scenario case141_3min_final --qweight 0.01 --voltage-barrier-type l2 --test-mode test_data
```
The meanings of the arguments:
* `--alg` is the MARL algorithm, e.g. `maddpg`, `matd3`, `icstransmaddpg`, `icstransmatd3`.
* `--alias` is the alias to distinguish different experiments.
* `--scenario` is the power system on which you like to train, e.g. `case141_3min_final`, `case322_3min_final`.
* `--qweight` is the q_weight used in training. We recommend 0.01 for case141 and 0.1 for case322.
* `--voltage-barrier-type` is the voltage barrier function in training, e.g. `l1`, `l2`, `bowl`.
* `--save-path` is the path to save the model and configures.
* `--test-mode` is the test mode, e.g. `single`, `test_data`. `test_data` means use test_data.csv to evaluate algorithms.
* `--test-day` is the day that you would like to do the test. Note that it is only activated if the `--test-mode` is `single`.
* `--render` indicates activating the rendering of the environment.

