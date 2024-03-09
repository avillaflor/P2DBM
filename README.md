# Tractable Joint *P*rediction and *P*lanning over *D*iscrete *B*ehavior *M*odes for Urban Driving: P2DBM
## Installation
```
cd $HOME
git clone https://github.com/avillaflor/P2DBM.git
cd P2DBM
conda env create -f environment.yml
conda activate p2dbm
pip install -e .
cd $HOME
git clone https://github.com/autonomousvision/plant.git
```

### Installing CARLA

Install CARLA v0.9.10 (https://carla.org/2020/09/25/release-0.9.10/) for which the binaries are available here: (https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.10.tar.gz)

```
mkdir $HOME/carla910
cd $HOME/carla910
wget "https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.10.tar.gz"
tar -xvzf CARLA_0.9.10.tar.gz
wget "https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.10.tar.gz"
tar -xvzf AdditionalMaps_0.9.10.tar.gz
rm CARLA_0.9.10.tar.gz
rm AdditionalMaps_0.9.10.tar.gz
```

Install CARLA v0.9.11 (https://carla.org/2020/12/22/release-0.9.11/) for which the binaries are available here: (https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.11.tar.gz)

```
mkdir $HOME/carla911
cd $HOME/carla911
wget "https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.11.tar.gz"
tar -xvzf CARLA_0.9.11.tar.gz
wget "https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.11.tar.gz"
tar -xvzf AdditionalMaps_0.9.11.tar.gz
rm CARLA_0.9.11.tar.gz
rm AdditionalMaps_0.9.11.tar.gz
```

Add the following to your `.bashrc`:

```
export CARLA_9_10_PATH=$HOME/carla910
export CARLA_9_10_PYTHONPATH=$CARLA_9_10_PATH/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export CARLA_9_11_PATH=$HOME/carla911
export CARLA_9_11_PYTHONPATH=$CARLA_9_11_PATH/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg
```

## Merge Scenarios
### Data Collection
```
./scripts/merge_scenarios/collect_data.sh
python scripts/merge_scenarios/create_h5_dataset.py
```
### Training
```
python scripts/merge_scenarios/ours/train.py gpu=[0]
```
### Closed-Loop Planner (Ours) Eval
```
./scripts/merge_scenarios/ours/closed_planner_eval.sh 'MODEL_CKPT_LOCATION'
```
### IL Eval
```
./scripts/merge_scenarios/ours/eval.sh 'MODEL_CKPT_LOCATION'
```
### Open-Loop Planner Eval
```
./scripts/merge_scenarios/ours/open_planner_eval.sh 'MODEL_CKPT_LOCATION'
```
Fill in `MODEL_CKPT_LOCATION` with location of `.ckpt` file for model you want to evaluate.

### CVAE
```
python scripts/merge_scenarios/vae/vae_train.py gpu=[0]
./scripts/merge_scenarios/vae/vae_closed_planner_eval.sh 'MODEL_CKPT_LOCATION'
```

### PlanT
```
python scripts/merge_scenarios/plant/train.py gpu=[0]
./scripts/merge_scenarios/plant/eval.sh 'MODEL_CKPT_LOCATION'
```

# Longest6
### Setting configs
Add user config at `scripts/leaderboard/config/user/$USER.yaml` and fill in `working_dir` with path of P2DBM directory, `carla_path` with path to CARLA version 0.9.10, and `plant_dir` with path to PlanT directory.
### Running CARLA
For running longest6 experiments, you need a separate CARLA instance running when collecting data or running evaluations.
```
cd $HOME/carla910
SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 ./CarlaUE4.sh --world-port=2000 -traffic-port=8000 -opengl
```
### Data Collection
```
./scripts/leaderboard/datagen.sh 2000 8000 1
./scripts/leaderboard/datagen.sh 2000 8000 2
./scripts/leaderboard/datagen.sh 2000 8000 3
./scripts/leaderboard/datagen_valid.sh 2000 8000
python scripts/leaderboard/ours/create_h5_dataset.py
```
### Training
```
python scripts/leaderboard/ours/train.py gpu=[0]
```
### Closed-Loop Planner (Ours) Eval
```
python scripts/leaderboard/run_evaluation.py user=$USER experiments=closed_planner eval=longest6 port=2000 trafficManagerPort=8000 CUDA_VISIBLE_DEVICES=0 experiments.model_checkpoint=MODEL_CKPT_LOCATION
```
### IL Eval
```
python scripts/leaderboard/run_evaluation.py user=$USER experiments=IL eval=longest6 port=2000 trafficManagerPort=8000 CUDA_VISIBLE_DEVICES=0 experiments.model_checkpoint=MODEL_CKPT_LOCATION
```
Fill in `MODEL_CKPT_LOCATION` with location of `.ckpt` file for model you want to evaluate.

## Acknowledgements

Code for running leaderboard and PlanT comparisons comes from [PlanT](https://github.com/autonomousvision/plant) repo.

Transformer code initially adapted from [Autobots](https://github.com/roggirg/AutoBots) repo.

Testing environments are based on the [CARLA](https://carla.org/) simulator and [Leaderboard](https://leaderboard.carla.org/get_started_v1/) v1.0 benchmark.
