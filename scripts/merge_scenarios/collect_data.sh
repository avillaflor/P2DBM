N_TRAIN_SAMPLES=${1:-10000}
let N_VAL_SAMPLES=$N_TRAIN_SAMPLES/10
GPU=${2:-0}
python scripts/merge_scenarios/collect_data.py datasets/merge_scenarios  3_1 --n_samples $N_TRAIN_SAMPLES --gpu $GPU
python scripts/merge_scenarios/collect_data.py datasets/merge_scenarios  3_2 --n_samples $N_TRAIN_SAMPLES --gpu $GPU
python scripts/merge_scenarios/collect_data.py datasets/merge_scenarios  3_3 --n_samples $N_TRAIN_SAMPLES --gpu $GPU
python scripts/merge_scenarios/collect_data.py datasets/merge_scenarios  4_1 --n_samples $N_TRAIN_SAMPLES --gpu $GPU
python scripts/merge_scenarios/collect_data.py datasets/merge_scenarios  4_2 --n_samples $N_TRAIN_SAMPLES --gpu $GPU
python scripts/merge_scenarios/collect_data.py datasets/merge_scenarios  6_1 --n_samples $N_TRAIN_SAMPLES --gpu $GPU
python scripts/merge_scenarios/collect_data.py datasets/merge_scenarios  6_2 --n_samples $N_TRAIN_SAMPLES --gpu $GPU
python scripts/merge_scenarios/collect_data.py datasets/merge_scenarios  6_3 --n_samples $N_TRAIN_SAMPLES --gpu $GPU
python scripts/merge_scenarios/collect_data.py datasets/merge_scenarios  6_4 --n_samples $N_TRAIN_SAMPLES --gpu $GPU
python scripts/merge_scenarios/collect_data.py datasets/merge_scenarios  6_5 --n_samples $N_TRAIN_SAMPLES --gpu $GPU
python scripts/merge_scenarios/collect_data.py datasets/merge_scenarios_valid  3_1 --n_samples $N_VAL_SAMPLES --gpu $GPU
python scripts/merge_scenarios/collect_data.py datasets/merge_scenarios_valid  3_2 --n_samples $N_VAL_SAMPLES --gpu $GPU
python scripts/merge_scenarios/collect_data.py datasets/merge_scenarios_valid  3_3 --n_samples $N_VAL_SAMPLES --gpu $GPU
python scripts/merge_scenarios/collect_data.py datasets/merge_scenarios_valid  4_1 --n_samples $N_VAL_SAMPLES --gpu $GPU
python scripts/merge_scenarios/collect_data.py datasets/merge_scenarios_valid  4_2 --n_samples $N_VAL_SAMPLES --gpu $GPU
python scripts/merge_scenarios/collect_data.py datasets/merge_scenarios_valid  6_1 --n_samples $N_VAL_SAMPLES --gpu $GPU
python scripts/merge_scenarios/collect_data.py datasets/merge_scenarios_valid  6_2 --n_samples $N_VAL_SAMPLES --gpu $GPU
python scripts/merge_scenarios/collect_data.py datasets/merge_scenarios_valid  6_3 --n_samples $N_VAL_SAMPLES --gpu $GPU
python scripts/merge_scenarios/collect_data.py datasets/merge_scenarios_valid  6_4 --n_samples $N_VAL_SAMPLES --gpu $GPU
python scripts/merge_scenarios/collect_data.py datasets/merge_scenarios_valid  6_5 --n_samples $N_VAL_SAMPLES --gpu $GPU
