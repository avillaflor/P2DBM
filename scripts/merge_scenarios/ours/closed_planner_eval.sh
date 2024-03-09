MODEL_CKPT_LOCATION=$1
GPU=${2:-0}
python scripts/merge_scenarios/ours/closed_planner_eval.py gpu=[$GPU] scenario=65 model_checkpoint=$MODEL_CKPT_LOCATION
python scripts/merge_scenarios/ours/closed_planner_eval.py gpu=[$GPU] scenario=64 model_checkpoint=$MODEL_CKPT_LOCATION
python scripts/merge_scenarios/ours/closed_planner_eval.py gpu=[$GPU] scenario=63 model_checkpoint=$MODEL_CKPT_LOCATION
python scripts/merge_scenarios/ours/closed_planner_eval.py gpu=[$GPU] scenario=62 model_checkpoint=$MODEL_CKPT_LOCATION
python scripts/merge_scenarios/ours/closed_planner_eval.py gpu=[$GPU] scenario=61 model_checkpoint=$MODEL_CKPT_LOCATION
python scripts/merge_scenarios/ours/closed_planner_eval.py gpu=[$GPU] scenario=42 model_checkpoint=$MODEL_CKPT_LOCATION
python scripts/merge_scenarios/ours/closed_planner_eval.py gpu=[$GPU] scenario=41 model_checkpoint=$MODEL_CKPT_LOCATION
python scripts/merge_scenarios/ours/closed_planner_eval.py gpu=[$GPU] scenario=33 model_checkpoint=$MODEL_CKPT_LOCATION
python scripts/merge_scenarios/ours/closed_planner_eval.py gpu=[$GPU] scenario=32 model_checkpoint=$MODEL_CKPT_LOCATION
python scripts/merge_scenarios/ours/closed_planner_eval.py gpu=[$GPU] scenario=31 model_checkpoint=$MODEL_CKPT_LOCATION
