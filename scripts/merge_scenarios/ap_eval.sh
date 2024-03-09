GPU=${1:-0}
BEHAVIOR=${2:-normal}
python scripts/merge_scenarios/ap_eval.py --gpu $GPU --scenario 65 --behavior $BEHAVIOR
# python scripts/merge_scenarios/ap_eval.py --gpu $GPU --scenario 64 --behavior $BEHAVIOR
# python scripts/merge_scenarios/ap_eval.py --gpu $GPU --scenario 63 --behavior $BEHAVIOR
# python scripts/merge_scenarios/ap_eval.py --gpu $GPU --scenario 62 --behavior $BEHAVIOR
# python scripts/merge_scenarios/ap_eval.py --gpu $GPU --scenario 61 --behavior $BEHAVIOR
# python scripts/merge_scenarios/ap_eval.py --gpu $GPU --scenario 42 --behavior $BEHAVIOR
# python scripts/merge_scenarios/ap_eval.py --gpu $GPU --scenario 41 --behavior $BEHAVIOR
# python scripts/merge_scenarios/ap_eval.py --gpu $GPU --scenario 33 --behavior $BEHAVIOR
# python scripts/merge_scenarios/ap_eval.py --gpu $GPU --scenario 32 --behavior $BEHAVIOR
# python scripts/merge_scenarios/ap_eval.py --gpu $GPU --scenario 31 --behavior $BEHAVIOR
