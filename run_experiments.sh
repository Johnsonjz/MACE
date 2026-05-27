set -euo pipefail

# Ensure we are in the right directory
cd /data/zyjin/mace/mace

RUN_ROOT="/data/zyjin/mace/mace_input/mag/runs/magmoms_2x2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_ROOT"

echo "RUN_ROOT=$RUN_ROOT"

DATASETS=("/data/zyjin/mace/mace_input/mag/dataseteasy.xyz" "/data/zyjin/mace/mace_input/mag/datasethard.xyz")
MODELS=("MACE" "MACESOG")

for ds in "${DATASETS[@]}"; do
  dname="$(basename "$ds" .xyz)"
  for model in "${MODELS[@]}"; do
    run_name="${dname}_${model,,}_magmoms"
    work_dir="$RUN_ROOT/$run_name"
    mkdir -p "$work_dir"
    echo "[run] $run_name"

    # Using python -m mace.cli.run_train or full path
    # Set stress_weight to 0.0 and ignore stress_key to avoid parsing error if it's not 3x3
    python mace/cli/run_train.py \
      --name "$run_name" \
      --work_dir "$work_dir" \
      --model "$model" \
      --train_file "$ds" \
      --valid_fraction 0.1 \
      --test_file "$ds" \
      --loss energy_forces_magmoms \
      --error_table EnergyForcesMagmomsRMSE \
      --E0s average \
      --energy_key energy \
      --forces_key forces \
      --magmoms_key mag \
      --r_max 5.5 \
      --num_interactions 2 \
      --max_ell 3 \
      --max_L 1 \
      --num_channels 128 \
      --hidden_irreps 128x0e+128x1o \
      --MLP_irreps 16x0e \
      --correlation 2 \
      --batch_size 4 \
      --valid_batch_size 4 \
      --num_workers 0 \
      --max_num_epochs 120 \
      --energy_weight 1.0 \
      --forces_weight 100.0 \
      --magmoms_weight 1.0 \
      --stress_weight 0.0 \
      --default_dtype float32 \
      --device cuda \
      --seed 123 \
      --ema \
      2>&1 | tee "$work_dir/console.log"
  done
done

echo "===== SUMMARY FROM LOGS ====="
for log in "$RUN_ROOT"/*/console.log; do
  run_name="$(basename "$(dirname "$log")")"
  best_line="$(grep -E "(Final|Test|Evaluating|RMSE_E_per_atom|RMSE_F|RMSE_M)" "$log" | tail -n 1 || true)"
  echo "[$run_name] $best_line"
  echo "[$run_name] log=$log"
done

echo "===== RUN_ROOT ====="
echo "$RUN_ROOT"

# Print directory structure for verification
for d in "$RUN_ROOT"/*; do
  if [[ -d "$d" ]]; then
    echo "[files] $d"
    find "$d" -maxdepth 3 -type f | sort
  fi
done
