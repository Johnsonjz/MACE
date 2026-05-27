set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# source /data/zyjin/mace/.venv/bin/activate || conda activate mace

RUN_ROOT="/data/zyjin/mace/mace_input/mag/runs/magmoms_2x2_$(date +%Y%m%d_%H%M%S)"
CLEAN_DIR="$RUN_ROOT/cleaned_data"
mkdir -p "$CLEAN_DIR"

# Cleaning data
python3 - <<PY
from pathlib import Path
import re

run_root = Path(r"$RUN_ROOT")
clean_dir = Path(r"$CLEAN_DIR")
src_files = [
    Path('/data/zyjin/mace/mace_input/mag/dataseteasy.xyz'),
    Path('/data/zyjin/mace/mace_input/mag/datasethard.xyz'),
]

for src in src_files:
    dst = clean_dir / src.name
    lines = src.read_text().splitlines()
    out = []
    i = 0
    while i < len(lines):
        n_line = lines[i]
        out.append(n_line)
        i += 1
        if i >= len(lines):
            break
        header = lines[i]
        # Remove the stress key-value pair in the comment line
        header = re.sub(r'\s+stress="[^"]*"', '', header)
        out.append(header)
        i += 1
        try:
            n_atoms = int(n_line.strip())
        except ValueError:
            n_atoms = 0
        for _ in range(n_atoms):
            if i < len(lines):
                out.append(lines[i])
                i += 1
    dst.write_text("\n".join(out) + "\n")
    print(f"cleaned: {src} -> {dst}")
PY

echo "RUN_ROOT=$RUN_ROOT"
echo "CLEAN_DIR=$CLEAN_DIR"

DATASETS=("$CLEAN_DIR/dataseteasy.xyz" "$CLEAN_DIR/datasethard.xyz")
# MODES
MODELS=("MACE" "MACESOG")

for ds in "${DATASETS[@]}"; do
  dname="$(basename "$ds" .xyz)"
  for model in "${MODELS[@]}"; do
    run_name="${dname}_${model,,}_magmoms"
    work_dir="$RUN_ROOT/$run_name"
    mkdir -p "$work_dir"
    echo "[run] $run_name"

    python3 mace/cli/run_train.py \
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
      --batch_size 2 \
      --valid_batch_size 2 \
      --num_workers 0 \
      --max_num_epochs 120 \
      --energy_weight 1.0 \
      --forces_weight 100.0 \
      --magmoms_weight 1.0 \
      --default_dtype float32 \
      --device cuda \
      --seed 123 \
      --ema \
      2>&1 | tee "$work_dir/console.log"
  done
done

# Summary
python3 - <<PY
from pathlib import Path
import re

run_root = Path(r"$RUN_ROOT")
print("===== METRIC SUMMARY =====")
pat = re.compile(r"Test: head: .*RMSE_E_per_atom=\s*([0-9.]+) meV, RMSE_F=\s*([0-9.]+) meV / A, RMSE_M=\s*([0-9.]+)")

for run_dir in sorted([p for p in run_root.iterdir() if p.is_dir() and p.name != 'cleaned_data']):
    log = run_dir / 'console.log'
    best = None
    if log.exists():
        for line in log.read_text(errors='ignore').splitlines():
            m = pat.search(line)
            if m:
                best = (float(m.group(1)), float(m.group(2)), float(m.group(3)), line.strip())
    if best is None:
        print(f"{run_dir.name}\tNO_TEST_LINE\tlog={log}")
    else:
        e, f, mval, raw = best
        print(f"{run_dir.name}\tRMSE_E_meV_per_atom={e:.4f}\tRMSE_F_meV_per_A={f:.4f}\tRMSE_M={mval:.6f}")

print("===== RUN_ROOT =====")
print(run_root)
PY
