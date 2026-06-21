#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$(id -u)" -eq 0 ]]; then
  echo "[ERROR] Do not run this script with sudo."
  echo "[ERROR] IsaacLab must run as user 'shenji'; sudo changes HOME to /root and breaks the IsaacLab path."
  exit 1
fi

if [[ "${TERM:-}" == "dumb" || -z "${TERM:-}" ]]; then
  export TERM=xterm
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SEED="${SEED:-42}"
ROOT="${ROOT:-$REPO_ROOT/Paper/论点证明/phase1_dec_coup_full_grid/seed${SEED}}"
DEC_RUN="${DEC_RUN:-$REPO_ROOT/logs/rsl_rl/Encoder_DataCollectionMLW/2026-04-19_23-47-50}"
COUP_RUN="${COUP_RUN:-$REPO_ROOT/logs/rsl_rl/Encoder_DataCollectionMLW/2026-04-21_21-25-05}"
PLOT="${PLOT:-$REPO_ROOT/source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_lab/plot/IsaaclabPlot12.5.py}"
ISAACLAB="${ISAACLAB:-$HOME/IsaacLab/isaaclab.sh}"
CKPTS="${CKPTS:-15500 15000 14500}"
HEADLESS_FLAG="${HEADLESS_FLAG:---headless}"

mkdir -p "$ROOT"
LOG="$ROOT/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "[INFO] repo: $REPO_ROOT"
echo "[INFO] output: $ROOT"
echo "[INFO] log: $LOG"
echo "[INFO] ckpts: $CKPTS"
echo

if [[ ! -x "$ISAACLAB" ]]; then
  echo "[ERROR] IsaacLab launcher not found or not executable: $ISAACLAB"
  exit 1
fi

run_play() {
  local name="$1"
  local run_dir="$2"
  local ckpt="$3"
  local anchor="$4"
  local coef="$5"
  local out_csv="$6"

  local ckpt_path="$run_dir/model_${ckpt}.pt"
  if [[ ! -f "$ckpt_path" ]]; then
    echo "[ERROR] Missing checkpoint: $ckpt_path"
    return 1
  fi

  rm -f "$run_dir/payload_data.csv"

  echo "[RUN] $name checkpoint=$ckpt"
  "$ISAACLAB" -p scripts/rsl_rl/play.py \
    --task Isaac-Uav-Meta-v0 \
    --num_envs 1 \
    --seed "$SEED" \
    --checkpoint "$ckpt_path" \
    $HEADLESS_FLAG \
    env.rma_use_physics_anchor="$anchor" \
    env.rma_phys_anchor_coef="$coef" \
    env.scene.clone_in_fabric=false

  if [[ ! -s "$run_dir/payload_data.csv" ]]; then
    echo "[ERROR] play finished but CSV was not created: $run_dir/payload_data.csv"
    return 1
  fi

  cp "$run_dir/payload_data.csv" "$out_csv"
  echo "[OK] copied: $out_csv"
}

for CKPT in $CKPTS; do
  echo
  echo "========== checkpoint episode $CKPT =========="
  OUT_DIR="$ROOT/episode_${CKPT}"
  mkdir -p "$OUT_DIR/Decoupled" "$OUT_DIR/Coupled" "$OUT_DIR/comparison"

  run_play "Decoupled" "$DEC_RUN" "$CKPT" true 1.0 "$OUT_DIR/Decoupled/payload_data.csv"
  run_play "Coupled" "$COUP_RUN" "$CKPT" false 0.0 "$OUT_DIR/Coupled/payload_data.csv"

  echo "[PLOT] checkpoint=$CKPT"
  "$ISAACLAB" -p "$PLOT" \
    --csv \
    "$OUT_DIR/Coupled/payload_data.csv" \
    "$OUT_DIR/Decoupled/payload_data.csv" \
    --labels Coupled Decoupled \
    --out_dir "$OUT_DIR/comparison" \
    --time_window 5

  ls -lh "$OUT_DIR"/{Coupled,Decoupled}/payload_data.csv
done

echo
echo "[DONE] Decoupled vs Coupled outputs are under: $ROOT"
echo "[DONE] Full log: $LOG"
