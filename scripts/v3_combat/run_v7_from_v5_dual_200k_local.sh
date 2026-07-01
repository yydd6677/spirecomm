#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PYTHON="${PYTHON:-/home/yydd/miniforge3/envs/spirecomm-rl/bin/python}"
JOB_NAME="${JOB_NAME:-v7_from_v5_dual_200k_local}"
SHARD_DIR="${SHARD_DIR:-data/v3_combat_teacher/v7_next_from_v5_dual_semantic_legacy_gate_200k}"
SOURCE_LIST="${SOURCE_LIST:-_cache/${JOB_NAME}_source_shards.txt}"
VALIDATION_SOURCES="${VALIDATION_SOURCES:-data/v3_combat_tensor/${JOB_NAME}.validation_every7.txt}"
TENSOR="${TENSOR:-data/v3_combat_tensor/${JOB_NAME}.pt}"
OUTPUT="${OUTPUT:-models/v3_combat_transformer_${JOB_NAME}.pt}"
EPOCH_DIR="${EPOCH_DIR:-${OUTPUT}.epochs}"
LOG="${LOG:-logs/${JOB_NAME}.log}"
PIDFILE="${PIDFILE:-logs/${JOB_NAME}.pid}"
CACHE_WORKERS="${CACHE_WORKERS:-6}"
BATCH_SIZE="${BATCH_SIZE:-64}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

mkdir -p "$(dirname "$SOURCE_LIST")" "$(dirname "$VALIDATION_SOURCES")" "$(dirname "$TENSOR")" "$(dirname "$OUTPUT")" "$EPOCH_DIR" logs
echo "$$" > "$PIDFILE"
exec >>"$LOG" 2>&1

for var_name in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS; do
  var_value="${!var_name:-}"
  if [[ -z "$var_value" || "$var_value" == *[^0-9]* ]]; then
    export "$var_name=1"
  fi
done

echo "[$(date '+%F %T')] start $JOB_NAME pid=$$"
echo "python=$PYTHON"
echo "shard_dir=$SHARD_DIR"
echo "tensor=$TENSOR"
echo "output=$OUTPUT"
echo "cache_workers=$CACHE_WORKERS batch_size=$BATCH_SIZE"

"$PYTHON" - "$SHARD_DIR" "$SOURCE_LIST" "$VALIDATION_SOURCES" <<'PY'
import re
import sys
from pathlib import Path

shard_dir = Path(sys.argv[1])
source_list = Path(sys.argv[2])
validation_sources = Path(sys.argv[3])
shards = sorted(shard_dir.glob("shard_*.pt"))
if not shards:
    raise SystemExit(f"no shards found under {shard_dir}")
source_list.write_text("\n".join(str(path) for path in shards) + "\n", encoding="utf-8")

selected = []
for path in shards:
    match = re.search(r"shard_(\d+)\.pt$", path.name)
    shard_index = int(match.group(1)) if match else len(selected)
    # Spread validation across the whole run instead of taking an early contiguous block.
    if shard_index % 7 == 0:
        selected.append(path)
if not selected:
    selected = shards[: max(1, len(shards) // 7)]
validation_sources.write_text("\n".join(str(path) for path in selected) + "\n", encoding="utf-8")
print(f"source_shards={len(shards)} validation_shards={len(selected)}")
PY

need_cache=1
if [[ -f "$TENSOR" ]]; then
  if "$PYTHON" - "$TENSOR" <<'PY'
import sys
from pathlib import Path
from spirecomm.ai.torch_compat import torch

tensor = Path(sys.argv[1])
payload = torch.load(tensor, map_location="cpu", weights_only=False)
metadata = dict(payload.get("metadata") or {})
token_schema = dict(payload.get("token_schema") or {})
ok = (
    str(payload.get("token_schema_version") or token_schema.get("version") or "")
    == "v3_combat_transformer_tokens_v7_action_binding"
    and int(metadata.get("root_count") or 0) >= 200000
    and int(metadata.get("sequence_length") or 0) == 90
    and len(payload.get("entity_vocab") or []) >= 450
    and len(payload.get("chunks") or []) > 0
)
if not ok:
    raise SystemExit(1)
print(f"reuse existing cache: {tensor} roots={metadata.get('root_count')} seq={metadata.get('sequence_length')} entity_vocab={len(payload.get('entity_vocab') or [])}")
PY
  then
    need_cache=0
  fi
fi

if [[ "$need_cache" -eq 1 ]]; then
  echo "[$(date '+%F %T')] rebuild tensor cache"
  rm -f "$TENSOR" "${TENSOR}.summary.json"
  rm -rf "${TENSOR}.chunks"
  mapfile -t SHARDS < "$SOURCE_LIST"
  "$PYTHON" scripts/v3_combat/cache_v3_combat_transformer_dataset.py \
    --shards "${SHARDS[@]}" \
    --output "$TENSOR" \
    --summary "${TENSOR}.summary.json" \
    --dtype float16 \
    --chunked \
    --workers "$CACHE_WORKERS" \
    --token-schema-version v3_combat_transformer_tokens_v7_action_binding
fi

"$PYTHON" - "$TENSOR" <<'PY'
import sys
from pathlib import Path
from spirecomm.ai.torch_compat import torch

tensor = Path(sys.argv[1])
payload = torch.load(tensor, map_location="cpu", weights_only=False)
metadata = dict(payload.get("metadata") or {})
token_schema = dict(payload.get("token_schema") or {})
entity_vocab = list(payload.get("entity_vocab") or [])
if str(payload.get("token_schema_version") or token_schema.get("version") or "") != "v3_combat_transformer_tokens_v7_action_binding":
    raise SystemExit("bad token schema")
if int(metadata.get("sequence_length") or 0) != 90:
    raise SystemExit(f"bad sequence_length: {metadata.get('sequence_length')}")
if len(entity_vocab) < 450:
    raise SystemExit(f"suspicious entity_vocab len: {len(entity_vocab)}")
if int(metadata.get("root_count") or 0) < 200000:
    raise SystemExit(f"insufficient roots: {metadata.get('root_count')}")
print(f"cache schema ok roots={metadata.get('root_count')} candidates={metadata.get('candidate_count')} chunks={len(payload.get('chunks') or [])} entity_vocab={len(entity_vocab)}")
PY

echo "[$(date '+%F %T')] start training"
resume_args=()
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  if [[ ! -f "$RESUME_CHECKPOINT" ]]; then
    echo "resume checkpoint not found: $RESUME_CHECKPOINT" >&2
    exit 2
  fi
  resume_args=(--resume-checkpoint "$RESUME_CHECKPOINT")
  echo "resume_checkpoint=$RESUME_CHECKPOINT"
fi

"$PYTHON" -u scripts/v3_combat/train_v3_combat_transformer_scorer.py \
  --tensor-dataset "$TENSOR" \
  --validation-source-shards-file "$VALIDATION_SOURCES" \
  --output "$OUTPUT" \
  "${resume_args[@]}" \
  --epoch-output-dir "$EPOCH_DIR" \
  --architecture candidate \
  --candidate-head-variant dual-action-binding-gate \
  --action-set-layers 1 \
  --legacy-dropout 0.1 \
  --token-type-vocab-size 25 \
  --epochs "${EPOCHS:-30}" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "${LEARNING_RATE:-5e-5}" \
  --min-learning-rate "${MIN_LEARNING_RATE:-1e-5}" \
  --weight-decay "${WEIGHT_DECAY:-1e-4}" \
  --device "${DEVICE:-cuda}" \
  --amp-dtype "${AMP_DTYPE:-bfloat16}" \
  --allow-tf32 \
  --length-bucket-batches \
  --length-bucket-window "${LENGTH_BUCKET_WINDOW:-64}" \
  --stage-chunks-on-device \
  --potion-vs-non-potion-weight "${POTION_PAIR_WEIGHT:-0.5}" \
  --potion-vs-non-potion-margin "${POTION_PAIR_MARGIN:-0.15}" \
  --potion-vs-non-potion-min-teacher-gap "${POTION_PAIR_MIN_GAP:-0.5}" \
  --elite-boss-top-potion-root-weight "${ELITE_BOSS_TOP_POTION_ROOT_WEIGHT:-6.0}" \
  --gap-q-weight "${GAP_Q_WEIGHT:-0.0}" \
  --gap-q-transform "${GAP_Q_TRANSFORM:-sqrt}" \
  --gap-q-loss "${GAP_Q_LOSS:-l1}" \
  --gap-q-hard-negative-threshold "${GAP_Q_HARD_NEGATIVE_THRESHOLD:-10.0}" \
  --gap-q-hard-negative-weight "${GAP_Q_HARD_NEGATIVE_WEIGHT:-5.0}" \
  --early-stop-patience "${EARLY_STOP_PATIENCE:-5}" \
  --early-stop-min-delta "${EARLY_STOP_MIN_DELTA:-0.0005}" \
  --save-each-epoch \
  --progress-interval-chunks "${PROGRESS_INTERVAL_CHUNKS:-5}" \
  --progress-interval-seconds "${PROGRESS_INTERVAL_SECONDS:-15}" \
  --min-mem-available-gb "${MIN_MEM_AVAILABLE_GB:-1.0}" \
  --min-roots 200000 \
  --memory-log "logs/${JOB_NAME}.memory.jsonl"

echo "[$(date '+%F %T')] finished $JOB_NAME"
