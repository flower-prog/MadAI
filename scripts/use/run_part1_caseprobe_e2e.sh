#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

CASE_FILE="${CASE_FILE:-$ROOT_DIR/数据/虚拟病例_trial_af78.txt}"
OUTPUT="${OUTPUT:-$ROOT_DIR/outputs/part1_caseprobe_e2e_result.json}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-$ROOT_DIR/outputs/snapshots_part1_caseprobe_e2e}"
TRIAL_OUTPUT_ROOT="${TRIAL_OUTPUT_ROOT:-$ROOT_DIR/vector_stores/trials/part1_caseprobe/kb}"
TRIAL_QDRANT_MODE="${TRIAL_QDRANT_MODE:-embedded}"
TRIAL_QDRANT_URL="${TRIAL_QDRANT_URL:-${MEDAI_TRIAL_QDRANT_URL:-http://127.0.0.1:6333}}"
TRIAL_QDRANT_PATH="${TRIAL_QDRANT_PATH:-$ROOT_DIR/vector_stores/trials/part1_caseprobe/qdrant_embedded}"
TRIAL_QDRANT_COLLECTION="${TRIAL_QDRANT_COLLECTION:-trial_chunks_medcpt_part1_caseprobe}"
RETRIEVER_BACKEND="${RETRIEVER_BACKEND:-hybrid}"

log() {
  printf '[run_part1_caseprobe_e2e] %s\n' "$1"
}

if [ ! -x "$PYTHON_BIN" ]; then
  log "python not found or not executable: $PYTHON_BIN"
  exit 1
fi

if [ ! -f "$CASE_FILE" ]; then
  log "case file not found: $CASE_FILE"
  exit 1
fi

if [ ! -f "$TRIAL_OUTPUT_ROOT/trial_record.jsonl" ] || [ ! -f "$TRIAL_OUTPUT_ROOT/trial_chunk.jsonl" ]; then
  log "trial KB files not found under: $TRIAL_OUTPUT_ROOT"
  log "expected trial_record.jsonl and trial_chunk.jsonl"
  exit 1
fi

QDRANT_ARGS=(
  --trial-vector-store qdrant
  --trial-qdrant-collection "$TRIAL_QDRANT_COLLECTION"
)

case "$TRIAL_QDRANT_MODE" in
  server)
    if [ -z "$TRIAL_QDRANT_URL" ]; then
      log "TRIAL_QDRANT_URL is required when TRIAL_QDRANT_MODE=server"
      exit 1
    fi
    QDRANT_ARGS+=(--trial-qdrant-url "$TRIAL_QDRANT_URL")
    ;;
  embedded)
    if [ ! -f "$TRIAL_QDRANT_PATH/meta.json" ]; then
      log "embedded Qdrant metadata not found: $TRIAL_QDRANT_PATH/meta.json"
      log "run scripts/use/build_part1_caseprobe_qdrant.sh first"
      exit 1
    fi
    QDRANT_ARGS+=(--trial-qdrant-path "$TRIAL_QDRANT_PATH")
    ;;
  *)
    log "unsupported TRIAL_QDRANT_MODE=$TRIAL_QDRANT_MODE (expected embedded or server)"
    exit 1
    ;;
esac

if [ "$TRIAL_QDRANT_MODE" = "server" ]; then
  if command -v curl >/dev/null 2>&1; then
    if ! curl -fsS "$TRIAL_QDRANT_URL/collections/$TRIAL_QDRANT_COLLECTION" >/dev/null; then
      log "Qdrant server collection is not reachable: $TRIAL_QDRANT_URL/collections/$TRIAL_QDRANT_COLLECTION"
      log "run TRIAL_QDRANT_MODE=server scripts/use/build_part1_caseprobe_qdrant.sh first"
      exit 1
    fi
  else
    log "curl not found; skipping Qdrant server preflight"
  fi
fi

log "case_file=$CASE_FILE"
log "trial_output_root=$TRIAL_OUTPUT_ROOT"
log "trial_qdrant_mode=$TRIAL_QDRANT_MODE"
if [ "$TRIAL_QDRANT_MODE" = "server" ]; then
  log "trial_qdrant_url=$TRIAL_QDRANT_URL"
else
  log "trial_qdrant_path=$TRIAL_QDRANT_PATH"
fi
log "trial_qdrant_collection=$TRIAL_QDRANT_COLLECTION"
log "retriever_backend=$RETRIEVER_BACKEND"
log "output=$OUTPUT"

"$PYTHON_BIN" "$ROOT_DIR/scripts/use/run_single_case_e2e.py" \
  --case-file "$CASE_FILE" \
  --retriever-backend "$RETRIEVER_BACKEND" \
  --trial-output-root "$TRIAL_OUTPUT_ROOT" \
  "${QDRANT_ARGS[@]}" \
  --output "$OUTPUT" \
  --snapshot-dir "$SNAPSHOT_DIR" \
  "$@"
