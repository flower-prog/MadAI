#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/vector_stores/trials/part1_caseprobe/kb}"
TRIAL_QDRANT_MODE="${TRIAL_QDRANT_MODE:-embedded}"
QDRANT_URL="${QDRANT_URL:-${TRIAL_QDRANT_URL:-${MEDAI_TRIAL_QDRANT_URL:-http://127.0.0.1:6333}}}"
QDRANT_PATH="${QDRANT_PATH:-$ROOT_DIR/vector_stores/trials/part1_caseprobe/qdrant_embedded}"
COLLECTION_NAME="${COLLECTION_NAME:-trial_chunks_medcpt_part1_caseprobe}"
BATCH_SIZE="${BATCH_SIZE:-64}"
RECREATE="${RECREATE:-1}"
RESUME_PREFIX_COUNT="${RESUME_PREFIX_COUNT:-0}"

log() {
  printf '[build_part1_caseprobe_qdrant] %s\n' "$1"
}

if [ ! -f "$OUTPUT_DIR/trial_record.jsonl" ] || [ ! -f "$OUTPUT_DIR/trial_chunk.jsonl" ]; then
  log "missing trial KB under $OUTPUT_DIR"
  log "expected trial_record.jsonl and trial_chunk.jsonl"
  exit 1
fi

QDRANT_ARGS=(--collection-name "$COLLECTION_NAME")

case "$TRIAL_QDRANT_MODE" in
  server)
    if [ -z "$QDRANT_URL" ]; then
      log "QDRANT_URL or TRIAL_QDRANT_URL is required when TRIAL_QDRANT_MODE=server"
      exit 1
    fi
    QDRANT_ARGS+=(--qdrant-url "$QDRANT_URL")
    ;;
  embedded)
    mkdir -p "$QDRANT_PATH"
    QDRANT_ARGS+=(--qdrant-path "$QDRANT_PATH")
    ;;
  *)
    log "unsupported TRIAL_QDRANT_MODE=$TRIAL_QDRANT_MODE (expected embedded or server)"
    exit 1
    ;;
esac

log "syncing part1_caseprobe chunk embeddings into $TRIAL_QDRANT_MODE qdrant"
log "kb=$OUTPUT_DIR"
if [ "$TRIAL_QDRANT_MODE" = "server" ]; then
  log "qdrant_url=$QDRANT_URL"
else
  log "qdrant_path=$QDRANT_PATH"
fi
log "collection=$COLLECTION_NAME"
log "batch_size=$BATCH_SIZE"
log "recreate=$RECREATE"
log "resume_prefix_count=$RESUME_PREFIX_COUNT"

SYNC_ARGS=()
if [ "$RECREATE" = "1" ] || [ "$RECREATE" = "true" ]; then
  SYNC_ARGS+=(--recreate)
fi
if [ "$RESUME_PREFIX_COUNT" = "1" ] || [ "$RESUME_PREFIX_COUNT" = "true" ]; then
  SYNC_ARGS+=(--resume-prefix-count)
fi

"$PYTHON_BIN" -u -m agent.retrieval.qdrant \
  --skip-build \
  --output-dir "$OUTPUT_DIR" \
  "${QDRANT_ARGS[@]}" \
  "${SYNC_ARGS[@]}" \
  --batch-size "$BATCH_SIZE"

log "completed"
