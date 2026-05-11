#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/vector_stores/trials/full/kb}"
QDRANT_PATH="${QDRANT_PATH:-$ROOT_DIR/vector_stores/trials/full/qdrant_embedded}"
COLLECTION_NAME="${COLLECTION_NAME:-trial_chunks_medcpt_full}"
BATCH_SIZE="${BATCH_SIZE:-64}"
ZIP_LIST_FILE="${ZIP_LIST_FILE:-$ROOT_DIR/outputs/logs/full_trial_zip_paths.txt}"

log() {
  printf '[build_full_trial_qdrant] %s\n' "$1"
}

download_fresh() {
  local url="$1"
  local destination="$2"
  local tmp_path="${destination}.download"
  mkdir -p "$(dirname "$destination")"
  log "downloading $url -> $destination"
  wget -nv -O "$tmp_path" "$url"
  mv "$tmp_path" "$destination"
}

ZIP_PATHS=(
  "$ROOT_DIR/数据/totrials/corpus_2021_2022/ClinicalTrials.2021-04-27.part1.zip"
  "$ROOT_DIR/数据/totrials/corpus_2021_2022/ClinicalTrials.2021-04-27.part2.zip"
  "$ROOT_DIR/数据/totrials/corpus_2021_2022/ClinicalTrials.2021-04-27.part3.zip"
  "$ROOT_DIR/数据/totrials/corpus_2021_2022/ClinicalTrials.2021-04-27.part4.zip"
  "$ROOT_DIR/数据/totrials/corpus_2021_2022/ClinicalTrials.2021-04-27.part5.zip"
  "$ROOT_DIR/数据/totrials/corpus_2023/ClinicalTrials.2023-05-08.trials0.zip"
  "$ROOT_DIR/数据/totrials/corpus_2023/ClinicalTrials.2023-05-08.trials1.zip"
  "$ROOT_DIR/数据/totrials/corpus_2023/ClinicalTrials.2023-05-08.trials2.zip"
  "$ROOT_DIR/数据/totrials/corpus_2023/ClinicalTrials.2023-05-08.trials3.zip"
  "$ROOT_DIR/数据/totrials/corpus_2023/ClinicalTrials.2023-05-08.trials4.zip"
  "$ROOT_DIR/数据/totrials/corpus_2023/ClinicalTrials.2023-05-08.trials5.zip"
)

REPAIR_URLS=(
  "https://www.trec-cds.org/2021_data/ClinicalTrials.2021-04-27.part3.zip"
  "https://www.trec-cds.org/2023_data/ClinicalTrials.2023-05-08.trials0.zip"
  "https://www.trec-cds.org/2023_data/ClinicalTrials.2023-05-08.trials1.zip"
  "https://www.trec-cds.org/2023_data/ClinicalTrials.2023-05-08.trials2.zip"
  "https://www.trec-cds.org/2023_data/ClinicalTrials.2023-05-08.trials3.zip"
  "https://www.trec-cds.org/2023_data/ClinicalTrials.2023-05-08.trials4.zip"
  "https://www.trec-cds.org/2023_data/ClinicalTrials.2023-05-08.trials5.zip"
)

REPAIR_DESTINATIONS=(
  "$ROOT_DIR/数据/totrials/corpus_2021_2022/ClinicalTrials.2021-04-27.part3.zip"
  "$ROOT_DIR/数据/totrials/corpus_2023/ClinicalTrials.2023-05-08.trials0.zip"
  "$ROOT_DIR/数据/totrials/corpus_2023/ClinicalTrials.2023-05-08.trials1.zip"
  "$ROOT_DIR/数据/totrials/corpus_2023/ClinicalTrials.2023-05-08.trials2.zip"
  "$ROOT_DIR/数据/totrials/corpus_2023/ClinicalTrials.2023-05-08.trials3.zip"
  "$ROOT_DIR/数据/totrials/corpus_2023/ClinicalTrials.2023-05-08.trials4.zip"
  "$ROOT_DIR/数据/totrials/corpus_2023/ClinicalTrials.2023-05-08.trials5.zip"
)

mkdir -p "$(dirname "$ZIP_LIST_FILE")"
printf '%s\n' "${ZIP_PATHS[@]}" > "$ZIP_LIST_FILE"

for idx in "${!REPAIR_URLS[@]}"; do
  download_fresh "${REPAIR_URLS[$idx]}" "${REPAIR_DESTINATIONS[$idx]}"
done

log "validating corpus shards"
ZIP_LIST_FILE="$ZIP_LIST_FILE" "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import json
import os
import zipfile

zip_list_file = Path(os.environ["ZIP_LIST_FILE"])
zip_paths = [
    Path(line.strip())
    for line in zip_list_file.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
bad_archives: list[dict[str, str]] = []
summary: list[dict[str, object]] = []
for path in zip_paths:
    try:
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
            xml_count = sum(1 for name in archive.namelist() if name.lower().endswith(".xml"))
        if bad_member is not None:
            bad_archives.append({"path": str(path), "bad_member": str(bad_member)})
        summary.append({"path": str(path), "xml_count": xml_count})
    except Exception as exc:
        bad_archives.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})

print(
    json.dumps(
        {
            "checked_zip_count": len(zip_paths),
            "summary": summary,
            "bad_archives": bad_archives,
        },
        ensure_ascii=False,
        indent=2,
    )
)
if bad_archives:
    raise SystemExit(1)
PY

log "building full trial vector kb under $OUTPUT_DIR"
ZIP_LIST_FILE="$ZIP_LIST_FILE" OUTPUT_DIR="$OUTPUT_DIR" "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import json
import os

from agent.trial_vector_kb import build_trial_vector_kb

zip_list_file = Path(os.environ["ZIP_LIST_FILE"])
output_dir = os.environ["OUTPUT_DIR"]
zip_paths = [
    line.strip()
    for line in zip_list_file.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
manifest = build_trial_vector_kb(
    input_paths=zip_paths,
    output_dir=output_dir,
)
print(json.dumps(manifest, ensure_ascii=False, indent=2))
PY

log "syncing chunk embeddings into embedded qdrant under $QDRANT_PATH"
"$PYTHON_BIN" -u -m agent.retrieval.qdrant \
  --skip-build \
  --output-dir "$OUTPUT_DIR" \
  --qdrant-path "$QDRANT_PATH" \
  --collection-name "$COLLECTION_NAME" \
  --recreate \
  --batch-size "$BATCH_SIZE"

log "completed"
