#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
import xml.etree.ElementTree as ET


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def summarize_lengths(items: list[int]) -> dict[str, float | int]:
    if not items:
        return {"count": 0, "min": 0, "max": 0, "avg": 0.0, "median": 0.0}
    return {
        "count": len(items),
        "min": min(items),
        "max": max(items),
        "avg": round(sum(items) / len(items), 2),
        "median": round(float(median(items)), 2),
    }


def parse_topics_file(path: Path) -> dict:
    root = ET.parse(path).getroot()
    topics = root.findall("topic")
    year = path.stem.replace("topics", "")
    summary: dict[str, object] = {
        "file": str(path),
        "topic_count": len(topics),
        "topic_numbers": [int(topic.attrib["number"]) for topic in topics],
    }

    if year in {"2021", "2022"}:
        topic_rows = []
        lengths = []
        for topic in topics:
            text = normalize_text(" ".join(topic.itertext()))
            lengths.append(len(text))
            topic_rows.append(
                {
                    "number": int(topic.attrib["number"]),
                    "text_length": len(text),
                    "text_preview": text[:240],
                }
            )
        summary["format"] = "free_text_case_description"
        summary["text_length_stats"] = summarize_lengths(lengths)
        summary["samples"] = topic_rows[:3]
        return summary

    template_counter = Counter()
    field_counter = Counter()
    missing_counter = Counter()
    field_cardinality: Counter[str] = Counter()
    topic_rows = []

    for topic in topics:
        template = topic.attrib.get("template", "")
        template_counter[template] += 1
        fields = {}
        non_empty_fields = 0
        for field in topic.findall("field"):
            name = field.attrib.get("name", "")
            value = normalize_text(field.text or "")
            field_counter[name] += 1
            if not value:
                missing_counter[name] += 1
            else:
                non_empty_fields += 1
                field_cardinality[name] += 1
            fields[name] = value
        topic_rows.append(
            {
                "number": int(topic.attrib["number"]),
                "template": template,
                "field_count": len(fields),
                "non_empty_field_count": non_empty_fields,
                "fields": fields,
            }
        )

    summary["format"] = "templated_multifield_cases"
    summary["template_distribution"] = dict(template_counter)
    summary["field_frequency"] = dict(field_counter.most_common())
    summary["field_non_empty_frequency"] = dict(field_cardinality.most_common())
    summary["field_missing_frequency"] = dict(missing_counter.most_common())
    summary["samples"] = topic_rows[:3]
    return summary


def parse_qrels_file(path: Path) -> dict:
    rows = 0
    label_counter = Counter()
    docs_per_topic = Counter()
    unique_docs = set()

    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                continue
            topic, _, doc_id, label = parts
            rows += 1
            label_counter[label] += 1
            docs_per_topic[topic] += 1
            unique_docs.add(doc_id)

    values = list(docs_per_topic.values())
    positive = label_counter.get("1", 0) + label_counter.get("2", 0)

    return {
        "file": str(path),
        "row_count": rows,
        "topic_count": len(docs_per_topic),
        "unique_doc_count": len(unique_docs),
        "label_distribution": dict(sorted(label_counter.items())),
        "positive_fraction": round(positive / rows, 4) if rows else 0.0,
        "docs_per_topic_stats": summarize_lengths(values),
    }


def inspect_archive(path: Path) -> dict:
    info = {
        "file": str(path),
        "size_bytes": path.stat().st_size,
        "readable": False,
    }
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            xml_names = [name for name in names if name.endswith(".xml")]
            info["readable"] = True
            info["entry_count"] = len(names)
            info["xml_entry_count"] = len(xml_names)
            info["sample_entries"] = names[:5]
    except zipfile.BadZipFile as exc:
        info["error"] = f"BadZipFile: {exc}"
    except Exception as exc:  # pragma: no cover - defensive for local data oddities
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def analyze_benchmark(base_dir: Path) -> dict:
    years = {}
    for year in ("2021", "2022", "2023"):
        year_dir = base_dir / year
        years[year] = {
            "topics": parse_topics_file(year_dir / f"topics{year}.xml"),
            "qrels": parse_qrels_file(year_dir / f"qrels{year}.txt"),
        }

    corpus_dirs = {}
    for corpus_dir in ("corpus_2021_2022", "corpus_2023"):
        folder = base_dir / corpus_dir
        archives = sorted(folder.glob("*.zip"))
        corpus_dirs[corpus_dir] = {
            "archive_count": len(archives),
            "archives": [inspect_archive(path) for path in archives],
        }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_dir": str(base_dir),
        "years": years,
        "corpora": corpus_dirs,
    }


def print_console_summary(summary: dict) -> None:
    print(f"Base directory: {summary['base_dir']}")
    print()
    for year, year_summary in summary["years"].items():
        topics = year_summary["topics"]
        qrels = year_summary["qrels"]
        print(f"[{year}]")
        print(
            f"  topics={topics['topic_count']} format={topics['format']} "
            f"qrels={qrels['row_count']} unique_docs={qrels['unique_doc_count']}"
        )
        print(
            "  labels="
            + ", ".join(
                f"{label}:{count}" for label, count in qrels["label_distribution"].items()
            )
        )
        if year == "2023":
            templates = topics["template_distribution"]
            print(
                "  templates="
                + ", ".join(f"{name}:{count}" for name, count in templates.items())
            )
        else:
            length_stats = topics["text_length_stats"]
            print(
                f"  topic_text_length avg={length_stats['avg']} "
                f"median={length_stats['median']}"
            )
        print()

    for corpus_name, corpus_summary in summary["corpora"].items():
        readable = sum(1 for item in corpus_summary["archives"] if item["readable"])
        print(
            f"[{corpus_name}] archives={corpus_summary['archive_count']} readable={readable}"
        )
        unreadable = [item for item in corpus_summary["archives"] if not item["readable"]]
        for item in unreadable[:5]:
            print(f"  unreadable: {Path(item['file']).name} -> {item.get('error', 'unknown')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the TREC Clinical Trials benchmark.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/home/yuanzy/MadAI/数据/totrials"),
        help="Path to the benchmark directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/yuanzy/MadAI/outputs/totrials_summary.json"),
        help="Where to write the JSON summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze_benchmark(args.base_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print_console_summary(summary)
    print()
    print(f"Wrote JSON summary to: {args.output}")


if __name__ == "__main__":
    main()
