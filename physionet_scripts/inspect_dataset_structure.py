#!/usr/bin/env python3
"""
Generate a structural report for MIMIC-IV clinical CSVs and MIMIC-IV waveform files.

The script scans:
- CSV files in hosp and icu folders (column headers)
- WFDB .hea files (record metadata, segment/signal structure)
- WFDB .dat files (file inventory and size)

Outputs:
- <output-prefix>.md
- <output-prefix>.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class CsvFileSummary:
    file: str
    columns: list[str] = field(default_factory=list)
    column_count: int = 0
    size_bytes: int = 0
    read_error: str | None = None


@dataclass
class RecordLineSummary:
    record_name: str | None = None
    num_segments: int | None = None
    num_signals: int | None = None
    sampling_frequency_hz: float | None = None
    samples_per_signal: int | None = None
    base_time: str | None = None
    base_date: str | None = None


@dataclass
class SegmentSummary:
    segment_name: str
    samples: int | None = None


@dataclass
class SignalSummary:
    data_file: str | None = None
    format: str | None = None
    gain_and_units: str | None = None
    adc_resolution: str | None = None
    adc_zero: str | None = None
    initial_value: str | None = None
    checksum: str | None = None
    block_size: str | None = None
    signal_name: str | None = None


@dataclass
class HeaFileSummary:
    file: str
    size_bytes: int = 0
    record: RecordLineSummary = field(default_factory=RecordLineSummary)
    comments: list[str] = field(default_factory=list)
    segments: list[SegmentSummary] = field(default_factory=list)
    signals: list[SignalSummary] = field(default_factory=list)
    referenced_dat_files: list[str] = field(default_factory=list)
    parse_error: str | None = None


@dataclass
class DatFileSummary:
    file: str
    size_bytes: int


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_hosp_dir = script_dir / "files" / "mimiciv" / "3.1" / "hosp"
    default_icu_dir = script_dir / "files" / "mimiciv" / "3.1" / "icu"

    preferred_wave_root = (
        script_dir
        / "files"
        / "mimic4wdb"
        / "0.1.0"
        / "waves"
        / "p100"
        / "p10014354"
        / "81739927"
    )
    fallback_wave_root = script_dir / "files" / "mimic4wdb" / "0.1.0" / "waves"
    default_wave_root = preferred_wave_root if preferred_wave_root.exists() else fallback_wave_root

    parser = argparse.ArgumentParser(
        description="Generate dataset structure report for MIMIC-IV CSV and WFDB files."
    )
    parser.add_argument("--hosp-dir", type=Path, default=default_hosp_dir)
    parser.add_argument("--icu-dir", type=Path, default=default_icu_dir)
    parser.add_argument("--wave-root", type=Path, default=default_wave_root)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=script_dir / "dataset_structure_report",
        help="Output path prefix. Produces <prefix>.md and <prefix>.json",
    )
    return parser.parse_args()


def read_csv_header(csv_path: Path) -> CsvFileSummary:
    summary = CsvFileSummary(file=str(csv_path))
    try:
        summary.size_bytes = csv_path.stat().st_size
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            row = next(reader, [])
            summary.columns = [col.strip() for col in row]
            summary.column_count = len(summary.columns)
    except StopIteration:
        summary.columns = []
        summary.column_count = 0
    except Exception as exc:
        summary.read_error = str(exc)
    return summary


def read_csv_or_gz_header(path: Path) -> CsvFileSummary:
    if path.suffix.lower() != ".gz":
        return read_csv_header(path)

    summary = CsvFileSummary(file=str(path))
    try:
        summary.size_bytes = path.stat().st_size
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            row = next(reader, [])
            summary.columns = [col.strip() for col in row]
            summary.column_count = len(summary.columns)
    except StopIteration:
        summary.columns = []
        summary.column_count = 0
    except Exception as exc:
        summary.read_error = str(exc)
    return summary


def parse_int(value: str) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def parse_float(value: str) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def parse_record_line(line: str) -> RecordLineSummary:
    tokens = line.split()
    summary = RecordLineSummary()

    if not tokens:
        return summary

    record_token = tokens[0]
    if "/" in record_token:
        record_name, maybe_segments = record_token.split("/", 1)
        summary.record_name = record_name
        summary.num_segments = parse_int(maybe_segments)
    else:
        summary.record_name = record_token

    if len(tokens) > 1:
        summary.num_signals = parse_int(tokens[1])

    if len(tokens) > 2:
        fs_token = tokens[2].split("/", 1)[0]
        summary.sampling_frequency_hz = parse_float(fs_token)

    if len(tokens) > 3:
        summary.samples_per_signal = parse_int(tokens[3])

    if len(tokens) > 4:
        summary.base_time = tokens[4]

    if len(tokens) > 5:
        summary.base_date = tokens[5]

    return summary


def parse_segment_line(line: str) -> SegmentSummary:
    tokens = line.split()
    if not tokens:
        return SegmentSummary(segment_name="")

    seg_name = tokens[0]
    samples = parse_int(tokens[1]) if len(tokens) > 1 else None
    return SegmentSummary(segment_name=seg_name, samples=samples)


def parse_signal_line(line: str) -> SignalSummary:
    tokens = line.split()
    signal = SignalSummary()

    if len(tokens) > 0:
        signal.data_file = tokens[0]
    if len(tokens) > 1:
        signal.format = tokens[1]
    if len(tokens) > 2:
        signal.gain_and_units = tokens[2]
    if len(tokens) > 3:
        signal.adc_resolution = tokens[3]
    if len(tokens) > 4:
        signal.adc_zero = tokens[4]
    if len(tokens) > 5:
        signal.initial_value = tokens[5]
    if len(tokens) > 6:
        signal.checksum = tokens[6]
    if len(tokens) > 7:
        signal.block_size = tokens[7]
    if len(tokens) > 8:
        signal.signal_name = " ".join(tokens[8:])

    return signal


def parse_hea_file(hea_path: Path) -> HeaFileSummary:
    summary = HeaFileSummary(file=str(hea_path))
    try:
        summary.size_bytes = hea_path.stat().st_size
        lines = [line.strip() for line in hea_path.read_text(encoding="utf-8").splitlines()]
        lines = [line for line in lines if line]

        comments = [line for line in lines if line.startswith("#")]
        summary.comments = comments
        data_lines = [line for line in lines if not line.startswith("#")]

        if not data_lines:
            summary.parse_error = "No WFDB header content found after comments"
            return summary

        summary.record = parse_record_line(data_lines[0])
        payload_lines = data_lines[1:]

        if summary.record.num_segments is not None:
            max_segments = summary.record.num_segments
            for line in payload_lines[:max_segments]:
                seg = parse_segment_line(line)
                if seg.segment_name:
                    summary.segments.append(seg)
        else:
            n_sig = summary.record.num_signals or 0
            for line in payload_lines[:n_sig]:
                signal = parse_signal_line(line)
                summary.signals.append(signal)
                if signal.data_file:
                    summary.referenced_dat_files.append(signal.data_file)

        summary.referenced_dat_files = sorted(set(summary.referenced_dat_files))

    except Exception as exc:
        summary.parse_error = str(exc)

    return summary


def summarize_csv_folder(folder: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "folder": str(folder),
        "exists": folder.exists(),
        "csv_files": [],
        "csv_file_count": 0,
    }

    if not folder.exists():
        return result

    files = sorted([p for p in folder.iterdir() if p.is_file() and (p.suffix == ".csv")])
    summaries = [read_csv_header(path) for path in files]

    result["csv_files"] = [asdict(item) for item in summaries]
    result["csv_file_count"] = len(summaries)
    return result


def summarize_wave_folder(wave_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "wave_root": str(wave_root),
        "exists": wave_root.exists(),
        "hea_files": [],
        "dat_files": [],
        "hea_file_count": 0,
        "dat_file_count": 0,
    }

    if not wave_root.exists():
        return result

    hea_paths = sorted(wave_root.rglob("*.hea"))
    dat_paths = sorted(wave_root.rglob("*.dat"))

    hea_summaries = [parse_hea_file(path) for path in hea_paths]
    dat_summaries = [DatFileSummary(file=str(path), size_bytes=path.stat().st_size) for path in dat_paths]

    result["hea_files"] = [asdict(item) for item in hea_summaries]
    result["dat_files"] = [asdict(item) for item in dat_summaries]
    result["hea_file_count"] = len(hea_summaries)
    result["dat_file_count"] = len(dat_summaries)

    dat_index = {Path(item.file).name for item in dat_summaries}
    for hea in result["hea_files"]:
        referenced = hea.get("referenced_dat_files", [])
        missing = sorted([name for name in referenced if name not in dat_index])
        hea["missing_dat_files"] = missing

    return result


def build_markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Dataset Structure Report")
    lines.append("")
    lines.append(f"Generated at (UTC): {report['generated_at_utc']}")
    lines.append("")

    lines.append("## Clinical CSV Structure")
    lines.append("")

    for section_name in ["hosp", "icu"]:
        section = report["csv_sections"][section_name]
        lines.append(f"### {section_name.upper()} Folder")
        lines.append("")
        lines.append(f"Path: {section['folder']}")
        lines.append(f"Exists: {section['exists']}")
        lines.append(f"CSV files: {section['csv_file_count']}")
        lines.append("")

        for item in section["csv_files"]:
            lines.append(f"- File: {item['file']}")
            lines.append(f"  - Size (bytes): {item['size_bytes']}")
            if item.get("read_error"):
                lines.append(f"  - Read error: {item['read_error']}")
            else:
                lines.append(f"  - Column count: {item['column_count']}")
                lines.append("  - Columns: " + ", ".join(item["columns"]))
        lines.append("")

    wave = report["wave_section"]
    lines.append("## Waveform Structure (.hea and .dat)")
    lines.append("")
    lines.append(f"Path: {wave['wave_root']}")
    lines.append(f"Exists: {wave['exists']}")
    lines.append(f".hea files: {wave['hea_file_count']}")
    lines.append(f".dat files: {wave['dat_file_count']}")
    lines.append("")

    lines.append("### .hea Files")
    lines.append("")
    for hea in wave["hea_files"]:
        lines.append(f"- File: {hea['file']}")
        lines.append(f"  - Size (bytes): {hea['size_bytes']}")

        record = hea["record"]
        lines.append(f"  - Record name: {record.get('record_name')}")
        lines.append(f"  - Number of segments: {record.get('num_segments')}")
        lines.append(f"  - Number of signals: {record.get('num_signals')}")
        lines.append(f"  - Sampling frequency (Hz): {record.get('sampling_frequency_hz')}")
        lines.append(f"  - Samples per signal: {record.get('samples_per_signal')}")
        lines.append(f"  - Base time: {record.get('base_time')}")
        lines.append(f"  - Base date: {record.get('base_date')}")

        if hea.get("parse_error"):
            lines.append(f"  - Parse error: {hea['parse_error']}")

        if hea.get("comments"):
            lines.append("  - Comments:")
            for comment in hea["comments"]:
                lines.append(f"    - {comment}")

        if hea.get("segments"):
            lines.append("  - Segments:")
            for seg in hea["segments"]:
                lines.append(f"    - {seg['segment_name']}: samples={seg.get('samples')}")

        if hea.get("signals"):
            lines.append("  - Signals:")
            for sig in hea["signals"]:
                lines.append(
                    "    - "
                    f"name={sig.get('signal_name')}, "
                    f"data_file={sig.get('data_file')}, "
                    f"format={sig.get('format')}, "
                    f"gain_units={sig.get('gain_and_units')}"
                )

        referenced = hea.get("referenced_dat_files", [])
        lines.append("  - Referenced .dat files: " + (", ".join(referenced) if referenced else "None"))

        missing = hea.get("missing_dat_files", [])
        lines.append("  - Missing referenced .dat files: " + (", ".join(missing) if missing else "None"))
        lines.append("")

    lines.append("### .dat Files")
    lines.append("")
    for dat in wave["dat_files"]:
        lines.append(f"- File: {dat['file']}")
        lines.append(f"  - Size (bytes): {dat['size_bytes']}")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "csv_sections": {
            "hosp": summarize_csv_folder(args.hosp_dir),
            "icu": summarize_csv_folder(args.icu_dir),
        },
        "wave_section": summarize_wave_folder(args.wave_root),
    }

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    json_path = output_prefix.with_suffix(".json")
    md_path = output_prefix.with_suffix(".md")

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown_report(report), encoding="utf-8")

    print("Dataset structure report generated:")
    print(f"- {md_path}")
    print(f"- {json_path}")


if __name__ == "__main__":
    main()
