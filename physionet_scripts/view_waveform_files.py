#!/usr/bin/env python3
"""
VIEW WAVEFORM FILES
Inspect .hea (header) and .dat (data) files from MIMIC-IV Waveform Database
"""

import os
import struct
import numpy as np
import pandas as pd
from pathlib import Path
import wfdb


def summarize_signal(signal_data):
    """Return NaN-safe summary stats and valid sample coverage for a 1D signal."""
    total = signal_data.size
    valid_mask = np.isfinite(signal_data)
    valid_count = int(np.count_nonzero(valid_mask))
    coverage_pct = (100.0 * valid_count / total) if total > 0 else 0.0

    if valid_count == 0:
        return {
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "coverage_pct": coverage_pct,
            "valid_count": valid_count,
            "total_count": total,
        }

    valid_data = signal_data[valid_mask]
    return {
        "min": float(np.min(valid_data)),
        "max": float(np.max(valid_data)),
        "mean": float(np.mean(valid_data)),
        "std": float(np.std(valid_data)),
        "coverage_pct": coverage_pct,
        "valid_count": valid_count,
        "total_count": total,
    }

print("\n" + "="*90)
print("WAVEFORM FILE VIEWER: Patient P10494990, Record 88374538")
print("="*90)

# ============================================================================
# PART 1: EXPLORE DIRECTORY STRUCTURE
# ============================================================================

print("\n" + "#"*90)
print("# PART 1: EXPLORING WAVEFORM DIRECTORY STRUCTURE")
print("#"*90)

waveform_dir = "/Users/mahdiya/physionet.org/files/mimic4wdb/0.1.0/waves/p104/p10494990/88374538"

print(f"\nDirectory: {waveform_dir}\n")

if os.path.exists(waveform_dir):
    files = os.listdir(waveform_dir)
    files.sort()
    
    print(f"Found {len(files)} items:\n")
    
    hea_files = []
    dat_files = []
    other_files = []
    
    for f in files:
        fpath = os.path.join(waveform_dir, f)
        
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            
            if f.endswith('.hea'):
                hea_files.append((f, size))
                print(f"  [HEA] {f:<30} {size:>15,} bytes (header)")
            elif f.endswith('.dat'):
                dat_files.append((f, size))
                print(f"  [DAT] {f:<30} {size:>15,} bytes (data)")
            else:
                other_files.append((f, size))
                print(f"  [---] {f:<30} {size:>15,} bytes")
    
    print(f"\n✓ Summary:")
    print(f"  • .hea files: {len(hea_files)}")
    print(f"  • .dat files: {len(dat_files)}")
    print(f"  • Other files: {len(other_files)}")
else:
    print(f"✗ Directory not found: {waveform_dir}")
    exit(1)

# ============================================================================
# PART 2: PARSE MAIN HEADER FILE (.hea)
# ============================================================================

print("\n\n" + "#"*90)
print("# PART 2: PARSING MAIN HEADER FILE (88374538.hea)")
print("#"*90)

main_hea_file = os.path.join(waveform_dir, "88374538.hea")

if os.path.exists(main_hea_file):
    print(f"\n✓ Found: {main_hea_file}\n")
    
    print("-"*90)
    print("CONTENT OF 88374538.hea:")
    print("-"*90)
    
    with open(main_hea_file, 'r') as f:
        hea_content = f.read()
        print(hea_content)
    
    print("-"*90)
    print("PARSED HEADER INFORMATION:")
    print("-"*90)

    raw_lines = [line.strip() for line in hea_content.splitlines() if line.strip()]
    comment_lines = [line for line in raw_lines if line.startswith("#")]
    data_lines = [line for line in raw_lines if not line.startswith("#")]

    if len(data_lines) == 0:
        print("✗ No WFDB header content found after comments")
    else:
        header_parts = data_lines[0].split()
        record_token = header_parts[0]

        # Multi-segment records encode segment count as record_name/num_segments.
        record_name = record_token
        num_segments = None
        if "/" in record_token:
            record_name_parts = record_token.split("/", 1)
            record_name = record_name_parts[0]
            if record_name_parts[1].isdigit():
                num_segments = int(record_name_parts[1])

        num_signals = int(header_parts[1]) if len(header_parts) > 1 else None

        # The fs field can be like "62.4725/999.56"; the value before '/' is fs.
        sampling_freq = None
        if len(header_parts) > 2:
            fs_token = header_parts[2].split("/", 1)[0]
            try:
                sampling_freq = float(fs_token)
            except ValueError:
                sampling_freq = None

        num_samples = None
        if len(header_parts) > 3:
            try:
                num_samples = int(header_parts[3])
            except ValueError:
                num_samples = None

        print(f"\nRecord name: {record_name}")
        if num_segments is not None:
            print(f"Number of segments: {num_segments}")
        if num_signals is not None:
            print(f"Number of signals: {num_signals}")
        if sampling_freq is not None:
            print(f"Sampling frequency: {sampling_freq} Hz")
        if num_samples is not None and sampling_freq is not None:
            duration_hours = num_samples / sampling_freq / 3600
            duration_minutes = num_samples / sampling_freq / 60
            print(f"Total samples: {num_samples:,}")
            print(f"Duration: {duration_hours:.2f} hours ({duration_minutes:.1f} minutes)")

        if len(comment_lines) > 0:
            print("\nHeader comments:")
            for comment in comment_lines:
                print(f"  {comment}")

        if num_segments is not None:
            print(f"\n{'Segment':<10} {'Entry':<25} {'Samples':<15} {'Duration (s)':<15} {'Type':<10}")
            print("-" * 80)

            segment_lines = data_lines[1:1 + num_segments]
            total_segment_samples = 0

            for idx, segment_line in enumerate(segment_lines):
                parts = segment_line.split()
                segment_name = parts[0] if len(parts) > 0 else f"unknown_{idx:04d}"

                segment_samples = None
                if len(parts) > 1:
                    try:
                        segment_samples = int(parts[1])
                    except ValueError:
                        segment_samples = None

                duration_seconds = None
                if segment_samples is not None and sampling_freq is not None:
                    duration_seconds = segment_samples / sampling_freq
                    total_segment_samples += segment_samples

                entry_type = "gap" if segment_name == "~" else "segment"
                samples_display = f"{segment_samples:,}" if segment_samples is not None else "?"
                duration_display = f"{duration_seconds:.2f}" if duration_seconds is not None else "?"

                print(
                    f"{idx:<10} {segment_name:<25} {samples_display:<15} {duration_display:<15} {entry_type:<10}"
                )

            if total_segment_samples > 0 and sampling_freq is not None:
                total_duration_hours = total_segment_samples / sampling_freq / 3600
                print(f"\nTotal segment samples (including gaps): {total_segment_samples:,}")
                print(f"Total segment duration: {total_duration_hours:.2f} hours")
        else:
            print(f"\n{'Signal':<15} {'File':<25} {'Format':<15} {'Samples':<20} {'Description':<20}")
            print("-" * 95)

            signal_lines = data_lines[1:]
            n_to_show = len(signal_lines) if num_signals is None else min(len(signal_lines), num_signals)

            for i, signal_line in enumerate(signal_lines[:n_to_show]):
                parts = signal_line.split()

                if len(parts) >= 2:
                    filename = parts[0]
                    fmt = parts[1].split('x')[0] if 'x' in parts[1] else parts[1]
                    num_samp = parts[2] if len(parts) > 2 else "?"

                    signal_name = f"Signal {i}"
                    if len(parts) > 3:
                        signal_name = ' '.join(parts[3:])

                    print(f"Signal {i:<7} {filename:<25} {fmt:<15} {num_samp:<20} {signal_name:<20}")
else:
    print(f"✗ Main header file not found: {main_hea_file}")

# ============================================================================
# PART 3: PARSE SEGMENT HEADER FILES
# ============================================================================

print("\n\n" + "#"*90)
print("# PART 3: PARSING SEGMENT HEADER FILES (88374538_0000.hea, etc.)")
print("#"*90)

segment_hea_files = [f for f in hea_files if '_' in f[0]]

print(f"\nFound {len(segment_hea_files)} segment files\n")

for i, (seg_file, size) in enumerate(segment_hea_files[:5]):  # Show first 5
    seg_path = os.path.join(waveform_dir, seg_file)
    
    print(f"\n{'-'*90}")
    print(f"File: {seg_file}")
    print(f"{'-'*90}")
    
    with open(seg_path, 'r') as f:
        seg_content = f.read()
    
    # Show first few lines
    seg_lines = seg_content.strip().split('\n')
    for line in seg_lines[:10]:
        print(f"  {line}")
    
    if len(seg_lines) > 10:
        print(f"  ... ({len(seg_lines) - 10} more lines)")

if len(segment_hea_files) > 5:
    print(f"\n... and {len(segment_hea_files) - 5} more segment files")

# ============================================================================
# PART 4: USE WFDB TO READ AND DISPLAY WAVEFORM DATA
# ============================================================================

print("\n\n" + "#"*90)
print("# PART 4: READING WAVEFORM DATA WITH WFDB LIBRARY")
print("#"*90)

record_path = os.path.join(waveform_dir, "88374538")

print(f"\nLoading record: {record_path}\n")

try:
    record = wfdb.rdrecord(record_path)
    
    print(f"✓ Successfully loaded waveform record\n")
    
    print(f"{'-'*90}")
    print("RECORD METADATA:")
    print(f"{'-'*90}")
    print(f"Record name: {record.record_name}")
    print(f"Channels: {record.sig_name}")
    print(f"Number of signals: {len(record.sig_name)}")
    print(f"Sampling frequency: {record.fs} Hz")
    print(f"Total samples: {len(record.p_signal):,}")
    print(f"Duration: {len(record.p_signal) / record.fs / 3600:.2f} hours")
    
    # Signal details
    print(f"\n{'-'*90}")
    print("SIGNAL DETAILS:")
    print(f"{'-'*90}")
    print(f"{'#':<5} {'Name':<20} {'Min Value':<15} {'Max Value':<15} {'Mean':<15} {'Std Dev':<15} {'Coverage %':<12}")
    print("-" * 100)
    
    for i, sig_name in enumerate(record.sig_name):
        signal_data = record.p_signal[:, i]
        stats = summarize_signal(signal_data)
        print(
            f"{i:<5} {sig_name:<20} {stats['min']:<15.4f} {stats['max']:<15.4f} "
            f"{stats['mean']:<15.4f} {stats['std']:<15.4f} {stats['coverage_pct']:<12.2f}"
        )
    
    # ========================================================================
    # PART 5: EXTRACT AND DISPLAY SAMPLE DATA
    # ========================================================================
    
    print("\n\n" + "#"*90)
    print("# PART 5: SAMPLE WAVEFORM DATA")
    print("#"*90)
    
    # First second of each signal
    samples_per_sec = int(record.fs)
    
    print(f"\nFirst second of data (sample 0-{samples_per_sec}):\n")
    
    first_second_data = record.p_signal[:samples_per_sec, :]
    
    # Create a DataFrame for display
    df_first_sec = pd.DataFrame(first_second_data, columns=record.sig_name)
    df_first_sec.index.name = "Sample"
    
    print(df_first_sec.head(20).to_string())
    print(f"\n... ({len(df_first_sec) - 20} more samples in first second)")
    
    # ========================================================================
    # PART 6: HOURLY STATISTICS
    # ========================================================================
    
    print("\n\n" + "#"*90)
    print("# PART 6: HOURLY STATISTICS")
    print("#"*90)
    
    n_hours = int(len(record.p_signal) / record.fs / 3600)
    print(f"\nTotal hours of data: {n_hours}\n")
    
    hourly_stats = []
    
    for hour in range(min(n_hours, 24)):  # First 24 hours
        start_idx = int(hour * 3600 * record.fs)
        end_idx = int((hour + 1) * 3600 * record.fs)
        
        hour_data = record.p_signal[start_idx:end_idx, :]
        
        hour_stats = {'Hour': hour}
        
        for i, sig_name in enumerate(record.sig_name):
            sig_data = hour_data[:, i]
            stats = summarize_signal(sig_data)
            hour_stats[f"{sig_name}_min"] = stats['min']
            hour_stats[f"{sig_name}_max"] = stats['max']
            hour_stats[f"{sig_name}_mean"] = stats['mean']
            hour_stats[f"{sig_name}_std"] = stats['std']
            hour_stats[f"{sig_name}_coverage_pct"] = stats['coverage_pct']
        
        hourly_stats.append(hour_stats)
    
    df_hourly = pd.DataFrame(hourly_stats)
    
    print("-"*90)
    print("HOURLY STATISTICS (First 10 hours):")
    print("-"*90)
    print(df_hourly.head(10).to_string())
    
    if n_hours > 10:
        print(f"\n... and {n_hours - 10} more hours")
    
    # ========================================================================
    # PART 7: SAVE TO CSV
    # ========================================================================
    
    print("\n\n" + "#"*90)
    print("# PART 7: SAVING DATA")
    print("#"*90)
    
    output_dir = "/tmp/p10494990_waveform_88374538"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save first hour details
    first_hour_data = record.p_signal[:int(record.fs * 3600), :]
    df_first_hour = pd.DataFrame(first_hour_data, columns=record.sig_name)
    df_first_hour.to_csv(os.path.join(output_dir, 'first_hour_waveform.csv'), index=False)
    print(f"\n✓ Saved first hour data to: {output_dir}/first_hour_waveform.csv")
    
    # Save hourly statistics
    df_hourly.to_csv(os.path.join(output_dir, 'hourly_statistics.csv'), index=False)
    print(f"✓ Saved hourly statistics to: {output_dir}/hourly_statistics.csv")
    
    # Save record info
    with open(os.path.join(output_dir, 'record_info.txt'), 'w') as f:
        f.write(f"Record: {record.record_name}\n")
        f.write(f"Signals: {', '.join(record.sig_name)}\n")
        f.write(f"Sampling frequency: {record.fs} Hz\n")
        f.write(f"Total samples: {len(record.p_signal):,}\n")
        f.write(f"Duration: {len(record.p_signal) / record.fs / 3600:.2f} hours\n")
    
    print(f"✓ Saved record info to: {output_dir}/record_info.txt")
    
except Exception as e:
    print(f"✗ Error reading waveform: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*90)
print("WAVEFORM FILE VIEWING COMPLETE")
print("="*90 + "\n")
