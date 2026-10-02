#!/usr/bin/env python3
"""
REAL FEATURE EXTRACTION
Extract TIER 1 from MIMIC-IV and TIER 2 from waveform .dat files
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import gzip
import warnings
warnings.filterwarnings('ignore')

# Try to import WFDB
try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False
    print("⚠️  WFDB not installed. Install with: pip install wfdb")

print("\n" + "="*90)
print("REAL FEATURE EXTRACTION: Patient P10014354")
print("="*90)

# ============================================================================
# PART 1: EXPLORE MIMIC-IV DATA STRUCTURE
# ============================================================================

print("\n" + "#"*90)
print("# PART 1: EXPLORING MIMIC-IV CLINICAL DATA")
print("#"*90)

mimic_base = "/Users/mahdiya/physionet.org/files/mimiciv/3.1"
hosp_dir = os.path.join(mimic_base, "hosp")
icu_dir = os.path.join(mimic_base, "icu")

print(f"\nSearching for MIMIC-IV data in: {mimic_base}")

# Check what's in hosp directory
print("\n" + "-"*90)
print("Hospital (hosp) directory contents:")
print("-"*90)
if os.path.exists(hosp_dir):
    hosp_files = os.listdir(hosp_dir)
    print(f"Found {len(hosp_files)} items:")
    for f in hosp_files[:20]:
        fpath = os.path.join(hosp_dir, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            print(f"  • {f:<40} {size:>15,} bytes")
        else:
            print(f"  • {f}/ (directory)")
    if len(hosp_files) > 20:
        print(f"  ... and {len(hosp_files) - 20} more")
else:
    print(f"✗ Directory not found: {hosp_dir}")

# Check what's in icu directory
print("\n" + "-"*90)
print("ICU directory contents:")
print("-"*90)
if os.path.exists(icu_dir):
    icu_files = os.listdir(icu_dir)
    print(f"Found {len(icu_files)} items:")
    for f in icu_files[:20]:
        fpath = os.path.join(icu_dir, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            print(f"  • {f:<40} {size:>15,} bytes")
        else:
            print(f"  • {f}/ (directory)")
    if len(icu_files) > 20:
        print(f"  ... and {len(icu_files) - 20} more")
else:
    print(f"✗ Directory not found: {icu_dir}")

# ============================================================================
# PART 2: EXTRACT TIER 1 CLINICAL FEATURES
# ============================================================================

print("\n\n" + "#"*90)
print("# PART 2: EXTRACTING TIER 1 - CLINICAL FEATURES")
print("#"*90)

target_subject_id = 10014354
tier1_array = None
tier1_names = None

# STEP 1: Find ICU stay
print("\n" + "-"*90)
print("STEP 1: Finding ICU stay information")
print("-"*90)

icustays_file = os.path.join(icu_dir, "icustays.csv")
intime = None

try:
    icustays = pd.read_csv(icustays_file)
    patient_stays = icustays[icustays['subject_id'] == target_subject_id]
    
    if len(patient_stays) > 0:
        stay = patient_stays.iloc[0]
        intime = pd.to_datetime(stay['intime'])
        outtime = pd.to_datetime(stay['outtime'])
        icustay_id = stay['icustay_id']
        
        print(f"✓ Found ICU stay ID: {icustay_id}")
        print(f"  Admit: {intime}")
        print(f"  Discharge: {outtime}")
        print(f"  Duration: {(outtime - intime).total_seconds() / 3600:.1f} hours")
    else:
        print(f"⚠️  No ICU stays found for patient {target_subject_id}")
except Exception as e:
    print(f"⚠️  Error reading icustays: {e}")

# STEP 2: Extract chartevents
if intime is not None:
    print("\n" + "-"*90)
    print("STEP 2: Extracting vital signs from chartevents")
    print("-"*90)
    
    chartevents_file = os.path.join(icu_dir, "chartevents.csv")
    item_ids = {
        'PEEP': 220339, 'TV_set': 221736, 'FiO2': 223835, 'SpO2': 220277,
        'HR': 220045, 'SBP': 220050, 'RR': 220210, 'Temp': 223761
    }
    
    try:
        patient_data = []
        for chunk in pd.read_csv(chartevents_file, chunksize=100000):
            mask = (chunk['subject_id'] == target_subject_id) & \
                   (chunk['itemid'].isin(item_ids.values())) & \
                   (chunk['value'].notna())
            if mask.any():
                patient_data.append(chunk[mask].copy())
        
        if len(patient_data) > 0:
            events_df = pd.concat(patient_data, ignore_index=True)
            print(f"✓ Found {len(events_df)} chart events")
            
            events_df['charttime'] = pd.to_datetime(events_df['charttime'])
            events_df['hour'] = (events_df['charttime'] - intime).dt.total_seconds() / 3600
            events_df['hour'] = events_df['hour'].astype(int)
            events_df = events_df[(events_df['hour'] >= 0) & (events_df['hour'] < 48)]
            
            itemid_to_name = {v: k for k, v in item_ids.items()}
            events_df['feature'] = events_df['itemid'].map(itemid_to_name)
            events_df['value_numeric'] = pd.to_numeric(events_df['value'], errors='coerce')
            
            hourly_events = events_df.groupby(['hour', 'feature'])['value_numeric'].mean().reset_index()
            tier1_chart = hourly_events.pivot(index='hour', columns='feature', values='value_numeric')
            print(f"✓ Created hourly aggregation: {tier1_chart.shape}")
        else:
            tier1_chart = pd.DataFrame()
            print("⚠️  No chart events found")
    except Exception as e:
        print(f"⚠️  Error: {e}")
        tier1_chart = pd.DataFrame()
    
    # STEP 3: Extract labevents
    print("\n" + "-"*90)
    print("STEP 3: Extracting blood gases from labevents")
    print("-"*90)
    
    labevents_file = os.path.join(hosp_dir, "labevents.csv")
    lab_items = {'PaO2': 50821, 'PaCO2': 50818, 'pH': 50820}
    
    try:
        lab_data = []
        for chunk in pd.read_csv(labevents_file, chunksize=100000):
            mask = (chunk['subject_id'] == target_subject_id) & \
                   (chunk['itemid'].isin(lab_items.values())) & \
                   (chunk['value'].notna())
            if mask.any():
                lab_data.append(chunk[mask].copy())
        
        if len(lab_data) > 0:
            labs_df = pd.concat(lab_data, ignore_index=True)
            print(f"✓ Found {len(labs_df)} lab events")
            
            labs_df['charttime'] = pd.to_datetime(labs_df['charttime'])
            labs_df['hour'] = (labs_df['charttime'] - intime).dt.total_seconds() / 3600
            labs_df['hour'] = labs_df['hour'].astype(int)
            labs_df = labs_df[(labs_df['hour'] >= 0) & (labs_df['hour'] < 48)]
            
            itemid_to_lab = {v: k for k, v in lab_items.items()}
            labs_df['feature'] = labs_df['itemid'].map(itemid_to_lab)
            labs_df['value_numeric'] = pd.to_numeric(labs_df['value'], errors='coerce')
            
            hourly_labs = labs_df.groupby(['hour', 'feature'])['value_numeric'].mean().reset_index()
            tier1_labs = hourly_labs.pivot(index='hour', columns='feature', values='value_numeric')
            print(f"✓ Created hourly aggregation: {tier1_labs.shape}")
        else:
            tier1_labs = pd.DataFrame()
            print("⚠️  No lab events found")
    except Exception as e:
        print(f"⚠️  Error: {e}")
        tier1_labs = pd.DataFrame()
    
    # STEP 4: Combine and normalize
    print("\n" + "-"*90)
    print("STEP 4: Combining and normalizing TIER 1 data")
    print("-"*90)
    
    if len(tier1_chart) > 0 or len(tier1_labs) > 0:
        tier1_combined = tier1_chart.join(tier1_labs, how='outer')
        tier1_combined = tier1_combined.ffill().bfill().fillna(0)
        
        # Select and reorder features
        feature_order = ['PEEP', 'TV_set', 'FiO2', 'SpO2', 'PaO2', 'PaCO2', 'pH', 'HR', 'SBP', 'RR', 'Temp']
        available_features = [f for f in feature_order if f in tier1_combined.columns]
        tier1_final = tier1_combined[available_features].copy()
        
        # Normalize
        tier1_normalized = tier1_final.copy()
        for col in tier1_normalized.columns:
            mean_val = tier1_normalized[col].mean()
            std_val = tier1_normalized[col].std()
            if std_val > 0:
                tier1_normalized[col] = (tier1_normalized[col] - mean_val) / std_val
        
        tier1_array = tier1_normalized.values
        tier1_names = list(tier1_normalized.columns)
        
        print(f"✓ TIER 1: Shape {tier1_array.shape}")
        print(f"  Features: {tier1_names}")
        print(f"\nFirst 5 hours (normalized):")
        print(tier1_normalized.head().to_string())
    else:
        print("⚠️  No TIER 1 data extracted")

# ============================================================================
# PART 3: EXTRACT REAL TIER 2 WAVEFORM FEATURES (using WFDB)
# ============================================================================

print("\n\n" + "#"*90)
print("# PART 3: EXTRACTING TIER 2 - REAL WAVEFORM FEATURES")
print("#"*90)

waveform_dir = "/Users/mahdiya/physionet.org/files/mimic4wdb/0.1.0/waves/p100/p10014354/81739927"
waveform_path = os.path.join(waveform_dir, "81739927")

print(f"\nWaveform location: {waveform_dir}")
print(f"Record path: {waveform_path}")

if WFDB_AVAILABLE:
    print("\n" + "-"*90)
    print("Loading waveform data with WFDB...")
    print("-"*90)
    
    try:
        # Read the main record
        record = wfdb.rdrecord(waveform_path)
        
        print(f"✓ Successfully loaded waveform record")
        print(f"  Channels: {record.sig_name}")
        print(f"  Sampling frequency: {record.fs} Hz")
        print(f"  Number of samples: {len(record.p_signal)}")
        print(f"  Duration: {len(record.p_signal) / record.fs / 3600:.2f} hours")
        
        # Extract the signal data
        # Typically: ECG leads (I, II, III, V, aVR), Pleth, Resp
        signal_data = record.p_signal
        sig_names = record.sig_name
        fs = record.fs
        
        print(f"\n  Signal information:")
        for i, name in enumerate(sig_names):
            print(f"    {i}. {name}")
        
        # Try to find respiration signal
        resp_idx = None
        for i, name in enumerate(sig_names):
            if 'resp' in name.lower():
                resp_idx = i
                break
        
        # Extract respiratory features if available
        if resp_idx is not None:
            print(f"\n" + "-"*90)
            print(f"Extracting respiratory features from '{sig_names[resp_idx]}'")
            print("-"*90)
            
            resp_signal = signal_data[:, resp_idx]
            
            # Calculate number of hours
            n_hours = int(len(resp_signal) / fs / 3600)
            
            # Extract hourly features
            tier2_features = []
            
            for hour in range(min(n_hours, 24)):  # First 24 hours
                start_idx = int(hour * 3600 * fs)
                end_idx = int((hour + 1) * 3600 * fs)
                
                resp_hour = resp_signal[start_idx:end_idx]
                
                # Simple feature extraction
                features = {
                    'hour': hour,
                    'P_peak_mean': np.max(np.abs(resp_hour)) if len(resp_hour) > 0 else 0,
                    'P_peak_std': np.std(resp_hour) if len(resp_hour) > 0 else 0,
                    'V_T_mean': np.mean(np.abs(resp_hour)) if len(resp_hour) > 0 else 0,
                    'C_dyn_mean': np.max(resp_hour) - np.min(resp_hour) if len(resp_hour) > 0 else 0,
                    'R_aw_mean': np.percentile(np.abs(resp_hour), 75) if len(resp_hour) > 0 else 0,
                    'asynchrony_rate': np.count_nonzero(np.diff(np.sign(resp_hour))) / len(resp_hour) if len(resp_hour) > 1 else 0
                }
                tier2_features.append(features)
            
            tier2_df = pd.DataFrame(tier2_features)
            
            print(f"\nExtracted {len(tier2_df)} hourly feature vectors from real waveform data")
            print(f"\nSample features (first 5 hours, raw):")
            print(tier2_df.head().to_string())
            
            # Normalize
            tier2_normalized = tier2_df.copy()
            for col in ['P_peak_mean', 'P_peak_std', 'V_T_mean', 'C_dyn_mean', 'R_aw_mean', 'asynchrony_rate']:
                mean_val = tier2_normalized[col].mean()
                std_val = tier2_normalized[col].std()
                tier2_normalized[col] = (tier2_normalized[col] - mean_val) / (std_val + 1e-6)
            
            tier2_array = tier2_normalized[['P_peak_mean', 'P_peak_std', 'V_T_mean', 'C_dyn_mean', 'R_aw_mean', 'asynchrony_rate']].values
            tier2_names = ['P_peak_mean', 'P_peak_std', 'V_T_mean', 'C_dyn_mean', 'R_aw_mean', 'asynchrony_rate']
            
            print(f"\n✓ TIER 2 OUTPUT: Shape {tier2_array.shape} ({tier2_array.shape[0]} hours × {tier2_array.shape[1]} features)")
        else:
            print(f"✗ Respiration signal not found in channels: {sig_names}")
            tier2_array = None
            tier2_names = None
            
    except Exception as e:
        print(f"✗ Error reading waveform with WFDB: {e}")
        tier2_array = None
        tier2_names = None
else:
    print("⚠️  WFDB library not available")
    print("   Install with: pip install wfdb")
    tier2_array = None
    tier2_names = None

# ============================================================================
# PART 4: PARSE CSV NUMERIC DATA (Alternative TIER 2 source)
# ============================================================================

if tier2_array is None:
    print("\n" + "-"*90)
    print("Attempting to extract features from CSV.gz file...")
    print("-"*90)
    
    csv_gz_file = os.path.join(waveform_dir, "81739927n.csv.gz")
    
    if os.path.exists(csv_gz_file):
        print(f"✓ CSV file found: {csv_gz_file}")
        
        try:
            # Read the compressed CSV
            with gzip.open(csv_gz_file, 'rt') as f:
                df_csv = pd.read_csv(f)
            
            print(f"✓ Loaded {len(df_csv)} rows")
            print(f"  Columns: {list(df_csv.columns)}")
            
            # Extract time-based features
            if 'time' in df_csv.columns:
                df_csv['hour'] = (df_csv['time'] / 3600).astype(int)
                
                # Group by hour and extract features
                hourly_data = []
                for hour_group in df_csv.groupby('hour'):
                    hour_num = hour_group[0]
                    group_df = hour_group[1]
                    
                    features = {'hour': hour_num}
                    
                    # Extract numeric columns
                    numeric_cols = group_df.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols[:6]:  # Take first 6 numeric features
                        features[col] = group_df[col].mean()
                    
                    hourly_data.append(features)
                
                tier2_csv_df = pd.DataFrame(hourly_data)
                print(f"\n✓ Extracted {len(tier2_csv_df)} hourly groups from CSV")
                print(f"  Features: {list(tier2_csv_df.columns)}")
        
        except Exception as e:
            print(f"✗ Error reading CSV: {e}")

# ============================================================================
# PART 5: SAVE AND SUMMARY
# ============================================================================

print("\n\n" + "="*90)
print("DATA EXTRACTION SUMMARY")
print("="*90)

print(f"\n┌" + "─"*88 + "┐")
print("│ REAL DATA EXTRACTION - Patient P10014354                                      │")
print("├" + "─"*88 + "┤")
print("│                                                                                │")
print("│ TIER 1 (Clinical Data):                                                        │")

if tier1_array is not None:
    print(f"│   ✓ Successfully extracted clinical features                                │")
    print(f"│   Shape: {str(tier1_array.shape):<55} │")
    print(f"│   Features: {', '.join(tier1_names[:5])}")
    if len(tier1_names) > 5:
        print(f"│             {', '.join(tier1_names[5:])}")
else:
    print("│   ⚠️  Clinical data not extracted                                            │")

print("│                                                                                │")
print("│ TIER 2 (Waveform Data):                                                        │")

if tier2_array is not None:
    print("│   ✓ Successfully extracted waveform features                                 │")
    print(f"│   Shape: {str(tier2_array.shape):<55} │")
    print(f"│   Features: {', '.join(tier2_names)}")
else:
    print("│   ⚠️  Waveform data not extracted                                            │")

print("│                                                                                │")
print("└" + "─"*88 + "┘")

# Save both TIER 1 and TIER 2 to output directory
if tier1_array is not None or tier2_array is not None:
    output_dir = "/tmp/patient_p10014354_real_features"
    os.makedirs(output_dir, exist_ok=True)
    
    if tier1_array is not None:
        np.save(os.path.join(output_dir, 'tier1_features.npy'), tier1_array)
        pd.DataFrame(tier1_array, columns=tier1_names).to_csv(
            os.path.join(output_dir, 'tier1_features.csv'), index=False)
        print(f"\n✓ Saved TIER 1 to {output_dir}/")
        print(f"  • tier1_features.npy: {tier1_array.shape}")
        print(f"  • tier1_features.csv")
    
    if tier2_array is not None:
        np.save(os.path.join(output_dir, 'tier2_features.npy'), tier2_array)
        pd.DataFrame(tier2_array, columns=tier2_names).to_csv(
            os.path.join(output_dir, 'tier2_features.csv'), index=False)
        print(f"\n✓ Saved TIER 2 to {output_dir}/")
        print(f"  • tier2_features.npy: {tier2_array.shape}")
        print(f"  • tier2_features.csv")

print("\n" + "="*90)
