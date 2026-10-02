import os
import re

def scan_waves():
    root_dir = 'files/mimic4wdb/0.1.0/waves'
    pressure_terms = ['paw', 'airway pressure', 'airwaypressure', 'p_ao', 'pip', 'pplat', 'pmean', 'pressure airway']
    flow_terms = ['airflow', 'flow', 'inspflow', 'expflow', 'vflow', 'flow airway']
    
    pressure_regex = re.compile('|'.join(re.escape(t) for t in pressure_terms), re.IGNORECASE)
    flow_regex = re.compile('|'.join(re.escape(t) for t in flow_terms), re.IGNORECASE)
    
    hea_files = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.hea'):
                hea_files.append(os.path.join(root, f))
    
    hea_files.sort()
    
    total_scanned = len(hea_files)
    pressure_files = []
    flow_files = []
    both_files = []
    
    def get_base_record(filepath):
        stem = os.path.splitext(os.path.basename(filepath))[0]
        base = re.sub(r'_\d{4}$', '', stem)
        return base

    for filepath in hea_files:
        has_p = False
        has_f = False
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    # Look for signal names which are usually at the end of the line
                    # But we scan the whole line as per requirements
                    if pressure_regex.search(line):
                        has_p = True
                    if flow_regex.search(line):
                        has_f = True
        except Exception:
            continue
            
        if has_p:
            pressure_files.append(filepath)
        if has_f:
            flow_files.append(filepath)
        if has_p and has_f:
            both_files.append(filepath)

    def get_base_counts(file_list):
        bases = {get_base_record(f) for f in file_list}
        return len(bases)

    print(f"Total .hea files scanned: {total_scanned}")
    print(f"Count of files with pressure term: {len(pressure_files)}")
    print(f"Count of files with flow term: {len(flow_files)}")
    print(f"Count of files with both terms: {len(both_files)}")
    
    print(f"Base-record count (pressure): {get_base_counts(pressure_files)}")
    print(f"Base-record count (flow): {get_base_counts(flow_files)}")
    print(f"Base-record count (both): {get_base_counts(both_files)}")
    
    print("\nPressure matching files (up to 30):")
    for f in sorted(pressure_files)[:30]:
        print(f)
        
    print("\nFlow matching files (up to 30):")
    for f in sorted(flow_files)[:30]:
        print(f)
        
    print("\nBoth matching files (up to 30):")
    for f in sorted(both_files)[:30]:
        print(f)

if __name__ == "__main__":
    scan_waves()
