import os

files_to_check = [
    'paper_english/paper.tex',
    'results/lookback_ablation.tex',
    'results/sliding_window.tex',
    'results/cost_sensitivity.tex',
    'results/turnover_rates.tex',
]

for fpath in files_to_check:
    if not os.path.exists(fpath):
        print(f"[MISSING] {fpath}")
        continue
    with open(fpath, 'rb') as f:
        raw = f.read()

    issues = []

    # BOM check
    if raw[:3] == b'\xef\xbb\xbf':
        issues.append('UTF-8 BOM')
    if raw[:2] == b'\xff\xfe':
        issues.append('UTF-16 LE BOM')

    # UTF-8 validity
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError as e:
        issues.append(f'UTF-8 decode error: {e}')

    # Check for U+FFFD in decoded text
    decoded = raw.decode('utf-8', errors='replace')
    if '�' in decoded:
        issues.append('Contains U+FFFD replacement character')

    # Find non-printable control chars (not tab/LF/CR)
    for i, byte in enumerate(raw):
        if byte < 0x20 and byte not in (0x09, 0x0a, 0x0d):
            issues.append(f'Control char 0x{byte:02x} at offset {i}')

    # Find non-ASCII chars and report them
    decoded = raw.decode('utf-8')
    non_ascii = [(i, ch, hex(ord(ch))) for i, ch in enumerate(decoded) if ord(ch) > 127]
    if non_ascii:
        issues.append(f'{len(non_ascii)} non-ASCII chars, first 10: {non_ascii[:10]}')

    status = 'ISSUES: ' + '; '.join(issues) if issues else 'CLEAN'
    print(f'{fpath}: {status}')
