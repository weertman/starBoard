"""Debug script to analyze date formats in the dataset."""
import pandas as pd
from pathlib import Path

# Load metadata
df = pd.read_csv('star_dataset/metadata_temporal.csv')

print("=" * 80)
print("DATE/OUTING FORMAT ANALYSIS")
print("=" * 80)

for ds in sorted(df['dataset'].unique()):
    ds_df = df[df['dataset'] == ds]
    outings = ds_df['outing'].unique().tolist()
    dates = ds_df['date'].unique().tolist()
    has_valid_dates = ds_df['date'].notna().any()
    
    # Check if outings should collapse
    n_outings = len(outings)
    n_dates = len([d for d in dates if pd.notna(d)])
    
    print(f"\n{ds}:")
    print(f"  Outings ({n_outings}): {outings}")
    print(f"  Dates ({n_dates}): {dates}")
    print(f"  Has valid dates: {has_valid_dates}")
    
    if has_valid_dates and n_outings > n_dates:
        print(f"  ⚠️  ISSUE: {n_outings} outings but only {n_dates} unique dates!")
        print(f"      Outings should collapse by date.")
    elif not has_valid_dates and n_outings > 1:
        print(f"  ⚠️  ISSUE: {n_outings} outings but NO parseable dates!")
        print(f"      Cannot determine if outings should collapse.")

print("\n" + "=" * 80)
print("PROBLEMATIC DATASETS SUMMARY")
print("=" * 80)

problems = []
for ds in df['dataset'].unique():
    ds_df = df[df['dataset'] == ds]
    outings = ds_df['outing'].unique()
    dates = [d for d in ds_df['date'].unique() if pd.notna(d)]
    
    if len(outings) > len(dates) if dates else len(outings) > 1:
        train_pct = (ds_df['split'] == 'train').mean() * 100
        problems.append({
            'dataset': ds,
            'n_outings': len(outings),
            'n_dates': len(dates),
            'train_pct': train_pct,
            'issue': 'No dates' if not dates else 'Outings > Dates'
        })

for p in problems:
    print(f"\n{p['dataset']}:")
    print(f"  Outings: {p['n_outings']}, Unique Dates: {p['n_dates']}")
    print(f"  Issue: {p['issue']}")
    print(f"  Current train%: {p['train_pct']:.1f}%")


