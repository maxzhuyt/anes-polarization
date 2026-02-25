"""
Check which GSS variables from the demographic and lifestyle variable lists
are present (have non-missing data) in all 3 years (2021, 2022, 2024).
"""

import pandas as pd
import numpy as np
import json

# --- Load the GSS data ---
print("Loading GSS data...")
gss = pd.read_csv('/project/jevans/maxzhuyt/gss-depth/gss_2021_2024.csv')
print(f"  GSS shape: {gss.shape}")
print(f"  Years: {sorted(gss['year'].unique())}")
print(f"  Rows per year: {dict(gss['year'].value_counts().sort_index())}")

# --- Load variable lists ---
demo_df = pd.read_csv('/project/jevans/maxzhuyt/claude_polarization/anes-polarization/gss_question_lists/gss_demographic_variables.csv')
life_df = pd.read_csv('/project/jevans/maxzhuyt/claude_polarization/anes-polarization/gss_question_lists/gss_politicized_lifestyle_variables.csv')

demo_vars = demo_df['VariableName'].tolist()
life_vars = life_df['VariableName'].tolist()

print(f"\nDemographic variables requested: {len(demo_vars)}")
print(f"Lifestyle variables requested: {len(life_vars)}")

gss_columns_lower = {c.lower(): c for c in gss.columns}

years = [2021, 2022, 2024]

def check_vars(var_list, label):
    """For each variable, check non-missing count per year."""
    results = []
    not_in_dataset = []

    for var in var_list:
        # Try exact match, then case-insensitive
        if var in gss.columns:
            col = var
        elif var.lower() in gss_columns_lower:
            col = gss_columns_lower[var.lower()]
        else:
            not_in_dataset.append(var)
            continue

        year_counts = {}
        for y in years:
            subset = gss[gss['year'] == y][col]
            non_missing = subset.notna().sum()
            # Also check if the column might be all-NaN strings like "NA", ".d", ".i", etc.
            # For safety, also count non-empty strings
            if subset.dtype == object:
                non_missing = ((subset.notna()) & (subset != '') &
                               (~subset.isin(['.d', '.i', '.n', '.s', '.r', '.y', '.z', 'NA', 'nan']))).sum()
            year_counts[y] = int(non_missing)

        present_years = [y for y in years if year_counts[y] > 0]
        missing_years = [y for y in years if year_counts[y] == 0]

        results.append({
            'variable': var,
            'column_used': col,
            'counts': year_counts,
            'present_years': present_years,
            'missing_years': missing_years,
            'in_all_3': len(present_years) == 3
        })

    return results, not_in_dataset


# --- Check demographic variables ---
print("\n" + "="*80)
print(f"DEMOGRAPHIC VARIABLES ANALYSIS")
print("="*80)

demo_results, demo_missing_from_data = check_vars(demo_vars, "Demographic")

demo_in_all = [r for r in demo_results if r['in_all_3']]
demo_not_all = [r for r in demo_results if not r['in_all_3']]

print(f"\nTotal demographic variables in list: {len(demo_vars)}")
print(f"Variables NOT found in GSS dataset at all: {len(demo_missing_from_data)}")
if demo_missing_from_data:
    for v in demo_missing_from_data:
        print(f"  - {v}")

print(f"\nVariables found in dataset: {len(demo_results)}")
print(f"  Present in ALL 3 years (2021, 2022, 2024): {len(demo_in_all)}")
print(f"  Missing from at least one year: {len(demo_not_all)}")

if demo_not_all:
    print(f"\n  Variables missing from some years:")
    for r in sorted(demo_not_all, key=lambda x: len(x['missing_years']), reverse=True):
        present_str = ', '.join(str(y) for y in r['present_years']) if r['present_years'] else 'none'
        missing_str = ', '.join(str(y) for y in r['missing_years'])
        counts_str = ', '.join(f"{y}:{r['counts'][y]}" for y in years)
        print(f"    {r['variable']:20s}  missing from: [{missing_str}]  counts: ({counts_str})")

print(f"\n  Variables present in all 3 years:")
for r in sorted(demo_in_all, key=lambda x: x['variable']):
    counts_str = ', '.join(f"{y}:{r['counts'][y]}" for y in years)
    print(f"    {r['variable']:20s}  counts: ({counts_str})")


# --- Check lifestyle variables ---
print("\n" + "="*80)
print(f"POLITICIZED LIFESTYLE VARIABLES ANALYSIS")
print("="*80)

life_results, life_missing_from_data = check_vars(life_vars, "Lifestyle")

life_in_all = [r for r in life_results if r['in_all_3']]
life_not_all = [r for r in life_results if not r['in_all_3']]

print(f"\nTotal lifestyle variables in list: {len(life_vars)}")
print(f"Variables NOT found in GSS dataset at all: {len(life_missing_from_data)}")
if life_missing_from_data:
    for v in life_missing_from_data:
        print(f"  - {v}")

print(f"\nVariables found in dataset: {len(life_results)}")
print(f"  Present in ALL 3 years (2021, 2022, 2024): {len(life_in_all)}")
print(f"  Missing from at least one year: {len(life_not_all)}")

if life_not_all:
    print(f"\n  Variables missing from some years:")
    for r in sorted(life_not_all, key=lambda x: len(x['missing_years']), reverse=True):
        present_str = ', '.join(str(y) for y in r['present_years']) if r['present_years'] else 'none'
        missing_str = ', '.join(str(y) for y in r['missing_years'])
        counts_str = ', '.join(f"{y}:{r['counts'][y]}" for y in years)
        print(f"    {r['variable']:20s}  missing from: [{missing_str}]  counts: ({counts_str})")

print(f"\n  Variables present in all 3 years:")
for r in sorted(life_in_all, key=lambda x: x['variable']):
    counts_str = ', '.join(f"{y}:{r['counts'][y]}" for y in years)
    print(f"    {r['variable']:20s}  counts: ({counts_str})")


# --- Save the final lists ---
demographic_vars_all_years = sorted([r['variable'] for r in demo_in_all])
lifestyle_vars_all_years = sorted([r['variable'] for r in life_in_all])

output = {
    'demographic_vars_all_years': demographic_vars_all_years,
    'lifestyle_vars_all_years': lifestyle_vars_all_years,
    'demographic_vars_missing_some_years': sorted([r['variable'] for r in demo_not_all]),
    'lifestyle_vars_missing_some_years': sorted([r['variable'] for r in life_not_all]),
    'demographic_vars_not_in_dataset': sorted(demo_missing_from_data),
    'lifestyle_vars_not_in_dataset': sorted(life_missing_from_data),
}

output_path = '/project/jevans/maxzhuyt/gss_vars_presence_by_year.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n{'='*80}")
print(f"Results saved to: {output_path}")
print(f"  demographic_vars_all_years: {len(demographic_vars_all_years)} variables")
print(f"  lifestyle_vars_all_years: {len(lifestyle_vars_all_years)} variables")
print(f"{'='*80}")
