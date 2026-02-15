import pandas as pd
import re

public_opinion_variables = [
    # ideological
    "partyid",

    # Government Spending & Priorities
    "natspac", "natenvir", "natheal", "natcity", "natcrime", "natdrug", 
    "nateduc", "natrace", "natarms", "nataid", "natfare", "natroad", 
    "natsoc", "natspacy", "natenviy", "nathealy", "natcityy", "natcrimy", 
    "natdrugy", "nateducy", "natracey", "natarmsy", "nataidy", "natfarey", 
    "natmass", "natpark", "natsci", "natenrgy", 

    # Institutional Confidence
    "conbus", "conclerg", "coneduc", "confed", "conjudge", "conlegis", 
    "conarmy", "confinan", 

    # Civil Liberties & Freedom of Speech
    "spkath", "colath", "libath", "spkcom", "colcom", "libcom", 
    "spkrac", "colrac", "librac", "spkhomo", "colhomo", "libhomo", 
    "spkmslm", "colmslm", "libmslm", 

    # Societal Issues, Laws & Policy
    "polviews", "cappun", "gunlaw", "courts", "grass", "eqwlth", 
    "prayer", "sexeduc", "pillok", "spanking", "letdie1", "letdie1y", 
    "uswary", "getahead",

    # Racial & Gender Equity / Public Policy
    "racopen", "helppoor", "helpnot", "helpsick", "helpblk", "racdif1", 
    "racdif2", "racdif3", "racdif4", "wlthwhts", "workwhts", "discaff", 
    "affrmact", "wrkwayup", "intlwhts", "fejobaff", "discaffm", "fehire", 
    "natchld", "marasian", "marwht", "discaffw",

    # Reproductive Rights (Public/Legal Policy)
    "abdefect", "abnomore", "abhlth", "abpoor", "abrape", "absingle", "abany"
]

def load_and_process_gss(file_list, target_vars):
    all_dfs = []
    for file_path in file_list:
        print(f"Loading {file_path}...")
        year_match = re.search(r'\d{4}', file_path)
        year_str = year_match.group() if year_match else "Unknown"
        df = pd.read_stata(file_path, convert_categoricals=False)
        if 'id' in df.columns:
            df['id'] = df['id'].astype(str) + '_' + year_str
        df['year_label'] = year_str
        cols_to_extract = ['id', 'year_label'] + [v for v in target_vars if v in df.columns]
        all_dfs.append(df[cols_to_extract])
    return pd.concat(all_dfs, ignore_index=True)

files = ['/Users/maxzhu/Downloads/GSS/GSS2021.dta', '/Users/maxzhu/Downloads/GSS/GSS2022.dta', '/Users/maxzhu/Downloads/GSS/GSS2024.dta']
df = load_and_process_gss(files, public_opinion_variables)

# Apply the filter mentioned in the notebook: df = df[df['partyid'].between(0, 6)]
if 'partyid' in df.columns:
    df = df[df['partyid'].between(0, 6)]

print("\nAnalysis of Variables (Valid >= 0):\n")
variable_cols = [col for col in df.columns if col not in ['id', 'year_label']]

for col in variable_cols:
    if pd.api.types.is_numeric_dtype(df[col]):
        valid_data = df[df[col] >= 0][col]
        num_valid = len(valid_data)
        
        print(f"Variable: {col}")
        print(f"  Valid Count (>= 0): {num_valid}")
        
        if num_valid > 0:
            dist = valid_data.value_counts().sort_index()
            print("  Distribution:")
            for val, count in dist.items():
                pct = (count / num_valid) * 100
                print(f"    {val:4.0f}: {count:6.0f} ({pct:5.1f}%)")
        print("-" * 30)
