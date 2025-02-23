import pandas as pd
import os

def filter_situation_all(input_file: str):
    """Filter rows where 'situation' is 'all'."""
    df_sample = pd.read_csv(input_file, nrows=5)
    expected_columns = len(df_sample.columns)

    print(f"Expected column count: {expected_columns}")

    df = pd.read_csv(input_file, header=0, dtype=str, on_bad_lines="skip")

    df = df.dropna(subset=['situation'])

    if df.shape[1] != expected_columns:
        print(f"Warning: Dataset has {df.shape[1]} columns instead of {expected_columns}. Some rows may have been removed.")

    print(f"Rows before filtering: {len(df)}")

    if 'situation' not in df.columns:
        print("Error: Column 'situation' not found in dataset.")
        return None

    df_filtered = df[df['situation'] == 'all']
    print(f"Rows after filtering by 'situation': {len(df_filtered)}")

    output_file_all = os.path.join(os.path.dirname(input_file), "filtered_" + os.path.basename(input_file))
    df_filtered.to_csv(output_file_all, index=False)

    print(f"Filtered (situation == 'all') file saved as: {output_file_all}")
    return output_file_all

def filter_only_name(input_file: str):
    """Keep only the 'name' column from the filtered dataset."""
    df = pd.read_csv(input_file, header=0, dtype=str, on_bad_lines="skip")

    if 'name' not in df.columns:
        print("Error: Column 'name' not found in dataset.")
        return None

    df_name_only = df[['name']]

    output_file_name = os.path.join(os.path.dirname(input_file), "name_only_" + os.path.basename(input_file))
    df_name_only.to_csv(output_file_name, index=False)

    print(f"Filtered (name only) file saved as: {output_file_name}")
    return output_file_name

if __name__ == "__main__":
    input_path = r"C:\Users\souha\OneDrive\ドキュメント\Lewandowski (Datathon)\moneypuck downloaded - player - skaters.csv"
    
    filtered_all_file = filter_situation_all(input_path)
    
    if filtered_all_file:
        filter_only_name(filtered_all_file)
