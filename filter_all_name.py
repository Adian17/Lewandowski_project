import pandas as pd
import os

def filter_situation_all(input_file: str):
    """Filter rows where 'situation' is 'all'."""
    # Detect correct number of columns by reading a small portion of the file
    df_sample = pd.read_csv(input_file, nrows=5)  # Read first 5 rows
    expected_columns = len(df_sample.columns)  # Get expected column count

    print(f"Expected column count: {expected_columns}")

    # Load full dataset while handling errors
    df = pd.read_csv(input_file, header=0, dtype=str, on_bad_lines="skip")  # Load all as string

    # Drop rows with missing 'situation' column values
    df = df.dropna(subset=['situation'])

    # Check column count
    if df.shape[1] != expected_columns:
        print(f"Warning: Dataset has {df.shape[1]} columns instead of {expected_columns}. Some rows may have been removed.")

    print(f"Rows before filtering: {len(df)}")

    # Ensure 'situation' column exists
    if 'situation' not in df.columns:
        print("Error: Column 'situation' not found in dataset.")
        return None

    # Filter by 'situation' == 'all'
    df_filtered = df[df['situation'] == 'all']
    print(f"Rows after filtering by 'situation': {len(df_filtered)}")

    # Save the first filtered dataset
    output_file_all = os.path.join(os.path.dirname(input_file), "filtered_" + os.path.basename(input_file))
    df_filtered.to_csv(output_file_all, index=False)

    print(f"Filtered (situation == 'all') file saved as: {output_file_all}")
    return output_file_all  # Return the filtered file path for further filtering

def filter_only_name(input_file: str):
    """Keep only the 'name' column from the filtered dataset."""
    # Load the already filtered file
    df = pd.read_csv(input_file, header=0, dtype=str, on_bad_lines="skip")

    # Check if 'name' column exists
    if 'name' not in df.columns:
        print("Error: Column 'name' not found in dataset.")
        return None

    # Keep only the 'name' column
    df_name_only = df[['name']]

    # Save the final filtered dataset
    output_file_name = os.path.join(os.path.dirname(input_file), "name_only_" + os.path.basename(input_file))
    df_name_only.to_csv(output_file_name, index=False)

    print(f"Filtered (name only) file saved as: {output_file_name}")
    return output_file_name

if __name__ == "__main__":
    input_path = r"C:\Users\souha\OneDrive\ドキュメント\Lewandowski (Datathon)\moneypuck downloaded - player - skaters.csv"
    
    # Step 1: Filter by 'situation' == 'all'
    filtered_all_file = filter_situation_all(input_path)
    
    # Step 2: Keep only 'name' column from the filtered file
    if filtered_all_file:
        filter_only_name(filtered_all_file)
