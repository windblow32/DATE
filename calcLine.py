import pandas as pd
import argparse

def count_csv_rows(file_path):
    """
    Count the number of rows in a CSV file (excluding header).
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        int: Number of rows (excluding header)
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Get the number of rows (excluding header)
        num_rows = len(df)
        
        return num_rows
        
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Count rows in a CSV file (excluding header)')
    parser.add_argument('file_path', type=str, help='Path to the CSV file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Count rows
    row_count = count_csv_rows(args.file_path)
    
    if row_count is not None:
        print(f"Number of rows (excluding header): {row_count}")

if __name__ == "__main__":
    main()