import pandas as pd

def calculate_error_rate(file1_path, file2_path):
    """
    Calculate the error rate if all predictions were 0 based on the last column of two CSV files.
    
    Args:
        file1_path (str): Path to the first CSV file
        file2_path (str): Path to the second CSV file
    
    Returns:
        dict: Dictionary containing counts and error rate information
    """
    try:
        # Read both CSV files
        df1 = pd.read_csv(file1_path, header=None)
        df2 = pd.read_csv(file2_path, header=None)
        
        # Get the last column from each dataframe
        last_col1 = df1.iloc[:, -1]
        last_col2 = df2.iloc[:, -1]
        
        # Combine both columns
        combined = pd.concat([last_col1, last_col2])
        
        # Count 0s and 1s
        count_0 = (combined == "0").sum()
        count_1 = (combined == "1").sum()
        total = len(combined)
        
        # Calculate error rate if all predictions are 0
        # Error occurs when actual value is 1
        error_rate = count_1 / total if total > 0 else 0
        
        return {
            'total_samples': int(total),
            'count_0': int(count_0),
            'count_1': int(count_1),
            'error_rate': error_rate,
            'acc': 1-error_rate,
            'error_rate_percentage': f"{error_rate * 100:.2f}%",
            'acc_percentage': f"{(1-error_rate) * 100:.2f}%"
        }
        
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        return None

def main():
    # Example usage
    file1 = "dataset/clf_num/electricity.csv"
    # file2 = "dataset/generatedData/EPIC/electricity_samples2.csv"
    file2 = "dataset/generatedData/EPIC/test.csv"
    
    result = calculate_error_rate(file1, file2)
    
    if result:
        print("Error Rate Analysis")
        print("-------------------")
        print(f"Total samples: {result['total_samples']}")
        print(f"Number of 0s: {result['count_0']}")
        print(f"Number of 1s: {result['count_1']}")
        print(f"Error rate if all predictions are 0: {result['error_rate_percentage']}")

if __name__ == "__main__":
    main()