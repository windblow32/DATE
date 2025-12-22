from tkinter import N
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def preprocess_data(df):
    """Preprocess the data by handling categorical variables"""
    # Make a copy to avoid modifying original data
    df_processed = df.copy()
    
    # Convert categorical columns to numeric using LabelEncoder
    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
    
    return df_processed, label_encoders

def main():
    datasetName = "MagicTelescope"
    # Read the main dataset
    try:
        df_main = pd.read_csv('dataset/clf_num/'+datasetName+'.csv')
        print(f"Main dataset shape: {df_main.shape}")
    except FileNotFoundError:
        print("Error: Could not find 'dataset/clf_num/bank-marketing.csv'")
        return

    # Read additional dataset
    additional_path = 'dataset/generatedData/QAgenera/'+datasetName+'GT.csv'  # Change this to your actual path
    
    # Preprocess the main dataset
    df_main_processed, main_encoders = preprocess_data(df_main)
    
    # Split into train:validation:test = 3:1:1 (total 5 parts)
    # This means 60% train, 20% validation, 20% test
    train_val_df, test_df = train_test_split(df_main_processed, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)  # 0.25 of 80% = 20%
    
    print(f"Train set shape: {train_df.shape}")
    print(f"Validation set shape: {val_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    

    try:
        # Read additional dataset without headers, using main dataset columns
        df_additional = pd.read_csv(additional_path)
        df_additional.columns = df_main.columns  # Use main dataset column names
        print(f"Additional dataset shape: {df_additional.shape}")
        
        # Preprocess additional dataset using the same encoders
        df_additional_processed = df_additional.copy()
        for col, le in main_encoders.items():
            if col in df_additional_processed.columns:
                # Handle unseen labels by mapping them to a new category
                unique_vals = set(df_additional_processed[col].astype(str).unique())
                known_vals = set(le.classes_)
                unseen_vals = unique_vals - known_vals
                
                if unseen_vals:
                    # Add unseen labels to the encoder
                    new_labels = list(le.classes_) + list(unseen_vals)
                    le.classes_ = np.array(new_labels)
                
                df_additional_processed[col] = le.transform(df_additional_processed[col].astype(str))
        
        # Combine with training data
        combined_train_df = pd.concat([train_df, df_additional_processed], ignore_index=True)
        print(f"Combined training set shape: {combined_train_df.shape}")
        
        # Remove rows with NaN values in target
        combined_train_df = combined_train_df.dropna(subset=[combined_train_df.columns[-1]])
        print(f"Combined training set shape after removing NaN targets: {combined_train_df.shape}")
        
    except FileNotFoundError:
        print(f"Warning: Could not find additional dataset at '{additional_path}'")
        print("Proceeding with original training data only...")
        combined_train_df = train_df
    
    # Separate features and target
    # Assuming the last column is the target
    X_train = combined_train_df.iloc[:, :-1]
    y_train = combined_train_df.iloc[:, -1]
    
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]
    
    print(f"Training features shape: {X_train.shape}")
    print(f"Training target shape: {y_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    print(f"Test target shape: {y_test.shape}")
    
    # Fit Decision Tree model
    print("\nTraining Decision Tree Classifier model...")
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = dt_model.predict(X_test)
    
    # Calculate accuracy and 1-accuracy
    accuracy = accuracy_score(y_test, y_pred)
    one_minus_accuracy = 1 - accuracy
    
    print(f"\nResults:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"1 - Accuracy: {one_minus_accuracy:.4f}")
    
    # Additional model information
    print(f"\nModel details:")
    print(f"Tree depth: {dt_model.get_depth()}")
    print(f"Number of leaves: {dt_model.get_n_leaves()}")

if __name__ == "__main__":
    main()