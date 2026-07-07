from src.data_preprocessing import load_and_preprocess_data
from src.model_training import train_and_evaluate_model

def main():
    print("--- PROJECT 2: DATA CLASSIFICATION USING AI ---")
    print("Initializing IPO Framework (Input -> Process -> Output)...\n")

    # INPUT & PREPROCESSING
    print("[1/3] Loading Iris dataset and applying StandardScaler...")
    X_train, X_test, y_train, y_test, target_names = load_and_preprocess_data()
    print(f"Data split successful: 80% Training ({len(X_train)} samples), 20% Testing ({len(X_test)} samples).\n")

    # PROCESS & OUTPUT
    print("[2/3] Instantiating KNN Algorithm (K=5) and fitting the model...")
    model, cm, f1, report = train_and_evaluate_model(
        X_train, X_test, y_train, y_test, target_names
    )

    # RESULTS
    print("[3/3] Model Evaluation Complete.\n")
    print("=== CONFUSION MATRIX ===")
    print("(Format: True labels on the left, Predicted labels on the top)")
    print(cm)
    print("\n=== CLASSIFICATION REPORT (Precision, Recall, F1 Score) ===")
    print(report)
    print(f"Weighted F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()