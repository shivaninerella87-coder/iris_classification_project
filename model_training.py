from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score

def train_and_evaluate_model(X_train, X_test, y_train, y_test, target_names):
    # 1. INSTANTIATE (Build the frame)
    # Using K=5 as our nearest neighbors baseline
    model = KNeighborsClassifier(n_neighbors=5)

    # 2. FIT (Memorize the map)
    model.fit(X_train, y_train)

    # 3. PREDICT (Apply logic)
    predictions = model.predict(X_test)

    # 4. OUTPUT VALIDATION
    # The Diagnostic Tool: Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    
    # Calculate F1 Score (The Harmonic Mean)
    f1 = f1_score(y_test, predictions, average='weighted')
    
    # Full classification report (includes Precision and Recall)
    report = classification_report(y_test, predictions, target_names=target_names)

    return model, cm, f1, report