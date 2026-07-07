from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data():
    # 1. Load the Iris Domain data
    iris = load_iris()
    X = iris.data   # Features: Sepal & Petal dimensions
    y = iris.target # Classes: Setosa, Versicolor, Virginica
    target_names = iris.target_names

    # 2. Structural Integrity: Train-Test Split
    # Shuffling is applied automatically to remove order bias. We use an 80/20 split.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=True
    )

    # 3. The Gatekeeper Rule: Scaling
    # Using StandardScaler to achieve Mean = 0, Variance = 1
    scaler = StandardScaler()
    
    # Fit the scaler only on the training data, then transform both train and test sets
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, target_names