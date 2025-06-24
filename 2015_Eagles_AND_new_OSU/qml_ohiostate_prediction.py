import pandas as pd
import numpy as np
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pennylane

# Optional matplotlib import with error handling
try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    print("WARNING: matplotlib.pyplot not found. Plotting will be skipped.")
    PLOTTING_AVAILABLE = False

# Print Pennylane version for debugging
print(f"Pennylane version: {pennylane.__version__}")

# --- 1. Configuration ---
INPUT_FILENAME = 'qml_ohiostate_enhanced.csv'
NUM_QUBITS = 10  # Matches 10 features
NUM_LAYERS = 6   # For data re-uploading
DEV = qml.device('default.qubit', wires=NUM_QUBITS)
SEED = 42
np.random.seed(SEED)

print("--- Starting QML Prediction for Ohio State Play Type ---")

# --- 2. Data Loading and Preprocessing ---
try:
    df = pd.read_csv(INPUT_FILENAME)
    print(f"Loaded '{INPUT_FILENAME}' with {len(df)} records.")
except FileNotFoundError:
    print(f"ERROR: '{INPUT_FILENAME}' not found. Ensure the file exists in the working directory.")
    exit()

# Filter for pass and run plays only
valid_plays = df['play_type'].isin(['Rush', 'Pass Reception', 'Pass Incompletion', 'Passing Touchdown', 'Rushing Touchdown'])
df = df[valid_plays].copy()
print(f"Filtered to {len(df)} pass/run plays.")

# Create binary target: 1 for pass, 0 for run
df['play_type_numeric'] = df['play_type'].apply(lambda x: 1 if x in ['Pass Reception', 'Pass Incompletion', 'Passing Touchdown'] else 0)

# Compute ScoreDiff if not present
if 'ScoreDiff' not in df.columns:
    df['ScoreDiff'] = df['offense_score'] - df['defense_score']

# Features selection
features = [
    'period', 'down', 'yards_to_goal', 'distance', 'ScoreDiff',
    'ppa', 'drive_time_minutes_elapsed', 'drive_time_seconds_elapsed',
    'offense_timeouts', 'defense_timeouts'
]
X = df[features].copy()

# Handle missing values
X = X.fillna(X.mean(numeric_only=True))
print(f"Missing values filled with column means.")

# Convert to numpy array
X = X.values
y = df['play_type_numeric'].values

# Verify feature count
print(f"Number of features: {X.shape[1]}")
if X.shape[1] != NUM_QUBITS:
    print(f"ERROR: Expected {NUM_QUBITS} features, got {X.shape[1]}.")
    exit()

# Scale features to [0, pi] for angle encoding
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X)
print(f"X_scaled shape: {X_scaled.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")

# Convert labels to {-1.0, 1.0} floats for quantum circuit
y_train_pm = np.array([1.0 if label == 1 else -1.0 for label in y_train])
y_test_pm = np.array([1.0 if label == 1 else -1.0 for label in y_test])

# --- 3. Variational Quantum Classifier with Data Re-uploading ---
@qml.qnode(DEV)
def circuit(weights, x):
    for l in range(NUM_LAYERS):
        qml.AngleEmbedding(x, wires=range(NUM_QUBITS), rotation='Y')
        for i in range(NUM_QUBITS):
            qml.RX(weights[l, i, 0], wires=i)
            qml.RY(weights[l, i, 1], wires=i)
            qml.RZ(weights[l, i, 2], wires=i)
        for i in range(NUM_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
    return qml.expval(qml.PauliZ(0))

def variational_classifier(weights, bias, x):
    return circuit(weights, x) + bias

def square_loss(labels, predictions):
    return np.mean((labels - predictions) ** 2)

def accuracy(labels, predictions):
    preds_pm = np.sign(predictions)
    acc = np.mean(preds_pm == labels)
    return acc

def cost(weights, bias, X, y):
    predictions = np.array([variational_classifier(weights, bias, x) for x in X])
    return square_loss(y, predictions)

# Initialize parameters
weights_shape = (NUM_LAYERS, NUM_QUBITS, 3)
weights = np.random.uniform(0, 2 * np.pi, weights_shape, requires_grad=True)
bias = np.array(0.0, requires_grad=True)
print(f"Initialized weights with shape: {weights.shape}")

# Training loop
print("\n--- Training Variational Quantum Classifier ---")
opt = AdamOptimizer(0.1)
batchsize = 32
num_steps = 100
train_acc_history, test_acc_history = [], []
for step in range(num_steps):
    indices = np.random.choice(len(X_train), batchsize)
    X_batch = X_train[indices]
    y_batch = y_train_pm[indices]

    # Optimize weights and bias
    weights, bias = opt.step(cost, weights, bias, X=X_batch, y=y_batch)

    if step % 10 == 0 or step == num_steps - 1:
        train_preds = np.array([variational_classifier(weights, bias, x) for x in X_train])
        test_preds = np.array([variational_classifier(weights, bias, x) for x in X_test])
        train_acc = accuracy(y_train_pm, train_preds)
        test_acc = accuracy(y_test_pm, test_preds)
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
        print(f"Step {step:3d}: Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")

# Final evaluation
vqc_test_preds = np.sign([variational_classifier(weights, bias, x) for x in X_test])
vqc_accuracy = accuracy_score(y_test, (vqc_test_preds + 1) / 2)  # Convert {-1,1} to {0,1}
print(f"\nVQC Final Test Accuracy: {vqc_accuracy:.4f}")

# --- 4. Quantum Kernel Method ---
@qml.qnode(DEV)
def kernel_circuit(x1, x2):
    qml.AngleEmbedding(x1, wires=range(NUM_QUBITS), rotation='Y')
    qml.adjoint(qml.AngleEmbedding)(x2, wires=range(NUM_QUBITS), rotation='Y')
    return qml.probs(wires=range(NUM_QUBITS))

def quantum_kernel(x1, x2):
    return kernel_circuit(x1, x2)[0].item()

def compute_kernel_matrix(X1, X2):
    n1, n2 = len(X1), len(X2)
    kernel_matrix = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            kernel_matrix[i, j] = quantum_kernel(X1[i], X2[j])
    return kernel_matrix

print("\n--- Training Quantum Kernel SVM ---")
try:
    print("Computing train kernel matrix...")
    kernel_matrix_train = compute_kernel_matrix(X_train, X_train)
    
    svm = SVC(kernel='precomputed')
    svm.fit(kernel_matrix_train, y_train)
    
    print("Computing test kernel matrix...")
    kernel_matrix_test = compute_kernel_matrix(X_test, X_train)
    
    kernel_preds = svm.predict(kernel_matrix_test)
    kernel_accuracy = accuracy_score(y_test, kernel_preds)
    print(f"Quantum Kernel SVM Test Accuracy: {kernel_accuracy:.4f}")
except Exception as e:
    print(f"ERROR in Quantum Kernel SVM: {e}")
    kernel_accuracy = None

# --- 5. Classical Baseline (Random Forest) ---
print("\n--- Training Classical Random Forest ---")
rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_preds)
print(f"Random Forest Test Accuracy: {rf_accuracy:.4f}")

# --- 6. Plotting Results (Optional) ---
if PLOTTING_AVAILABLE:
    plt.figure(figsize=(10, 6))
    steps = list(range(0, num_steps, 10))
    if num_steps - 1 not in steps:
        steps.append(num_steps - 1)
    plt.plot(steps, train_acc_history, marker='o', label='VQC Train Accuracy')
    plt.plot(steps, test_acc_history, marker='o', label='VQC Test Accuracy')
    plt.axhline(y=rf_accuracy, color='r', linestyle='--', label=f'Random Forest Accuracy ({rf_accuracy:.4f})')
    if kernel_accuracy is not None:
        plt.axhline(y=kernel_accuracy, color='g', linestyle='--', label=f'Quantum Kernel SVM Accuracy ({kernel_accuracy:.4f})')
    plt.xlabel('Training Step')
    plt.ylabel('Accuracy')
    plt.title('Ohio State Play Type Prediction Performance')
    plt.legend()
    plt.grid(True)
    plt.ylim(0.4, 1.0)
    plt.savefig("ohiostate_model_performance.png")
    print("\nPlot saved to ohiostate_model_performance.png")
else:
    print("\nSkipping plot generation due to missing matplotlib.")