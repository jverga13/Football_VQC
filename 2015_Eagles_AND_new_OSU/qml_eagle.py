import pandas as pd
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp  # Use PennyLane's wrapped numpy
import jax
from jax import numpy as jnp
import jaxopt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# --- 1. Load and Prepare the Data ---
print("--- Loading and Preparing Data ---")
DATA_FILENAME = 'qml_real_data_eagles_2015.csv'
try:
    df = pd.read_csv(DATA_FILENAME)
    print(f"Successfully loaded '{DATA_FILENAME}'.")
except FileNotFoundError:
    print(f"FATAL ERROR: The data file '{DATA_FILENAME}' was not found.")
    print("Please ensure the data preparation script has been run successfully.")
    exit()

# Separate features (X) from the target (y)
X = df.drop('play_type_numeric', axis=1).values
y = df['play_type_numeric'].values

# The number of features determines the number of qubits we need
n_features = X.shape[1]
n_wires = n_features
print(f"Dataset has {n_features} features, using {n_wires} qubits.")

# Scale features to the range [0, pi] for angle encoding in the quantum circuit
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X)

# Map target labels from {0, 1} to {-1, 1} to match the quantum model's output
# Run (0) -> -1; Pass (1) -> 1
y_mapped = np.array([1 if label == 1 else -1 for label in y])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_mapped, test_size=0.25, random_state=42
)

# Convert data to JAX arrays for use with JAX optimizers
X_train_jnp = jnp.array(X_train)
y_train_jnp = jnp.array(y_train)
X_test_jnp = jnp.array(X_test)
y_test_jnp = jnp.array(y_test)

print(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")

# --- 2. Define the Quantum Circuit and Model ---
print("\n--- Setting up Quantum Model ---")
# Define the quantum device (simulator)
dev = qml.device("default.qubit", wires=n_wires)

@qml.qnode(dev, interface="jax")
def circuit(weights, features):
    """Quantum circuit for classification."""
    # Encode features using angle embedding (RY gates)
    qml.AngleEmbedding(features, wires=range(n_wires))
    
    # Trainable ansatz (variational circuit) with weights
    qml.BasicEntanglerLayers(weights, wires=range(n_wires))
    
    # Return the expectation value of the Pauli-Z operator on the first qubit.
    # This will be a value between -1 and 1.
    return qml.expval(qml.PauliZ(0))

def quantum_model(weights, features):
    """A simple wrapper for the circuit, vectorizing over the features."""
    # We use jax.vmap to run the circuit for a whole batch of features at once.
    # It passes 'weights' as the first argument and each sample of 'features' as the second.
    return jax.vmap(circuit, in_axes=(None, 0))(weights, features)

# --- 3. Define Loss Function and Initialize Parameters ---

# The square loss function measures the squared difference between predictions and targets.
def loss_fn(params, features, targets):
    """Square loss function."""
    predictions = quantum_model(params['weights'], features)
    loss = jnp.mean((predictions - targets) ** 2)
    return loss

# Initialize random weights for the trainable layers in the circuit
# The shape matches the BasicEntanglerLayers expectation (num_layers, num_qubits)
num_layers = 4
key = jax.random.PRNGKey(0)  # JAX random key
weights_shape = (num_layers, n_wires)
initial_weights = jax.random.uniform(key, shape=weights_shape) * 2 * np.pi
params = {'weights': initial_weights}

print("Initial loss:", loss_fn(params, X_train_jnp, y_train_jnp))

# --- 4. Train the Model ---
print("\n--- Starting Model Training ---")

# Set up the optimizer
optimizer = jaxopt.GradientDescent(fun=loss_fn, maxiter=200, stepsize=0.1)

# Set up batching for training
batch_size = 32
num_batches = len(X_train) // batch_size

# JIT-compile the update step for performance
@jax.jit
def update_step(params, opt_state, X_batch, y_batch):
    params, opt_state = optimizer.update(params, opt_state, features=X_batch, targets=y_batch)
    return params, opt_state

# Initialize the optimizer state
opt_state = optimizer.init_state(params, features=X_train_jnp, targets=y_train_jnp)

# Training loop
for i in range(optimizer.maxiter):
    # Create a random permutation of training data for each epoch
    perm = np.random.permutation(len(X_train))
    X_perm = X_train_jnp[perm]
    y_perm = y_train_jnp[perm]

    for j in range(num_batches):
        # Get the current batch
        X_batch = X_perm[j*batch_size : (j+1)*batch_size]
        y_batch = y_perm[j*batch_size : (j+1)*batch_size]
        
        # Perform one optimization step
        params, opt_state = update_step(params, opt_state, X_batch, y_batch)

    # Compute and print the loss at intervals to monitor training
    if (i + 1) % 10 == 0:
        train_loss = loss_fn(params, X_train_jnp, y_train_jnp)
        test_loss = loss_fn(params, X_test_jnp, y_test_jnp)
        print(f"Step {i+1:3d} | Train Loss: {train_loss:.7f} | Test Loss: {test_loss:.7f}")

# --- 5. Evaluate the Trained Model ---
print("\n--- Evaluating Model Performance ---")

# Get predictions on the test set using the final trained parameters
test_predictions = quantum_model(params['weights'], X_test_jnp)

# Convert model output (-1 to 1) to class labels (0 or 1)
# Prediction > 0 -> Pass (1), Prediction <= 0 -> Run (0)
predicted_labels = [1 if p > 0 else -1 for p in test_predictions]

# Calculate the accuracy
correct_predictions = np.sum(predicted_labels == y_test)
accuracy = correct_predictions / len(y_test)

print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")

# Show some example predictions
print("\nExample Predictions (Actual vs. Predicted):")
for i in range(10):
    actual = "Pass" if y_test[i] == 1 else "Run"
    predicted = "Pass" if predicted_labels[i] == 1 else "Run"
    print(f"  - Play {i+1}: Actual='{actual}', Predicted='{predicted}'")