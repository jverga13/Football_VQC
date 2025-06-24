import pandas as pd
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import jax
import jax.numpy as jnp
import optax
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
    exit()

# Check class balance
print("Class distribution:", np.bincount(df['play_type_numeric']))

X = df.drop('play_type_numeric', axis=1).values
y = df['play_type_numeric'].values
n_features = X.shape[1]
n_wires = n_features
print(f"Dataset has {n_features} features, using {n_wires} qubits.")

scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X)
y_mapped = np.array([1 if label == 1 else -1 for label in y])

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_mapped, test_size=0.25, random_state=42
)
X_train_jnp = jnp.array(X_train)
y_train_jnp = jnp.array(y_train)
X_test_jnp = jnp.array(X_test)
y_test_jnp = jnp.array(y_test)
print(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")

# --- 2. Define the Quantum Circuit and Model ---
print("\n--- Setting up Quantum Model ---")
dev = qml.device("default.qubit", wires=n_wires)
# dev = qml.device("lightning.qubit", wires=n_wires)  # Revert once confirmed working

@qml.qnode(dev, interface="jax")
def circuit(weights, features):
    # Use AngleEmbedding to apply RY rotations to all wires
    qml.AngleEmbedding(features, wires=range(n_wires), rotation='Y')
    qml.StronglyEntanglingLayers(weights, wires=range(n_wires))
    return qml.expval(qml.PauliZ(0))

def quantum_model(weights, features):
    # Vectorize circuit execution over batch dimension
    return jax.vmap(circuit, in_axes=(None, 0))(weights, features)

# --- 3. Define Loss Function and Initialize Parameters ---
def loss_fn(params, features, targets):
    predictions = quantum_model(params['weights'], features)
    loss = jnp.mean((predictions - targets) ** 2)
    l2_reg = 0.01 * jnp.sum(params['weights'] ** 2)
    return loss + l2_reg

num_layers = 6
key = jax.random.PRNGKey(0)
weights_shape = (num_layers, n_wires, 3)
initial_weights = jax.random.uniform(key, shape=weights_shape) * 2 * np.pi
params = {'weights': initial_weights}
print("Initial loss:", float(loss_fn(params, X_train_jnp, y_train_jnp)))

# --- 4. Train the Model ---
print("\n--- Starting Model Training ---")
optimizer = optax.adam(learning_rate=0.1)
opt_state = optimizer.init(params)
batch_size = 32
num_batches = len(X_train) // batch_size

@jax.jit
def update_step(params, opt_state, X_batch, y_batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, X_batch, y_batch)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

for i in range(500):
    perm = np.random.permutation(len(X_train))
    X_perm = X_train_jnp[perm]
    y_perm = y_train_jnp[perm]
    for j in range(num_batches):
        X_batch = X_perm[j*batch_size : (j+1)*batch_size]
        y_batch = y_perm[j*batch_size : (j+1)*batch_size]
        params, opt_state, _ = update_step(params, opt_state, X_batch, y_batch)
    if (i + 1) % 10 == 0:
        train_loss = loss_fn(params, X_train_jnp, y_train_jnp)
        test_loss = loss_fn(params, X_test_jnp, y_test_jnp)
        test_preds = quantum_model(params['weights'], X_test_jnp)
        test_labels = [1 if p > 0 else -1 for p in test_preds]
        test_acc = np.sum(np.array(test_labels) == y_test) / len(y_test)
        print(f"Step {i+1:3d} | Train Loss: {train_loss:.7f} | Test Loss: {test_loss:.7f} | Test Acc: {test_acc*100:.2f}%")

# --- 5. Evaluate the trained Model ---
print("\n--- Evaluating Model Performance ---")
test_predictions = quantum_model(params['weights'], X_test_jnp)
predicted_labels = [1 if p > 0 else -1 for p in test_predictions]
correct_predictions = np.sum(np.array(predicted_labels) == y_test)
accuracy = correct_predictions / len(y_test)
print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")

print("\nExample Predictions (Actual vs. Predicted):")
for i in range(10):
    actual = "Pass" if y_test[i] == 1 else "Run"
    predicted = "Pass" if predicted_labels[i] == 1 else "Run"
    print(f"  - Play {i+1}: Actual='{actual}', Predicted='{predicted}'")