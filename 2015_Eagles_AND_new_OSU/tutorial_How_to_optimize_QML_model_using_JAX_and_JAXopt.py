import numpy as np
import pennylane as qml
import jax
from jax import numpy as jnp
import jaxopt

jax.config.update("jax_platform_name", "cpu")

n_wires = 5
# Correctly define data and targets once using jax.numpy
data = jnp.sin(jnp.mgrid[-2:2:0.2].reshape(n_wires, -1)) ** 3
targets = jnp.array([-0.2, 0.4, 0.35, 0.2])

dev = qml.device("default.qubit", wires=n_wires)

@qml.qnode(dev)
def circuit(data, weights):
    """Quantum circuit ansatz"""

    # data embedding
    for i in range(n_wires):
        # data[i] will be of shape (4,); we are
        # taking advantage of operation vectorization here
        qml.RY(data[i], wires=i)

    # trainable ansatz
    for i in range(n_wires):
        qml.RX(weights[i, 0], wires=i)
        qml.RY(weights[i, 1], wires=i)
        qml.RX(weights[i, 2], wires=i)
        qml.CNOT(wires=[i, (i + 1) % n_wires])

    # we use a sum of local Z's as an observable since a
    # local Z would only be affected by params on that qubit.
    return qml.expval(qml.sum(*[qml.PauliZ(i) for i in range(n_wires)]))

def my_model(data, weights, bias):
    return circuit(data, weights) + bias

######################################################################
# We will define a simple cost function that computes the overlap between model output and target
# data, and `just-in-time (JIT) compile <https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html>`__ it:
#

@jax.jit
def loss_fn(params, data, targets):
    predictions = my_model(data, params["weights"], params["bias"])
    loss = jnp.sum((targets - predictions) ** 2 / len(data))
    return loss

######################################################################
# Note that the model above is just an example for demonstration – there are important considerations
# that must be taken into account when performing QML research, including methods for data embedding,
# circuit architecture, and cost function, in order to build models that may have use. This is still
# an active area of research; see our `demonstrations <https://pennylane.ai/qml/demonstrations>`__ for
# details.
#

######################################################################
# Initialize your parameters
# --------------------------
#

######################################################################
# Now, we can generate our trainable parameters ``weights`` and ``bias`` that will be used to train
# our QML model.
#

weights = jnp.ones([n_wires, 3])
bias = jnp.array(0.)
params = {"weights": weights, "bias": bias}

######################################################################
# Plugging the trainable parameters, data, and target labels into our cost function, we can see the
# current loss as well as the parameter gradients:
#

print(loss_fn(params, data, targets))

print(jax.grad(loss_fn)(params, data, targets))

######################################################################
# Create the optimizer
# --------------------
#

######################################################################
# We can now use JAXopt to create a gradient descent optimizer, and train our circuit.
#
# To do so, we first create a function that returns the loss value *and* the gradient value during
# training; this allows us to track and print out the loss during training within JAXopt’s internal
# optimization loop.
#

def loss_and_grad(params, data, targets, print_training, i):
    loss_val, grad_val = jax.value_and_grad(loss_fn)(params, data, targets)

    def print_fn():
        jax.debug.print("Step: {i}  Loss: {loss_val}", i=i, loss_val=loss_val)

    # if print_training=True, print the loss every 5 steps
    jax.lax.cond((jnp.mod(i, 5) == 0) & print_training, print_fn, lambda: None)

    return loss_val, grad_val

######################################################################
# Note that we use a couple of JAX specific functions here:
#
# -  ``jax.lax.cond`` instead of a Python ``if`` statement
# -  ``jax.debug.print`` instead of a Python ``print`` function
#
# These JAX compatible functions are needed because JAXopt will automatically JIT compile the
# optimizer update step.
#

opt = jaxopt.GradientDescent(loss_and_grad, stepsize=0.3, value_and_grad=True)
opt_state = opt.init_state(params)

for i in range(100):
    params, opt_state = opt.update(params, opt_state, data, targets, True, i)


######################################################################
# Jitting the optimization loop
# -----------------------------
#

######################################################################
# In the above example, we JIT compiled our cost function ``loss_fn`` (and JAXopt automatically JIT compiled the `loss_and_grad` function behind the scenes). However, we can
# also JIT compile the entire optimization loop; this means that the for-loop around optimization is
# not happening in Python, but is compiled and executed natively. This avoids (potentially costly) data
# transfer between Python and our JIT compiled cost function with each update step.
#

@jax.jit
def optimization_jit(params, data, targets, print_training=False):
    opt = jaxopt.GradientDescent(loss_and_grad, stepsize=0.3, value_and_grad=True)
    opt_state = opt.init_state(params)

    def update(i, args):
        params, opt_state = opt.update(*args, i)
        return (params, opt_state, *args[2:])

    args = (params, opt_state, data, targets, print_training)
    (params, opt_state, _, _, _) = jax.lax.fori_loop(0, 100, update, args)

    return params

######################################################################
# Note that -- similar to above -- we use ``jax.lax.fori_loop`` and ``jax.lax.cond``, rather than a standard Python for loop
# and if statement, to allow the control flow to be JIT compatible.
#

params = {"weights": weights, "bias": bias}
optimization_jit(params, data, targets, print_training=True)

######################################################################
# Appendix: Timing the two approaches
# -----------------------------------
#
# We can time the two approaches (JIT compiling just the cost function, vs JIT compiling the entire
# optimization loop) to explore the differences in performance:
#

from timeit import repeat

def optimization(params, data, targets):
    opt = jaxopt.GradientDescent(loss_and_grad, stepsize=0.3, value_and_grad=True)
    opt_state = opt.init_state(params)

    for i in range(100):
        params, opt_state = opt.update(params, opt_state, data, targets, False, i)

    return params

reps = 5
num = 2

times = repeat("optimization(params, data, targets)", globals=globals(), number=num, repeat=reps)
result = min(times) / num

print(f"Jitting just the cost (best of {reps}): {result} sec per loop")

times = repeat("optimization_jit(params, data, targets)", globals=globals(), number=num, repeat=reps)
result = min(times) / num

print(f"Jitting the entire optimization (best of {reps}): {result} sec per loop")

######################################################################
# In this example, JIT compiling the entire optimization loop
# is significantly more performant.
#
# About the authors
# -----------------
#
