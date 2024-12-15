import jax
import jax.numpy as jnp

n_samples = 1000

# Generate actual binary outcome
key = jax.random.PRNGKey(0)

key, subkey = jax.random.split(key)

u = jax.random.normal(subkey, shape=(n_samples, 1))
v = jax.random.normal(subkey, shape=(n_samples, 1))

#Simulate selection mechanism (Collider-bias)
S = jnp.where(u + v > 1, 1, 0)

key, subkey = jax.random.split(key)
true_conversion = 0.3 * u + 0.7 * v + jax.random.normal(subkey, shape=(n_samples, 1))

def sigmoid(x):
    return jnp.exp(x) / (1 + jnp.exp(x))

# Generate actual binary outcome
key, subkey = jax.random.split(key)
Y = jax.random.bernoulli(subkey, sigmoid(true_conversion))
