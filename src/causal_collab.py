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

def sigmoid(x):
    return jnp.exp(x) / (1 + jnp.exp(x))

def generate_binary_conversion(key):
    key, subkey1 = jax.random.split(key)
    # Generate noise for true_conversion
    noise = jax.random.normal(subkey1, shape=(n_samples, 1))
    true_conversion = 0.3 * u + 0.7 * v + noise
    
    key, subkey2 = jax.random.split(key)
    Y = jax.random.bernoulli(subkey2, sigmoid(true_conversion))
    return Y

n_sets = 1000

Y_n = jax.vmap(generate_binary_conversion, in_axes=0)(jax.random.split(key, n_sets))

print(Y_n.shape)