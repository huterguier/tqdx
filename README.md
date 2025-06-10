<div align="center">
    <img src="https://github.com/huterguier/tqdx/blob/main/images/tqdx.png" width="200">
</div>

# tqdx
Adds `tqdm` progress bars to `jax.lax.scan` and `jax.lax.fori_loop`. Progress bars commonly used in Python, such as tqdm, are not compatible with JAX's jit-compiled functions due to restrictions on side effects like printing. `tqdx` addresses this limitation by using callbacks to update progress bars created on the host.

```python
import tqdx

...
carry, ys = tqdx.scan(f, init, xs)
```
```
Processing: 100%|████████████████████████████████████████████████████████| 50/50 [02:38<00:00,  3.20s/it]
```
## Features

- **Progress bars for JAX**: See the progress of your computations when using `jax.lax.scan` and `jax.lax.fori_loop`.
- **Works with `jax.jit`**: Progress bars show up even inside jit-compiled code.
- **Minimal syntax change**: Just replace your calls to `jax.lax.scan` and `jax.lax.fori_loop` with `tqdx.scan` and `tqdx.fori_loop`.
- **No extra dependencies**: Only requires JAX and tqdm.

## Usage

### Progress bar for scan

```python
import tqdx
import jax.numpy as jnp

def step(carry, x):
    return carry + x, carry + x

xs = jnp.arange(100)
carry_init = 0

carry, ys = tqdx.scan(step, carry_init, xs)
```

### Progress bar for fori_loop

```python
import tqdx

def body_fun(i, val):
    return val + i

result = tqdx.fori_loop(0, 100, body_fun, 0)
```

## Installation

```bash
pip install tqdx
```

