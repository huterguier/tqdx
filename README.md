<div align="center">
    <img src="https://github.com/huterguier/tqdx/blob/main/images/tqdx.gif" width="220">
</div>

# tqdx
Adds `tqdm` progress bars to `jax.lax.scan` and `jax.lax.fori_loop`. Progress bars commonly used in Python, such as tqdm, are not compatible with JAX's jit-compiled functions due to restrictions on side effects like printing. `tqdx` addresses this limitation by using callbacks to update progress bars created on the host.

```python
import tqdx

...
carry, ys = tqdx.scan(f, init, xs)
```
```
Processing: 100%|███████████████████████████████████████████| 50/50 [02:38<00:00,  3.20s/it]
```
## Features

- **Progress bars for JAX**: See the progress of your computations when using `jax.lax.scan` and `jax.lax.fori_loop`.
- **Works with `jax.jit`**: Progress bars show up even inside jit-compiled code.
- **Minimal syntax change**: Just replace your calls to `jax.lax.scan` and `jax.lax.fori_loop` with `tqdx.scan` and `tqdx.fori_loop`.
- **No extra dependencies**: Only requires JAX and tqdm.

## Usage
The following example demonstrates how to use `tqdx` with `jax.lax.scan` and `jax.lax.fori_loop`. You can arbitrarily nest these functions, and the progress bars will still work correctly.
```python
import jax
import tqdx
from time import sleep

def step(carry, x):
    def body_fun(i, val):
        jax.debug.callback(lambda: sleep(0.5))
        return val + i
    jax.debug.callback(lambda: sleep(0.5))
    carry = tqdx.fori_loop(0, 10, body_fun, carry)
    return carry, x + 1

def f(xs):
    return tqdx.scan(step, 0, xs)


xs = jax.numpy.arange(10)
result, _ = jax.jit(f)(xs)
```

## Installation

```bash
pip install tqdx
```
