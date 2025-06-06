import jax
from typing import Callable, TypeVar
from .callbacks import init_pbar, update_pbar, close_pbar


Carry = TypeVar('Carry')
X = TypeVar('X')
Y = TypeVar('Y')


def scan(f: Callable[[Carry, X], tuple[Carry, Y]],
         init: Carry,
         xs: X | None = None,
         length: int | None = None,
         reverse: bool = False,
         unroll: int | bool = 1,
         _split_transpose: bool = False) -> tuple[Carry, Y]: 
    """A wrapper around jax.lax.scan that adds a progress bar."""
    if length is None:
        length = len(xs)
    id = init_pbar(length)

    def wrapped_f(carry, x):
        out = f(carry, x)
        update_pbar(id)
        return out

    out = jax.lax.scan(
        wrapped_f,
        init,
        xs,
        length=length,
        reverse=reverse,
        unroll=unroll,
        _split_transpose=_split_transpose
    )
    close_pbar(id)
    return out

def fori_loop(lower, upper, body_fun, init_val,
              *, unroll: int | bool | None = None):
    """A wrapper around jax.lax.fori_loop that adds a progress bar."""
    length = upper - lower
    id = init_pbar(length)

    def wrapped_body_fun(i, val):
        out = body_fun(i, val)
        update_pbar(id)
        return out

    out = jax.lax.fori_loop(
        lower,
        upper,
        wrapped_body_fun,
        init_val,
        unroll=unroll
    )
    close_pbar(id)
    return out


def tqdx(f):
    """A decorator that adds a progress to `jax.lax.scan` or `jax.lax.fori_loop`."""
    if f is jax.lax.scan:
        return scan
    elif f is jax.lax.fori_loop:
        return fori_loop
    else:
        raise ValueError("Function must be jax.lax.scan or jax.lax.fori_loop")
