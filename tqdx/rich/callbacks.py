from functools import partial
from itertools import count

import jax
import jax.experimental
import tqdm
import tqdm.rich as rich

pbars: dict[str, tqdm.std.tqdm] = {}
pbar_ids: count = count()


def init_pbar(**kwargs) -> int:
    """Initialize a progress bar with the given length."""

    def callback(**kwargs):
        pbar = rich.tqdm(**kwargs)
        id = next(pbar_ids)
        pbars[str(id)] = pbar
        return id

    id = jax.experimental.io_callback(
        partial(callback, **kwargs),
        result_shape_dtypes=jax.ShapeDtypeStruct((), jax.numpy.int32),
    )
    return id


def update_pbar(id: int):
    """Update the progress bar with the given id."""

    def callback(id):
        id = int(id)
        if str(id) in pbars:
            pbars[str(id)].update(1)
        return id

    id = jax.experimental.io_callback(
        callback, result_shape_dtypes=jax.ShapeDtypeStruct((), jax.numpy.int32), id=id
    )
    return id


def close_pbar(id: int):
    """Close the progress bar with the given id."""

    def callback(id):
        global pbars
        id = int(id)
        if str(id) in pbars:
            pbars[str(id)].close()
            del pbars[str(id)]
        return id

    id = jax.experimental.io_callback(
        callback, result_shape_dtypes=jax.ShapeDtypeStruct((), jax.numpy.int32), id=id
    )
    return id
