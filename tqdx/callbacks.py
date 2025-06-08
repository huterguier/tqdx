import jax
import jax.experimental
import tqdm
from itertools import count


pbars: dict[str, tqdm.std.tqdm] = {}
pbar_ids: count = count()


def init_pbar(length: int) -> int:
    """Initialize a progress bar with the given length."""
    def callback(length):
        pbar = tqdm.tqdm(total=int(length), desc="Processing")
        id = next(pbar_ids)
        pbars[str(id)] = pbar
        return id

    id = jax.experimental.io_callback(
        callback,
        result_shape_dtypes=jax.ShapeDtypeStruct((), jax.numpy.int32),
        length=length)
    return id


def update_pbar(id: int):
    """Update the progress bar with the given id."""
    def callback(id):
        id = int(id)
        if str(id) in pbars:
            pbars[str(id)].update(1)
    jax.debug.callback(callback, id)


def close_pbar(id: int):
    """Close the progress bar with the given id."""
    def callback(id):
        global pbars
        id = int(id)
        if id in pbars:
            pbars[id].close()
            del pbars[id]
    jax.debug.callback(callback, id)
