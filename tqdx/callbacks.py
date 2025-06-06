import jax
import jax.experimental
import tqdm
from .pbars import pbars, next_id


def init_pbar(length: int) -> int:
    """Initialize a progress bar with the given length."""
    print(length)
    def callback(length):
        global next_id, pbars
        pbar = tqdm.tqdm(total=int(length), desc="Processing")
        id = next_id
        print(id)
        next_id += 1
        pbars[id] = pbar
        print(pbars)
        return id

    id = jax.experimental.io_callback(
        callback, 
        result_shape_dtypes=jax.ShapeDtypeStruct((), jax.numpy.int32),
        length=length)
    print(pbars)
    return id


def update_pbar(id: int) -> None:
    """Update the progress bar with the given id."""
    def callback(id):
        global pbars
        id = int(id)
        print(id)
        print(pbars)
        if id in pbars:
            print("Updating progress bar", id)
            pbars[id].update(1)
    jax.debug.callback(callback, id)


def close_pbar(id: int) -> None:
    """Close the progress bar with the given id."""
    def callback(id):
        global pbars
        id = int(id)
        if id in pbars:
            pbars[id].close()
            del pbars[id]
    jax.debug.callback(callback, id)
