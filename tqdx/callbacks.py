from itertools import count

import jax
import jax.experimental
import tqdm
import tqdm.rich as rich

pbars: dict[str, tqdm.std.tqdm] = {}
pbar_ids: count = count()


def init_pbar(length: int, disable: bool = False, use_rich: bool = False) -> int:
    """Initialize a progress bar with the given length."""

    def callback(length):
        cls = rich if use_rich else tqdm
        pbar = cls.tqdm(total=int(length), desc="Processing", disable=disable)
        id = next(pbar_ids)
        pbars[str(id)] = pbar
        return id

    id = jax.experimental.io_callback(
        callback,
        result_shape_dtypes=jax.ShapeDtypeStruct((), jax.numpy.int32),
        length=length,
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


if __name__ == "__main__":
    import time

    def process(length: int, use_rich: bool):
        bar_id = init_pbar(length, disable=False, use_rich=use_rich)
        for _ in range(length):
            time.sleep(0.05)
            bar_id = update_pbar(bar_id)
        close_pbar(bar_id)

    print("\nTesting with standard tqdm:")
    process(30, use_rich=False)

    print("\nTesting with rich-tqdm:")
    process(30, use_rich=True)
