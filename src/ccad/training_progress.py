"""Observe completed Sparsify updates without modifying its optimization loop."""
from contextlib import contextmanager


@contextmanager
def completed_step_callback(callback):
    # In the pinned trainer, pbar.update follows the optimizer, scheduler,
    # dead-feature bookkeeping and global_step increment. Saving here avoids
    # capturing a half-updated continuation state in an optimizer post-hook.
    import sparsify.trainer as implementation
    original = implementation.tqdm

    def tracked_progress(*args, **kwargs):
        bar = original(*args, **kwargs)
        update = bar.update
        def tracked_update(*values, **options):
            result = update(*values, **options)
            callback()
            return result
        bar.update = tracked_update
        return bar

    implementation.tqdm = tracked_progress
    try:
        yield
    finally:
        implementation.tqdm = original
