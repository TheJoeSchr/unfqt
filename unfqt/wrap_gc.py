import logging
import os
import psutil
import time

from functools import wraps


def to_mb(mb):
    return f'{mb/1024/1024:.2f} MB'


def wrap_gc(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        logger = logging.getLogger(__name__)
        process = psutil.Process(os.getpid())
        mb = process.memory_info().rss
        # gc.set_threshold(50_000, 500, 1000)
        # gc.set_debug(gc.DEBUG_STATS)
        start = time.time()
        res = f(*args, **kwargs)
        end = time.time()
        mb_a = process.memory_info().rss
        logger.info(
            f"took {end - start:,.2f} ms."
            f"Total memory used: {to_mb(mb)} -> {to_mb(mb_a)} "
            f"({to_mb(mb_a - mb)})")
        return res
    return wrapped
