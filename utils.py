import logging
import time


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        logging.info(f"\U00002B50 {self.name}:")
        self.t = time.time()

    def __exit__(self, *args, **kwargs):
        elapsed_time = time.time() - self.t
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        # Timedelta makes for human readable format - not safe for maths operations
        logging.info(
            f"\U0001F551 Elapsed time for {self.name}: {elapsed_time} (HH:MM:SS)"
        )
