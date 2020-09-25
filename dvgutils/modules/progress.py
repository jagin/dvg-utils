import sys

from tqdm import tqdm
from tqdm.contrib import DummyTqdmFile


class Progress:
    def __init__(self, **kwargs):
        # See: https://github.com/tqdm/tqdm#redirecting-writing
        self.orig_std = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = map(DummyTqdmFile, self.orig_std)
        self.progress = tqdm(file=self.orig_std[0], dynamic_ncols=True, **kwargs)

    def update(self, n=1):
        self.progress.update(n)

    def close(self):
        sys.stdout, sys.stderr = self.orig_std
