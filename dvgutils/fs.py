"""File system helpers"""

import os


def walk_to_level(path, level=None):
    """Directory tree generator down to the selected level.

    If level is not provided it works the same as :func:`os.walk`.

    :returns: yields a 3-tuple: dirpath, dirnames, filenames
    :rtype: (str, str, str)
    """
    if level is None:
        yield from os.walk(path)
        return

    path = path.rstrip(os.path.sep)
    num_sep = path.count(os.path.sep)
    for root, dirs, files in os.walk(path):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            # When some directory on or below the desired level is found, all
            # of its subdirs are removed from the list of subdirs to search next.
            # So they won't be walked.
            del dirs[:]


def list_files(path, valid_exts=None, contains=None, level=None):
    """List files generator.

    :param str path: starting path
    :param str | (str, ...) | None valid_exts: valid file extension(s)
    :param str | None contains: string to be contained in the filename
    :param int | None level: go down to the selected level

    :returns: yields file path
    :rtype: str
    """

    # Add a dot to selected file extension(s)
    if isinstance(valid_exts, list):
        valid_exts = (f".{ext}" for ext in valid_exts)
    else:
        valid_exts = (f".{valid_exts}",)

    # Loop over the input directory structure
    for (root_dir, dir_names, filenames) in walk_to_level(path, level):
        for filename in sorted(filenames):
            # ignore the file if not contains the string
            if contains is not None and contains not in filename:
                continue

            # ignore the file if extension not valid
            if valid_exts is not None and not os.path.splitext(filename)[1].endswith(valid_exts):
                continue

            # Construct the path to the file and yield it
            file = os.path.join(root_dir, filename)
            yield file
