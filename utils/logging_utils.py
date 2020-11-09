import sys


def log(msg):
    if msg[-1] != '\n':
        msg += '\n'
    sys.stderr.write(msg)
    sys.stderr.flush()

