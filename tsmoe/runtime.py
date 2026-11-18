"""Shared runtime helpers: cooperative Ctrl+C for long train/prepare loops."""
import signal
import sys

# Long torch/CUDA ops defer SIGINT until the next Python boundary. First Ctrl+C
# sets a flag checked between batches; a second forces an immediate interrupt.
_STOP = False


def _on_sigint(signum, frame):
    global _STOP
    if _STOP:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        raise KeyboardInterrupt
    _STOP = True
    print("\nCtrl+C received; stopping after the current step "
          "(Ctrl+C again to force exit).", file=sys.stderr, flush=True)


def install_interrupt_handler():
    """Install cooperative SIGINT handling; returns the previous handler."""
    global _STOP
    _STOP = False
    return signal.signal(signal.SIGINT, _on_sigint)


def check_interrupt():
    if _STOP:
        raise KeyboardInterrupt
