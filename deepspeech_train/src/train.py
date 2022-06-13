try:
    import sys
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    from .utils.measure import timer
    from deepspeech_training import train as ds_train
except ImportError:
    sys.exit(1)


@timer
def start_training(command):
    sys.argv.extend(command)
    ds_train.run_script()
