import tensorflow as tf

# List available GPU
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)

# Test if GPU is being used
if gpus:
    tf.debugging.set_log_device_placement(True)
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        print(tf.matmul(a, b))
else:
    print("No GPU detected.")
