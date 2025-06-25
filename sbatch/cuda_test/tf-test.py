import tensorflow as tf

# Check TensorFlow version
print(f"TensorFlow Version: {tf.__version__}")

# Check if GPU is available
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

# List all physical devices
physical_devices = tf.config.list_physical_devices()
print("\nAll physical devices:")
for device in physical_devices:
    print(device)

# Additional GPU details if available
if tf.config.list_physical_devices('GPU'):
    print("\nGPU Details:")
    gpu_devices = tf.config.list_physical_devices('GPU')
    for gpu in gpu_devices:
        print(gpu)
    
    # Run a simple computation on GPU
    print("\nRunning test computation...")
    with tf.device('/GPU:0'):
        a = tf.random.normal([100, 100])
        b = tf.random.normal([100, 100])
        c = tf.matmul(a, b)
    print("Test computation completed successfully using GPU!")
else:
    print("\nNo GPU devices found. TensorFlow is using the CPU.")