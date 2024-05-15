import os
import tensorflow as tf
import numpy as np

# Specify the directory path
directory = "logs/E_R2/"

# Iterate through all subdirectories and files
for root, dirs, files in os.walk(directory):
    for dir in dirs:
        for root2, dirs2, files2 in os.walk(os.path.join(root,dir+"/1/images")):            
            for file in files2:
                if file.endswith('.tensor'):
                    # Get the file path
                    file_path = os.path.join(root2, file)
                    
                    # Load the tensor data
                    serialized_tensor = tf.io.read_file(file_path)

                    # Parse the serialized tensor
                    restored_tensor = tf.io.parse_tensor(serialized_tensor, out_type=tf.float32)

                    # Convert the restored tensor to a NumPy array
                    numpy_array = np.array(restored_tensor)
                    
                    # Create the new file path with .npy extension
                    new_file_path = file_path.replace('.tensor', '.npy')
                    
                    # Save the tensor data in .npy format
                    np.save(new_file_path, numpy_array)