###############################################################################
##### MatrixMultiplication4 Example                                       #####
###############################################################################

# Use Apache Ant to build and run example

# Clean all files
ant clean

# Run example
ant run

 # Run GPU Kernel
# java -Xmx4G -jar MatrixMultiplication4-GPU.jar \
  [n=4 m=4 l=4 debug=true]

# java -Xmx4G -jar MatrixMultiplication4-GPU.jar 4 4 4 true
# java -Xmx4G -jar MatrixMultiplication4-GPU.jar 1024 1024 1024 false
# java -Xmx4G -jar MatrixMultiplication4-GPU.jar 2048 2048 2048 false
# java -Xmx4G -jar MatrixMultiplication4-GPU.jar 3072 3072 3072 false
# java -Xmx8G -jar MatrixMultiplication4-GPU.jar 4096 4096 4096 false

###############################################################################
