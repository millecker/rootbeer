###############################################################################
##### MatrixMultiplication3 Example                                       #####
###############################################################################

# Use Apache Ant to build and run example

# Clean all files
ant clean

# Run example
ant run

 # Run GPU Kernel
# java -Xmx4G -jar MatrixMultiplication3-GPU.jar \
  [gridSize=14 blockSize=256 n=4 m=4 debug=true]

# java -Xmx4G -jar MatrixMultiplication3-GPU.jar 14 256 4 4 true

###############################################################################
