###############################################################################
##### MatrixMultiplication1 Example                                       #####
###############################################################################

# Use Apache Ant to build and run example

# Clean all files
ant clean

# Run example
ant run

 # Run GPU Kernel
# java -Xmx4g -jar MatrixMultiplication1-GPU.jar \
  [blockSize=256 gridSize=14 n=1024 debug=false]

# java -Xmx4g -jar MatrixMultiplication1-GPU.jar 256 14 1024 false
# java -Xms8g -Xmx8g -jar MatrixMultiplication1-GPU.jar 1024 14 4096 false

###############################################################################
