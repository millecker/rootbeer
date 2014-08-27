###############################################################################
##### MatrixMultiplication2 Example                                       #####
###############################################################################

# Use Apache Ant to build and run example

# Clean all files
ant clean

# Run example
ant run

 # Run GPU Kernel
# java -Xmx4g -jar MatrixMultiplication2-GPU.jar \
  [n=32*32 debug=false]

# java -Xms4g -Xmx4g -jar MatrixMultiplication2-GPU.jar 32 false
# java -Xms8g -Xmx8g -jar MatrixMultiplication2-GPU.jar 64 false

###############################################################################
