###############################################################################
##### MatrixMultiplication5 Example                                       #####
###############################################################################

# Use Apache Ant to build and run example

# Clean all files
ant clean

# Run example
ant run

 # Run GPU Kernel
# java -Xmx4g -jar MatrixMultiplication5-GPU.jar \
  [n=4 m=4 l=4 debug=true]

# java -Xms4g -Xmx4g -jar MatrixMultiplication5-GPU.jar 4 4 4 true
# java -Xms4g -Xmx4g -jar MatrixMultiplication5-GPU.jar 1024 1024 1024 false
# java -Xms4g -Xmx4g -jar MatrixMultiplication5-GPU.jar 2048 2048 2048 false
# java -Xms4g -Xmx4g -jar MatrixMultiplication5-GPU.jar 3072 3072 3072 false
# java -Xms8g -Xmx8g -jar MatrixMultiplication5-GPU.jar 4096 4096 4096 false

###############################################################################
