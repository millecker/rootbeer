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
  [tileWidth=32 n=32 m=32 l=32 debug=true]

# java -Xms4g -Xmx4g -jar MatrixMultiplication5-GPU.jar 10 10 10 10 false
# java -Xms4g -Xmx4g -jar MatrixMultiplication5-GPU.jar 12 12 12 12 false
# java -Xms4g -Xmx4g -jar MatrixMultiplication5-GPU.jar 32 32 32 32 false
# java -Xms4g -Xmx4g -jar MatrixMultiplication5-GPU.jar 32 1024 1024 1024 false
# java -Xms4g -Xmx4g -jar MatrixMultiplication5-GPU.jar 32 2048 2048 2048 false
# java -Xms4g -Xmx4g -jar MatrixMultiplication5-GPU.jar 32 3072 3072 3072 false
# java -Xms8g -Xmx8g -jar MatrixMultiplication5-GPU.jar 32 4096 4096 4096 false

###############################################################################
