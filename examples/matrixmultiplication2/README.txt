###############################################################################
##### MatrixMultiplication2 Example                                       #####
###############################################################################

# Use Apache Ant to build and run example

# Clean all files
ant clean

# Run example
ant run

 # Run GPU Kernel
# java -Xmx4G -jar MatrixMultiplication2-GPU.jar \
  [n=32*32 debug=false]

# java -Xmx4G -jar MatrixMultiplication2-GPU.jar 32 false

###############################################################################
