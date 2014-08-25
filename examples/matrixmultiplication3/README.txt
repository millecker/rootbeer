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
  [n=4 m=4 debug=false]

# java -Xmx4G -jar MatrixMultiplication3-GPU.jar 4 4 true

###############################################################################
