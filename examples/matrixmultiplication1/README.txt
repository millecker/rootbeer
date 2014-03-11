###############################################################################
##### MatrixMultiplication Example                                        #####
###############################################################################

# Use Apache Ant to build and run example

# Clean all files
ant clean

# Run example
ant run

 # Run GPU Kernel
# java -Xmx4G -jar PiEstimator-GPU.jar \
  [blockSize=256 gridSize=14]

# java -Xmx4G -jar PiEstimator-GPU.jar 256 14

###############################################################################
