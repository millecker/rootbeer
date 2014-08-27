###############################################################################
##### Simple Example                                                      #####
###############################################################################

# Use Apache Ant to build and run example

# Clean all files
ant clean

# Run example
ant run

 # Run GPU Kernel
# java -Xmx4g -jar SimpleExample-GPU.jar \
  [gridSize=1 blockSize=2]

# java -Xmx4g -jar SimpleExample-GPU.jar 1 2

###############################################################################
