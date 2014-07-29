###############################################################################
##### Simple Example                                                      #####
###############################################################################

# Use Apache Ant to build and run example

# Clean all files
ant clean

# Run example
ant run

 # Run GPU Kernel
# java -Xmx4G -jar SimpleExample-GPU.jar \
  [blockSize=2 gridSize=1]

# java -Xmx4G -jar SimpleExample-GPU.jar 2 1

###############################################################################
