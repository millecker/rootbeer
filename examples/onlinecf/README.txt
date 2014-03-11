###############################################################################
##### Collaborative Filtering Example                                     #####
###############################################################################

# Use Apache Ant to build and run example

# Clean all files
ant clean

# Run example
ant run

 # Run GPU Kernel
# java -Xmx4G -jar OnlineCF-GPU.jar [-DblockSize=2 -DgridSize=1]

# java -Xmx4G -jar OnlineCF-GPU.jar 256 14

###############################################################################
