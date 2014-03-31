###############################################################################
##### Test2KeyMap Example                                                 #####
###############################################################################

# Use Apache Ant to build and run example

# Clean all files
ant clean

# Run example
ant run

 # Run GPU Kernel
# java -Xmx4G -jar Test2KeyMap-GPU.jar [-DblockSize=2 -DgridSize=1]

# java -Xmx4G -jar Test2KeyMap-GPU.jar 2 1

###############################################################################
