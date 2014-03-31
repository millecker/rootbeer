###############################################################################
##### TestPrintln Example                                                 #####
###############################################################################

# Use Apache Ant to build and run example

# Clean all files
ant clean

# Run example
ant run

 # Run GPU Kernel
# java -Xmx4G -jar TestPrintln-GPU.jar [-DblockSize=256 -DgridSize=14]

# java -Xmx4G -jar TestPrintln-GPU.jar 256 14

###############################################################################
