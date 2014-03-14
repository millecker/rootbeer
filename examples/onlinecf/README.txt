###############################################################################
##### Collaborative Filtering Example                                     #####
###############################################################################

# Use Apache Ant to build and run example

# Clean all files
ant clean

# Run example
ant run

# Run GPU Kernel
# java -Xmx4G -jar OnlineCF-GPU.jar \
  [blockSize=256 gridSize=14 matrixRank=3 maxIterations=1 debug=false]
  [inputFile=/home/user/Downloads/ml-100k/u.data]

# java -Xmx4G -jar OnlineCF-GPU.jar 256 14 3 1 true
# java -Xmx4G -jar OnlineCF-GPU.jar 256 14 3 1 false \
  /home/user/Downloads/ml-100k/u.data

###############################################################################
