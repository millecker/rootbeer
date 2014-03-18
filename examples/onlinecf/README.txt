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
  [useCPU=false]
  [inputFile=/home/USERNAME/Downloads/ml-100k/u.data]
  [separator=::]

# java -Xmx4G -jar OnlineCF-GPU.jar 256 14 3 1 true

# java -Xmx4G -jar OnlineCF-GPU.jar 256 14 3 1 false false \
  /home/USERNAME/Downloads/ml-100k/u.data

# java -Xmx4G -jar OnlineCF-GPU.jar 256 14 3 1 false false \
  /home/USERNAME/Downloads/ml-1m/ratings.dat ::

# java -Xmx8G -jar OnlineCF-GPU.jar 256 14 3 1 false false \
  /home/USERNAME/Downloads/ml-10M100K/ratings.dat ::

###############################################################################
