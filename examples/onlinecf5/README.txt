###############################################################################
##### Collaborative Filtering Example                                     #####
###############################################################################

# Use Apache Ant to build and run example

# Clean all files
ant clean

# Run example
ant run

# Run GPU Kernel
# java -Xmx4G -jar OnlineCF5-GPU.jar \
  [blockSize=256 gridSize=14 matrixRank=3 maxIterations=150 debug=false]
  [useCPU=false] [CPUemulatesGPU=false] [ALPHA=0.001] 
  [userCount=0] [itemCount=0] [percentNonZeroValues=50]
  [inputFile=/home/USERNAME/Downloads/ml-100k/u.data]
  [separator=::]

# java -Xmx4G -jar OnlineCF5-GPU.jar 256 14 3 150 true

# java -Xmx4G -jar OnlineCF5-GPU.jar 256 14 3 1 false false false \
  0.001 0 0 0 \
  /home/USERNAME/Downloads/ml-100k/u.data

# java -Xmx4G -jar OnlineCF5-GPU.jar 256 14 3 1 false false false \
  0.001 0 0 0 \
  /home/USERNAME/Downloads/ml-1m/ratings.dat ::

# java -Xmx16G -jar OnlineCF5-GPU.jar 256 14 3 1 false false false \
  0.001 0 0 0 \
  /home/USERNAME/Downloads/ml-10M100K/ratings.dat ::

###############################################################################
