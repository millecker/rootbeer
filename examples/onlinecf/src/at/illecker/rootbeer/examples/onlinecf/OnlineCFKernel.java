/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package at.illecker.rootbeer.examples.onlinecf;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;

import org.trifort.rootbeer.runtime.Context;
import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.Rootbeer;
import org.trifort.rootbeer.runtime.RootbeerGpu;
import org.trifort.rootbeer.runtime.StatsRow;
import org.trifort.rootbeer.runtime.ThreadConfig;
import org.trifort.rootbeer.runtime.util.Stopwatch;

/**
 * Collaborative Filtering based on
 * 
 * Singular Value Decomposition for Collaborative Filtering on a GPU
 * http://iopscience.iop.org/1757-899X/10/1/012017/pdf/1757-899X_10_1_012017.pdf
 * 
 */
public class OnlineCFKernel implements Kernel {

  private GpuUserItemMap m_userItemMap;
  private GpuVectorMap m_usersMatrix;
  private GpuVectorMap m_itemsMatrix;
  private long m_N;
  private long m_M;
  private double m_ALPHA;
  private int m_matrixRank;
  private int m_maxIterations;

  public OnlineCFKernel(GpuUserItemMap userItemMap, GpuVectorMap usersMatrix,
      GpuVectorMap itemsMatrix, long n, long m, double alpha, int matrixRank,
      int maxIterations) {
    this.m_userItemMap = userItemMap;
    this.m_usersMatrix = usersMatrix;
    this.m_itemsMatrix = itemsMatrix;
    this.m_N = n;
    this.m_M = m;
    this.m_ALPHA = alpha;
    this.m_matrixRank = matrixRank;
    this.m_maxIterations = maxIterations;
  }

  public void gpuMethod() {
    int blockSize = RootbeerGpu.getBlockDimx();
    int gridSize = RootbeerGpu.getGridDimx();
    int block_idxx = RootbeerGpu.getBlockIdxx();
    int thread_idxx = RootbeerGpu.getThreadIdxx();

    if (blockSize < m_matrixRank) {
      return; // TODO Error
    }

    long usersPerBlock = divup(m_N, gridSize);
    long itemsPerBlock = divup(m_M, gridSize);

    // SharedMemory per block
    int shmStartPos = 0;
    // userVector: matrixRank x Doubles (m_matrixRank * 8 bytes)
    int shmUserVectorStartPos = shmStartPos;
    // itemVector: matrixRank x Doubles (m_matrixRank * 8 bytes)
    int shmItemVectorStartPos = shmUserVectorStartPos + m_matrixRank * 8;
    // multVector: matrixRank x Doubles (m_matrixRank * 8 bytes)
    int shmMultVectorStartPos = shmItemVectorStartPos + m_matrixRank * 8;
    // 1 x Double (8 bytes)
    int shmExpectedScoreStartPos = shmMultVectorStartPos + m_matrixRank * 8;
    // 1 x Long (8 bytes)
    int shmInputIdStartPos = shmExpectedScoreStartPos + 8;
    // 1 x Boolean (1 byte)
    int shmInputIsNullStartPos = shmInputIdStartPos + 8;

    // DEBUG
    if (RootbeerGpu.getThreadId() == 0) {
      System.out.println("blockSize: " + blockSize);
      System.out.println("gridSize: " + gridSize);
      System.out.println("usersPerBlock: " + usersPerBlock);
      System.out.println("itemsPerBlock: " + itemsPerBlock);
    }

    // Start OnlineCF algorithm
    for (int i = 0; i < m_maxIterations; i++) {

      // **********************************************************************
      // Compute U (Users)
      // **********************************************************************
      // Loop over all usersPerBlock
      for (long u = 0; u < usersPerBlock; u++) {

        // Thread 0 of each block prepare SharedMemory
        if (thread_idxx == 0) {

          long userId = (block_idxx * usersPerBlock) + u + 1; // starting with 1
          RootbeerGpu.setSharedLong(shmInputIdStartPos, userId);

          if (userId <= m_N) {
            double[] userVector = m_usersMatrix.get(userId);
            if (userVector != null) {
              RootbeerGpu.setSharedBoolean(shmInputIsNullStartPos, false);

              // Setup userVector
              for (int j = 0; j < m_matrixRank; j++) {
                int userVectorIndex = shmUserVectorStartPos + j * 8;
                RootbeerGpu.setSharedDouble(userVectorIndex, userVector[j]);
              }

              // Init multVector
              for (int j = 0; j < m_matrixRank; j++) {
                int multVectorIndex = shmMultVectorStartPos + j * 8;
                RootbeerGpu.setSharedDouble(multVectorIndex, 0);
              }

            } else {
              RootbeerGpu.setSharedBoolean(shmInputIsNullStartPos, true);
            }
          } else { // userId > m_N
            RootbeerGpu.setSharedBoolean(shmInputIsNullStartPos, true);
          }

        }
        // Sync all threads within a block
        RootbeerGpu.syncthreads();

        // if userVector != null
        if (!RootbeerGpu.getSharedBoolean(shmInputIsNullStartPos)) {

          // Each user loops over all items
          for (long itemId = 1; itemId <= m_M; itemId++) {

            if (thread_idxx == 0) {

              // Setup expectedScore
              Double expectedScore = m_userItemMap.get(
                  RootbeerGpu.getSharedLong(shmInputIdStartPos), itemId);
              if (expectedScore != null) {
                RootbeerGpu.setSharedDouble(shmExpectedScoreStartPos,
                    expectedScore);

                // Setup itemVector on SharedMemory
                double[] itemVector = m_itemsMatrix.get(itemId);
                for (int j = 0; j < m_matrixRank; j++) {
                  int itemVectorIndex = shmItemVectorStartPos + j * 8;
                  RootbeerGpu.setSharedDouble(itemVectorIndex, itemVector[j]);
                }

              } else {
                RootbeerGpu.setSharedDouble(shmExpectedScoreStartPos, 0);
              }
            }

            // Sync all threads within a block
            RootbeerGpu.syncthreads();

            // if expectedScore != 0
            if (RootbeerGpu.getSharedDouble(shmExpectedScoreStartPos) != 0) {

              // Each thread within a block computes one multiplication
              if (thread_idxx < m_matrixRank) {

                int userVectorIndex = shmUserVectorStartPos + thread_idxx * 8;
                double userVal = RootbeerGpu.getSharedDouble(userVectorIndex);

                int itemVectorIndex = shmItemVectorStartPos + thread_idxx * 8;
                double itemVal = RootbeerGpu.getSharedDouble(itemVectorIndex);

                int multVectorIndex = shmMultVectorStartPos + thread_idxx * 8;
                RootbeerGpu.setSharedDouble(multVectorIndex, userVal * itemVal);
              }

              // Ensure all multiplications were saved in SharedMemory
              // RootbeerGpu.threadfenceBlock();

              // Sync all threads within a block
              RootbeerGpu.syncthreads();

              // Calculate score by summing up multiplications
              // do reduction in shared memory
              // 1-bit right shift = divide by two to the power 1
              int shmMultVectorEndPos = shmMultVectorStartPos + m_matrixRank
                  * 8;
              for (int s = (int) divup(m_matrixRank, 2); s > 0; s >>= 1) {

                if (thread_idxx < s) {
                  // sh_mem[ltid] += sh_mem[ltid + s];
                  int multVectorIndex1 = shmMultVectorStartPos + thread_idxx
                      * 8;
                  int multVectorIndex2 = shmMultVectorStartPos
                      + (thread_idxx + s) * 8;
                  double val1 = RootbeerGpu.getSharedDouble(multVectorIndex1);
                  double val2 = 0;
                  if (multVectorIndex2 < shmMultVectorEndPos) {
                    val2 = RootbeerGpu.getSharedDouble(multVectorIndex2);
                  }
                  RootbeerGpu.setSharedDouble(multVectorIndex1, val1 + val2);
                }
                // Sync all threads within a block
                RootbeerGpu.syncthreads();
              }

              // Calculate new userVector
              // Each thread does one update operation of vector u
              if (thread_idxx < m_matrixRank) {

                int userVectorIndex = shmUserVectorStartPos + thread_idxx * 8;
                double userVal = RootbeerGpu.getSharedDouble(userVectorIndex);

                int itemVectorIndex = shmItemVectorStartPos + thread_idxx * 8;
                double itemVal = RootbeerGpu.getSharedDouble(itemVectorIndex);

                double expectedScore = RootbeerGpu
                    .getSharedDouble(shmExpectedScoreStartPos);

                double calculatedScore = RootbeerGpu
                    .getSharedDouble(shmMultVectorStartPos);

                userVal += 2 * m_ALPHA * itemVal
                    * (expectedScore - calculatedScore);

                RootbeerGpu.setSharedDouble(userVectorIndex, userVal);
              }

              // Sync all threads within a block
              RootbeerGpu.syncthreads();

            } // if expectedScore != 0

          } // loop over all items

          // Thread 0 of each block updates userVector
          if (thread_idxx == 0) {
            double[] newUserVector = new double[m_matrixRank];
            for (int j = 0; j < m_matrixRank; j++) {
              int userVectorIndex = shmUserVectorStartPos + j * 8;
              newUserVector[j] = RootbeerGpu.getSharedDouble(userVectorIndex);
            }

            m_usersMatrix.put(RootbeerGpu.getSharedLong(shmInputIdStartPos),
                newUserVector);
          }

        } // if userVector != null

      } // loop over all usersPerBlock

      // Ensure usersMatrix is updated on every block
      RootbeerGpu.threadfenceSystem();

      // Sync all blocks Inter-Block Synchronization
      RootbeerGpu.syncblocks(1);

      // **********************************************************************
      // Compute V (Items)
      // **********************************************************************
      // Loop over all itemsPerBlock
      for (long v = 0; v < itemsPerBlock; v++) {

        // Thread 0 of each block prepare SharedMemory
        if (thread_idxx == 0) {

          long itemId = (block_idxx * itemsPerBlock) + v + 1; // starting with 1
          RootbeerGpu.setSharedLong(shmInputIdStartPos, itemId);

          if (itemId <= m_M) {
            double[] itemVector = m_itemsMatrix.get(itemId);
            if (itemVector != null) {
              RootbeerGpu.setSharedBoolean(shmInputIsNullStartPos, false);

              // Setup itemVector
              for (int j = 0; j < m_matrixRank; j++) {
                int itemVectorIndex = shmItemVectorStartPos + j * 8;
                RootbeerGpu.setSharedDouble(itemVectorIndex, itemVector[j]);
              }

              // Init multVector
              for (int j = 0; j < m_matrixRank; j++) {
                int multVectorIndex = shmMultVectorStartPos + j * 8;
                RootbeerGpu.setSharedDouble(multVectorIndex, 0);
              }
            } else {
              RootbeerGpu.setSharedBoolean(shmInputIsNullStartPos, true);
            }
          } else { // itemId > m_M
            RootbeerGpu.setSharedBoolean(shmInputIsNullStartPos, true);
          }

        }
        // Sync all threads within a block
        RootbeerGpu.syncthreads();

        // if itemVector != null
        if (!RootbeerGpu.getSharedBoolean(shmInputIsNullStartPos)) {

          // Each user loops over all items
          for (long userId = 1; userId <= m_N; userId++) {

            if (thread_idxx == 0) {

              // Setup expectedScore
              Double expectedScore = m_userItemMap.get(userId,
                  RootbeerGpu.getSharedLong(shmInputIdStartPos));
              if (expectedScore != null) {
                RootbeerGpu.setSharedDouble(shmExpectedScoreStartPos,
                    expectedScore);

                // Setup userVector on SharedMemory
                double[] userVector = m_usersMatrix.get(userId);
                for (int j = 0; j < m_matrixRank; j++) {
                  int userVectorIndex = shmUserVectorStartPos + j * 8;
                  RootbeerGpu.setSharedDouble(userVectorIndex, userVector[j]);
                }

              } else {
                RootbeerGpu.setSharedDouble(shmExpectedScoreStartPos, 0);
              }
            }

            // Sync all threads within a block
            RootbeerGpu.syncthreads();

            // if expectedScore != 0
            if (RootbeerGpu.getSharedDouble(shmExpectedScoreStartPos) != 0) {

              // Each thread within a block computes one multiplication
              if (thread_idxx < m_matrixRank) {

                int itemVectorIndex = shmItemVectorStartPos + thread_idxx * 8;
                double itemVal = RootbeerGpu.getSharedDouble(itemVectorIndex);

                int userVectorIndex = shmUserVectorStartPos + thread_idxx * 8;
                double userVal = RootbeerGpu.getSharedDouble(userVectorIndex);

                int multVectorIndex = shmMultVectorStartPos + thread_idxx * 8;
                RootbeerGpu.setSharedDouble(multVectorIndex, itemVal * userVal);
              }

              // Ensure all multiplications were saved in SharedMemory
              // RootbeerGpu.threadfenceBlock();

              // Sync all threads within a block
              RootbeerGpu.syncthreads();

              // Calculate score by summing up multiplications
              // do reduction in shared memory
              // 1-bit right shift = divide by two to the power 1
              int shmMultVectorEndPos = shmMultVectorStartPos + m_matrixRank
                  * 8;
              for (int s = (int) divup(m_matrixRank, 2); s > 0; s >>= 1) {

                if (thread_idxx < s) {
                  // sh_mem[ltid] += sh_mem[ltid + s];
                  int multVectorIndex1 = shmMultVectorStartPos + thread_idxx
                      * 8;
                  int multVectorIndex2 = shmMultVectorStartPos
                      + (thread_idxx + s) * 8;
                  double val1 = RootbeerGpu.getSharedDouble(multVectorIndex1);
                  double val2 = 0;
                  if (multVectorIndex2 < shmMultVectorEndPos) {
                    val2 = RootbeerGpu.getSharedDouble(multVectorIndex2);
                  }
                  RootbeerGpu.setSharedDouble(multVectorIndex1, val1 + val2);
                }
                // Sync all threads within a block
                RootbeerGpu.syncthreads();
              }

              // Calculate new itemVector
              // Each thread does one update operation of vector u
              if (thread_idxx < m_matrixRank) {

                int itemVectorIndex = shmItemVectorStartPos + thread_idxx * 8;
                double itemVal = RootbeerGpu.getSharedDouble(itemVectorIndex);

                int userVectorIndex = shmUserVectorStartPos + thread_idxx * 8;
                double userVal = RootbeerGpu.getSharedDouble(userVectorIndex);

                double expectedScore = RootbeerGpu
                    .getSharedDouble(shmExpectedScoreStartPos);

                double calculatedScore = RootbeerGpu
                    .getSharedDouble(shmMultVectorStartPos);

                itemVal += 2 * m_ALPHA * userVal
                    * (expectedScore - calculatedScore);

                RootbeerGpu.setSharedDouble(itemVectorIndex, itemVal);
              }

              // Sync all threads within a block
              RootbeerGpu.syncthreads();

            } // if expectedScore != 0

          } // loop over all items

          // Thread 0 of each block updates itemVector
          if (thread_idxx == 0) {
            double[] newItemVector = new double[m_matrixRank];
            for (int j = 0; j < m_matrixRank; j++) {
              int itemVectorIndex = shmItemVectorStartPos + j * 8;
              newItemVector[j] = RootbeerGpu.getSharedDouble(itemVectorIndex);
            }

            m_itemsMatrix.put(RootbeerGpu.getSharedLong(shmInputIdStartPos),
                newItemVector);
          }

        } // if itemVector != null

      } // loop over all itemsPerBlock

      // Ensure itemsMatrix is updated on every block
      RootbeerGpu.threadfenceSystem();

      // Sync all blocks Inter-Block Synchronization
      RootbeerGpu.syncblocks(2);
    }
  }

  private long divup(long x, long y) {
    if (x % y != 0) {
      return ((x + y - 1) / y); // round up
    } else {
      return x / y;
    }
  }

  private String arrayToString(double[] arr) {
    if (arr != null) {
      String result = "";
      for (int i = 0; i < arr.length; i++) {
        result += (i + 1 == arr.length) ? arr[i] : (arr[i] + ",");
      }
      return result;
    }
    return "null";
  }

  public static GpuUserItemMap getUserItemMap() {
    GpuUserItemMap userItemMap = new GpuUserItemMap(13);
    userItemMap.put(1, 1, 4);
    userItemMap.put(1, 2, 2.5);
    userItemMap.put(1, 3, 3.5);
    userItemMap.put(1, 4, 1);
    userItemMap.put(1, 5, 3.5);

    userItemMap.put(2, 1, 4);
    userItemMap.put(2, 2, 2.5);
    userItemMap.put(2, 3, 3.5);
    userItemMap.put(2, 4, 1);
    userItemMap.put(2, 5, 3.5);

    userItemMap.put(3, 1, 4);
    userItemMap.put(3, 2, 2.5);
    userItemMap.put(3, 3, 3.5);

    return userItemMap;
  }

  public static double[] getRandomArray(Random rand, int size) {
    double[] arr = new double[size];
    for (int i = 0; i < size; i++) {
      arr[i] = rand.nextDouble();
    }
    return arr;
  }

  public static GpuVectorMap getVectorMap(Random rand, int size, int matrixRank) {
    GpuVectorMap vectorMap = new GpuVectorMap(size);
    for (int i = 1; i <= size; i++) {
      vectorMap.put(i, getRandomArray(rand, matrixRank));
    }
    return vectorMap;
  }

  public static void main(String[] args) {

    int blockSize = 256;
    int gridSize = 14;
    boolean isDebbuging = false;
    Random rand = new Random(32L);

    final double ALPHA = 0.01;
    int matrixRank = 3;
    int maxIterations = 1;
    int userCount = 3;
    int itemCount = 5;
    String inputFile = "";

    // parse arguments
    if ((args.length > 0) && (args.length >= 5)) {
      blockSize = Integer.parseInt(args[0]);
      gridSize = Integer.parseInt(args[1]);
      matrixRank = Integer.parseInt(args[2]);
      maxIterations = Integer.parseInt(args[3]);
      isDebbuging = Boolean.parseBoolean(args[4]);
      // optional parameters
      if (args.length > 5) {
        inputFile = args[5];
      }
    } else {
      System.out.println("Wrong argument size!");
      System.out.println("    Argument1=blockSize");
      System.out.println("    Argument2=gridSize");
      System.out.println("    Argument3=matrixRank");
      System.out.println("    Argument4=maxIterations");
      System.out.println("    Argument5=debug(true|false=default)");
      System.out
          .println("    Argument6=inputFile (optional) | MovieLens inputFile");
      return;
    }

    // Check if inputFile exists
    if ((!inputFile.isEmpty()) && (!new File(inputFile).exists())) {
      System.out.println("Error: inputFile: " + inputFile + " does not exist!");
      return;
    }

    if (blockSize < matrixRank) {
      System.err.println("blockSize < matrixRank");
      return;
    }

    System.out.println("blockSize: " + blockSize);
    System.out.println("gridSize: " + gridSize);
    System.out.println("matrixRank: " + matrixRank);
    System.out.println("maxIterations: " + maxIterations);

    // Prepare input
    GpuUserItemMap userItemMap = null;
    GpuVectorMap usersMap = null;
    GpuVectorMap itemsMap = null;

    if (inputFile.isEmpty()) { // no inputFile

      userItemMap = getUserItemMap();
      usersMap = getVectorMap(rand, userCount, matrixRank);
      itemsMap = getVectorMap(rand, itemCount, matrixRank);

    } else { // parse inputFile

      List<double[]> preferences = new ArrayList<double[]>();
      HashMap<Long, double[]> usersMatrix = new HashMap<Long, double[]>();
      HashMap<Long, double[]> itemsMatrix = new HashMap<Long, double[]>();

      try {

        BufferedReader br = new BufferedReader(new FileReader(inputFile));
        String line;
        while ((line = br.readLine()) != null) {
          String[] values = line.split("\\t");
          long userId = Long.parseLong(values[0]);
          long itemId = Long.parseLong(values[1]);
          double rating = Double.parseDouble(values[2]);
          // System.out.println("userId: " + userId + " itemId: " + itemId
          // + " rating: " + rating);

          // Add User vector
          if (usersMatrix.containsKey(userId) == false) {
            double[] vals = new double[matrixRank];
            for (int i = 0; i < matrixRank; i++) {
              vals[i] = rand.nextDouble();
            }
            usersMatrix.put(userId, vals);
          }

          // Add Item vector
          if (itemsMatrix.containsKey(itemId) == false) {
            double[] vals = new double[matrixRank];
            for (int i = 0; i < matrixRank; i++) {
              vals[i] = rand.nextDouble();
            }
            itemsMatrix.put(itemId, vals);
          }

          // Add preference
          double vector[] = new double[3];
          vector[0] = userId;
          vector[1] = itemId;
          vector[2] = rating;
          preferences.add(vector);
        }
        br.close();

      } catch (NumberFormatException e) {
        e.printStackTrace();
      } catch (IOException e) {
        e.printStackTrace();
      }

      // Convert preferences to UserItemMap
      userItemMap = new GpuUserItemMap(preferences.size());
      System.out.println("userItemMap: length: " + preferences.size());
      for (double[] v : preferences) {
        userItemMap.put((long) v[0], (long) v[1], v[2]);
        if (isDebbuging) {
          System.out.println("userItemMap userId: '" + v[0] + "' itemId: '"
              + v[1] + "' value: '" + v[2]);
        }
      }

      // Convert usersMatrix to GpuVectorMap
      usersMap = new GpuVectorMap(usersMatrix.size());
      System.out.println("usersMap: length: " + usersMatrix.size());
      Iterator<Entry<Long, double[]>> userIt = usersMatrix.entrySet()
          .iterator();
      while (userIt.hasNext()) {
        Entry<Long, double[]> entry = userIt.next();
        long userId = entry.getKey();
        double[] vector = entry.getValue();
        usersMap.put(userId, vector);

        if (isDebbuging) {
          System.out.println("usersMap userId: '" + userId + " value: '"
              + Arrays.toString(vector));
        }
      }

      // Convert itemsMatrix to GpuVectorMap
      itemsMap = new GpuVectorMap(itemsMatrix.size());
      System.out.println("itemsMap: length: " + itemsMatrix.size());
      Iterator<Entry<Long, double[]>> itemIt = itemsMatrix.entrySet()
          .iterator();
      while (itemIt.hasNext()) {
        Entry<Long, double[]> entry = itemIt.next();
        long itemId = entry.getKey();
        double[] vector = entry.getValue();
        itemsMap.put(itemId, vector);

        if (isDebbuging) {
          System.out.println("itemsMap itemId: '" + itemId + " value: '"
              + Arrays.toString(vector));
        }
      }
    }

    // Debug users
    if (isDebbuging) {
      System.out.println(usersMap.size() + " users");
      for (int i = 1; i <= userCount; i++) {
        System.out.println("user: " + i + " vector: "
            + Arrays.toString(usersMap.get(i)));
      }
    }
    // Debug items
    if (isDebbuging) {
      System.out.println(itemsMap.size() + " items");
      for (int i = 1; i <= itemCount; i++) {
        System.out.println("item: " + i + " vector: "
            + Arrays.toString(itemsMap.get(i)));
      }
    }

    // Run GPU Kernels
    OnlineCFKernel kernel = new OnlineCFKernel(userItemMap, usersMap, itemsMap,
        userCount, itemCount, ALPHA, matrixRank, maxIterations);

    Rootbeer rootbeer = new Rootbeer();
    Context context = rootbeer.createDefaultContext();
    Stopwatch watch = new Stopwatch();
    watch.start();
    rootbeer.run(kernel, new ThreadConfig(blockSize, gridSize, blockSize
        * gridSize), context);
    watch.stop();

    List<StatsRow> stats = context.getStats();
    for (StatsRow row : stats) {
      System.out.println("  StatsRow:");
      System.out.println("    serial time: " + row.getSerializationTime());
      System.out.println("    exec time: " + row.getExecutionTime());
      System.out.println("    deserial time: " + row.getDeserializationTime());
      System.out.println("    num blocks: " + row.getNumBlocks());
      System.out.println("    num threads: " + row.getNumThreads());
      System.out.println("GPUTime: " + watch.elapsedTimeMillis() + " ms");
    }

    // Debug user information
    if (isDebbuging) {
      System.out.println(usersMap.size() + " users");
      for (int i = 1; i <= userCount; i++) {
        System.out.println("user: " + i + " vector: "
            + Arrays.toString(usersMap.get(i)));
      }
    }
    // Debug item information
    if (isDebbuging) {
      System.out.println(itemsMap.size() + " items");
      for (int i = 1; i <= itemCount; i++) {
        System.out.println("item: " + i + " vector: "
            + Arrays.toString(itemsMap.get(i)));
      }
    }

  }
}
