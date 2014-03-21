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
package at.illecker.rootbeer.examples.onlinecf2;

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

  private double[][] m_userItemMatrix;
  private double[][] m_usersMatrix;
  private double[][] m_itemsMatrix;
  private int m_N;
  private int m_M;
  private double m_ALPHA;
  private int m_matrixRank;
  private int m_maxIterations;

  public OnlineCFKernel(double[][] userItemMatrix, double[][] usersMatrix,
      double[][] itemsMatrix, int n, int m, double alpha, int matrixRank,
      int maxIterations) {
    this.m_userItemMatrix = userItemMatrix;
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

    int usersPerBlock = divup(m_N, gridSize);
    int itemsPerBlock = divup(m_M, gridSize);

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
    // 1 x Integer (4 bytes)
    int shmInputIdStartPos = shmExpectedScoreStartPos + 8;
    // 1 x Boolean (1 byte)
    int shmInputIsNullStartPos = shmInputIdStartPos + 4;

    // DEBUG
    if (RootbeerGpu.getThreadId() == 0) {
      System.out.println("blockSize: " + blockSize);
      System.out.println("gridSize: " + gridSize);
      System.out.println("users(N): " + m_N);
      System.out.println("items(M): " + m_M);
      System.out.println("usersPerBlock: " + usersPerBlock);
      System.out.println("itemsPerBlock: " + itemsPerBlock);
    }

    // Start OnlineCF algorithm
    for (int i = 0; i < m_maxIterations; i++) {

      // **********************************************************************
      // Compute U (Users)
      // **********************************************************************
      // Loop over all usersPerBlock
      for (int u = 0; u < usersPerBlock; u++) {

        // Thread 0 of each block prepare SharedMemory
        if (thread_idxx == 0) {

          int userId = (block_idxx * usersPerBlock) + u + 1; // starting with 1
          RootbeerGpu.setSharedInteger(shmInputIdStartPos, userId);

          if (userId <= m_N) {
            RootbeerGpu.setSharedBoolean(shmInputIsNullStartPos, false);

            // Setup userVector
            for (int j = 0; j < m_matrixRank; j++) {
              int userVectorIndex = shmUserVectorStartPos + j * 8;
              RootbeerGpu.setSharedDouble(userVectorIndex,
                  m_usersMatrix[userId][j]);
            }

            // Init multVector
            for (int j = 0; j < m_matrixRank; j++) {
              int multVectorIndex = shmMultVectorStartPos + j * 8;
              RootbeerGpu.setSharedDouble(multVectorIndex, 0);
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
          for (int itemId = 1; itemId <= m_M; itemId++) {

            if (thread_idxx == 0) {

              // Setup expectedScore
              double expectedScore = m_userItemMatrix[RootbeerGpu
                  .getSharedInteger(shmInputIdStartPos)][itemId];

              RootbeerGpu.setSharedDouble(shmExpectedScoreStartPos,
                  expectedScore);

              if (expectedScore != 0) {
                // Setup itemVector on SharedMemory
                for (int j = 0; j < m_matrixRank; j++) {
                  int itemVectorIndex = shmItemVectorStartPos + j * 8;
                  RootbeerGpu.setSharedDouble(itemVectorIndex,
                      m_itemsMatrix[itemId][j]);
                }
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
            for (int j = 0; j < m_matrixRank; j++) {
              int userVectorIndex = shmUserVectorStartPos + j * 8;
              m_usersMatrix[RootbeerGpu.getSharedInteger(shmInputIdStartPos)][j] = RootbeerGpu
                  .getSharedDouble(userVectorIndex);
            }
          }

        } // if userVector != null

      } // loop over all usersPerBlock

      // Sync all blocks Inter-Block Synchronization
      RootbeerGpu.syncblocks(1);

      // **********************************************************************
      // Compute V (Items)
      // **********************************************************************
      // Loop over all itemsPerBlock
      for (int v = 0; v < itemsPerBlock; v++) {

        // Thread 0 of each block prepare SharedMemory
        if (thread_idxx == 0) {

          int itemId = (block_idxx * itemsPerBlock) + v + 1; // starting with 1
          RootbeerGpu.setSharedInteger(shmInputIdStartPos, itemId);

          if (itemId <= m_M) {
            RootbeerGpu.setSharedBoolean(shmInputIsNullStartPos, false);

            // Setup itemVector
            for (int j = 0; j < m_matrixRank; j++) {
              int itemVectorIndex = shmItemVectorStartPos + j * 8;
              RootbeerGpu.setSharedDouble(itemVectorIndex,
                  m_itemsMatrix[itemId][j]);
            }

            // Init multVector
            for (int j = 0; j < m_matrixRank; j++) {
              int multVectorIndex = shmMultVectorStartPos + j * 8;
              RootbeerGpu.setSharedDouble(multVectorIndex, 0);
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
          for (int userId = 1; userId <= m_N; userId++) {

            if (thread_idxx == 0) {

              // Setup expectedScore
              double expectedScore = m_userItemMatrix[userId][RootbeerGpu
                  .getSharedInteger(shmInputIdStartPos)];

              RootbeerGpu.setSharedDouble(shmExpectedScoreStartPos,
                  expectedScore);

              if (expectedScore != 0) {
                // Setup userVector on SharedMemory
                for (int j = 0; j < m_matrixRank; j++) {
                  int userVectorIndex = shmUserVectorStartPos + j * 8;
                  RootbeerGpu.setSharedDouble(userVectorIndex,
                      m_usersMatrix[userId][j]);
                }
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
            for (int j = 0; j < m_matrixRank; j++) {
              int itemVectorIndex = shmItemVectorStartPos + j * 8;
              m_itemsMatrix[RootbeerGpu.getSharedInteger(shmInputIdStartPos)][j] = RootbeerGpu
                  .getSharedDouble(itemVectorIndex);
            }
          }

        } // if itemVector != null

      } // loop over all itemsPerBlock

      // Sync all blocks Inter-Block Synchronization
      RootbeerGpu.syncblocks(2);
    }
  }

  private int divup(int x, int y) {
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

  public static List<double[]> getUserItems() {
    List<double[]> userItems = new ArrayList<double[]>();
    userItems.add(new double[] { 1, 1, 4 });
    userItems.add(new double[] { 1, 2, 2.5 });
    userItems.add(new double[] { 1, 3, 3.5 });
    userItems.add(new double[] { 1, 4, 1 });
    userItems.add(new double[] { 1, 5, 3.5 });

    userItems.add(new double[] { 2, 1, 4 });
    userItems.add(new double[] { 2, 2, 2.5 });
    userItems.add(new double[] { 2, 3, 3.5 });
    userItems.add(new double[] { 2, 4, 1 });
    userItems.add(new double[] { 2, 5, 3.5 });

    userItems.add(new double[] { 3, 1, 4 });
    userItems.add(new double[] { 3, 2, 2.5 });
    userItems.add(new double[] { 3, 3, 3.5 });

    return userItems;
  }

  public static double[] getRandomArray(Random rand, int size) {
    double[] arr = new double[size];
    for (int i = 0; i < size; i++) {
      arr[i] = rand.nextDouble();
    }
    return arr;
  }

  public static HashMap<Long, double[]> getVectorMap(Random rand, int size,
      int matrixRank) {
    HashMap<Long, double[]> vectorMap = new HashMap<Long, double[]>(size);
    for (long i = 1; i <= size; i++) {
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
    String inputFile = "";
    String separator = "\\t";
    boolean useCPU = false;

    // parse arguments
    if ((args.length > 0) && (args.length >= 5)) {
      blockSize = Integer.parseInt(args[0]);
      gridSize = Integer.parseInt(args[1]);
      matrixRank = Integer.parseInt(args[2]);
      maxIterations = Integer.parseInt(args[3]);
      isDebbuging = Boolean.parseBoolean(args[4]);
      // optional parameters
      if (args.length > 5) {
        useCPU = Boolean.parseBoolean(args[5]);
      }
      if (args.length > 6) {
        inputFile = args[6];
      }
      if (args.length > 7) {
        separator = args[7];
      }
    } else {
      System.out.println("Wrong argument size!");
      System.out.println("    Argument1=blockSize");
      System.out.println("    Argument2=gridSize");
      System.out.println("    Argument3=matrixRank");
      System.out.println("    Argument4=maxIterations");
      System.out.println("    Argument5=debug(true|false=default)");
      System.out.println("    Argument6=useCPU (optional) | default (" + useCPU
          + ")");
      System.out
          .println("    Argument7=inputFile (optional) | MovieLens inputFile");
      System.out.println("    Argument8=Separator (optional) | default '"
          + separator + "' ");
      return;
    }

    // Check if inputFile exists
    if ((!inputFile.isEmpty()) && (!new File(inputFile).exists())) {
      System.out.println("Error: inputFile: " + inputFile + " does not exist!");
      return;
    }

    if ((!useCPU) && (blockSize < matrixRank)) {
      System.err.println("blockSize < matrixRank");
      return;
    }

    // Debug output
    System.out.println("useCPU: " + useCPU);
    if (!useCPU) {
      System.out.println("blockSize: " + blockSize);
      System.out.println("gridSize: " + gridSize);
    }
    System.out.println("matrixRank: " + matrixRank);
    System.out.println("maxIterations: " + maxIterations);
    if (!inputFile.isEmpty()) {
      System.out.println("inputFile: " + inputFile);
      System.out.println("separator: '" + separator + "'");
    }

    // Prepare input
    List<double[]> preferences = null;
    HashMap<Long, double[]> usersMatrix = null;
    HashMap<Long, double[]> itemsMatrix = null;

    if (inputFile.isEmpty()) { // no inputFile

      preferences = getUserItems();
      usersMatrix = getVectorMap(rand, 3, matrixRank);
      itemsMatrix = getVectorMap(rand, 5, matrixRank);

    } else { // parse inputFile

      preferences = new ArrayList<double[]>();
      usersMatrix = new HashMap<Long, double[]>();
      itemsMatrix = new HashMap<Long, double[]>();

      try {

        BufferedReader br = new BufferedReader(new FileReader(inputFile));
        String line;
        while ((line = br.readLine()) != null) {
          String[] values = line.split(separator);
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
    }

    if (!useCPU) {

      // Convert preferences to double[][]
      double[][] userItemMatrix = new double[usersMatrix.size() + 1][itemsMatrix
          .size() + 1];
      System.out.println("userItemMap: length: " + preferences.size());
      for (double[] v : preferences) {
        int userId = (int) v[0];
        int itemId = (int) v[1];
        userItemMatrix[userId][itemId] = v[2];
        if (isDebbuging) {
          System.out.println("userItemMap userId: '" + v[0] + "' itemId: '"
              + v[1] + "' value: '" + v[2]);
        }
      }

      // Convert usersMatrix to double[][]
      double[][] userMatrix = new double[usersMatrix.size() + 1][matrixRank];
      System.out.println("usersMap: length: " + usersMatrix.size());
      Iterator<Entry<Long, double[]>> userIt = usersMatrix.entrySet()
          .iterator();
      while (userIt.hasNext()) {
        Entry<Long, double[]> entry = userIt.next();
        int userId = entry.getKey().intValue();
        double[] vector = entry.getValue();
        for (int i = 0; i < matrixRank; i++) {
          userMatrix[userId][i] = vector[i];
        }
        if (isDebbuging) {
          System.out.println("usersMap userId: '" + userId + " value: '"
              + Arrays.toString(vector));
        }
      }

      // Convert itemsMatrix to double[][]
      double[][] itemMatrix = new double[itemsMatrix.size() + 1][matrixRank];
      System.out.println("itemsMap: length: " + itemsMatrix.size());
      Iterator<Entry<Long, double[]>> itemIt = itemsMatrix.entrySet()
          .iterator();
      while (itemIt.hasNext()) {
        Entry<Long, double[]> entry = itemIt.next();
        int itemId = entry.getKey().intValue();
        double[] vector = entry.getValue();
        for (int i = 0; i < matrixRank; i++) {
          itemMatrix[itemId][i] = vector[i];
        }
        if (isDebbuging) {
          System.out.println("itemsMap itemId: '" + itemId + " value: '"
              + Arrays.toString(vector));
        }
      }

      // Run GPU Kernels
      System.out.println("Run on GPU");
      OnlineCFKernel kernel = new OnlineCFKernel(userItemMatrix, userMatrix,
          itemMatrix, usersMatrix.size(), itemsMatrix.size(), ALPHA,
          matrixRank, maxIterations);

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
        System.out
            .println("    deserial time: " + row.getDeserializationTime());
        System.out.println("    num blocks: " + row.getNumBlocks());
        System.out.println("    num threads: " + row.getNumThreads());
        System.out.println("GPUTime: " + watch.elapsedTimeMillis() + " ms");
      }

      // Debug users
      if (isDebbuging) {
        System.out.println(usersMatrix.size() + " users");
        for (int i = 1; i <= usersMatrix.size(); i++) {
          System.out.println("user: " + i + " vector: "
              + Arrays.toString(kernel.m_usersMatrix[i]));
        }
      }
      // Debug items
      if (isDebbuging) {
        System.out.println(itemsMatrix.size() + " items");
        for (int i = 1; i <= itemsMatrix.size(); i++) {
          System.out.println("item: " + i + " vector: "
              + Arrays.toString(kernel.m_itemsMatrix[i]));
        }
      }

    } else { // run on CPU

      // Debug input
      System.out.println("preferences: length: " + preferences.size());
      if (isDebbuging) {
        for (double[] v : preferences) {
          System.out.println("preferences userId: '" + v[0] + "' itemId: '"
              + v[1] + "' value: '" + v[2]);
        }
      }
      System.out.println("usersMatrix: length: " + usersMatrix.size());
      if (isDebbuging) {
        Iterator<Entry<Long, double[]>> userIt = usersMatrix.entrySet()
            .iterator();
        while (userIt.hasNext()) {
          Entry<Long, double[]> entry = userIt.next();
          long userId = entry.getKey();
          double[] vector = entry.getValue();
          System.out.println("usersMatrix userId: '" + userId + " value: '"
              + Arrays.toString(vector));

        }
      }
      System.out.println("itemsMatrix: length: " + itemsMatrix.size());
      if (isDebbuging) {
        Iterator<Entry<Long, double[]>> itemIt = itemsMatrix.entrySet()
            .iterator();
        while (itemIt.hasNext()) {
          Entry<Long, double[]> entry = itemIt.next();
          long itemId = entry.getKey();
          double[] vector = entry.getValue();
          System.out.println("itemsMatrix itemId: '" + itemId + " value: '"
              + Arrays.toString(vector));
        }
      }

      // Run CPU
      System.out.println("Run on CPU");
      OnlineCF onlineCF = new OnlineCF(preferences, usersMatrix, itemsMatrix,
          ALPHA, matrixRank, maxIterations);

      long startTime = System.currentTimeMillis();
      onlineCF.compute();
      long endTime = System.currentTimeMillis() - startTime;
      System.out.println("GPUTime: " + endTime + " ms");

      // Debug output
      if (isDebbuging) {
        System.out.println(onlineCF.m_usersMatrix.size() + " users");
        Iterator<Entry<Long, double[]>> userIt = onlineCF.m_usersMatrix
            .entrySet().iterator();
        while (userIt.hasNext()) {
          Entry<Long, double[]> entry = userIt.next();
          long userId = entry.getKey();
          double[] vector = entry.getValue();
          System.out.println("usersMatrix userId: '" + userId + " value: '"
              + Arrays.toString(vector));

        }

        System.out.println(onlineCF.m_itemsMatrix.size() + " items");
        Iterator<Entry<Long, double[]>> itemIt = onlineCF.m_itemsMatrix
            .entrySet().iterator();
        while (itemIt.hasNext()) {
          Entry<Long, double[]> entry = itemIt.next();
          long itemId = entry.getKey();
          double[] vector = entry.getValue();
          System.out.println("itemsMatrix itemId: '" + itemId + " value: '"
              + Arrays.toString(vector));
        }
      }
    }

  }
}
