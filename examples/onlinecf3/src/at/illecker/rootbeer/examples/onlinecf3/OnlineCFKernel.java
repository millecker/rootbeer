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
package at.illecker.rootbeer.examples.onlinecf3;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
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

    // SharedMemory per block (max 12 + 1024 * 8 = 8204 bytes)
    // e.g., maxtrixRank 256 => 12 + 256 * 8 = 2060 bytes
    int shmStartPos = 0;
    // multVector: matrixRank x Doubles (m_matrixRank * 8 bytes)
    int shmMultVectorStartPos = shmStartPos;

    // DEBUG
    // if (RootbeerGpu.getThreadId() == 0) {
    // System.out.println("blockSize: " + blockSize);
    // System.out.println("gridSize: " + gridSize);
    // System.out.println("users(N): " + m_N);
    // System.out.println("items(M): " + m_M);
    // System.out.println("usersPerBlock: " + usersPerBlock);
    // System.out.println("itemsPerBlock: " + itemsPerBlock);
    // }

    // Start OnlineCF algorithm
    for (int i = 0; i < m_maxIterations; i++) {

      // **********************************************************************
      // Compute U (Users)
      // **********************************************************************
      // Loop over all usersPerBlock
      for (int u = 0; u < usersPerBlock; u++) {

        int userId = (usersPerBlock * u) + block_idxx;
        if (userId < m_N) {

          // Each user loops over all items
          for (int itemId = 0; itemId < m_M; itemId++) {

            double expectedScore = m_userItemMatrix[userId][itemId];
            if (expectedScore != 0) {

              // Each thread within a block computes one multiplication
              if (thread_idxx < m_matrixRank) {

                RootbeerGpu.setSharedDouble(shmMultVectorStartPos + thread_idxx
                    * 8, m_usersMatrix[userId][thread_idxx]
                    * m_itemsMatrix[itemId][thread_idxx]);
              }

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

                double calculatedScore = RootbeerGpu
                    .getSharedDouble(shmMultVectorStartPos);

                m_usersMatrix[userId][thread_idxx] += 2 * m_ALPHA
                    * m_itemsMatrix[itemId][thread_idxx]
                    * (expectedScore - calculatedScore);
              }

              // Sync all threads within a block
              RootbeerGpu.syncthreads();

            } // if expectedScore != 0

          } // loop over all items

        } // if userId < m_N

      } // loop over all usersPerBlock

      // Sync all blocks Inter-Block Synchronization
      RootbeerGpu.syncblocks(1);

      // **********************************************************************
      // Compute V (Items)
      // **********************************************************************
      // Loop over all itemsPerBlock
      for (int v = 0; v < itemsPerBlock; v++) {

        int itemId = (itemsPerBlock * v) + block_idxx;
        if (itemId < m_M) {

          // Each user loops over all items
          for (int userId = 0; userId < m_N; userId++) {

            double expectedScore = m_userItemMatrix[userId][itemId];
            if (expectedScore != 0) {

              // Each thread within a block computes one multiplication
              if (thread_idxx < m_matrixRank) {

                RootbeerGpu.setSharedDouble(shmMultVectorStartPos + thread_idxx
                    * 8, m_itemsMatrix[itemId][thread_idxx]
                    * m_usersMatrix[userId][thread_idxx]);
              }

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

                double calculatedScore = RootbeerGpu
                    .getSharedDouble(shmMultVectorStartPos);

                m_itemsMatrix[itemId][thread_idxx] += 2 * m_ALPHA
                    * m_usersMatrix[userId][thread_idxx]
                    * (expectedScore - calculatedScore);
              }

              // Sync all threads within a block
              RootbeerGpu.syncthreads();

            } // if expectedScore != 0

          } // loop over all items

        } // if (itemId < m_M)

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

    userItems.add(new double[] { 2, 1, 4 });
    userItems.add(new double[] { 2, 2, 2.5 });
    userItems.add(new double[] { 2, 3, 3.5 });
    userItems.add(new double[] { 2, 4, 1 });
    userItems.add(new double[] { 2, 5, 3.5 });

    userItems.add(new double[] { 3, 1, 4 });
    userItems.add(new double[] { 3, 2, 2.5 });
    userItems.add(new double[] { 3, 3, 3.5 });
    userItems.add(new double[] { 3, 4, 1 });
    userItems.add(new double[] { 3, 5, 3.5 });

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

  public static <K extends Comparable, V extends Comparable> Map<K, V> sortByKeys(
      Map<K, V> map) {

    List<K> keys = new LinkedList<K>(map.keySet());
    Collections.sort(keys);

    // LinkedHashMap will keep the keys in the order they are inserted
    // which is currently sorted on natural ordering
    Map<K, V> sortedMap = new LinkedHashMap<K, V>();
    for (K key : keys) {
      sortedMap.put(key, map.get(key));
    }

    return sortedMap;
  }

  public static <K extends Comparable, V extends Comparable> Map<K, V> sortByValues(
      Map<K, V> map) {

    List<Map.Entry<K, V>> entries = new LinkedList<Map.Entry<K, V>>(
        map.entrySet());

    Collections.sort(entries, new Comparator<Map.Entry<K, V>>() {
      @Override
      public int compare(Entry<K, V> o1, Entry<K, V> o2) {
        return o2.getValue().compareTo(o1.getValue());
      }
    });

    // LinkedHashMap will keep the keys in the order they are inserted
    // which is currently sorted on natural ordering
    Map<K, V> sortedMap = new LinkedHashMap<K, V>();
    for (Map.Entry<K, V> entry : entries) {
      sortedMap.put(entry.getKey(), entry.getValue());
    }

    return sortedMap;
  }

  public static void main(String[] args) {

    int blockSize = 1024;
    int gridSize = 14;
    boolean isDebbuging = false;
    int debugLines = 10;
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
    Map<Long, HashMap<Long, Double>> preferencesMap = new HashMap<Long, HashMap<Long, Double>>();
    Map<Long, double[]> usersMatrix = null;
    Map<Long, double[]> itemsMatrix = null;

    Map<Long, Long> userRatingCount = new HashMap<Long, Long>();
    Map<Long, Long> itemRatingCount = new HashMap<Long, Long>();

    if (inputFile.isEmpty()) { // no inputFile

      preferences = getUserItems();
      usersMatrix = getVectorMap(rand, 3, matrixRank);
      itemsMatrix = getVectorMap(rand, 5, matrixRank);

      // used on GPU only
      if (!useCPU) {

        for (double[] pref : preferences) {
          long userId = (long) pref[0];
          long itemId = (long) pref[1];
          double rating = pref[2];

          // Add preferencesMap which is used on GPU only
          if (preferencesMap.containsKey(userId) == false) {
            HashMap<Long, Double> map = new HashMap<Long, Double>();
            map.put(itemId, rating);
            preferencesMap.put(userId, map);
          } else {
            preferencesMap.get(userId).put(itemId, rating);
          }

          // Increase userRatingCount which is used on GPU only
          if (userRatingCount.containsKey(userId) == false) {
            userRatingCount.put(userId, 1l);
          } else {
            userRatingCount.put(userId, userRatingCount.get(userId) + 1);
          }

          // Increase itemRatingCount which is used on GPU only
          if (itemRatingCount.containsKey(itemId) == false) {
            itemRatingCount.put(itemId, 1l);
          } else {
            itemRatingCount.put(itemId, itemRatingCount.get(itemId) + 1);
          }
        }

      }

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
            userRatingCount.put(userId, 1l);

          } else {
            userRatingCount.put(userId, userRatingCount.get(userId) + 1);
          }

          // Add Item vector
          if (itemsMatrix.containsKey(itemId) == false) {
            double[] vals = new double[matrixRank];
            for (int i = 0; i < matrixRank; i++) {
              vals[i] = rand.nextDouble();
            }
            itemsMatrix.put(itemId, vals);
            itemRatingCount.put(itemId, 1l);

          } else {
            itemRatingCount.put(itemId, itemRatingCount.get(itemId) + 1);
          }

          // Add preference
          double vector[] = new double[3];
          vector[0] = userId;
          vector[1] = itemId;
          vector[2] = rating;
          preferences.add(vector);

          // Add to preferencesMap which is used on GPU only
          if (preferencesMap.containsKey(userId) == false) {
            HashMap<Long, Double> map = new HashMap<Long, Double>();
            map.put(itemId, rating);
            preferencesMap.put(userId, map);
          } else {
            preferencesMap.get(userId).put(itemId, rating);
          }
        }
        br.close();

      } catch (NumberFormatException e) {
        e.printStackTrace();
      } catch (IOException e) {
        e.printStackTrace();
      }
    }

    if (!useCPU) {

      Map<Long, Long> sortedUserRatingCount = sortByValues(userRatingCount);
      Map<Long, Long> sortedItemRatingCount = sortByValues(itemRatingCount);

      // Convert preferences to double[][]
      double[][] userItemMatrix = new double[usersMatrix.size()][itemsMatrix
          .size()];
      Map<Long, Integer> userItemMatrixUserRowMap = new HashMap<Long, Integer>();
      Map<Long, Integer> userItemMatrixItemColMap = new HashMap<Long, Integer>();

      System.out.println("userItemMatrix: (m x n): " + usersMatrix.size()
          + " x " + itemsMatrix.size());
      int rowId = 0;
      for (Long userId : sortedUserRatingCount.keySet()) {
        userItemMatrixUserRowMap.put(userId, rowId);
        int colId = 0;
        for (Long itemId : sortedItemRatingCount.keySet()) {
          if (rowId == 0) {
            userItemMatrixItemColMap.put(itemId, colId);
          }
          if (preferencesMap.get(userId).containsKey(itemId)) {
            userItemMatrix[rowId][colId] = preferencesMap.get(userId).get(
                itemId);
          }
          colId++;
        }
        if ((isDebbuging) && (rowId < debugLines)) {
          System.out.println("userItemMatrix userId: "
              + userId
              + " row["
              + rowId
              + "]: "
              + Arrays.toString(Arrays.copyOfRange(userItemMatrix[rowId], 0,
                  Math.min(itemsMatrix.size(), debugLines))));
        }
        rowId++;
      }

      // Convert usersMatrix to double[][]
      double[][] userMatrix = new double[usersMatrix.size()][matrixRank];
      System.out.println("userMatrix: length: " + usersMatrix.size());
      rowId = 0;
      for (Long userId : sortedUserRatingCount.keySet()) {
        double[] vector = usersMatrix.get(userId);
        for (int i = 0; i < matrixRank; i++) {
          userMatrix[rowId][i] = vector[i];
        }
        if ((isDebbuging) && (rowId < debugLines)) {
          System.out.println("userMatrix userId: " + userId + " "
              + Arrays.toString(vector));
        }
        rowId++;
      }

      // Convert itemsMatrix to double[][]
      double[][] itemMatrix = new double[itemsMatrix.size()][matrixRank];
      System.out.println("itemMatrix: length: " + itemsMatrix.size());
      rowId = 0;
      for (Long itemId : sortedItemRatingCount.keySet()) {
        double[] vector = itemsMatrix.get(itemId);
        for (int i = 0; i < matrixRank; i++) {
          itemMatrix[rowId][i] = vector[i];
        }
        if ((isDebbuging) && (rowId < debugLines)) {
          System.out.println("itemMatrix itemId: " + itemId + " "
              + Arrays.toString(vector));
        }
        rowId++;
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
        System.out.println("GPU Time: " + watch.elapsedTimeMillis() + " ms");
      }

      // Debug users
      if (isDebbuging) {
        System.out.println(usersMatrix.size() + " users");
        int maxOutput = Math.min(usersMatrix.size(), debugLines);
        for (Map.Entry<Long, Integer> entry : userItemMatrixUserRowMap
            .entrySet()) {
          if (--maxOutput < 0) {
            break;
          }
          System.out.println("userId: " + entry.getKey() + " "
              + Arrays.toString(kernel.m_usersMatrix[entry.getValue()]));
        }
      }
      // Debug items
      if (isDebbuging) {
        System.out.println(itemsMatrix.size() + " items");
        int maxOutput = Math.min(itemsMatrix.size(), debugLines);
        for (Map.Entry<Long, Integer> entry : userItemMatrixItemColMap
            .entrySet()) {
          if (--maxOutput < 0) {
            break;
          }
          System.out.println("itemId: " + entry.getKey() + " "
              + Arrays.toString(kernel.m_itemsMatrix[entry.getValue()]));
        }
      }

      // Test example output
      if ((isDebbuging) && (inputFile.isEmpty())) {
        double score = 0;
        double error = 0;
        // 1, 1, 4
        score = OnlineCF.computeScore(
            kernel.m_usersMatrix[userItemMatrixUserRowMap.get(1l)],
            kernel.m_itemsMatrix[userItemMatrixItemColMap.get(1l)], matrixRank);
        error += Math.abs(4 - score);
        System.out.println("(1, 1, 4): " + score + " error: "
            + Math.abs(4 - score));
        // 1, 2, 2.5
        score = OnlineCF.computeScore(
            kernel.m_usersMatrix[userItemMatrixUserRowMap.get(1l)],
            kernel.m_itemsMatrix[userItemMatrixItemColMap.get(2l)], matrixRank);
        error += Math.abs(2.5 - score);
        System.out.println("(1, 2, 2.5): " + score + " error: "
            + Math.abs(2.5 - score));
        // 1, 3, 3.5
        score = OnlineCF.computeScore(
            kernel.m_usersMatrix[userItemMatrixUserRowMap.get(1l)],
            kernel.m_itemsMatrix[userItemMatrixItemColMap.get(3l)], matrixRank);
        error += Math.abs(3.5 - score);
        System.out.println("(1, 3, 3.5): " + score + " error: "
            + Math.abs(3.5 - score));
        // 1, 4, 1
        score = OnlineCF.computeScore(
            kernel.m_usersMatrix[userItemMatrixUserRowMap.get(1l)],
            kernel.m_itemsMatrix[userItemMatrixItemColMap.get(4l)], matrixRank);
        error += Math.abs(1 - score);
        System.out.println("(1, 4, 1): " + score + " error: "
            + Math.abs(1 - score));
        // 1, 5, 3.5
        score = OnlineCF.computeScore(
            kernel.m_usersMatrix[userItemMatrixUserRowMap.get(1l)],
            kernel.m_itemsMatrix[userItemMatrixItemColMap.get(5l)], matrixRank);
        error += Math.abs(3.5 - score);
        System.out.println("(1, 5, 3.5): " + score + " error: "
            + Math.abs(3.5 - score));
        System.out.println("Total error: " + error);
      }

    } else { // run on CPU

      // Debug input
      System.out.println("preferences: length: " + preferences.size());
      if (isDebbuging) {
        for (int i = 0; i < Math.min(preferences.size(), debugLines); i++) {
          System.out.println("preferences userId: '" + preferences.get(i)[0]
              + "' itemId: '" + preferences.get(i)[1] + "' value: '"
              + preferences.get(i)[2]);
        }
      }
      System.out.println("usersMatrix: length: " + usersMatrix.size());
      if (isDebbuging) {
        int i = 0;
        Iterator<Entry<Long, double[]>> userIt = usersMatrix.entrySet()
            .iterator();
        while ((userIt.hasNext()) && (i < debugLines)) {
          Entry<Long, double[]> entry = userIt.next();
          long userId = entry.getKey();
          double[] vector = entry.getValue();
          System.out.println("usersMatrix userId: '" + userId + " value: '"
              + Arrays.toString(vector));
          i++;
        }
      }
      System.out.println("itemsMatrix: length: " + itemsMatrix.size());
      if (isDebbuging) {
        int i = 0;
        Iterator<Entry<Long, double[]>> itemIt = itemsMatrix.entrySet()
            .iterator();
        while ((itemIt.hasNext()) && (i < debugLines)) {
          Entry<Long, double[]> entry = itemIt.next();
          long itemId = entry.getKey();
          double[] vector = entry.getValue();
          System.out.println("itemsMatrix itemId: '" + itemId + " value: '"
              + Arrays.toString(vector));
          i++;
        }
      }

      // Run CPU
      System.out.println("Run on CPU");
      OnlineCF onlineCF = new OnlineCF(preferences, usersMatrix, itemsMatrix,
          ALPHA, matrixRank, maxIterations);

      long startTime = System.currentTimeMillis();
      onlineCF.compute();
      long endTime = System.currentTimeMillis() - startTime;
      System.out.println("CPU Time: " + endTime + " ms");

      // Debug output
      if (isDebbuging) {
        System.out.println(onlineCF.m_usersMatrix.size() + " users");
        int i = 0;
        Iterator<Entry<Long, double[]>> userIt = onlineCF.m_usersMatrix
            .entrySet().iterator();
        while ((userIt.hasNext()) && (i < debugLines)) {
          Entry<Long, double[]> entry = userIt.next();
          long userId = entry.getKey();
          double[] vector = entry.getValue();
          System.out.println("usersMatrix userId: '" + userId + " value: '"
              + Arrays.toString(vector));
          i++;
        }

        System.out.println(onlineCF.m_itemsMatrix.size() + " items");
        i = 0;
        Iterator<Entry<Long, double[]>> itemIt = onlineCF.m_itemsMatrix
            .entrySet().iterator();
        while ((itemIt.hasNext()) && (i < debugLines)) {
          Entry<Long, double[]> entry = itemIt.next();
          long itemId = entry.getKey();
          double[] vector = entry.getValue();
          System.out.println("itemsMatrix itemId: '" + itemId + " value: '"
              + Arrays.toString(vector));
          i++;
        }
      }

      // Test example output
      if ((isDebbuging) && (inputFile.isEmpty())) {
        double score = 0;
        double error = 0;
        // 1, 1, 4
        score = OnlineCF.computeScore(onlineCF.m_usersMatrix.get(1l),
            onlineCF.m_itemsMatrix.get(1l), matrixRank);
        error += Math.abs(4 - score);
        System.out.println("(1, 1, 4): " + score + " error: "
            + Math.abs(4 - score));
        // 1, 2, 2.5
        score = OnlineCF.computeScore(onlineCF.m_usersMatrix.get(1l),
            onlineCF.m_itemsMatrix.get(2l), matrixRank);
        error += Math.abs(2.5 - score);
        System.out.println("(1, 2, 2.5): " + score + " error: "
            + Math.abs(2.5 - score));
        // 1, 3, 3.5
        score = OnlineCF.computeScore(onlineCF.m_usersMatrix.get(1l),
            onlineCF.m_itemsMatrix.get(3l), matrixRank);
        error += Math.abs(3.5 - score);
        System.out.println("(1, 3, 3.5): " + score + " error: "
            + Math.abs(3.5 - score));
        // 1, 4, 1
        score = OnlineCF.computeScore(onlineCF.m_usersMatrix.get(1l),
            onlineCF.m_itemsMatrix.get(4l), matrixRank);
        error += Math.abs(1 - score);
        System.out.println("(1, 4, 1): " + score + " error: "
            + Math.abs(1 - score));
        // 1, 5, 3.5
        score = OnlineCF.computeScore(onlineCF.m_usersMatrix.get(1l),
            onlineCF.m_itemsMatrix.get(5l), matrixRank);
        error += Math.abs(3.5 - score);
        System.out.println("(1, 5, 3.5): " + score + " error: "
            + Math.abs(3.5 - score));
        System.out.println("Total error: " + error);
      }
    } // run on CPU

  }

}
