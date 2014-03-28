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
package at.illecker.rootbeer.examples.onlinecf5;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

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
  private int[][] m_userHelper;
  private int[][] m_itemHelper;
  private double[][] m_usersMatrix;
  private double[][] m_itemsMatrix;
  private int m_N;
  private int m_M;
  private double m_ALPHA;
  private int m_matrixRank;
  private int m_maxIterations;

  public OnlineCFKernel(double[][] userItemMatrix, int[][] userHelper,
      int[][] itemHelper, double[][] usersMatrix, double[][] itemsMatrix,
      int n, int m, double alpha, int matrixRank, int maxIterations) {
    this.m_userItemMatrix = userItemMatrix;
    this.m_userHelper = userHelper;
    this.m_itemHelper = itemHelper;
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
      return;
    }

    int usersPerBlock = divup(m_N, gridSize);
    int itemsPerBlock = divup(m_M, gridSize);
    int reductionStart = roundUpToNextPowerOfTwo(divup(m_matrixRank, 2));

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

        int userId = (gridSize * u) + block_idxx;
        if (userId < m_N) {

          // Each user loops over all items which have a rating
          for (int itemId = 0; itemId < m_userHelper[userId][0]; itemId++) {

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
            for (int s = reductionStart; s > 0; s >>= 1) {

              if ((thread_idxx < s) && (thread_idxx + s) < m_matrixRank) {
                // sh_mem[tid] += sh_mem[tid + s];
                RootbeerGpu.setSharedDouble(
                    shmMultVectorStartPos + thread_idxx * 8,
                    RootbeerGpu.getSharedDouble(shmMultVectorStartPos
                        + thread_idxx * 8)
                        + RootbeerGpu.getSharedDouble(shmMultVectorStartPos
                            + (thread_idxx + s) * 8));
              }

              // Sync all threads within a block
              RootbeerGpu.syncthreads();
            }

            // Calculate new userVector
            // Each thread does one update operation of vector u
            if (thread_idxx < m_matrixRank) {
              m_usersMatrix[userId][thread_idxx] += m_itemsMatrix[itemId][thread_idxx]
                  * 2
                  * m_ALPHA
                  * (m_userItemMatrix[userId][m_userHelper[userId][itemId]] - RootbeerGpu
                      .getSharedDouble(shmMultVectorStartPos));
            }

            // Sync all threads within a block
            RootbeerGpu.syncthreads();

          } // loop over all items which have a rating

        } // if userId < m_N

      } // loop over all usersPerBlock

      // Sync all blocks Inter-Block Synchronization
      RootbeerGpu.syncblocks(1);

      // **********************************************************************
      // Compute V (Items)
      // **********************************************************************
      // Loop over all itemsPerBlock
      for (int v = 0; v < itemsPerBlock; v++) {

        int itemId = (gridSize * v) + block_idxx;
        if (itemId < m_M) {

          // Each user loops over all users which have a rating
          for (int userId = 0; userId < m_itemHelper[itemId][0]; userId++) {

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
            for (int s = reductionStart; s > 0; s >>= 1) {

              if ((thread_idxx < s) && (thread_idxx + s) < m_matrixRank) {
                // sh_mem[tid] += sh_mem[tid + s];
                RootbeerGpu.setSharedDouble(
                    shmMultVectorStartPos + thread_idxx * 8,
                    RootbeerGpu.getSharedDouble(shmMultVectorStartPos
                        + thread_idxx * 8)
                        + RootbeerGpu.getSharedDouble(shmMultVectorStartPos
                            + (thread_idxx + s) * 8));
              }

              // Sync all threads within a block
              RootbeerGpu.syncthreads();
            }

            // Calculate new userVector
            // Each thread does one update operation of vector u
            if (thread_idxx < m_matrixRank) {
              m_itemsMatrix[itemId][thread_idxx] += m_usersMatrix[userId][thread_idxx]
                  * 2
                  * m_ALPHA
                  * (m_userItemMatrix[userId][m_itemHelper[itemId][userId]] - RootbeerGpu
                      .getSharedDouble(shmMultVectorStartPos));
            }

            // Sync all threads within a block
            RootbeerGpu.syncthreads();

          } // loop over all users which have a rating

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

  private int roundUpToNextPowerOfTwo(int x) {
    x--;
    x |= x >> 1; // handle 2 bit numbers
    x |= x >> 2; // handle 4 bit numbers
    x |= x >> 4; // handle 8 bit numbers
    x |= x >> 8; // handle 16 bit numbers
    x |= x >> 16; // handle 32 bit numbers
    x++;
    return x;
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

  // **********************************************************************
  // Generate input
  // **********************************************************************
  public static List<double[]> getUserItems() {
    List<double[]> userItems = new ArrayList<double[]>();
    userItems.add(new double[] { 0, 0, 4 });
    userItems.add(new double[] { 0, 1, 2.5 });
    userItems.add(new double[] { 0, 2, 3.5 });

    userItems.add(new double[] { 1, 0, 4 });
    userItems.add(new double[] { 1, 1, 2.5 });
    userItems.add(new double[] { 1, 2, 3.5 });
    userItems.add(new double[] { 1, 3, 1 });
    userItems.add(new double[] { 1, 4, 3.5 });

    userItems.add(new double[] { 2, 0, 4 });
    userItems.add(new double[] { 2, 1, 2.5 });
    userItems.add(new double[] { 2, 2, 3.5 });
    userItems.add(new double[] { 2, 3, 1 });
    userItems.add(new double[] { 2, 4, 3.5 });

    return userItems;
  }

  public static List<double[]> getTestUserItems() {
    List<double[]> testUserItems = new ArrayList<double[]>();
    testUserItems.add(new double[] { 0, 0, 4 });
    testUserItems.add(new double[] { 0, 1, 2.5 });
    testUserItems.add(new double[] { 0, 2, 3.5 });
    testUserItems.add(new double[] { 0, 3, 1 });
    testUserItems.add(new double[] { 0, 4, 3.5 });

    return testUserItems;
  }

  public static List<double[]> getRandomUserItems(Random rand, int userCount,
      int itemCount, int percentNonZeroValues) {

    List<double[]> userItems = new ArrayList<double[]>();
    int possibleUserItemRatings = userCount * itemCount;
    int userItemRatings = possibleUserItemRatings * percentNonZeroValues / 100;
    System.out.println("possibleRatings: " + possibleUserItemRatings
        + " ratings: " + userItemRatings);
    Set<Map.Entry<Integer, Integer>> userItemPairs = new HashSet<Map.Entry<Integer, Integer>>();

    for (int i = 0; i < userItemRatings; i++) {

      Map.Entry<Integer, Integer> userItemPair;
      do {
        int userId = rand.nextInt(userCount);
        int itemId = rand.nextInt(itemCount);
        userItemPair = new AbstractMap.SimpleImmutableEntry<Integer, Integer>(
            userId, itemId);
      } while (userItemPairs.contains(userItemPair));

      userItemPairs.add(userItemPair);
      userItems.add(new double[] { userItemPair.getKey(),
          userItemPair.getValue(), (rand.nextInt(5) + 1) });
    }
    return userItems;
  }

  public static HashMap<Long, double[]> getVectorMap(Random rand, int size,
      int matrixRank) {
    HashMap<Long, double[]> vectorMap = new HashMap<Long, double[]>(size);
    for (long i = 0; i < size; i++) {
      vectorMap.put(i, getRandomArray(rand, matrixRank));
    }
    return vectorMap;
  }

  public static double[] getRandomArray(Random rand, int size) {
    double[] arr = new double[size];
    for (int i = 0; i < size; i++) {
      arr[i] = rand.nextDouble();
    }
    return arr;
  }

  // **********************************************************************
  // Sort Map by keys
  // **********************************************************************
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

  // **********************************************************************
  // Sort Map by values
  // **********************************************************************
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

  // **********************************************************************
  // Main methods
  // **********************************************************************
  public static void main(String[] args) {
    Random rand = new Random(32L);
    boolean isDebbuging = false;
    int debugLines = 10;

    int blockSize = 1024;
    int gridSize = 14;

    int matrixRank = 3;
    int maxIterations = 1;
    double ALPHA = 0.001;
    int userCount = 0;
    int itemCount = 0;
    int percentNonZeroValues = 50;

    String inputFile = "";
    String separator = "\\t";

    boolean useCPU = false;
    boolean cpuEmulatesGPU = false;

    // **********************************************************************
    // Parse command line arguments
    // **********************************************************************
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
        cpuEmulatesGPU = Boolean.parseBoolean(args[6]);
      }
      if (args.length > 7) {
        double alpha = Double.parseDouble(args[7]);
        if (alpha > 0) {
          ALPHA = alpha;
        }
      }
      if (args.length > 8) {
        userCount = Integer.parseInt(args[8]);
      }
      if (args.length > 9) {
        itemCount = Integer.parseInt(args[9]);
      }
      if (args.length > 10) {
        int percent = Integer.parseInt(args[10]);
        if ((percent > 0) && (percent <= 100)) {
          percentNonZeroValues = percent;
        }
      }
      if (args.length > 11) {
        inputFile = args[11];
      }
      if (args.length > 12) {
        separator = args[12];
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
      System.out.println("    Argument7=CPUemulatesGPU (optional) | default ("
          + cpuEmulatesGPU + ")");
      System.out.println("    Argument8=ALPHA (optional) | default (" + ALPHA
          + ")");
      System.out.println("    Argument9=userCount (optional) | default ("
          + userCount + ")");
      System.out.println("    Argument10=itemCount (optional) | default ("
          + itemCount + ")");
      System.out
          .println("    Argument11=percentNonZeroValues (optional) | default ("
              + percentNonZeroValues + "%)");
      System.out
          .println("    Argument12=inputFile (optional) | MovieLens inputFile");
      System.out.println("    Argument13=separator (optional) | default '"
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

    // **********************************************************************
    // Debug infos
    // **********************************************************************
    System.out.println("useCPU: " + useCPU);
    if (!useCPU) {
      System.out.println("blockSize: " + blockSize);
      System.out.println("gridSize: " + gridSize);
    } else {
      System.out.println("cpuEmulatesGPU: " + cpuEmulatesGPU);
    }
    System.out.println("matrixRank: " + matrixRank);
    System.out.println("maxIterations: " + maxIterations);
    System.out.println("ALPHA: " + ALPHA);
    if (inputFile.isEmpty()) {
      if (userCount > 0) {
        System.out.println("userCount: " + userCount);
      }
      if (itemCount > 0) {
        System.out.println("itemCount: " + itemCount);
      }
      System.out.println("percentNonZeroValues: " + percentNonZeroValues + "%");
    } else {
      System.out.println("inputFile: " + inputFile);
      System.out.println("separator: '" + separator + "'");
    }

    // **********************************************************************
    // Prepare input
    // **********************************************************************
    List<double[]> preferences = null;
    List<double[]> testPreferences = null;
    Map<Long, double[]> usersMatrix = null;
    Map<Long, double[]> itemsMatrix = null;
    Map<Long, HashMap<Long, Double>> preferencesMap = new HashMap<Long, HashMap<Long, Double>>();
    Map<Long, Long> userRatingCount = new HashMap<Long, Long>();
    Map<Long, Long> itemRatingCount = new HashMap<Long, Long>();

    if (inputFile.isEmpty()) { // no inputFile

      if ((userCount > 0) && (itemCount > 0)) {
        preferences = getRandomUserItems(rand, userCount, itemCount,
            percentNonZeroValues);
        testPreferences = preferences;
        usersMatrix = getVectorMap(rand, userCount, matrixRank);
        itemsMatrix = getVectorMap(rand, itemCount, matrixRank);
      } else {
        preferences = getUserItems();
        testPreferences = getTestUserItems();
        usersMatrix = getVectorMap(rand, 3, matrixRank);
        itemsMatrix = getVectorMap(rand, 5, matrixRank);
      }

      // Build preferencesMap, userRatingCount and itemRatingCount
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

        // Set testPreferences to all preferences
        testPreferences = preferences;

        // Debug infos
        double totalRatings = usersMatrix.size() * itemsMatrix.size();
        double nonZeroValues = (preferences.size() / totalRatings) * 100;
        System.out.println("ratings: " + preferences.size()
            + " possibleRatings: " + (long) totalRatings);
        System.out.print("percentNonZeroValues: ");
        System.out.format("%.2f%% %n", nonZeroValues);

      } catch (NumberFormatException e) {
        e.printStackTrace();
      } catch (IOException e) {
        e.printStackTrace();
      }
    }

    // **********************************************************************
    // Run application
    // **********************************************************************
    if (!useCPU) { // use GPU

      Map<Long, Long> sortedUserRatingCount = sortByValues(userRatingCount);
      Map<Long, Long> sortedItemRatingCount = sortByValues(itemRatingCount);

      // Convert preferences to userItemMatrix double[][]
      // TODO Skip zero rows and cols
      // sortedUserRatingCount.size() x sortedItemRatingCount.size()
      System.out.println("userItemMatrix: (m x n): " + usersMatrix.size()
          + " x " + itemsMatrix.size());
      double[][] userItemMatrix = new double[usersMatrix.size()][itemsMatrix
          .size()];
      Map<Long, Integer> userItemMatrixUserRowMap = new HashMap<Long, Integer>();
      Map<Long, Integer> userItemMatrixItemColMap = new HashMap<Long, Integer>();
      // Create userHelper to int[][]
      // userHelper[userId][0] = userRatingCount
      // userHelper[userId][1] = colId of userItemMatrix
      int[][] userHelper = null;
      // Create itemHelper to int[][]
      // itemHelper[itemId][0] = itemRatingCount
      // itemHelper[userId][1] = rowId of userItemMatrix
      int[][] itemHelper = null;
      Map<Long, Integer> itemHelperId = new HashMap<Long, Integer>();

      int rowId = 0;
      for (Long userId : sortedUserRatingCount.keySet()) {

        // Map userId to rowId in userItemMatrixUserRowMap
        userItemMatrixUserRowMap.put(userId, rowId);

        // Setup userHelper
        if (userHelper == null) {
          // TODO sortedUserRatingCount.size()
          userHelper = new int[usersMatrix.size()][sortedUserRatingCount.get(
              userId).intValue() + 1];
        }
        userHelper[rowId][0] = sortedUserRatingCount.get(userId).intValue();

        int colId = 0;
        int userHelperId = 1;
        for (Long itemId : sortedItemRatingCount.keySet()) {

          // Map itemId to colId in userItemMatrixItemColMap
          if (rowId == 0) {
            userItemMatrixItemColMap.put(itemId, colId);
          }

          // Setup itemHelper
          if (itemHelper == null) {
            // TODO sortedItemRatingCount.size()
            itemHelper = new int[itemsMatrix.size()][sortedItemRatingCount.get(
                itemId).intValue() + 1];
          }
          itemHelper[colId][0] = sortedItemRatingCount.get(itemId).intValue();

          if (preferencesMap.get(userId).containsKey(itemId)) {
            // Add userItemMatrix
            userItemMatrix[rowId][colId] = preferencesMap.get(userId).get(
                itemId);

            // Add userHelper
            userHelper[rowId][userHelperId] = colId;
            userHelperId++;

            // Add itemHelper
            if (itemHelperId.containsKey(itemId)) {
              int idx = itemHelperId.get(itemId);
              itemHelper[colId][idx] = rowId;
              itemHelperId.put(itemId, idx + 1);
            } else {
              itemHelper[colId][1] = rowId;
              itemHelperId.put(itemId, 2);
            }

          }

          colId++;
        }

        // Debug userItemMatrix
        if ((isDebbuging) && (rowId < debugLines)) {
          System.out.println("userItemMatrix userId: " + userId + " row["
              + rowId + "]: " + Arrays.toString(userItemMatrix[rowId])
              + " userRatings: " + sortedUserRatingCount.get(userId));
        }
        rowId++;
      }

      if (isDebbuging) {
        // Debug userHelper
        // TODO sortedUserRatingCount.size()
        for (int i = 0; i < Math.min(usersMatrix.size(), debugLines); i++) {
          System.out.println("userHelper row " + i + ": "
              + Arrays.toString(userHelper[i]));
        }
        // Debug itemHelper
        // TODO sortedItemRatingCount.size()
        for (int i = 0; i < Math.min(itemsMatrix.size(), debugLines); i++) {
          System.out.println("itemHelper row " + i + ": "
              + Arrays.toString(itemHelper[i]));
        }
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
          System.out.println("userId: " + userId + " "
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
          System.out.println("itemId: " + itemId + " "
              + Arrays.toString(vector));
        }
        rowId++;
      }

      // Run GPU Kernels
      System.out.println("Run on GPU");
      OnlineCFKernel kernel = new OnlineCFKernel(userItemMatrix, userHelper,
          itemHelper, userMatrix, itemMatrix, usersMatrix.size(),
          itemsMatrix.size(), ALPHA, matrixRank, maxIterations);

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
      double totalError = 0;
      int i = 0;
      for (double[] testPref : testPreferences) {
        long userId = (long) testPref[0];
        long itemId = (long) testPref[1];
        double expectedScore = testPref[2];

        double score = OnlineCF.computeScore(
            kernel.m_usersMatrix[userItemMatrixUserRowMap.get(userId)],
            kernel.m_itemsMatrix[userItemMatrixItemColMap.get(itemId)],
            matrixRank);
        totalError += Math.abs(expectedScore - score);
        if (++i < debugLines) {
          System.out.println("(" + userId + ", " + itemId + ", "
              + expectedScore + "): " + score + " error: "
              + Math.abs(expectedScore - score));
        }
      }
      System.out.println("Total error: " + totalError + " avgError: "
          + (totalError / testPreferences.size()));

    } else { // run on CPU

      Map<Long, Long> sortedUserRatingCount = null;
      Map<Long, Long> sortedItemRatingCount = null;

      if (cpuEmulatesGPU) {
        // Reorder preferences
        sortedUserRatingCount = sortByValues(userRatingCount);
        sortedItemRatingCount = sortByValues(itemRatingCount);

        List<double[]> reorderedPreferences = new ArrayList<double[]>();
        for (Long userId : sortedUserRatingCount.keySet()) {
          for (Long itemId : sortedItemRatingCount.keySet()) {
            if (preferencesMap.get(userId).containsKey(itemId)) {
              reorderedPreferences.add(new double[] { userId, itemId,
                  preferencesMap.get(userId).get(itemId) });
            }
          }
        }
        preferences = reorderedPreferences;
      }

      // Debug input
      System.out.println("preferences: length: " + preferences.size());
      if (isDebbuging) {
        for (int i = 0; i < Math.min(preferences.size(), debugLines); i++) {
          System.out.println("(" + (long) preferences.get(i)[0] + ", "
              + (long) preferences.get(i)[1] + ", " + preferences.get(i)[2]
              + ")");
        }
      }

      System.out.println("usersMatrix: length: " + usersMatrix.size());
      if (isDebbuging) {
        int i = 0;
        if (cpuEmulatesGPU) {
          for (Long userId : sortedUserRatingCount.keySet()) {
            System.out.println("userId: " + userId + " value: "
                + Arrays.toString(usersMatrix.get(userId)));
            if (++i >= debugLines) {
              break;
            }
          }
        } else {
          Iterator<Entry<Long, double[]>> userIt = usersMatrix.entrySet()
              .iterator();
          while ((userIt.hasNext()) && (i < debugLines)) {
            Entry<Long, double[]> entry = userIt.next();
            System.out.println("userId: " + entry.getKey() + " value: "
                + Arrays.toString(entry.getValue()));
            i++;
          }
        }
      }
      System.out.println("itemsMatrix: length: " + itemsMatrix.size());
      if (isDebbuging) {
        int i = 0;
        if (cpuEmulatesGPU) {
          for (Long itemId : sortedItemRatingCount.keySet()) {
            System.out.println("itemId: " + itemId + " value: "
                + Arrays.toString(itemsMatrix.get(itemId)));
            if (++i >= debugLines) {
              break;
            }
          }
        } else {
          Iterator<Entry<Long, double[]>> itemIt = itemsMatrix.entrySet()
              .iterator();
          while ((itemIt.hasNext()) && (i < debugLines)) {
            Entry<Long, double[]> entry = itemIt.next();
            System.out.println("itemId: " + entry.getKey() + " value: "
                + Arrays.toString(entry.getValue()));
            i++;
          }
        }
      }

      // Run CPU
      System.out.println("Run on CPU");
      OnlineCF onlineCF = new OnlineCF(preferences, usersMatrix, itemsMatrix,
          ALPHA, matrixRank, maxIterations);

      long startTime = System.currentTimeMillis();
      onlineCF.compute(cpuEmulatesGPU);
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
          System.out.println("userId: " + userId + " value: "
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
          System.out.println("itemId: " + itemId + " value: "
              + Arrays.toString(vector));
          i++;
        }
      }

      // Test example output
      double totalError = 0;
      int i = 0;
      for (double[] testPref : testPreferences) {
        long userId = (long) testPref[0];
        long itemId = (long) testPref[1];
        double expectedScore = testPref[2];

        double score = OnlineCF.computeScore(
            onlineCF.m_usersMatrix.get(userId),
            onlineCF.m_itemsMatrix.get(itemId), matrixRank);
        totalError += Math.abs(expectedScore - score);
        if (++i < debugLines) {
          System.out.println("(" + userId + ", " + itemId + ", "
              + expectedScore + "): " + score + " error: "
              + Math.abs(expectedScore - score));
        }
      }
      System.out.println("Total error: " + totalError + " avgError: "
          + (totalError / testPreferences.size()));

    } // run on CPU

  }

}
