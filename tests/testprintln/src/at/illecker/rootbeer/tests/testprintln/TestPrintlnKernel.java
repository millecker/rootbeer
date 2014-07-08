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
package at.illecker.rootbeer.tests.testprintln;

import java.util.List;
import java.util.Random;

import org.trifort.rootbeer.runtime.Context;
import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.Rootbeer;
import org.trifort.rootbeer.runtime.RootbeerGpu;
import org.trifort.rootbeer.runtime.StatsRow;
import org.trifort.rootbeer.runtime.ThreadConfig;
import org.trifort.rootbeer.runtime.util.Stopwatch;

public class TestPrintlnKernel implements Kernel {
  // gridSize = amount of blocks and multiprocessors
  public static final int GRID_SIZE = 14;
  // blockSize = amount of threads
  public static final int BLOCK_SIZE = 256;

  private double[][] m_big_array1;
  private double[][] m_array2;
  private double[][] m_array3;

  public TestPrintlnKernel(double[][] big_array1, double[][] array2,
      double[][] array3) {
    this.m_big_array1 = big_array1;
    this.m_array2 = array2;
    this.m_array3 = array3;
  }

  @Override
  public void gpuMethod() {
    if (RootbeerGpu.getThreadId() == 0) {
      System.out.println("I will NOT throw java.lang.RuntimeException");

      int x = 0;
      // TODO ERROR occurs here
      StringBuilder sb = new StringBuilder("BUT I will throw it. ");
      // sb.append(x);
      // System.out.println(sb.toString());

      // Same as:
      // System.out.println("BUT I will throw it. " + x);
    }

    // Some dummy calculations on arrays
    int block_idxx = RootbeerGpu.getBlockIdxx();
    int thread_idxx = RootbeerGpu.getThreadIdxx();
    double value = m_big_array1[block_idxx][thread_idxx];
    if (value != 0) {
      m_array2[0][thread_idxx] += m_array3[0][thread_idxx] * value;
    }
  }

  public static void main(String[] args) {
    int blockSize = BLOCK_SIZE;
    int gridSize = GRID_SIZE;
    Random rand = new Random();
    int N = 16500;
    int M = 1000; // TODO java.lang.OutOfMemoryError when M=1500

    // parse arguments
    if ((args.length > 0) && (args.length == 2)) {
      blockSize = Integer.parseInt(args[0]);
      gridSize = Integer.parseInt(args[1]);
    } else {
      System.out.println("Wrong argument size!");
      System.out.println("    Argument1=blockSize");
      System.out.println("    Argument2=gridSize");
      return;
    }

    System.out.println("blockSize: " + blockSize);
    System.out.println("gridSize: " + gridSize);

    // Prepare arrays
    System.out.println("big_array: double[" + N + "][" + M + "]");
    double[][] big_array = new double[N][M];
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < M; j++) {
        int choice = rand.nextInt(2);
        if (choice == 0) { // 50% chance
          big_array[i][j] = rand.nextDouble();
        }
      }
    }
    System.out.println("array2: double[" + N + "][" + blockSize + "]");
    double[][] array2 = new double[N][blockSize];
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < blockSize; j++) {
        array2[i][j] = rand.nextDouble();
      }
    }
    System.out.println("array2: double[" + M + "][" + blockSize + "]");
    double[][] array3 = new double[M][blockSize];
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < blockSize; j++) {
        array3[i][j] = rand.nextDouble();
      }
    }

    // Run GPU Kernels
    Rootbeer rootbeer = new Rootbeer();
    TestPrintlnKernel kernel = new TestPrintlnKernel(big_array, array2, array3);
    Context context = rootbeer.createDefaultContext();
    Stopwatch watch = new Stopwatch();
    watch.start();
    rootbeer.run(kernel, new ThreadConfig(blockSize, gridSize, blockSize
        * gridSize), context);
    watch.stop();

    // Logging
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
  }

}
