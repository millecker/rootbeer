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
package at.illecker.rootbeer.tests.testmap1;

import java.util.List;

import org.trifort.rootbeer.runtime.Context;
import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.Rootbeer;
import org.trifort.rootbeer.runtime.StatsRow;
import org.trifort.rootbeer.runtime.ThreadConfig;
import org.trifort.rootbeer.runtime.util.Stopwatch;

public class TestMap1Kernel implements Kernel {
  // gridSize = amount of blocks and multiprocessors
  public static final int GRID_SIZE = 1; // 14;
  // blockSize = amount of threads
  public static final int BLOCK_SIZE = 1; // 1024;

  private int m_N;
  private GpuVectorMap m_map;

  public TestMap1Kernel(int n) {
    this.m_N = n;
    this.m_map = new GpuVectorMap(n);
  }

  @Override
  public void gpuMethod() {

    double[] vector1 = new double[m_N];
    for (int i = 0; i < m_N; i++) {
      vector1[i] = i;
    }
    m_map.put(0, vector1);

    double[] vector2 = m_map.get(0);
    System.out.println("vector.len: " + vector2.length);
    // System.out.println("vector: '" + arrayToString(vector2) + "'");
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

  public static void main(String[] args) {

    int blockSize = BLOCK_SIZE;
    int gridSize = GRID_SIZE;

    // parse arguments
    if (args.length > 0) {

      if (args.length == 2) {
        blockSize = Integer.parseInt(args[0]);
        gridSize = Integer.parseInt(args[1]);
      } else {
        System.out.println("Wrong argument size!");
        System.out.println("    Argument1=blockSize");
        System.out.println("    Argument2=gridSize");
        return;
      }
    }

    System.out.println("blockSize: " + blockSize);
    System.out.println("gridSize: " + gridSize);

    Rootbeer rootbeer = new Rootbeer();
    TestMap1Kernel kernel = new TestMap1Kernel(GRID_SIZE);

    // Run GPU Kernels
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
