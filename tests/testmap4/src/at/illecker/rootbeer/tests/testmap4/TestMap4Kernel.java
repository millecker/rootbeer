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
package at.illecker.rootbeer.tests.testmap4;

import java.util.List;

import org.trifort.rootbeer.runtime.Context;
import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.Rootbeer;
import org.trifort.rootbeer.runtime.RootbeerGpu;
import org.trifort.rootbeer.runtime.StatsRow;
import org.trifort.rootbeer.runtime.ThreadConfig;
import org.trifort.rootbeer.runtime.util.Stopwatch;

public class TestMap4Kernel implements Kernel {
  // gridSize = amount of blocks and multiprocessors
  public static final int GRID_SIZE = 1; // 14;
  // blockSize = amount of threads
  public static final int BLOCK_SIZE = 2; // 1024;

  private int m_N;
  private GpuIntegerMap m_map;

  public TestMap4Kernel(int n) {
    this.m_N = n;
    this.m_map = new GpuIntegerMap(n);
    for (int i = 0; i < m_N; i++) {
      m_map.put(i, i);
    }
  }

  @Override
  public void gpuMethod() {

    // Global thread 0 updates map
    if (RootbeerGpu.getThreadId() == 0) {
      for (int i = 0; i < m_N; i++) {
        System.out.println("map[" + i + "]: " + m_map.get(i));
        m_map.add(i, i);
        System.out.println("updated map[" + i + "]: " + m_map.get(i));
      }
    }

    // threadfence and sync
    RootbeerGpu.threadfenceSystem();
    RootbeerGpu.syncthreads();

    printElement(RootbeerGpu.getThreadId());
  }

  private synchronized void printElement(int index) {
    System.out.println("index: " + index + " value: " + m_map.get(index));
  }

  public static void main(String[] args) {
    int blockSize = BLOCK_SIZE;
    int gridSize = GRID_SIZE;

    // parse arguments
    if (args.length > 0) {
      if (args.length == 1) {
        blockSize = Integer.parseInt(args[0]);
      } else {
        System.out.println("Wrong argument size!");
        System.out.println("    Argument1=blockSize");
        return;
      }
    }

    System.out.println("blockSize: " + blockSize);
    System.out.println("gridSize: " + gridSize);

    TestMap4Kernel kernel = new TestMap4Kernel(blockSize);

    // Run GPU Kernels
    Rootbeer rootbeer = new Rootbeer();
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
