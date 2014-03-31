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
package at.illecker.rootbeer.examples.testsyncblocks;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.trifort.rootbeer.runtime.Context;
import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.Rootbeer;
import org.trifort.rootbeer.runtime.RootbeerGpu;
import org.trifort.rootbeer.runtime.StatsRow;
import org.trifort.rootbeer.runtime.ThreadConfig;
import org.trifort.rootbeer.runtime.util.Stopwatch;

public class TestSyncBlocks implements Kernel {
  // gridSize = amount of blocks and multiprocessors
  public static final int GRID_SIZE = 10;
  // blockSize = amount of threads
  public static final int BLOCK_SIZE = 10;

  private int[] m_array;

  public TestSyncBlocks(int[] array) {
    this.m_array = array;
  }

  @Override
  public void gpuMethod() {
    int thread_idxx = RootbeerGpu.getThreadIdxx();
    int block_idxx = RootbeerGpu.getBlockIdxx();

    if (thread_idxx == 0) {
      RootbeerGpu.setSharedInteger(0, m_array[block_idxx]);
      m_array[block_idxx] = block_idxx;
    }

    RootbeerGpu.syncblocks(1);

    if (thread_idxx == 0) {
      m_array[RootbeerGpu.getSharedInteger(0)]++;
    }
  }

  public static void main(String[] args) {

    int blockSize = BLOCK_SIZE;
    int gridSize = GRID_SIZE;

    // parse arguments
    if ((args.length > 0) && (args.length == 1)) {
      gridSize = Integer.parseInt(args[0]);
      blockSize = gridSize;
    } else {
      System.out.println("Wrong argument size!");
      System.out.println("    Argument1=gridSize");
      return;
    }

    System.out.println("blockSize: " + blockSize);
    System.out.println("gridSize: " + gridSize);

    boolean isDebugging = (gridSize < 20);

    // Prepare array
    int[] array = new int[gridSize];
    for (int i = 0; i < gridSize; i++) {
      array[i] = i;
    }
    // Shuffling indexes
    Random rand = new Random();
    for (int i = gridSize; i > 0; i--) {
      int index = Math.abs(rand.nextInt()) % i;
      int tmp = array[i - 1];
      array[i - 1] = array[index];
      array[index] = tmp;
    }
    // Debug
    if (isDebugging) {
      System.out.println("input: " + Arrays.toString(array));
    }

    // Run GPU Kernels
    Rootbeer rootbeer = new Rootbeer();
    TestSyncBlocks kernel = new TestSyncBlocks(array);
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

    // Debug
    if (isDebugging) {
      System.out.println("output: " + Arrays.toString(kernel.m_array));
    }

    // Verify
    boolean verified = true;
    for (int i = 0; i < gridSize; i++) {
      if (!verified) {
        break;
      }
      if (kernel.m_array[i] != (i + 1)) {
        System.out.println("Error at position: " + i + " value: "
            + kernel.m_array[i]);
        verified = false;
        break;
      }
    }
    if (verified) {
      System.out.println("Data verified!");
    } else {
      System.out.println("Error in verification!");
    }
  }

}
