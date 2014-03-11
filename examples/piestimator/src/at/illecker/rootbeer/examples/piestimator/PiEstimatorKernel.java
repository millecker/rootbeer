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
package at.illecker.rootbeer.examples.piestimator;

import java.util.List;

import org.trifort.rootbeer.runtime.Context;
import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.Rootbeer;
import org.trifort.rootbeer.runtime.RootbeerGpu;
import org.trifort.rootbeer.runtime.StatsRow;
import org.trifort.rootbeer.runtime.ThreadConfig;
import org.trifort.rootbeer.runtime.util.Stopwatch;

public class PiEstimatorKernel implements Kernel {

  private long m_iterations;
  private long m_seed;
  public ResultList m_resultList;

  public PiEstimatorKernel(long iterations, long seed) {
    this.m_iterations = iterations;
    this.m_seed = seed;
    this.m_resultList = new ResultList();
  }

  public void gpuMethod() {

    int thread_idxx = RootbeerGpu.getThreadIdxx();
    LinearCongruentialRandomGenerator lcg = new LinearCongruentialRandomGenerator(
        m_seed / RootbeerGpu.getThreadId());

    long hits = 0;
    for (int i = 0; i < m_iterations; i++) {
      double x = 2.0 * lcg.nextDouble() - 1.0; // value between -1 and 1
      double y = 2.0 * lcg.nextDouble() - 1.0; // value between -1 and 1

      if ((Math.sqrt(x * x + y * y) < 1.0)) {
        hits++;
      }
    }

    // write hits to shared memory
    RootbeerGpu.setSharedLong(thread_idxx * 8, hits);
    RootbeerGpu.syncthreads();

    // do reduction in shared memory
    // 1-bit right shift = divide by two to the power 1
    for (int s = RootbeerGpu.getBlockDimx() / 2; s > 0; s >>= 1) {

      if (thread_idxx < s) {
        // sh_mem[ltid] += sh_mem[ltid + s];
        long val1 = RootbeerGpu.getSharedLong(thread_idxx * 8);
        long val2 = RootbeerGpu.getSharedLong((thread_idxx + s) * 8);
        RootbeerGpu.setSharedLong(thread_idxx * 8, val1 + val2);
      }

      RootbeerGpu.syncthreads();
    }

    // thread 0 of each block adds result
    if (thread_idxx == 0) {
      Result result = new Result();
      result.hits = RootbeerGpu.getSharedLong(thread_idxx * 8);
      m_resultList.add(result);
    }
  }

  public static void main(String[] args) {
    // nvcc ~/.rootbeer/generated.cu --ptxas-options=-v -arch sm_35
    // ptxas info : Used 39 registers, 40984 bytes smem, 380 bytes cmem[0], 88
    // bytes cmem[2]

    // using -maxrregcount 32
    // using -shared-mem-size 1024*8 + 12 = 8192 + 12 = 8204
    // BlockSize = 1024
    // GridSize = 14

    long calculationsPerThread = 100000;
    int blockSize = 1024; // threads
    int gridSize = 14; // blocks

    // parse arguments
    if ((args.length > 0) && (args.length == 3)) {
      blockSize = Integer.parseInt(args[0]);
      gridSize = Integer.parseInt(args[1]);
      calculationsPerThread = Integer.parseInt(args[2]);
    } else {
      System.out.println("Wrong argument size!");
      System.out.println("    Argument1=blockSize");
      System.out.println("    Argument2=gridSize");
      System.out.println("    Argument3=calculationsPerThread");
      return;
    }

    PiEstimatorKernel kernel = new PiEstimatorKernel(calculationsPerThread,
        System.currentTimeMillis());

    // Run GPU Kernels
    Rootbeer rootbeer = new Rootbeer();
    Context context = rootbeer.createDefaultContext();
    Stopwatch watch = new Stopwatch();
    watch.start();
    rootbeer.run(kernel, new ThreadConfig(blockSize, gridSize, blockSize
        * gridSize), context);
    watch.stop();

    // Get GPU results
    long totalHits = 0;
    List<Result> resultList = kernel.m_resultList.getList();
    for (Result result : resultList) {
      totalHits += result.hits;
    }

    double result = 4.0 * totalHits
        / (calculationsPerThread * blockSize * gridSize);

    List<StatsRow> stats = context.getStats();
    for (StatsRow row : stats) {
      System.out.println("  StatsRow:");
      System.out.println("    serial time: " + row.getSerializationTime());
      System.out.println("    exec time: " + row.getExecutionTime());
      System.out.println("    deserial time: " + row.getDeserializationTime());
      System.out.println("    num blocks: " + row.getNumBlocks());
      System.out.println("    num threads: " + row.getNumThreads());
    }
    System.out.println("GPUTime=" + watch.elapsedTimeMillis() + "ms");
    System.out.println("Pi: " + result);
    System.out.println("totalHits: " + totalHits);
    System.out.println("calculationsPerThread: " + calculationsPerThread);
    System.out.println("results: " + resultList.size());
    System.out.println("calculationsTotal: " + calculationsPerThread
        * blockSize * gridSize);
  }
}
