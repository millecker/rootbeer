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
package at.illecker.rootbeer.examples.simple;

import java.util.List;

import org.trifort.rootbeer.runtime.Context;
import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.Rootbeer;
import org.trifort.rootbeer.runtime.RootbeerGpu;
import org.trifort.rootbeer.runtime.StatsRow;
import org.trifort.rootbeer.runtime.ThreadConfig;

public class SimpleKernel implements Kernel {
  private int[] m_memory;

  public SimpleKernel(int[] memory) {
    m_memory = memory;
  }

  public void gpuMethod() {
    int index = RootbeerGpu.getThreadIdxx();
    m_memory[index] = index;
    printHelloRootbeer(index);
  }

  private synchronized void printHelloRootbeer(int index) {
    System.out.println("Hello Rootbeer! threadIdx.x: " + index);
  }

  public static void main(String[] args) {
    int gridSize = 1; // amount of blocks
    int blockSize = 2; // amount of threads within a block
    // parse arguments
    if ((args.length > 0) && (args.length == 2)) {
      gridSize = Integer.parseInt(args[0]);
      blockSize = Integer.parseInt(args[1]);
    } else {
      System.out.println("Wrong argument size!");
      System.out.println("    Argument1=gridSize");
      System.out.println("    Argument2=blockSize");
      return;
    }

    int[] memory = new int[blockSize * gridSize];
    SimpleKernel kernel = new SimpleKernel(memory);

    Rootbeer rootbeer = new Rootbeer();
    Context context = rootbeer.createDefaultContext();
    rootbeer.run(kernel, new ThreadConfig(blockSize, gridSize, (long) blockSize
        * gridSize), context);

    for (int i = 0; i < memory.length; i++) {
      System.out.println("Kernel: " + i + " has index: " + memory[i]);
    }

    // DEBUG
    List<StatsRow> stats = context.getStats();
    for (StatsRow row : stats) {
      System.out.println("  StatsRow:");
      System.out.println("    serial time: " + row.getSerializationTime());
      System.out.println("    exec time: " + row.getExecutionTime());
      System.out.println("    deserial time: " + row.getDeserializationTime());
      System.out.println("    num blocks: " + row.getNumBlocks());
      System.out.println("    num threads: " + row.getNumThreads());
    }
  }
}
