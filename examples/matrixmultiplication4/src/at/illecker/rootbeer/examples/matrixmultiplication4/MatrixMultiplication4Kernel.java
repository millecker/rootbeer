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
package at.illecker.rootbeer.examples.matrixmultiplication4;

import java.util.List;
import java.util.Random;

import org.trifort.rootbeer.runtime.Context;
import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.Rootbeer;
import org.trifort.rootbeer.runtime.RootbeerGpu;
import org.trifort.rootbeer.runtime.StatsRow;
import org.trifort.rootbeer.runtime.ThreadConfig;
import org.trifort.rootbeer.runtime.util.Stopwatch;

public class MatrixMultiplication4Kernel implements Kernel {

  private double[][] m_matrixA; // matrix A is transposed
  private double[][] m_matrixB;
  private double[][] m_matrixC;

  private int m_gridSize;
  private int m_blockSize;
  private int m_N;
  private int m_M;
  private int m_L;
  private int m_columnsPerBlock;
  private int m_rowsPerThread;
  private int m_reductionLimit;
  private int m_reductionStart;

  public MatrixMultiplication4Kernel(double[][] transposedmatrixA,
      double[][] matrixB, double[][] matrixC, int gridSize, int blockSize,
      int n, int m, int l) {
    m_matrixA = transposedmatrixA; // m x n
    m_matrixB = matrixB; // m x l
    m_matrixC = matrixC; // n x l
    m_gridSize = gridSize;
    m_blockSize = blockSize;
    m_N = n;
    m_M = m;
    m_L = l;
    m_columnsPerBlock = divup(n, gridSize);
    m_rowsPerThread = divup(m, blockSize);
    if (m_rowsPerThread == 1) {
      m_reductionLimit = m;
    } else {
      m_reductionLimit = blockSize;
    }
    m_reductionStart = roundUpToNextPowerOfTwo(divup(m_reductionLimit, 2));
  }

  // A block handles a column of matrix A and a column of matrix B and each
  // thread within this block takes one row
  //
  // SharedMemory per block
  // e.g., max blockSize = 1024 and one intermediate double value
  // => 12 (needed by Rootbeer) + 8 + (1024 * 8) = 8212 bytes bytes
  //
  public void gpuMethod() {
    int block_idxx = RootbeerGpu.getBlockIdxx();
    int thread_idxx = RootbeerGpu.getThreadIdxx();

    // store fields into local variables
    // each read from a field hits global ram while a local variable
    // is most likely stored in a register
    int gridSize = m_gridSize;
    int blockSize = m_blockSize;
    int N = m_N;
    int M = m_M;
    int L = m_L;
    int columnsPerBlock = m_columnsPerBlock;
    int rowsPerThread = m_rowsPerThread;
    int reductionLimit = m_reductionLimit;
    int reductionStart = m_reductionStart;

    // store pointers to arrays in local variable
    double[][] matrixA = m_matrixA;
    double[][] matrixB = m_matrixB;
    double[][] matrixC = m_matrixC;

    // DEBUG
    // if (RootbeerGpu.getThreadId() == 0) {
    // System.out.println("columnsPerBlock: " + columnsPerBlock);
    // System.out.println("rowsPerThread: " + rowsPerThread);
    // System.out.println("reductionLimit: " + reductionLimit);
    // System.out.println("reductionStart: " + reductionStart);
    // }

    // Loop over all columns of matrix A
    for (int i = 0; i < columnsPerBlock; i++) {

      int colId = (gridSize * i) + block_idxx;
      if (colId < N) {

        // Loop over all columns of matrix B
        for (int j = 0; j < L; j++) {

          // Init intermediate result in shared memory
          if (thread_idxx == 0) {
            RootbeerGpu.setSharedDouble(0, 0);
          }

          // Loop over all rows
          for (int l = 0; l < rowsPerThread; l++) {

            int rowId = (blockSize * l) + thread_idxx;
            if (rowId < M) {
              RootbeerGpu.setSharedDouble(8 + thread_idxx * 8,
                  matrixA[rowId][colId] * matrixB[rowId][j]);
            }
            // Sync all threads within a block
            RootbeerGpu.syncthreads();

            // DEBUG
            // if (RootbeerGpu.getThreadId() == 0) {
            // for (int t = 0; t < m_N; t++) {
            // System.out.println("colId: " + colId + " j: " + j + " value: "
            // + RootbeerGpu.getSharedDouble(t * 8));
            // }
            // }
            // Sync all threads within a block
            // RootbeerGpu.syncthreads();

            // do reduction in shared memory
            // 1-bit right shift = divide by two to the power 1
            for (int s = reductionStart; s > 0; s >>= 1) {

              if ((thread_idxx < s) && (thread_idxx + s) < reductionLimit) {
                // sh_mem[tid] += sh_mem[tid + s];
                RootbeerGpu.setSharedDouble(
                    8 + thread_idxx * 8,
                    RootbeerGpu.getSharedDouble(8 + thread_idxx * 8)
                        + RootbeerGpu
                            .getSharedDouble(8 + (thread_idxx + s) * 8));
              }

              // Sync all threads within a block
              RootbeerGpu.syncthreads();
            }

            // DEBUG
            // if (RootbeerGpu.getThreadId() == 0) {
            // System.out.println("colId: " + colId + " j: " + j + " sum: "
            // + RootbeerGpu.getSharedDouble(0));
            // }

            if (thread_idxx == 0) {
              RootbeerGpu.setSharedDouble(0, RootbeerGpu.getSharedDouble(0)
                  + RootbeerGpu.getSharedDouble(8));
            }

            // Sync all threads within a block
            RootbeerGpu.syncthreads();

          } // for (int l = 0; l < rowsPerThread; l++)

          if (thread_idxx == 0) {
            matrixC[colId][j] = RootbeerGpu.getSharedDouble(0);
          }

          // Sync all threads within a block
          // RootbeerGpu.syncthreads();

        } // for (int j = 0; j < m_L; j++)

      } // if (colId < m_M)

    } // for (int i = 0; i < columnsPerBlock
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

  public static void main(String[] args) {
    int n = 4;
    int m = 4;
    int l = 4;
    boolean isDebugging = true;

    // parse arguments
    if (args.length > 0) {
      if (args.length == 4) {
        n = Integer.parseInt(args[0]);
        m = Integer.parseInt(args[1]);
        l = Integer.parseInt(args[2]);
        isDebugging = Boolean.parseBoolean(args[3]);
      } else {
        System.out.println("Wrong argument size!");
        System.out.println("    Argument1=n");
        System.out.println("    Argument2=m");
        System.out.println("    Argument3=l");
        System.out.println("    Argument4=debug(true|false)");
        return;
      }
    }

    int gridSize = n * l;
    int blockSize = 1024;

    System.out.println("gridSize: " + gridSize + " MaxInt: "
        + Integer.MAX_VALUE);
    System.out.println("blockSize: " + blockSize);
    System.out.println("n: " + n);
    System.out.println("m: " + m);
    System.out.println("l: " + l);

    double[][] matrixA = createRandomMatrix(n, m, new Random(42L));
    double[][] transposedMatrixA = transposeMatrix(matrixA);
    double[][] matrixB = createRandomMatrix(m, l, new Random(1337L));
    double[][] matrixC = new double[n][l];

    if (isDebugging) {
      System.out.println("MatrixA");
      printMatrix(matrixA);
      System.out.println("TransposedMatrixA");
      printMatrix(transposedMatrixA);
      System.out.println("MatrixB");
      printMatrix(matrixB);
      // System.out.println("MatrixC");
      // printArray(matrixC, n, n);
    }

    // Run GPU Kernels
    MatrixMultiplication4Kernel kernel = new MatrixMultiplication4Kernel(
        transposedMatrixA, matrixB, matrixC, gridSize, blockSize, n, m, l);

    Rootbeer rootbeer = new Rootbeer();
    Context context = rootbeer.createDefaultContext();
    Stopwatch watch = new Stopwatch();
    watch.start();
    rootbeer.run(kernel, new ThreadConfig(blockSize, gridSize, (long) blockSize
        * gridSize), context);
    watch.stop();

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
    System.out.println("GPU Time: " + watch.elapsedTimeMillis() + "ms");

    long startTime = System.currentTimeMillis();
    double[][] matrixD = multiply(matrixA, matrixB);
    System.out.println("CPU Time: " + (System.currentTimeMillis() - startTime)
        + "ms");

    boolean verifyResult = verify(matrixC, matrixD);
    if (verifyResult) {
      System.out.println("Verify PASSED!\n");
    } else {
      System.out.println("Verify FAILED!\n");

    }
    if (isDebugging) {
      System.out.println("MatrixC");
      printMatrix(matrixC);
      System.out.println("MatrixD");
      printMatrix(matrixD);
    }
  }

  static double[][] createRandomMatrix(int n, int m, Random rand) {
    final double matrix[][] = new double[n][m];
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        // matrix[i][j] = rand.nextDouble();
        matrix[i][j] = rand.nextInt(9) + 1; // between 1 and 10
      }
    }
    return matrix;
  }

  static double[][] transposeMatrix(double[][] matrix) {
    int n = matrix[0].length;
    int m = matrix.length;
    final double transposedMatrix[][] = new double[n][m];
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        transposedMatrix[i][j] = matrix[j][i];
      }
    }
    return transposedMatrix;
  }

  static void printMatrix(double[][] matrix) {
    int n = matrix.length;
    int m = matrix[0].length;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        if (j == m - 1) {
          System.out.println(matrix[i][j] + "]");
        } else if (j == 0) {
          System.out.print("[" + matrix[i][j] + ",");
        } else {
          System.out.print(matrix[i][j] + ",");
        }
      }
    }
    System.out.println();
  }

  static double[][] multiply(double[][] matrixA, double[][] matrixB) {
    int n = matrixA.length;
    int m = matrixA[0].length;
    int l = matrixB[0].length;
    final double matrix[][] = new double[n][l];
    for (int k = 0; k < m; k++) {
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < l; j++) {
          matrix[i][j] += matrixA[i][k] * matrixB[k][j];
        }
      }
    }
    return matrix;
  }

  static boolean verify(double[][] matrixA, double[][] matrixB) {
    int n = matrixA.length;
    int m = matrixA[0].length;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        if (matrixA[i][j] != matrixB[i][j]) {
          System.out.println("Verify error at [" + i + "," + j + "]: "
              + matrixA[i][j] + " != " + matrixB[i][j]);
          return false;
        }
      }
    }
    return true;
  }
}
