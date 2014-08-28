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
package at.illecker.rootbeer.examples.matrixmultiplication5;

import java.util.List;
import java.util.Random;

import org.trifort.rootbeer.runtime.Context;
import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.Rootbeer;
import org.trifort.rootbeer.runtime.RootbeerGpu;
import org.trifort.rootbeer.runtime.StatsRow;
import org.trifort.rootbeer.runtime.ThreadConfig;
import org.trifort.rootbeer.runtime.util.Stopwatch;

public class MatrixMultiplication5Kernel implements Kernel {

  // TILE_WITH denotes the size of one submatrix
  // 32 * 32 = 1024 threads matches the blocksize
  public static final int TILE_WIDTH = 2;

  private double[] m_matrixA; // matrix A is transposed
  private double[] m_matrixB;
  private double[] m_matrixC;
  private int m_N;
  private int m_M;
  private int m_L;
  private int m_gridSize;
  private int m_blockSize;

  private int m_tileWidth;
  private int m_subMatricesPerThread;

  public MatrixMultiplication5Kernel(double[] transposedmatrixA,
      double[] matrixB, double[] matrixC, int n, int m, int l, int gridSize,
      int blockSize, int tileWidth, int subMatricesPerThread) {
    m_matrixA = transposedmatrixA; // m x n
    m_matrixB = matrixB; // m x l
    m_matrixC = matrixC; // n x l
    m_N = n;
    m_M = m;
    m_L = l;
    m_gridSize = gridSize;
    m_blockSize = blockSize;
    m_tileWidth = tileWidth; // 32 by default
    m_subMatricesPerThread = subMatricesPerThread;
  }

  // SharedMemory per block
  // blockSize = 1024
  // => 12 (needed by Rootbeer) + (2 * 1024 * 8 (double)) = 16396 bytes
  //
  // based on
  // http://www.shodor.org/media/content//petascale/materials/UPModules/matrixMultiplication/moduleDocument.pdf
  //
  public void gpuMethod() {
    // get local blockIdx and threadIdx
    int block_idxx = RootbeerGpu.getBlockIdxx();
    int thread_idxx = RootbeerGpu.getThreadIdxx();

    // store fields into local variables
    // each read from a field hits global ram while a local variable
    // is most likely stored in a register
    // int gridSize = m_gridSize;
    // int blockSize = m_blockSize;
    // int N = m_N;
    int M = m_M;
    int L = m_L;
    int tileWidth = m_tileWidth;
    int subMatricesPerThread = m_subMatricesPerThread;

    // store pointers to arrays in local variable
    double[] matrixA = m_matrixA;
    double[] matrixB = m_matrixB;
    double[] matrixC = m_matrixC;

    // Convert block_idxx to a two dimensional index
    int blockRow = block_idxx / (L / tileWidth);
    int blockCol = block_idxx % (L / tileWidth);

    // Convert thread_idxx to a two dimensional index within submatrix
    int threadRow = thread_idxx / tileWidth;
    int threadCol = thread_idxx % tileWidth;

    // Calculate the index of the destination row and col within submatrix
    int destRow = (blockRow * tileWidth) + threadRow;
    int destCol = (blockCol * tileWidth) + threadCol;

    // print(RootbeerGpu.getThreadId(), colA, colB, 0, 0, 0);

    double sum = 0;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < subMatricesPerThread; m++) {
      int aRowIndex = (m * tileWidth) + threadRow;
      int aColIndex = (blockRow * tileWidth) + threadCol;
      int aValueIndex = (aRowIndex * M) + aColIndex;

      int bRowIndex = (m * tileWidth) + threadRow;
      int bColIndex = destCol;
      int bValueIndex = (bRowIndex * L) + bColIndex;

      double aValue = matrixA[aValueIndex];
      double bValue = matrixB[bValueIndex];

      // DEBUG
      // if (block_idxx == 1) {
      // print("blockId(x,y)=" + blockRow + "," + blockCol + " threadId: "
      // + RootbeerGpu.getThreadId() + " threadId(x,y)=" + threadRow + ","
      // + threadCol + " dest(x,y): " + destRow + "," + destCol + " m:" + m
      // + " A(x,y)=" + aRowIndex + "," + aColIndex + " aValueIndex: "
      // + aValueIndex + " B(x,y)=" + bRowIndex + "," + bColIndex
      // + " bValueIndex: " + bValueIndex + " aValue: " + aValue
      // + " bValue: " + bValue);
      // }

      // store the aValue into shared memory at location
      RootbeerGpu.setSharedDouble(thread_idxx * 8, aValue);
      // store the bValue into shared memory at location
      // 1024 is the offset for the row of matrix A
      RootbeerGpu.setSharedDouble(1024 + (thread_idxx * 8), bValue);

      // sync threads within a block to make sure the sub-matrices are loaded
      RootbeerGpu.syncthreads();

      // loop over all of aValues and bValues
      for (int k = 0; k < tileWidth; k++) {
        // read the aValue from shared memory
        aValue = RootbeerGpu.getSharedDouble((k * tileWidth + threadRow) * 8);
        // read the bValue from shared memory
        bValue = RootbeerGpu
            .getSharedDouble(1024 + (k * tileWidth + threadCol) * 8);

        // multiply aValue and bValue and accumulate
        sum += aValue * bValue;

        // if (block_idxx == 1) {
        // print("blockId(x,y)= " + blockRow + "," + blockCol + " threadId: "
        // + RootbeerGpu.getThreadId() + " aValue: " + aValue + " bValue: "
        // + bValue + " sum: " + sum);
        // }
      }

      // sync threads within a block
      RootbeerGpu.syncthreads();
    }

    int cValueIndex = destRow * L + destCol;
    // update the target cValue with the sum
    matrixC[cValueIndex] = sum;

    // if (block_idxx == 1) {
    // print("blockId(x,y)=" + blockRow + "," + blockCol + " threadId: "
    // + RootbeerGpu.getThreadId() + " threadId(x,y)=" + threadRow + ","
    // + threadCol + " dest(x,y): " + destRow + "," + destCol
    // + " setMatrixC at Index: " + cValueIndex + " sum: " + sum);
    // }

  }

  public synchronized void print(String s) {
    System.out.println(s);
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

    int subMatrixSize = TILE_WIDTH * TILE_WIDTH;
    int numberOfSubMatrices = divup(n * l, subMatrixSize);
    int gridSize = numberOfSubMatrices;
    int blockSize = subMatrixSize;

    // int subMatrixSize = tileWidth * tileWidth;
    // rows of A and cols of B per block
    int subMatricesPerThread = divup(m, TILE_WIDTH);

    System.out.println("gridSize: " + gridSize);
    System.out.println("blockSize: " + blockSize);
    System.out.println("n: " + n);
    System.out.println("m: " + m);
    System.out.println("l: " + l);

    double[] matrixA = createRandomMatrix(n, m, new Random(42L));
    double[] transposedMatrixAgpu = transposeMatrix(matrixA, n, m);
    double[] matrixB = createRandomMatrix(m, l, new Random(1337L));
    double[] matrixCgpu = new double[n * l];

    if (isDebugging) {
      System.out.println("MatrixA");
      printMatrix(matrixA, n, m);
      System.out.println("TransposedMatrixA");
      printMatrix(transposedMatrixAgpu, m, n);
      System.out.println("MatrixB");
      printMatrix(matrixB, m, l);
    }

    // Run GPU Kernels
    MatrixMultiplication5Kernel kernel = new MatrixMultiplication5Kernel(
        transposedMatrixAgpu, matrixB, matrixCgpu, n, m, l, gridSize,
        blockSize, TILE_WIDTH, subMatricesPerThread);

    Rootbeer rootbeer = new Rootbeer();
    Context context = rootbeer.createDefaultContext();
    Stopwatch watch = new Stopwatch();
    watch.start();
    rootbeer.run(kernel, new ThreadConfig(blockSize, gridSize, blockSize
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
    double[] matrixCcpu = multiply(matrixA, matrixB, n, m, l);
    System.out.println("CPU Time: " + (System.currentTimeMillis() - startTime)
        + "ms");

    boolean verifyResult = verify(matrixCgpu, matrixCcpu, n, l);
    if (verifyResult) {
      System.out.println("Verify PASSED!\n");
    } else {
      System.out.println("Verify FAILED!\n");

    }
    if (isDebugging) {
      System.out.println("MatrixC GPU");
      printMatrix(matrixCgpu, n, l);
      System.out.println("MatrixC CPU");
      printMatrix(matrixCcpu, n, l);
    }
  }

  static int divup(int x, int y) {
    if (x % y != 0) {
      return ((x + y - 1) / y); // round up
    } else {
      return x / y;
    }
  }

  static double[] createRandomMatrix(int n, int m, Random rand) {
    final double matrix[] = new double[n * m];
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        // matrix[i * m + j] = rand.nextDouble();
        matrix[i * m + j] = rand.nextInt(9) + 1; // between 1 and 10
      }
    }
    return matrix;
  }

  static double[] transposeMatrix(double[] matrix, int n, int m) {
    final double transposedMatrix[] = new double[m * n];
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        transposedMatrix[i * n + j] = matrix[j * m + i]; // M[i][j] = M[j][i]
      }
    }
    return transposedMatrix;
  }

  static void printMatrix(double[] matrix, int n, int m) {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        if (j == m - 1) {
          System.out.println(matrix[i * m + j] + "]");
        } else if (j == 0) {
          System.out.print("[" + matrix[i * m + j] + ",");
        } else {
          System.out.print(matrix[i * m + j] + ",");
        }
      }
    }
    System.out.println();
  }

  static double[] multiply(double[] matrixA, double[] matrixB, int n, int m,
      int l) {
    final double matrix[] = new double[n * l];
    for (int i = 0; i < n; i++) { // for each row of A
      for (int j = 0; j < l; j++) { // for each col of B
        int sum = 0;
        for (int k = 0; k < m; k++) { // for each col of A and row of B
          sum += (matrixA[i * m + k] * matrixB[k * l + j]); // A[i][k] * B[k][j]
        }
        matrix[i * l + j] = sum; // C[i][j] += A[i][k] * B[k][j]
      }
    }
    return matrix;
  }

  static boolean verify(double[] matrixA, double[] matrixB, int n, int l) {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < l; ++j) {
        if (matrixA[i * l + j] != matrixB[i * l + j]) {
          System.out.println("Verify error at [" + i + "," + j + "]: "
              + matrixA[i * l + j] + " != " + matrixB[i * l + j]);
          return false;
        }
      }
    }
    return true;
  }
}
