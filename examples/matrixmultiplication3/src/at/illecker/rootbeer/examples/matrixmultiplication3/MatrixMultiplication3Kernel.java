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
package at.illecker.rootbeer.examples.matrixmultiplication3;

import java.util.List;
import java.util.Random;

import org.trifort.rootbeer.runtime.Context;
import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.Rootbeer;
import org.trifort.rootbeer.runtime.RootbeerGpu;
import org.trifort.rootbeer.runtime.StatsRow;
import org.trifort.rootbeer.runtime.util.Stopwatch;

public class MatrixMultiplication3Kernel implements Kernel {

  // input
  private double[][] m_matrixA;
  private double[][] m_matrixB;

  // output
  public double[][] resultMatrix;

  public MatrixMultiplication3Kernel(double[][] matrixA, double[][] matrixB) {
    m_matrixA = matrixA;
    m_matrixB = matrixB;
  }

  public void gpuMethod() {
    int threadId = RootbeerGpu.getThreadIdxx();
    int blockId = RootbeerGpu.getBlockIdxx();

  }

  public static void main(String[] args) {
    int n = 2;
    int m = 2;
    boolean isDebugging = true;
    int gridSize = 14;
    int blockSize = 256;

    // parse arguments
    if (args.length > 0) {
      if (args.length == 3) {
        n = Integer.parseInt(args[0]);
        m = Integer.parseInt(args[1]);
        isDebugging = Boolean.parseBoolean(args[2]);
      } else {
        System.out.println("Wrong argument size!");
        System.out.println("    Argument1=n");
        System.out.println("    Argument2=m");
        System.out.println("    Argument3=debug(true|false=default)");
        return;
      }
    }

    System.out.println("n: " + n);
    System.out.println("m: " + m);
    System.out.println("gridSize: " + gridSize);
    System.out.println("blockSize: " + blockSize);

    double[][] matrixA = createRandomMatrix(n, m, new Random(42L));
    double[][] transposedMatrixA = transposeMatrix(matrixA);
    double[][] matrixB = createRandomMatrix(m, n, new Random(1337L));
    // double[][] matrixC = createConstantArray(n, n, 0);

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

    MatrixMultiplication3Kernel kernel = new MatrixMultiplication3Kernel(
        matrixA, matrixB);

    // Run GPU Kernels
    Rootbeer rootbeer = new Rootbeer();
    Context context = rootbeer.createDefaultContext();
    Stopwatch watch = new Stopwatch();
    watch.start();
    // rootbeer.run(kernel, new ThreadConfig(blockSize, gridSize, blockSize
    // * gridSize), context);
    watch.stop();

    // Get GPU Result
    // double[] matrixC = kernel.resultMatrix.matrix;
    // double[] matrixD = multiply(matrixA, matrixB, n, n, n);

    // Debug
    List<StatsRow> stats = context.getStats();
    for (StatsRow row : stats) {
      System.out.println("  StatsRow:");
      System.out.println("    serial time: " + row.getSerializationTime());
      System.out.println("    exec time: " + row.getExecutionTime());
      System.out.println("    deserial time: " + row.getDeserializationTime());
      System.out.println("    num blocks: " + row.getNumBlocks());
      System.out.println("    num threads: " + row.getNumThreads());
    }
    System.out.println("GPUTime: " + watch.elapsedTimeMillis() + "ms");

    /*
     * boolean verifyResult = verify(matrixC, matrixD, n, n); if (verifyResult)
     * { System.out.println("Verify PASSED!"); } else {
     * System.out.println("Verify FAILED!"); } if (isDebugging) {
     * System.out.println("MatrixC"); printArray(matrixC, n, n);
     * System.out.println("MatrixD"); printArray(matrixD, n, n); }
     */
  }

  static double[] createConstantArray(int n, int m, double value) {
    final double data[] = new double[n * m];
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < m; ++i) {
        data[j * m + i] = value;
      }
    }
    return data;
  }

  static double[] createRandomArray(int n, int m, Random rand) {
    final double data[] = new double[n * m];
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < m; ++i) {
        // matrix[i][j] = rand.nextDouble();
        data[j * m + j] = rand.nextInt(9) + 1; // between 1 and 10
      }
    }
    return data;
  }

  static double[][] createRandomMatrix(int n, int m, Random rand) {
    final double matrix[][] = new double[n][m];
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < m; ++i) {
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
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < m; ++i) {
        transposedMatrix[i][j] = matrix[j][i];
      }
    }
    return transposedMatrix;
  }

  static void printArray(double[] data, int n, int m) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < m; ++i) {
        if (i == m - 1) {
          System.out.println(data[j * m + i] + "]");
        } else if (i == 0) {
          System.out.print("[" + data[j * m + i] + ",");
        } else {
          System.out.print(data[j * m + i] + ",");
        }
      }
    }
    System.out.println();
  }

  static void printMatrix(double[][] matrix) {
    int n = matrix.length;
    int m = matrix[0].length;
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < m; ++i) {
        if (i == m - 1) {
          System.out.println(matrix[j][i] + "]");
        } else if (i == 0) {
          System.out.print("[" + matrix[j][i] + ",");
        } else {
          System.out.print(matrix[j][i] + ",");
        }
      }
    }
    System.out.println();
  }

  static double[] multiply(double[] matrixA, double[] matrixB, int a_rows,
      int a_cols, int b_cols) {
    final double data[] = new double[a_rows * b_cols];

    for (int k = 0; k < a_cols; k++) {
      for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < b_cols; j++) {
          data[i * b_cols + j] += matrixA[i * b_cols + k]
              * matrixB[k * a_rows + j];
        }
      }
    }
    return data;
  }

  static boolean verify(double[] matrixA, double[] matrixB, int n, int m) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < m; ++i) {
        if (matrixA[j * m + i] != matrixB[j * m + i]) {
          System.out.println("Verify ERROR at [" + j + "," + i + "]");
          return false;
        }
      }
    }
    return true;
  }

  static int divup(int x, int y) {
    if (x % y != 0) {
      // aufrunden
      return ((x + y - 1) / y);
    } else {
      return x / y;
    }
  }
}
