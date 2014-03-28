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

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class OnlineCF {
  public List<double[]> m_preferences;
  public Map<Long, double[]> m_usersMatrix;
  public Map<Long, double[]> m_itemsMatrix;
  public double m_ALPHA;
  public int m_matrixRank;
  public int m_maxIterations;
  private ArrayList<Integer> m_indexes = new ArrayList<Integer>();
  private Random m_rand = new Random(32L);

  public OnlineCF(List<double[]> preferences, Map<Long, double[]> usersMatrix,
      Map<Long, double[]> itemsMatrix, double ALPHA, int matrixRank,
      int maxIterations) {
    this.m_preferences = preferences;
    this.m_usersMatrix = usersMatrix;
    this.m_itemsMatrix = itemsMatrix;
    this.m_ALPHA = ALPHA;
    this.m_matrixRank = matrixRank;
    this.m_maxIterations = maxIterations;
    for (int i = 0; i < m_preferences.size(); i++) {
      m_indexes.add(i);
    }
  }

  public void compute(boolean cpuEmulatesGPU) {
    for (int i = 0; i < m_maxIterations; i++) {
      if (cpuEmulatesGPU) {
        computeUserValues();
        computeItemValues();
      } else {
        computeAllValues();
      }
    }
  }

  private void computeUserValues() {
    for (Integer prefIdx : m_indexes) {
      double[] pref = m_preferences.get(prefIdx);
      long userId = (long) pref[0];
      long itemId = (long) pref[1];
      double expectedScore = pref[2];

      double[] userVector = m_usersMatrix.get(userId);
      double[] itemVector = m_itemsMatrix.get(itemId);

      double calculatedScore = computeScore(userVector, itemVector,
          m_matrixRank);
      double scoreDifference = expectedScore - calculatedScore;

      // calculate new userVector
      for (int i = 0; i < m_matrixRank; i++) {
        userVector[i] += itemVector[i] * 2 * m_ALPHA * scoreDifference;
      }

      m_usersMatrix.put(userId, userVector);
    }
  }

  private void computeItemValues() {
    for (Integer prefIdx : m_indexes) {
      double[] pref = m_preferences.get(prefIdx);
      long userId = (long) pref[0];
      long itemId = (long) pref[1];
      double expectedScore = pref[2];

      double[] userVector = m_usersMatrix.get(userId);
      double[] itemVector = m_itemsMatrix.get(itemId);

      double calculatedScore = computeScore(userVector, itemVector,
          m_matrixRank);
      double scoreDifference = expectedScore - calculatedScore;

      // calculate new itemVector
      for (int i = 0; i < m_matrixRank; i++) {
        itemVector[i] += userVector[i] * 2 * m_ALPHA * scoreDifference;
      }

      m_itemsMatrix.put(itemId, itemVector);
    }
  }

  private void computeAllValues() {
    // shuffling indexes
    int idx = 0;
    int idxValue = 0;
    int tmp = 0;
    for (int i = m_indexes.size(); i > 0; i--) {
      idx = Math.abs(m_rand.nextInt()) % i;
      idxValue = m_indexes.get(idx);
      tmp = m_indexes.get(i - 1);
      m_indexes.set(i - 1, idxValue);
      m_indexes.set(idx, tmp);
    }

    // compute values
    for (Integer prefIdx : m_indexes) {
      double[] pref = m_preferences.get(prefIdx);
      long userId = (long) pref[0];
      long itemId = (long) pref[1];
      double expectedScore = pref[2];

      double[] userVector = m_usersMatrix.get(userId);
      double[] itemVector = m_itemsMatrix.get(itemId);

      double calculatedScore = computeScore(userVector, itemVector,
          m_matrixRank);
      double scoreDifference = expectedScore - calculatedScore;

      // calculate new userVector
      for (int i = 0; i < m_matrixRank; i++) {
        userVector[i] += itemVector[i] * 2 * m_ALPHA * scoreDifference;
      }

      // calculate new itemVector
      for (int i = 0; i < m_matrixRank; i++) {
        itemVector[i] += userVector[i] * 2 * m_ALPHA * scoreDifference;
      }

      m_usersMatrix.put(userId, userVector);
      m_itemsMatrix.put(itemId, itemVector);
    }
  }

  public static double computeScore(double[] vector1, double[] vector2,
      int matrixRank) {
    // calculated score
    double calculatedScore = 0;
    for (int i = 0; i < matrixRank; i++) {
      calculatedScore += vector1[i] * vector2[i];
    }
    return calculatedScore;
  }

}
