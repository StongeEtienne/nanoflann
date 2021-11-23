/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2011-2016 Jose Luis Blanco (joseluisblancoc@gmail.com).
 *   All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

#include <nanoflann.hpp>

#include <cstdlib>
#include <ctime>
#include <iostream>

#include <Eigen/Dense>

using namespace Eigen;
using namespace std;
using namespace nanoflann;

#define SAMPLES_DIM (size_t)9
//const size_t SAMPLES_DIM = 12;

template <typename Der>
void generateRandomPointCloud(Eigen::MatrixBase<Der> &mat, const size_t N) {
  mat.resize(N, SAMPLES_DIM);
  for (size_t i = 0; i < N; i++)
    for (size_t d = 0; d < SAMPLES_DIM; d++)
      mat(i, d) = i + d;
}

template <typename num_t>
void kdtree_demo(const size_t nSamples) {

  // Generate points:
  Eigen::Matrix<num_t, Dynamic, Dynamic> mat(nSamples, SAMPLES_DIM);
  generateRandomPointCloud(mat, nSamples);
  //std::cout << mat << std::endl;

  // Query point:
  std::vector<num_t> query_pt(SAMPLES_DIM);
  for (size_t d = 0; d < SAMPLES_DIM; d++)
  	query_pt[d] = 0;

  typedef KDTreeEigenMatrixAdaptor<Eigen::Matrix<num_t,Dynamic,Dynamic>, SAMPLES_DIM, nanoflann::metric_L21_3D>
	my_kd_tree_t;

  my_kd_tree_t mat_index(SAMPLES_DIM, std::cref(mat), 10 /* max leaf */);
  mat_index.index->buildIndex();

  // do a knn search
  const size_t num_results = 2000; // 3
  vector<size_t> ret_indexes(num_results);
  vector<num_t> out_dists_sqr(num_results);

  nanoflann::KNNResultSet<num_t> resultSet(num_results);

  resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
  mat_index.index->findNeighbors(resultSet, &query_pt[0],
                                 nanoflann::SearchParams());

  std::cout << "knnSearch(nn=" << num_results << "): \n";
  for (size_t i = 0; i < num_results; i++)
    std::cout << "ret_index[" << i << "]=" << ret_indexes[i]
              << " out_dist=" << out_dists_sqr[i] << endl;

	const num_t search_radius = static_cast<num_t>(4000.0);
	std::vector<std::pair<int64_t, num_t> > ret_matches;

	const size_t nMatches = mat_index.index->radiusSearch(&query_pt[0], search_radius, ret_matches, nanoflann::SearchParams());

	std::cout << "radiusSearch(): radius=" << search_radius << " -> " << nMatches << " matches" << std::endl;
	//for (size_t i = 0; i < nMatches; i++)
	//	std::cout << "idx["<< i << "]=" << ret_matches[i].first << " dist["<< i << "]=" << ret_matches[i].second << endl;

}

int main(int argc, char **argv) {
  // Randomize Seed
  srand(static_cast<size_t>(time(nullptr)));
  kdtree_demo<float>(434563 /* samples */);

  return 0;
}
