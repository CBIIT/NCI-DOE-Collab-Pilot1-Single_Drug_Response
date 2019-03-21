#include "xTx.h"
#include "mpi_util.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <list>
#include <math.h>
#include <mpi.h>

#include "shuffle.h"

// A way to potentially speed up the sorting and partitioning routines (which can consume a non-trivial
// amount of CPU time) is to use the g++-specific __gnu_parallel:: funcions instead
// of the std:: functions. Since __gnu_parallel:: is a drop in replacement for
// std::, include:
#include <parallel/algorithm>
// which defines _GLIBCXX_PARALLEL_PARALLEL_H and enables parallel sorting
// Please note that approach will not work on non-g++ compilers (i.e. clang or intel).

#ifdef _GLIBCXX_PARALLEL_PARALLEL_H

	#define	SORT		__gnu_parallel::sort
	#define	PARTITION	__gnu_parallel::partition
#else

	#define	SORT		std::sort
	#define	PARTITION	std::partition
#endif // _GLIBCXX_PARALLEL_PARALLEL_H

using namespace std;

extern int mpi_numtasks;
extern int mpi_rank;

float average_response(
	const vector<LabeledData>::const_iterator m_begin,
	const vector<LabeledData>::const_iterator m_end);	

float best_split(pair<unsigned int, float> &m_boundary, vector<unsigned int> &m_left, 
	RandomForest::FeatureType &m_best_feature,
	const size_t &m_leaf, 
	const vector<LabeledData>::const_iterator &m_begin, 
	const vector<LabeledData>::const_iterator &m_end, 
	const vector< vector<float> > &m_cell_features,
	const vector< vector<float> > &m_drug_features,
	const vector< pair<unsigned int, RandomForest::FeatureType> > &m_feature_indicies);
			
void RandomForest::build(vector<LabeledData> &m_data, 
	const vector< vector<float> > &m_cell_features,
	const vector< vector<float> > &m_drug_features)
{
	if(forest_size == 0){
		
		// when number of trees is less than the number if workers, an
		// individual work may have nothing to do.
		return;
	}
	
	if( (forest_feature_bag <= 0.0) || (forest_feature_bag > 1.0) ){
		throw __FILE__ ":RandomForest::build: Please specify 0 < forest_feature_bag <= 1.0";
	}
	
	const __ID num_cell = m_cell_features.size();
	const __ID num_drug = m_drug_features.size();
	
	// Make sure that all of the data has the same number of features (for both cell and drug)
	// and the same number of response values
	unsigned int num_cell_features = 0;
	unsigned int num_drug_features = 0;
	
	for(vector<LabeledData>::const_iterator i = m_data.begin();i != m_data.end();++i){
		
		if( (i->cell_id >= num_cell) || (i->drug_id >= num_drug) ){
			throw __FILE__ ":RandomForest::build: Feature index out of bounds";
		}
		
		if( i == m_data.begin() ){

			num_cell_features = m_cell_features[i->cell_id].size();
			num_drug_features = m_drug_features[i->drug_id].size();
		}
		
		if( num_cell_features != m_cell_features[i->cell_id].size() ){
			throw __FILE__ ":RandomForest::build: Variable number of cell features not allowed";
		}
		
		if( num_drug_features != m_drug_features[i->drug_id].size() ){
			throw __FILE__ ":RandomForest::build: Variable number of drug features not allowed";
		}
	}
		
        const unsigned int num_bagged_cell_features = 
		max( (unsigned int)(1), (unsigned int)(forest_feature_bag*num_cell_features) );

	const unsigned int num_bagged_drug_features = 
		max( (unsigned int)(1), (unsigned int)(forest_feature_bag*num_drug_features) );
	
	vector<unsigned char> cell_mask(num_cell_features, false);
	vector<unsigned char> drug_mask(num_drug_features, false);

	for(unsigned int i = 0;i < num_bagged_cell_features;++i){
		cell_mask[i] = true;
	}	

	for(unsigned int i = 0;i < num_bagged_drug_features;++i){
		drug_mask[i] = true;
	}
	
	vector< pair<unsigned int, FeatureType> > feature_indicies;
	
	feature_indicies.reserve(num_bagged_cell_features
		+ num_bagged_drug_features 
		+ 1 /*dose*/);
	
	forest.resize(forest_size);

	// Rank 0 will output progress to keep the user informed
	string info_buffer;
	
	for(size_t i = 0;i < forest_size;++i){
		
		randomize(cell_mask.begin(), cell_mask.end(), ptr_seed);
		randomize(drug_mask.begin(), drug_mask.end(), ptr_seed);
		
		// While we could just pass the feature masks, passing
		// the actual target indicies was intended to improve
		// OpenMPI-based multi-threaded load balancing
		feature_indicies.clear();
		
		// Always include the dose as a feature
		feature_indicies.push_back( make_pair(0 /*dummy index*/, DOSE_FEATURE) );
		
		for(unsigned int j = 0;j < num_cell_features;++j){
			
			if(cell_mask[j]){
				feature_indicies.push_back( make_pair(j, CELL_FEATURE) );
			}
		}
		
		for(unsigned int j = 0;j < num_drug_features;++j){
			
			if(drug_mask[j]){
				feature_indicies.push_back( make_pair(j, DRUG_FEATURE) );
			}
		}
		
		RandomForest::build_tree(forest[i], 
			m_data.begin(), m_data.end(), 
			m_cell_features, m_drug_features,
			feature_indicies);

		if(mpi_rank == 0){

			for(string::const_iterator j = info_buffer.begin();j != info_buffer.end();++j){
				cerr << '\b';
			}

			for(string::const_iterator j = info_buffer.begin();j != info_buffer.end();++j){
                                cerr << ' ';
                        }

			for(string::const_iterator j = info_buffer.begin();j != info_buffer.end();++j){
                                cerr << '\b';
                        }

			stringstream ssin;

			ssin << (100.0*i)/forest_size << '%';

			info_buffer = ssin.str();

			cerr << info_buffer;
		}	
	}
}

#define	HIGH_BIT 	0x80000000
#define	CLEAR_HIGH_BIT 	0x7FFFFFFF

struct IsLeft
{
	inline bool operator()(const LabeledData &m_x) const
	{
		return (m_x.cell_id & HIGH_BIT);
	};
};

void RandomForest::build_tree(Tree &m_tree, 
	vector<LabeledData>::iterator m_begin,
	vector<LabeledData>::iterator m_end,
	const vector< vector<float> > &m_cell_features,
	const vector< vector<float> > &m_drug_features,
	const vector< pair<unsigned int, FeatureType> > &m_feature_indicies)
{	
	if(m_end <= m_begin){
		throw __FILE__ ":RandomForest::build_tree: No data!";
	}
	
	const unsigned int num_data = m_end - m_begin;
	
	TreeNode local;
	
	m_tree.push_back(local);
	
	// Do we have enough data to make a split?
	if(num_data <= forest_leaf){
		
		// Return the average over all leaf members
		m_tree.back().prediction = average_response(m_begin, m_end);
		return;
	}
	
	// Search for the partition that obtains the smallest *weighted* mean square error
	pair<unsigned int, float> best_boundary;
	vector<unsigned int> best_left;
	FeatureType best_feature = RandomForest::UNKNOWN_FEATURE;
	
	best_split(best_boundary, best_left, best_feature, forest_leaf, 
		m_begin, m_end, 
		m_cell_features, m_drug_features, 
		m_feature_indicies);
	
	// We could not find a valid split
	if(best_feature == RandomForest::UNKNOWN_FEATURE){

		// Return the average over all leaf members
		m_tree.back().prediction = average_response(m_begin, m_end);
		return;
	}
	
	// Partition the data into left and right branches. Since a non-trivial amount of time
	// is spent paritioning the data, this code has some admittedly kludgy hacks to make it run as
	// fast as posible. A single bit (borrowed from the high order bit of the cell_id member 
	// varible) is used to indicate membership in the left hand set.	
	for(vector<unsigned int>::iterator i = best_left.begin();i != best_left.end();++i){
		(m_begin + *i)->cell_id |= HIGH_BIT;
	}
	
	PARTITION( m_begin, m_end, IsLeft() );
	
	vector<LabeledData>::iterator boundary_iter = m_begin + best_left.size();
	
	// Unset the high bit so we can use the cell_id variable normally
	for(vector<LabeledData>::iterator i = m_begin;i != boundary_iter;++i){
		i->cell_id &= CLEAR_HIGH_BIT;
	}
	
	const unsigned int node = m_tree.size() - 1;
	
	m_tree[node].boundary = best_boundary;
	m_tree[node].boundary_feature = best_feature;
	
	m_tree[node].left = m_tree.size();

	build_tree(m_tree, 
		m_begin, boundary_iter,
		m_cell_features, m_drug_features,
		m_feature_indicies);

	m_tree[node].right = m_tree.size();

	build_tree(m_tree, 
		boundary_iter, m_end,
		m_cell_features, m_drug_features,
		m_feature_indicies);
}

// Each rank computes the total predicted value for all of the trees that
// belong the rank. The ranks then share this total will each other (via
// AllReduce) and *all* ranks compute the average predicted value over all
// trees in the entire, distributed forest.
float RandomForest::predict(const float &m_conc, 
	const vector<float> &m_cell_features, 
	const vector<float> &m_drug_features) const
{

	if( forest_size != forest.size() ){
		throw __FILE__ ":RandomForest::predict: forest_size != forest.size()";
	}
	
	// Make sure that all ranks agree on the concentration value by
	// using the rank 0 concentration value
	float conc = m_conc;
	
	MPI_Bcast( &conc, 1, MPI_FLOAT, 0, MPI_COMM_WORLD );
	
	double sum = 0.0;
	
	// A quick bench mark suggests that parallelizing this for loop
	// is slow (by a factor of two) than the serial version.
	//#pragma omp parallel for reduction(+:sum)
	for(size_t i = 0;i < forest_size;++i){
		
		sum += predict_tree(conc, forest[i],
			m_cell_features, m_drug_features);
	}
	
	double sum_and_norm[2] = {sum, double(forest_size)};
	
	// All ranks share their prediction values
	if(MPI_Allreduce(MPI_IN_PLACE, sum_and_norm, 2, 
		MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS){
		throw __FILE__ ":RandomForest::predict: Error in MPI_Allreduce";
	}

	if(sum_and_norm[1] <= 0.0){
		return 0.0;
	}
	
	return sum_and_norm[0]/sum_and_norm[1];
}

float RandomForest::predict_tree(const float &m_conc, const Tree &m_tree,
	const vector<float> &m_cell_features, 
	const vector<float> &m_drug_features) const
{
	if( m_tree.empty() ){
		throw __FILE__ ":RandomForest::predict_tree: Empty tree!";
	}
	
	const unsigned int num_cell_features = m_cell_features.size();
	const unsigned int num_drug_features = m_drug_features.size();
	
	unsigned int index = 0;

	while(true){

		const TreeNode &node = m_tree[index];

		if( node.is_leaf() ){
			return node.prediction;
		}
		
		switch(node.boundary_feature){
			case DOSE_FEATURE:
				
				index = (m_conc < node.boundary.second) ? 
					node.left :
					node.right;
				break;
			case CELL_FEATURE:
			
				if(num_cell_features <= node.boundary.first){
					throw __FILE__ ":RandomForest::predict_tree: Cell feature index out of bounds!";
				}

				index = (m_cell_features[node.boundary.first] < node.boundary.second) ?
					node.left :
					node.right;
				break;
			case DRUG_FEATURE:
			
				if(num_drug_features <= node.boundary.first){
					throw __FILE__ ":RandomForest::predict_tree: Drug feature index out of bounds!";
				}

				index = (m_drug_features[node.boundary.first] < node.boundary.second) ?
					node.left :
					node.right;

				break;
			default:
				throw __FILE__ ":RandomForest::predict_tree: Unknown boundary feature type!";
		};
	}
	
	throw __FILE__ ":RandomRegressionForest::predict_tree: Should never get here!";
	
	return 0.0f;
}

float average_response(
	const vector<LabeledData>::const_iterator m_begin,
	const vector<LabeledData>::const_iterator m_end)
{	
	if(m_end <= m_begin){
		throw __FILE__ ":average_response: No data!";
	}
	
	const unsigned int N = m_end - m_begin;
	
	// Accumulate as double to avoid loss of precision for
	// large datasets
	double ret = 0.0;
	
	#pragma omp parallel for \
		reduction(+:ret)
	for(unsigned int i = 0;i < N;++i){
		ret += (m_begin + i)->response;
	}
	
	ret /= N;
	
	return ret;
}

float best_split(pair<unsigned int, float> &m_boundary, vector<unsigned int> &m_left, 
	RandomForest::FeatureType &m_best_feature,
	const size_t &m_leaf, 
	const vector<LabeledData>::const_iterator &m_begin, 
	const vector<LabeledData>::const_iterator &m_end, 
	const vector< vector<float> > &m_cell_features,
	const vector< vector<float> > &m_drug_features,
	const vector< pair<unsigned int, RandomForest::FeatureType> > &m_feature_indicies)
{
	#define		REALLY_LARGE_VALUE	1.0e20
	
	float best_score = REALLY_LARGE_VALUE; // <-- A really large value!
	
	if(m_end <= m_begin){
		throw __FILE__ ":best_split: No data!";
	}
	
	const unsigned int num_data = m_end - m_begin;
	
	if(m_leaf < 1){
		throw __FILE__ ":best_split: m_leaf < 1";
	}

	const unsigned int num_features_to_test = m_feature_indicies.size();
	
	#pragma omp parallel
	{
		// Store the values for the i^th independent feature. We can allocate
		// this memory outside the for loop, since the size does not change
		vector< pair<float, unsigned int> > feature_slice(num_data);

		float local_best_score = REALLY_LARGE_VALUE; // <-- A really large value!
		RandomForest::FeatureType local_best_feature = RandomForest::UNKNOWN_FEATURE;
		pair<unsigned int, float> local_boundary;
		vector<unsigned int> local_left;
		
		// Test each feature and every possible boundary value within a feature 
		#pragma omp for
		for(unsigned int f = 0;f < num_features_to_test;++f){

			const pair<unsigned int, RandomForest::FeatureType> &curr_feature = m_feature_indicies[f];
			
			switch(curr_feature.second){
				case RandomForest::DOSE_FEATURE:
					
					for(unsigned int i = 0;i < num_data;++i){
						feature_slice[i] = make_pair( (m_begin + i)->dose, i );
					}
					break;
				case RandomForest::CELL_FEATURE:
					
					for(unsigned int i = 0;i < num_data;++i){
						feature_slice[i] = 
							make_pair(m_cell_features[(m_begin + i)->cell_id][curr_feature.first], i);
					}
					break;
				case RandomForest::DRUG_FEATURE:
				
					for(unsigned int i = 0;i < num_data;++i){
						feature_slice[i] = 
							make_pair(m_drug_features[(m_begin + i)->drug_id][curr_feature.first], i);
					}
					break;
				default:
					throw __FILE__ ":best_split: Unknown feature type!";
			};

			// Sort the feature values in ascending order
			sort( feature_slice.begin(), feature_slice.end() );

			// To make the calculation efficient, track the running sum of y values in the left and 
			// right branches. Use double to accumulate the floating point moments
			double sum_left = 0.0f;
			double sum_right = 0.0f;

			double sum_left2 = 0.0;
			double sum_right2 = 0.0;

			unsigned int num_left = 0;
			unsigned int num_right = 0;

			// By default, all of the data starts in the *right* branch
			
			num_right = num_data;
			
			for(unsigned int i = 0;i < num_data;++i){

				const LabeledData &ref = *(m_begin + i);
				
				sum_right += ref.response;
				sum_right2 += ref.response*ref.response;
			}
			
			for(unsigned int i = 0;i < num_data;++i){

				// Move data point i from the right branch to the left branch
				const float y = (m_begin + feature_slice[i].second)->response;

				// Since the sums are unnormalized, we can simply remove a point from the right and
				// add it to the left
				sum_right -= y;
				sum_left += y;

				const float y2 = y*y;
				
				sum_right2 -= y2;
				sum_left2 += y2;

				--num_right;
				++num_left;

				if( (i < m_leaf) || (i >= (num_data - m_leaf) ) ){
					continue;
				}

				// Don't split on equal values! We can access element i + 1 of the
				// feature slice vector since m_leaf must be greater than 0 (and the
				// above test on i will trigger a "continue" at the end of the vector range)
				if(feature_slice[i].first == feature_slice[i + 1].first){
					continue;
				}
				
				if( (num_left <= 0) || (num_right <= 0) ){
					throw __FILE__ ":best_split: Unable to normalize split variance";
				}

				const float left_mean_square_error = (sum_left2 - sum_left*sum_left/num_left);
				const float right_mean_square_error = (sum_right2 - sum_right*sum_right/num_right);

				// Here is the original, more readable code:
				//left_mean_square_error /= num_left;
				//right_mean_square_error /= num_right;
				//
				//const float trial_mean_square_error = (num_left*left_mean_square_error + 
				//	num_right*right_mean_square_error)/num_data;

				// And here is the code when we cancel the factors of L and R
				const float trial_mean_square_error = 
					(left_mean_square_error + right_mean_square_error)/num_data;

				if(trial_mean_square_error < local_best_score){

					local_best_score = trial_mean_square_error;
					local_best_feature = curr_feature.second;
					
					// Place the boundary at the midpoint between the current and the
					// next feature value.
					local_boundary = make_pair(curr_feature.first, 
						0.5*(feature_slice[i].first + feature_slice[i + 1].first) );

					local_left.resize(i + 1);

					for(unsigned int j = 0;j <= i;++j){
						local_left[j] = feature_slice[j].second;
					}
					
					// If a data point is not in the left branch, it must be in the
					// right branch (so we don't need to explicitly store the data points
					// that belong to the right hand branch).
				}
			}
		}
		
		#pragma omp critical
		if(local_best_score < best_score){
			
			best_score = local_best_score;
			m_best_feature = local_best_feature;
			m_boundary = local_boundary;
			m_left = local_left;
		}
	}
	
	return best_score;
}
