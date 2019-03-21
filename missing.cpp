#include "xTx.h"
#include "mpi_util.h"

#include <math.h>
#include <algorithm>

using namespace std;

// Global variables for MPI
extern int mpi_numtasks;
extern int mpi_rank;

float feature_distance(const vector<float> &m_a, const vector<float> &m_b);

template<class T>
vector<T> mask_feature(const vector<T> &m_input, const vector<bool> &m_mask)
{
	// Count the number of unmasked features to get the final output size
	size_t len = 0;

	for(vector<bool>::const_iterator i = m_mask.begin();i != m_mask.end();++i){
		len += (*i == false) ? 1 : 0;
	}

	vector<T> ret(len);

	const size_t num_feature = m_mask.size();

	if( num_feature != m_input.size() ){
		throw __FILE__ ":mask_feature: |mask| != |input|";
	}

	size_t index = 0;

	for(size_t i = 0;i < num_feature;++i){
		
		if(m_mask[i] == false){
			ret[index++] = m_input[i];
		}
	}
	
	return ret;
}

size_t remove_missing_values(vector< vector<float> > &m_data, vector<string> &m_names){
	
	const size_t num_data = m_data.size();
	const size_t num_feature = m_names.size();

        vector<bool> missing_features(num_feature, false);
	size_t num_completely_missing = 0;

        for(size_t i = 0;i < num_feature;++i){

		bool is_completely_missing = true;

                for(size_t j = 0;j < num_data;++j){
                        if(m_data[j][i] != MISSING_DATA){
                                is_completely_missing = false;
                        }
                }

		missing_features[i] = is_completely_missing;

		num_completely_missing += is_completely_missing ? 1 : 0;
        }

	if(num_completely_missing == 0){
		return num_completely_missing;
	}

	m_names = mask_feature(m_names, missing_features);

	for(vector< vector<float> >::iterator i = m_data.begin();i != m_data.end();++i){
		*i = mask_feature(*i, missing_features);
	}

	return num_completely_missing;
}

void infer_missing_values(vector< vector<float> > &m_data)
{
	const size_t num_data = m_data.size();

	// Find all of the feature data with missing values
	deque<size_t> missing;
	
	for(size_t i = 0;i < num_data;++i){
		
		bool has_missing = false;
		
		for(vector<float>::const_iterator j = m_data[i].begin();
			( j != m_data[i].end() ) && !has_missing;++j){
			
			if(*j == MISSING_DATA){
				has_missing = true;
			}
		}
		
		if(has_missing){
			missing.push_back(i);
		}
	}
	
	const size_t num_missing = missing.size();
	
	// If none of the samples have missing data, we can return right away!
	if(num_missing == 0){
		return;
	}
	
	// Make sure that all ranks agree on the ordering of samples with missing values
	sort( missing.begin(), missing.end() );
	
	vector< vector<float> > new_data(num_data);
	
	for(size_t i = 0;i < num_missing;++i){
		
		// Each MPI rank is in charge of a unique subset of missing values
		if( int(i%mpi_numtasks) != mpi_rank){
			continue;
		}
		
		const size_t query_index = missing[i];
		
		if(query_index >= num_data){
			throw __FILE__ ":infer_missing_values: Index out of bounds";
		}

		vector<float> &query = m_data[query_index];
		
		// Compute the pairwise distance between the query and all other feature vectors
		deque< pair<float, size_t> > pair_distance;
		
		for(size_t j = 0;j < num_data;++j){
			
			// Don't compare a feature vector to itself or to an empty feature vector
			if( (j == query_index) || m_data[j].empty() ){
				continue;
			}
			
			const float d = feature_distance(query, m_data[j]);
			
			pair_distance.push_back( make_pair(d, j) );
		}
		
		// Sort by distance in ascending order
		sort( pair_distance.begin(), pair_distance.end() );
		
		const size_t num_features = query.size();
		
		vector<float> f(query);
		
		for(deque< pair<float, size_t> >::const_iterator j = pair_distance.begin();
			j != pair_distance.end();++j){
			
			if(j->second >= num_data){
				throw __FILE__ ":infer_missing_values: Subject index out of bounds";
			}
			
			const vector<float> &subject = m_data[j->second];

			bool has_missing = false;
			
			for(size_t k = 0;k < num_features;++k){
				
				if( (f[k] == MISSING_DATA) && (subject[k] != MISSING_DATA) ){
					f[k] = subject[k];
				}
				
				if(f[k] == MISSING_DATA){
					has_missing = true;
				}
			}
			
			if(!has_missing){
			
				// Stop early if we don't have any more missing values to infer
				break;
			}
		}
		
		new_data[ missing[i] ] = f;
	}
	
	// Collect the corrected data computed by each rank
	for(int i = 0;i < mpi_numtasks;++i){
		
		// Each rank broadcasts the all of the interpolated samples is has processed
		size_t buffer_len = (mpi_rank == i) ? mpi_size(new_data) : 0;
		
		MPI_Bcast( &buffer_len, sizeof(buffer_len), MPI_BYTE, i, MPI_COMM_WORLD );
		
		unsigned char* buffer = new unsigned char[buffer_len];
		
		if(buffer == NULL){
			throw __FILE__ ":infer_missing_values: Unable to allocate buffer";
		}
		
		if(mpi_rank == i){
			mpi_pack(buffer, new_data);
		}
		
		MPI_Bcast( buffer, buffer_len, MPI_BYTE, i, MPI_COMM_WORLD );
		
		vector< vector<float> > local;
		
		mpi_unpack(buffer, local);
		
		delete [] buffer;
		
		// Overwrite the original feature vectors with the newly-inferred feature vectors
		for(size_t j = 0;j < num_data;++j){
			
			// Skip the empty feature vectors
			if( local[j].empty() ){
				continue;
			}

			m_data[j].swap(local[j]);
		}
	}
}

// Return the average feature vector distance
float feature_distance(const vector<float> &m_a, const vector<float> &m_b)
{
	float ret = 0.0;
	
	const size_t len = m_a.size();
	
	if( len != m_b.size() ){
		throw __FILE__ ":feature_distance: feature vector size mismatch";
	}
	
	size_t num_valid = 0;
	
	for(size_t i = 0;i < len;++i){
		
		if( (m_a[i] != MISSING_DATA) && (m_b[i] != MISSING_DATA) ){
			
			const float delta = m_a[i] - m_b[i];
			
			ret += delta*delta;
			++num_valid;
		}
	}
	
	if(num_valid > 0){
		ret = sqrt(ret/num_valid);
	}
	else{
		// Return a large number!
		ret = 1.0e20;
	}
	
	return ret;
}
