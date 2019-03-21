#include "mpi_util.h"
#include "xTx.h"

#include <string.h>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////
// Specialization for std::string
/////////////////////////////////////////////////////////////////////////////////////////
template<>
size_t mpi_size(const string &m_str)
{
	return sizeof(size_t) + m_str.size();
}

template<>
unsigned char* mpi_pack(unsigned char* m_ptr, const string &m_str)
{
	size_t len = m_str.size();
	
	memcpy( m_ptr, &len, sizeof(size_t) );
	m_ptr += sizeof(size_t);
	
	memcpy(m_ptr, m_str.c_str(), len);
	m_ptr += len;
	
	return m_ptr;
}

template<>
unsigned char* mpi_unpack(unsigned char* m_ptr, string &m_str)
{
	size_t len;
	
	memcpy( &len, m_ptr, sizeof(size_t) );
	m_ptr += sizeof(size_t);
	
	m_str.assign( (char*)m_ptr, len );
	m_ptr += len;
	
	return m_ptr;
}

/////////////////////////////////////////////////////////////////////////////////////////
// Specialization for Options
/////////////////////////////////////////////////////////////////////////////////////////
template<>
size_t mpi_size(const Options &m_obj)
{
	return  mpi_size(m_obj.print_usage) + 
		mpi_size(m_obj.seed) + 
		mpi_size(m_obj.fold) + 
		mpi_size(m_obj.forest_size) + 
		mpi_size(m_obj.forest_bag_fraction) + 
		mpi_size(m_obj.forest_min_leaf) + 
		mpi_size(m_obj.num_response_sample) + 
		mpi_size(m_obj.min_response_sample) + 
		mpi_size(m_obj.max_response_sample) + 
		mpi_size(m_obj.cv_strategy) + 
		mpi_size(m_obj.training_datasets) + 
		mpi_size(m_obj.testing_datasets) + 
		mpi_size(m_obj.dose_response_file) + 
		mpi_size(m_obj.cell_feature_file) + 
		mpi_size(m_obj.drug_feature_file) + 
		mpi_size(m_obj.output_file) +
		mpi_size(m_obj.prediction_file) +
		mpi_size(m_obj.pdm_file);
}

template<>
unsigned char* mpi_pack(unsigned char* m_ptr, const Options &m_obj)
{
	m_ptr = mpi_pack(m_ptr, m_obj.print_usage);
	m_ptr = mpi_pack(m_ptr, m_obj.seed);
	m_ptr = mpi_pack(m_ptr, m_obj.fold);
	m_ptr = mpi_pack(m_ptr, m_obj.forest_size);
	m_ptr = mpi_pack(m_ptr, m_obj.forest_bag_fraction);
	m_ptr = mpi_pack(m_ptr, m_obj.forest_min_leaf);
	m_ptr = mpi_pack(m_ptr, m_obj.num_response_sample);
	m_ptr = mpi_pack(m_ptr, m_obj.min_response_sample);
	m_ptr = mpi_pack(m_ptr, m_obj.max_response_sample);
	m_ptr = mpi_pack(m_ptr, m_obj.cv_strategy);
	m_ptr = mpi_pack(m_ptr, m_obj.training_datasets);
	m_ptr = mpi_pack(m_ptr, m_obj.testing_datasets);
	m_ptr = mpi_pack(m_ptr, m_obj.dose_response_file);
	m_ptr = mpi_pack(m_ptr, m_obj.cell_feature_file);
	m_ptr = mpi_pack(m_ptr, m_obj.drug_feature_file);
	m_ptr = mpi_pack(m_ptr, m_obj.output_file);
	m_ptr = mpi_pack(m_ptr, m_obj.prediction_file);
	m_ptr = mpi_pack(m_ptr, m_obj.pdm_file);
	
	return m_ptr;
}

template<>
unsigned char* mpi_unpack(unsigned char* m_ptr, Options &m_obj)
{
	m_ptr = mpi_unpack(m_ptr, m_obj.print_usage);
	m_ptr = mpi_unpack(m_ptr, m_obj.seed);
	m_ptr = mpi_unpack(m_ptr, m_obj.fold);
	m_ptr = mpi_unpack(m_ptr, m_obj.forest_size);
	m_ptr = mpi_unpack(m_ptr, m_obj.forest_bag_fraction);
	m_ptr = mpi_unpack(m_ptr, m_obj.forest_min_leaf);
	m_ptr = mpi_unpack(m_ptr, m_obj.num_response_sample);
	m_ptr = mpi_unpack(m_ptr, m_obj.min_response_sample);
	m_ptr = mpi_unpack(m_ptr, m_obj.max_response_sample);
	m_ptr = mpi_unpack(m_ptr, m_obj.cv_strategy);
	m_ptr = mpi_unpack(m_ptr, m_obj.training_datasets);
	m_ptr = mpi_unpack(m_ptr, m_obj.testing_datasets);
	m_ptr = mpi_unpack(m_ptr, m_obj.dose_response_file);
	m_ptr = mpi_unpack(m_ptr, m_obj.cell_feature_file);
	m_ptr = mpi_unpack(m_ptr, m_obj.drug_feature_file);
	m_ptr = mpi_unpack(m_ptr, m_obj.output_file);
	m_ptr = mpi_unpack(m_ptr, m_obj.prediction_file);
	m_ptr = mpi_unpack(m_ptr, m_obj.pdm_file);
	
	return m_ptr;
}

/////////////////////////////////////////////////////////////////////////////////////////
// Specialization for TreeNode
/////////////////////////////////////////////////////////////////////////////////////////
template<>
size_t mpi_size(const RandomForest::TreeNode &m_obj)
{
	size_t ret = 
		mpi_size(m_obj.boundary) + 
		mpi_size(m_obj.boundary_feature) + 
		mpi_size(m_obj.prediction)+
		mpi_size(m_obj.left)+
		mpi_size(m_obj.right);
	
	return ret;
}

template<>
unsigned char* mpi_pack(unsigned char* m_ptr, const RandomForest::TreeNode &m_obj)
{
	m_ptr = mpi_pack(m_ptr, m_obj.boundary); 
	m_ptr = mpi_pack(m_ptr, m_obj.boundary_feature);
	m_ptr = mpi_pack(m_ptr, m_obj.prediction);
	m_ptr = mpi_pack(m_ptr, m_obj.left);
	m_ptr = mpi_pack(m_ptr, m_obj.right);
	
	return m_ptr;
}

template<>
unsigned char* mpi_unpack(unsigned char* m_ptr, RandomForest::TreeNode &m_obj)
{
	m_ptr = mpi_unpack(m_ptr, m_obj.boundary); 
	m_ptr = mpi_unpack(m_ptr, m_obj.boundary_feature);
	m_ptr = mpi_unpack(m_ptr, m_obj.prediction);
	m_ptr = mpi_unpack(m_ptr, m_obj.left);
	m_ptr = mpi_unpack(m_ptr, m_obj.right);
	
	return m_ptr;
}
