#ifndef __CROSS_TRAIN
#define __CROSS_TRAIN

#include <string>
#include <deque>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#define	CROSS_TRAIN_VERSION	"0.1"

#define	MAP std::unordered_map
#define	MULTIMAP std::unordered_multimap
#define	SET std::unordered_set

#define	MISSING_DATA	0xFFFFFFFF

// Note that the high order bit of cell_id is "borrowed" for
// the partitioning step in the random forest algorithm. This
// requires that we never actually need a cell id that is so
// large that it requires the high order bit.
typedef unsigned int __ID;
typedef __ID CellID;
typedef __ID DrugID;

// We have several options for partitioning the input data for cross validation:
// 1) OVERLAPPING
//    Randomly split the available pairs of cell-drug measurements.
//    The same cells and drugs can appear in both test *and* train.
// 2) DISJOINT_CELL
//    Randomly partition individual cells between test and training.
//    The same drugs can appear in both test and train.
// 3) DISJOINT_DRUG
//    Randomly partition individual drugs between test and training.
//    The same cells can appear in both test and train.
// 4) DISJOINT
//    Randomly partition individual cells between test and training, and
//    individual drugs. Test and train will *not* share any drug or cell line.

typedef enum {
	OVERLAPPING,
	DISJOINT_CELL,
	DISJOINT_DRUG,
	DISJOINT
} CrossValidationStratgy;

typedef std::deque< std::pair<float, float> > DoseResponse;

struct Options
{
	bool print_usage;
	
	unsigned int seed;
	unsigned int fold;
	
	unsigned int forest_size;
	double forest_bag_fraction;
	unsigned int forest_min_leaf;
	
	unsigned int num_response_sample;
	float min_response_sample;
	float max_response_sample;
	
	CrossValidationStratgy cv_strategy;
	
	std::deque<std::string> training_datasets;
	std::deque<std::string> testing_datasets;
	
	std::string dose_response_file;
	std::string cell_feature_file;
	std::string drug_feature_file;
	std::string output_file;
	std::string prediction_file;
	std::string pdm_file;
	
	Options()
	{
		// Do nothing!
	};
	
	Options(int argc, char* argv[])
	{
		load(argc, argv);
	};
	
	void load(int argc, char* argv[]);
	
	template<class T> friend 
		size_t mpi_size(const T &m_obj);
	template<class T> friend 
		unsigned char* mpi_pack(unsigned char* m_ptr, const T &m_obj);
	template<class T> friend 
		unsigned char* mpi_unpack(unsigned char* m_ptr, T &m_obj);
};

template<> size_t mpi_size(const Options &m_obj);
template<> unsigned char* mpi_pack(unsigned char* m_ptr, const Options &m_obj);
template<> unsigned char* mpi_unpack(unsigned char* m_ptr, Options &m_obj);

namespace std
{
	template<> struct hash< pair<__ID, __ID> >
	{
		size_t operator()(const pair<__ID, __ID>& m_pair) const
		{
			// Pack the two 32-bit IDs into a 64-bit hash key
			return (size_t(m_pair.first) << 32) | size_t(m_pair.second);
		};
	};
}

struct LabeledData
{
	// Note that the high order bit of cell_id is "borrowed" for
	// the partitioning step in the random forest algorithm. This
	// requires that we never actually need a cell id that is so
	// large that it requires the high order bit.
	CellID cell_id;
	DrugID drug_id;
	float dose;
	float response;
};

class RandomForest
{	
	public:
		typedef enum {
			DOSE_FEATURE,
			CELL_FEATURE,
			DRUG_FEATURE,
			UNKNOWN_FEATURE
		} FeatureType;

		struct TreeNode
		{

			TreeNode()
			{
				left = right = 0;
			};


			std::pair<unsigned int /*index*/, 
				float /*threshold*/> boundary;
			FeatureType boundary_feature;

			unsigned int left;
			unsigned int right;

			float prediction;

			inline bool is_leaf() const
			{
				return left == right;
			};

			template<class T> friend 
				size_t mpi_size(const T &m_obj);
			template<class T> friend 
				unsigned char* mpi_pack(unsigned char* m_ptr, const T &m_obj);
			template<class T> friend 
				unsigned char* mpi_unpack(unsigned char* m_ptr, T &m_obj);

		};


	private:
		
		typedef std::deque<TreeNode> Tree;
		
		std::vector<Tree> forest;
		
		size_t forest_size;
		size_t forest_leaf;
		float forest_feature_bag;
		drand48_data *ptr_seed;
		
		void build_tree(Tree &m_tree, 
			std::vector<LabeledData>::iterator m_begin,
			std::vector<LabeledData>::iterator m_end,
			const std::vector< std::vector<float> > &m_cell_features,
			const std::vector< std::vector<float> > &m_drug_features,
			const std::vector< std::pair<unsigned int, 
				FeatureType> > &m_feature_indicies);
			
		float predict_tree(const float &m_conc, const Tree &m_tree,
			const std::vector<float> &m_cell_features, 
			const std::vector<float> &m_drug_features) const;
		
	public:
	
		RandomForest(const size_t &m_forest_size, 
			const size_t &m_forest_leaf, 
			const float &m_forest_feature_bag,
			drand48_data *m_seed_ptr) :
			forest_size(m_forest_size),
			forest_leaf(m_forest_leaf),
			forest_feature_bag(m_forest_feature_bag),
			ptr_seed(m_seed_ptr)
		{
		};
		
		// m_data needs to be mutable to allow the tree building process
		// to reorder the data points "in place" as individual trees
		// are built.
		void build(std::vector<LabeledData> &m_data,
			const std::vector< std::vector<float> > &m_cell_features,
			const std::vector< std::vector<float> > &m_drug_features);
		
		float predict(const float &m_conc, 
			const std::vector<float> &m_cell_features, 
			const std::vector<float> &m_drug_features) const;
};

template<> size_t mpi_size(const RandomForest::TreeNode &m_obj);
template<> unsigned char* mpi_pack(unsigned char* m_ptr, const RandomForest::TreeNode &m_obj);
template<> unsigned char* mpi_unpack(unsigned char* m_ptr, RandomForest::TreeNode &m_obj);

// io.cpp
void parse_dose_response(const std::string &m_filename, 
	std::vector< MAP<DrugID, DoseResponse> > &m_data,
	MAP< std::string, SET< std::pair<CellID, DrugID> > > &m_source,
	const MAP<std::string, CellID> &m_cell_names,
	const MAP<std::string, DrugID> &m_drug_names);

void parse_doubling_time(const std::string &m_filename, 
	std::vector< MAP<DrugID, float> > &m_data,
	MAP< std::string, SET< std::pair<CellID, DrugID> > > &m_source,
	const MAP<std::string, CellID> &m_cell_names,
	const MAP<std::string, DrugID> &m_drug_names,
	MULTIMAP<std::string, CellID> &m_prefix_to_cell);
	
void parse_feature(const std::string &m_filename, MAP< std::string, std::vector<float> > &m_features, 
	std::vector<std::string> &m_feature_names);

// missing.cpp
void infer_missing_values(std::vector< std::vector<float> > &m_data);
size_t remove_missing_values(std::vector< std::vector<float> > &m_data, 
	std::vector<std::string> &m_names);

#endif // __CROSS_TRAIN
