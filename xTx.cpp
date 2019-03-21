#include <iostream>
#include <fstream>
#include <sstream>
#include <deque>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <signal.h>

#include "xTx.h"
#include "mpi_util.h"
#include "keys.h"
#include "shuffle.h"

#include "correlation.hpp"

using namespace std;

// Global variables for MPI
int mpi_numtasks;
int mpi_rank;
double start_time;

void terminate_program(int m_sig);
string report_run_time();
MAP< string, vector<float> > merge_features(
	const MAP< string, vector<float> > &m_features);
vector< MAP<DrugID, DoseResponse> > merge_dose_response(
	const vector< MAP<DrugID, DoseResponse> > &m_data);
vector< MAP<DrugID, float> > merge_doubling_time(
	const vector< MAP<DrugID, float> > &m_data);
MAP< string, SET< pair<CellID, DrugID> > > merge_source(
	const MAP< string, SET< pair<CellID, DrugID> > > &m_data);
float find_root(const RandomForest &m_rf, const vector<float> &m_cell_features, 
	const vector<float> &m_drug_features, const float &m_y);
float AUC(const RandomForest &m_rf, const vector<float> &m_cell_features, 
	const vector<float> &m_drug_features);
float p_value(const float &m_val, const deque<float> &m_dist);
bool is_matching_cell_name(const string &m_a, const string &m_b);
string mangle_cell_name(const string &m_name);
bool is_matching_drug(const vector<float> &m_a, const vector<float> &m_b);

int main(int argc, char *argv[])
{
	try{
		MPI_Init(&argc, &argv);
		MPI_Comm_size(MPI_COMM_WORLD, &mpi_numtasks);
		MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
		
		signal( SIGINT, terminate_program );
		signal( SIGTERM, terminate_program );
		signal( SIGSEGV, terminate_program );
		
		start_time = MPI_Wtime();
		
		Options opt;
		
		if(mpi_rank == 0){
			
			// Only the rank 0 processes command line options
			opt.load(argc, argv);
		}
		
		// Share the command line options with the workers
		broadcast(opt, mpi_rank == 0);
		
		if(opt.print_usage){
		
			MPI_Finalize();
			return EXIT_SUCCESS;
		}
		
		time_t profile;
		ofstream fout;
		ofstream fpredict;
		
		if(mpi_rank == 0){
			
			cerr << "Running with " << mpi_numtasks << " MPI rank(s)" << endl;
			
			fout.open( opt.output_file.c_str() );

			if(!fout){
				throw __FILE__ ":main: Unable to open output file for writing";
			}
			
			fout << "# xTx version " << CROSS_TRAIN_VERSION << endl;
		
			fout << '#';
		
			for(int i = 0;i < argc;++i){
				fout << ' ' << argv[i];
			}
		
			fout << endl;
		
			fout << "# Random number seed = " << opt.seed << endl;
			fout << "# Performing " << opt.fold << "-fold cross validation" << endl;
			fout << "# Random forest parameters:" << endl;
			fout << "#\tNumber of trees = " << opt.forest_size << endl;
			fout << "#\tMinimum leaf size = " << opt.forest_min_leaf << endl;
			fout << "#\tBagging fraction = " << opt.forest_bag_fraction << endl;
			
			switch(opt.cv_strategy){
				case OVERLAPPING:
					fout << "# Cross validate using overlapping stratification" << endl;
					break;
				case DISJOINT_CELL:
					fout << "# Cross validate using disjoint-cell stratification" << endl;
					break;
				case DISJOINT_DRUG:
					fout << "# Cross validate using disjoint-drug stratification" << endl;
					break;
				case DISJOINT:
					fout << "# Cross validate using disjoint stratification" << endl;
					break;
				default:
					throw __FILE__ ":main: Unknown cross validation strategy";
			};
			
			if( !opt.prediction_file.empty() ){
			
				fpredict.open(opt.prediction_file);
				
				if(!fpredict){
					throw __FILE__ ":main: Unable to open prediction output file for writing";
				}
			}
		}

		// Make sure that each rank receives a different seed
		drand48_data rand_seed;
		
		srand48_r(opt.seed + mpi_rank, &rand_seed);
		
		// The cell line dose-response data is indexed as: [cell name][drug name][DoseResponse object]
		// (we expect fewer cell names than drug names)
		vector< MAP<DrugID, DoseResponse> > dose_response;
		
		// The PDM doubling time data is indexed as: [cell name][drug name][float]
		// (we expect fewer cell names than drug names)
		vector< MAP<DrugID, float> > pdm_doubling_time;
		
		// The drug and cell line membership of each data source is stored as:
		// data source -> {CellName, DrugName}
		MAP< string, SET< pair<CellID, DrugID> > > source;
		
		// A temporary variable for parsing cell and drug features by string
		MAP< string, vector<float> > tmp_features;
		
		vector<string> cell_feature_names;
		vector< vector<float> > cell_features; // gene expression
		
		vector<string> drug_feature_names;
		vector< vector<float> > drug_features; // drug features and/or fingerprints
		
		// All ranks load the data in parallel!
		profile = time(NULL);
		
		//////////////////////////////////////////////////////////////////////////////////////
		// Parse the cell line features
		//////////////////////////////////////////////////////////////////////////////////////
		parse_feature(opt.cell_feature_file, tmp_features, cell_feature_names);
		
		// Since each rank parses a subset of features, we need to collect and distribute the 
		// feature data to all ranks.
		tmp_features = merge_features(tmp_features);
		
		// Assign every cell line name a unique id number
		const size_t num_cell = tmp_features.size();
		
		MAP<string, CellID> cell_name_to_id; // Use a map for fast lookup
		vector<string> cell_names;
		
		cell_features.resize(num_cell); // Populate by index
		cell_names.reserve(num_cell); // Populate by push_back
		
		for(MAP< string, vector<float> >::iterator i = tmp_features.begin();
			i != tmp_features.end();++i){
			
			cell_names.push_back(i->first);
		}
		
		// Make sure all ranks agree on the same cell line index <-> name mapping
		sort( cell_names.begin(), cell_names.end() );
		
		for(size_t i = 0;i < num_cell;++i){
		
			cell_name_to_id[ cell_names[i] ] = i;
			
			MAP< string, vector<float> >::iterator iter = 
				tmp_features.find(cell_names[i]);
				
			if( iter == tmp_features.end() ){
				throw __FILE__ ":main: Unable to find cell name";
			}
			
			cell_features[i].swap(iter->second);
		}
		
		tmp_features.clear();
		
		profile = time(NULL) - profile;
		
		if(mpi_rank == 0){
		
			cerr << "Found " << num_cell << " cell lines." << endl;	
			cerr << "Parsed cell line features (found " << cell_feature_names.size() << " features per cell in " 
				<< profile << " sec)." << endl;
				
			fout << "# Found " << num_cell << " cell lines with "
				<< cell_feature_names.size() << " features per cell line" << endl;
		}
		
		profile = time(NULL);
		
		parse_feature(opt.drug_feature_file, tmp_features, drug_feature_names);
		
		// Since each rank parses a subset of features, we need to collect and distribute the 
		// feature data to all ranks.
		tmp_features = merge_features(tmp_features);
		
		// Assign every cell line name a unique id number
		const size_t num_drug = tmp_features.size();
		
		MAP<string, DrugID> drug_name_to_id;
		vector<string> drug_names;
		
		drug_features.resize(num_drug); // Populate by index
		drug_names.reserve(num_drug); // Populate by push_back
		
		for(MAP< string, vector<float> >::iterator i = tmp_features.begin();
			i != tmp_features.end();++i){
			
			drug_names.push_back(i->first);
		}
		
		// Make sure all ranks agree on the same drug index <-> name mapping
		sort( drug_names.begin(), drug_names.end() );
		
		for(size_t i = 0;i < num_drug;++i){
			
			drug_name_to_id[ drug_names[i] ] = i;
			
			MAP< string, vector<float> >::iterator iter = 
				tmp_features.find(drug_names[i]);
				
			if( iter == tmp_features.end() ){
				throw __FILE__ ":main: Unable to find drug name";
			}
			
			drug_features[i].swap(iter->second);
		}
		
		tmp_features.clear();
		
		profile = time(NULL) - profile;
		
		if(mpi_rank == 0){
			
			cerr << "Found " << num_drug << " drugs." << endl;	
			cerr << "Parsed drug features (found " << drug_feature_names.size() << " feature per drug in " 
				<< profile << " sec)." << endl;
			
			fout << "# Found " << num_drug << " drugs with "
				<< drug_feature_names.size() << " features per drug" << endl;
		}
	
		const size_t num_completely_missing_cell_features = remove_missing_values(cell_features, cell_feature_names);
		const size_t num_completely_missing_drug_features = remove_missing_values(drug_features, drug_feature_names);

		if(mpi_rank == 0){
			
			cerr << "Found and removed " << num_completely_missing_cell_features
				<< " completely missing cell features" << endl;
			cerr << "Found and removed " << num_completely_missing_drug_features 
				<< " completely missing drug features" << endl;
		}
			
		profile = time(NULL);
		
		parse_dose_response(opt.dose_response_file, dose_response, source,
			cell_name_to_id, drug_name_to_id);

		// Since each rank parses a subset of dose response values, we need to collect and distribute the 
		// dose response data to all ranks.
		dose_response = merge_dose_response(dose_response);
		
		// Handle the PDM data as a special case, since it has associated gene expression and drug 
		// features, but instead of dose repsonse data, it has doubling time data. 
		// ** NOTE ** Since this file is quite small, each rank will parse the file independently!
		// Hence, no need to merge the doubling time data
		MULTIMAP<string, CellID> pdm_prefix_to_cell;
		
		parse_doubling_time(opt.pdm_file, pdm_doubling_time, source,
			cell_name_to_id, drug_name_to_id, pdm_prefix_to_cell);
		
		if( (mpi_rank == 0) && !pdm_prefix_to_cell.empty() ){
			
			const deque<string> pdm = keys(pdm_prefix_to_cell);
			
			cerr << "Found " << pdm.size() << " PDM prefixes" << endl;
			
			for(deque<string>::const_iterator i = pdm.begin();i != pdm.end();++i){
				
				typedef MULTIMAP<string, CellID>::const_iterator I;
				
				const pair<I, I> range = pdm_prefix_to_cell.equal_range(*i);
				
				size_t count = 0;
				
				for(I j = range.first;j != range.second;++j){
					++count;
				}
				
				cerr << "PDM prefix " << *i << " has " << count << " associated gene expression sets" << endl;
			}
		}
		
		// Since each rank parses a subset of the source data, we need to collect and distribute the 
		// source data to all ranks.
		source = merge_source(source);
		
		profile = time(NULL) - profile;
		
		if(mpi_rank == 0){
			cerr << "Parsed dose response data in " << profile << " sec." << endl;
		}
		
		if(mpi_rank == 0){
			
			// Verify that the test and training sets are present in this data
			// (using the drug and cell name prefixes).
			cerr << "Found " << source.size() << " data sources" << endl;
			
			// Summarize the number of cell lines and drugs that are available for each
			// data source
			cerr << "Data source summary:" << endl;
			fout << "Data source summary:" << endl;
			
			for(MAP< string, SET< pair<CellID, DrugID> > >::const_iterator i = source.begin();
				i != source.end();++i){
				
				SET<CellID> cells;
				SET<DrugID> drugs;
				
				for( SET< pair<CellID, DrugID> >::const_iterator j = i->second.begin();
					j != i->second.end();++j){
					
					cells.insert(j->first);
					drugs.insert(j->second);
				}
				
				cerr << '\t' << i->first << " contains " << cells.size() 
					<< " cells and " << drugs.size() << " drugs" << endl;
				fout << '\t' << i->first << " contains " << cells.size() 
					<< " cells and " << drugs.size() << " drugs" << endl;					
			}
			
			// Update the training data sources to account for missing data
			deque<string> new_training_datasets;
			
			for(deque<string>::const_iterator i = opt.training_datasets.begin();
				i != opt.training_datasets.end();++i){
				
				// Is this training source included in the actual data sources?
				if( source.find(*i) == source.end() ){
				
					cerr << "\t\tDid not find training source " << *i << " in the input data" << endl;
					fout << "\t\tDid not find training source " << *i << " in the input data" << endl;
				}
				else{
					new_training_datasets.push_back(*i);
				}
			}
			
			opt.training_datasets = new_training_datasets;
			
			// Update the testing data sources to account for missing data
			deque<string> new_testing_datasets;
			
			for(deque<string>::const_iterator i = opt.testing_datasets.begin();
				i != opt.testing_datasets.end();++i){
				
				// Is this training source included in the actual data sources?
				if( source.find(*i) == source.end() ){
				
					cerr << "\t\tDid not find testing source " << *i << " in the input data" << endl;
					fout << "\t\tDid not find testing source " << *i << " in the input data" << endl;
				}
				else{
					new_testing_datasets.push_back(*i);
				}
			}
			
			opt.testing_datasets = new_testing_datasets;
			
			cerr << "Training on:";
			fout << "Training on:";
			
			for(deque<string>::const_iterator i = opt.training_datasets.begin();
				i != opt.training_datasets.end();++i){
			
				cerr << ' ' << *i;
				fout << ' ' << *i;
			}
			
			cerr << endl;
			fout << endl;
			
			if( opt.testing_datasets.empty() ){
				
				if( pdm_doubling_time.empty() ){
				
					cerr << "Warning -- no valid testing datasets found!" << endl;
					fout << "Warning -- no valid testing datasets found!" << endl;
				}
			}
			else{
				cerr << "Testing on:";
				fout << "Testing on:";

				for(deque<string>::const_iterator i = opt.testing_datasets.begin();
					i != opt.testing_datasets.end();++i){

					cerr << ' ' << *i;
					fout << ' ' << *i;
				}

				cerr << endl;
				fout << endl;
			}
			
			if( !pdm_doubling_time.empty() ){
				
				cerr << "Testing on PDM doubling time data" << endl;
				fout << "Testing on PDM doubling time data" << endl;
			}
		}
		
		// Send the potentially updated source testing and training datasets
		broadcast(opt.training_datasets, mpi_rank == 0);
		broadcast(opt.testing_datasets, mpi_rank == 0);
		
		if( opt.training_datasets.empty() ){
			
			cerr << "\nNo training data sources found" << endl;
			cerr << "\t|training_datasets| = " << opt.training_datasets.size() << endl;
			
			MPI_Finalize();
			return EXIT_SUCCESS;
		}
		
		// Perform cross valiation on the training set before we make predictions for the test set
		if(opt.fold > 1){
			
			if(mpi_rank == 0){
				cerr << "Performing cross validation on the training set" << endl;
			}
			
			if( (mpi_rank == 0) && fpredict.is_open() ){
				fpredict << "# Cross validation predicted vs observed" << endl;
			}
		
			// The Mean Square Error values for each cross validation fold
			float total_mse = 0.0;

			// The Mean Absolute Error values for each cross validation fold
			float total_mae = 0.0;

			float total_mse_of_ave = 0.0;
			float total_num_response = 0.0;
			vector<float> cv_pearson(opt.fold);
			vector<float> cv_spearman(opt.fold);
			vector<size_t> cv_num_response(opt.fold);
			
			// All ranks need to agree on the order of the randomized test/train split
			deque< pair<CellID, DrugID> > cv_by_cell_and_drug;
			deque<DrugID> cv_by_drug;
			deque<CellID> cv_by_cell;
			
			// Rank 0 will build the lists of test and train broadcast to the workers
			for(deque<string>::const_iterator i = opt.training_datasets.begin();
				i != opt.training_datasets.end();++i){
				
				MAP< string, SET< pair<CellID, DrugID> > >::const_iterator iter = source.find(*i);
				
				if( iter == source.end() ){
					throw __FILE__ ":main: Unable to look up training source";
				}
				
				for(SET< pair<CellID, DrugID> >::const_iterator j = iter->second.begin();
					j != iter->second.end();++j){
					
					cv_by_cell.push_back(j->first);
					cv_by_drug.push_back(j->second);
					cv_by_cell_and_drug.push_back(*j);
				}
				
				// Make the lists unique frequently to keep the length reasonable
				make_set(cv_by_cell_and_drug);
				make_set(cv_by_drug);
				make_set(cv_by_cell);
			}

			// All ranks randomize (but will be overwritten by the rank 0 ordering)
			randomize(cv_by_cell_and_drug.begin(), cv_by_cell_and_drug.end(), &rand_seed);
			randomize(cv_by_drug.begin(), cv_by_drug.end(), &rand_seed);
			randomize(cv_by_cell.begin(), cv_by_cell.end(), &rand_seed);
			
			// We have several options for partitioning the input data for cross validation:
			// 1) "Overlapping" : Randomly split the available pairs of cell-drug measurements.
			//    The same cells and drugs can appear in both test *and* train.
			// 2) "disjoint cell" : Randomly partition individual cells between test and training.
			//    The same drugs can appear in both test and train.
			// 3) "disjoint drug" : Randomly partition individual drugs between test and training.
			//    The same cells can appear in both test and train.
			// 4) "disjoint" : Randomly partition individual cells between test and training, and
			//    individual drugs. Test and train will *not* share any drug or cell line.
			
			for(size_t fold = 0;fold < opt.fold;++fold){
				
				if(mpi_rank == 0){

					cerr << "Fold " << fold << endl;
					fout << "Fold " << fold << endl;
				}
				
				deque< pair<CellID, DrugID> > test;
				deque< pair<CellID, DrugID> > train;
				
				switch(opt.cv_strategy){
					case OVERLAPPING:
						
						for(size_t i = 0;i < cv_by_cell_and_drug.size();++i){
							
							if(i%opt.fold == fold){
								test.push_back(cv_by_cell_and_drug[i]);
							}
							else{
								train.push_back(cv_by_cell_and_drug[i]);
							}
						}
						
						break;
					case DISJOINT_CELL:
						
						for(size_t i = 0;i < cv_by_cell.size();++i){
							
							const CellID &cell = cv_by_cell[i];
							
							if(i%opt.fold == fold){
								
								for(deque< pair<CellID, DrugID> >::const_iterator j = cv_by_cell_and_drug.begin();
									j != cv_by_cell_and_drug.end();++j){
									
									if(j->first == cell){
										test.push_back(*j);
									}
								}
							}
							else{
								for(deque< pair<CellID, DrugID> >::const_iterator j = cv_by_cell_and_drug.begin();
									j != cv_by_cell_and_drug.end();++j){
									
									if(j->first == cell){
										train.push_back(*j);
									}
								}
							}
						}
						
						break;
					case DISJOINT_DRUG:
						
						for(size_t i = 0;i < cv_by_drug.size();++i){
							
							const DrugID &drug = cv_by_drug[i];
							
							if(i%opt.fold == fold){
								
								for(deque< pair<CellID, DrugID> >::const_iterator j = cv_by_cell_and_drug.begin();
									j != cv_by_cell_and_drug.end();++j){
									
									if(j->second == drug){
										test.push_back(*j);
									}
								}
							}
							else{
								for(deque< pair<CellID, DrugID> >::const_iterator j = cv_by_cell_and_drug.begin();
									j != cv_by_cell_and_drug.end();++j){
									
									if(j->second == drug){
										train.push_back(*j);
									}
								}
							}
						}
						
						break;
					case DISJOINT:
						for(size_t i = 0;i < cv_by_cell.size();++i){
							
							const CellID &cell = cv_by_cell[i];
							
							for(size_t j = 0;j < cv_by_drug.size();++j){
								
								const DrugID &drug = cv_by_drug[j];
							
								// For the completely disjoint stratification, there will be many
								// drug/cell combinations that are excluded from both the test and
								// the training sets.	
								if( (i%opt.fold == fold) && (j%opt.fold == fold) ){
									
									for(deque< pair<CellID, DrugID> >::const_iterator k = cv_by_cell_and_drug.begin();
										k != cv_by_cell_and_drug.end();++k){

										if( (k->first == cell) && (k->second == drug) ){
											test.push_back(*k);
										}
									}
								}
								else if( (i%opt.fold != fold) && (j%opt.fold != fold) ){

									for(deque< pair<CellID, DrugID> >::const_iterator k = cv_by_cell_and_drug.begin();
										k != cv_by_cell_and_drug.end();++k){

										if( (k->first == cell) && (k->second == drug) ){
											train.push_back(*k);
										}
									}
								}
							}
							
						}
						break;
					default:
						throw __FILE__ ":main: Unknown cross validation strategy";
				};
				
				// All ranks use the test and training set selected by rank 0
				broadcast(test, mpi_rank == 0);
				broadcast(train, mpi_rank == 0);
				
				if(mpi_rank == 0){
				
					cerr << "\t|train| = " << train.size() << endl;
					cerr << "\t|test| = " << test.size() << endl;
				}
			
				// Extract the cell and drug features for the training set so we can 
				// infer missing values. Since these vectors are "full length" (i.e. num_cell
				// and num_drug) there will be some empty feature vectors
				vector< vector<float> > train_cell_features(num_cell);
				vector< vector<float> > train_drug_features(num_drug);
				
				for(deque< pair<CellID, DrugID> >::const_iterator i = train.begin();i != train.end();++i){
					
					if( (i->first >= num_cell) || (i->second >= num_drug) ){
						throw __FILE__ ":main: Index out of bounds";
					}
						
					if( train_cell_features[i->first].empty() ){
						
						// Only copy the feature vectors that have not yet been copied
						train_cell_features[i->first] = cell_features[i->first];
					}
					
					if( train_drug_features[i->second].empty() ){
						
						// Only copy the feature vectors that have not yet been copied
						train_drug_features[i->second] = drug_features[i->second];
					}
				}
				
				// Infer missing values in the cell line features
				if(mpi_rank == 0){
					cerr << "\tInferring missing training feature values ... ";
				}
				
				infer_missing_values(train_cell_features);
				infer_missing_values(train_drug_features);
				
				if(mpi_rank == 0){
					cerr << "done." << endl;
				}
				
				vector<LabeledData> train_data;
				
				// A hopefully reasonable guess as to the amount of training data:
				// number of cell & drug combinations times the number of doses per
				// combination
				train_data.reserve( 10*train.size() );
				
				for(deque< pair<CellID, DrugID> >::const_iterator i = train.begin();
					i != train.end();++i){
					
					if(i->first >= num_cell){
						throw __FILE__ ":main: Cell index out of bounds";
					}
					
					MAP<DrugID, DoseResponse>::const_iterator drug_iter = 
						dose_response[i->first].find(i->second);
					
					if( drug_iter == dose_response[i->first].end() ){
						throw __FILE__ ":main: Unable to find drug id in dose response";
					}
					
					for(DoseResponse::const_iterator j = drug_iter->second.begin();
						j != drug_iter->second.end();++j){
						
						train_data.push_back( LabeledData() );
						
						LabeledData &ref = train_data.back();

						ref.cell_id = i->first;
						ref.drug_id = i->second;
						
						ref.dose = j->first;
						ref.response = j->second;
					}
				}
				
				if(mpi_rank == 0){
				
					cerr << "\tTraining set contains " << train_data.size() 
						<< " dose-repsonse pairs" << endl;
					cerr << "\tBuilding random forest ";
				}
				
				profile = time(NULL);
				
				// Each rank builds a subset of the trees in the forest 
				// (but not all ranks build the same number of trees since the
				// total number of trees is independent of the number of ranks).
				const size_t local_forest_size = opt.forest_size/mpi_numtasks + 
					(mpi_rank < int(opt.forest_size%mpi_numtasks) ? 1 : 0);
				
				RandomForest rf(local_forest_size,
					opt.forest_min_leaf,
					opt.forest_bag_fraction,
					&rand_seed);
				
				rf.build(train_data, 
					train_cell_features, 
					train_drug_features);
								
				profile = time(NULL) - profile;
				
				if(mpi_rank == 0){
				
					cerr << " done." << endl;	
					cerr << "\tRandom forest construction took " << profile << " sec" << endl;
					cerr << "\tPredicting test values" << endl;
				}
				
				// In order to evaluate the test data, we will need to add the test features
				// to the training features and re-interpolate the missing data
				for(deque< pair<CellID, DrugID> >::const_iterator i = test.begin();i != test.end();++i){
					
					const size_t &cell_id = i->first;
					const size_t &drug_id = i->second;
					
					if( (cell_id >= num_cell) || (drug_id >= num_drug) ){
						throw __FILE__ ":main: Index out of bounds";
					}
						
					if( train_cell_features[cell_id].empty() ){
						
						// Only copy the feature vectors that have not yet been copied
						train_cell_features[cell_id] = cell_features[cell_id];
					}
					
					if( train_drug_features[drug_id].empty() ){
						
						// Only copy the feature vectors that have not yet been copied
						train_drug_features[drug_id] = drug_features[drug_id];
					}
				}
				
				// Infer missing values in the cell line and drug features (for the combined
				// test and training data).
				if(mpi_rank == 0){
					cerr << "\tInferring missing training *and* test feature values ... ";
				}
				
				infer_missing_values(train_cell_features);
				infer_missing_values(train_drug_features);
				
				if(mpi_rank == 0){
					cerr << "done." << endl;
				}
				
				// Compute the average *test* response (needed to compute the scikit R2 score).
				float ave_response = 0.0;
				size_t num_response = 0;
				
				for(deque< pair<CellID, DrugID> >::const_iterator i = test.begin();i != test.end();++i){
					
					const size_t &cell_id = i->first;
					const size_t &drug_id = i->second;
					
					// Look up the dose response data for this pair
					if( (cell_id >= num_cell) || (drug_id >= num_drug) ){
						throw __FILE__ ":main: Index out of bounds";
					}
					
					MAP<DrugID, DoseResponse>::const_iterator drug_iter = 
						dose_response[cell_id].find(drug_id);
					
					if( drug_iter == dose_response[cell_id].end() ){
						throw __FILE__ ":main: Unable to find drug id in dose response";
					}
					
					for(DoseResponse::const_iterator j = drug_iter->second.begin();
						j != drug_iter->second.end();++j){
						
						ave_response += j->second;
						++num_response;
					}
				}

				// Normalize the per-dose average test response
				if(num_response > 0){
					ave_response /= num_response;
				}
				
				// The per-fold mean squared error
				float local_mse = 0.0;

				// The per-fold mean absolute error
				float local_mae = 0.0;

				float local_mse_of_ave = 0.0;
				deque< pair<float, float> > per_fold_xy;
								
				for(deque< pair<CellID, DrugID> >::const_iterator i = test.begin();i != test.end();++i){
					
					const size_t &cell_id = i->first;
					const size_t &drug_id = i->second;
					
					// Look up the dose response data for this pair
					if( (cell_id >= num_cell) || (drug_id >= num_drug) ){
						throw __FILE__ ":main: Index out of bounds";
					}
					
					MAP<DrugID, DoseResponse>::const_iterator drug_iter = 
						dose_response[cell_id].find(drug_id);
					
					if( drug_iter == dose_response[cell_id].end() ){
						throw __FILE__ ":main: Unable to find drug id in dose response";
					}
					
					for(DoseResponse::const_iterator j = drug_iter->second.begin();
						j != drug_iter->second.end();++j){
						
						const float pred_y = rf.predict(j->first, 
							train_cell_features[cell_id],
							train_drug_features[drug_id]);
						
						if( (mpi_rank == 0) && fpredict.is_open() ){
							fpredict << pred_y << '\t' << j->second << endl;
							//fpredict << j->first << '\t' << pred_y << '\t' << j->second << endl;
						}
						
						per_fold_xy.push_back( make_pair(pred_y, j->second) );
						
						float delta = pred_y - j->second;
						float delta2 = delta*delta;
	
						local_mse += delta2;
						local_mae += fabs(delta);
	
						// Compute the MSE of the average response
						delta = ave_response - j->second;
						delta2 = delta*delta;
	
						local_mse_of_ave += delta2;
					}
				}
				
				total_mse += local_mse;
				total_mae += local_mae;

				total_mse_of_ave += local_mse_of_ave;
				total_num_response += num_response;
				
				if(num_response > 0){
				
					local_mse /= num_response;
					local_mae /= num_response;
					local_mse_of_ave /= num_response;
				}
				
				const float local_r2 = (local_mse_of_ave > 0.0) ?
					(local_mse_of_ave - local_mse)/local_mse_of_ave :
					0.0;
				
				const float pearson = pearson_correlation(per_fold_xy);
				const float spearman = spearman_correlation(per_fold_xy);
				
				cv_pearson[fold] = pearson;
				cv_spearman[fold] = spearman;
				cv_num_response[fold] = num_response;
				
				if(mpi_rank == 0){
					
					cerr << "\tMSE = " << local_mse << endl;
					fout << "\tMSE = " << local_mse << endl;
				
					cerr << "\tMAE = " << local_mae << endl;
					fout << "\tMAE = " << local_mae << endl;
	
					cerr << "\tMSE of ave test response = " << local_mse_of_ave << endl;
					fout << "\tMSE of ave test response = " << local_mse_of_ave << endl;
					
					cerr << "\tR2 = " << local_r2 << endl;
					fout << "\tR2 = " << local_r2 << endl;
					
					cerr << "\tNumber of test responses = " << num_response << endl;
					fout << "\tNumber of test responses = " << num_response << endl;
					
					cerr << "\tpearson r = " << pearson << endl;
					fout << "\tpearson r = " << pearson << endl;
					
					cerr << "\tspearman r = " << spearman << endl;
					fout << "\tspearman r = " << spearman << endl;
				}
			}
			
			// Compute the average MSE over all cross validation folds
			if(mpi_rank == 0){
				
				float ave_pearson = 0.0;
				float ave_spearman = 0.0;
								
				for(size_t i = 0;i < opt.fold;++i){
				
					ave_pearson += cv_pearson[i]*cv_num_response[i];
					ave_spearman += cv_spearman[i]*cv_num_response[i];
				}
				
				if(total_num_response){
				
					total_mse /= total_num_response;
					total_mae /= total_num_response;

					total_mse_of_ave /= total_num_response;
					ave_pearson /= total_num_response;
					ave_spearman /= total_num_response;
				}
				
				const float total_r2 = (total_mse_of_ave > 0) ?
					(total_mse_of_ave - total_mse)/total_mse_of_ave :
					0.0;
				
				cerr << "Final MSE = " << total_mse << endl;
				fout << "Final MSE = " << total_mse << endl;

				cerr << "Final MAE = " << total_mae << endl;
				fout << "Final MAE = " << total_mae << endl;
				
				cerr << "Final MSE of ave test response = " << total_mse_of_ave<< endl;
				fout << "Final MSE of ave test response = " << total_mse_of_ave << endl;
				
				cerr << "Final R2 = " << total_r2 << endl;
				fout << "Final R2 = " << total_r2 << endl;
				
				cerr << "Number of test responses = " << total_num_response << endl;
				fout << "Number of test responses = " << total_num_response << endl;
				
				cerr << "<pearson> = " << ave_pearson << endl;
				fout << "<pearson> = " << ave_pearson << endl;
				
				cerr << "<spearman> = " << ave_spearman << endl;
				fout << "<spearman> = " << ave_spearman << endl;
			}
		}
		else{
			if(mpi_rank == 0){
				cerr << "Skipping cross validation on the training set" << endl;
			}
		}
		
		if( opt.testing_datasets.empty() && pdm_doubling_time.empty() ){
			
			if(mpi_rank == 0){
				cerr << report_run_time() << endl;
			}
			
			MPI_Finalize();
			return EXIT_SUCCESS;
		}		
		
		// Add a separator between the cross validation results and the transfer
		// learning results if needed.
		if( (mpi_rank == 0) && (opt.fold > 1) && fpredict.is_open() ){
			fpredict <<'\t' << endl;
		}
						
		// Train a random forest on the specified training set
		deque< pair<CellID, DrugID> > train;
		
		for(deque<string>::const_iterator i = opt.training_datasets.begin();
			i != opt.training_datasets.end();++i){

			MAP< string, SET< pair<CellID, DrugID> > >::const_iterator iter = source.find(*i);

			if( iter == source.end() ){
				throw __FILE__ ":main: Unable to look up training source";
			}

			for(SET< pair<CellID, DrugID> >::const_iterator j = iter->second.begin();
				j != iter->second.end();++j){
				train.push_back(*j);
			}

			// Make the train unique frequently to keep the length reasonable
			make_set(train);
		}

		// Extract the cell and drug features for the training set so we can 
		// infer missing values. Since these vectors are "full length" (i.e. num_cell
		// and num_drug) there will be some empty feature vectors
		vector< vector<float> > train_cell_features(num_cell);
		vector< vector<float> > train_drug_features(num_drug);

		for(deque< pair<CellID, DrugID> >::const_iterator i = train.begin();i != train.end();++i){

			if( (i->first >= num_cell) || (i->second >= num_drug) ){
				throw __FILE__ ":main: Index out of bounds";
			}

			if( train_cell_features[i->first].empty() ){

				// Only copy the feature vectors that have not yet been copied
				train_cell_features[i->first] = cell_features[i->first];
			}

			if( train_drug_features[i->second].empty() ){

				// Only copy the feature vectors that have not yet been copied
				train_drug_features[i->second] = drug_features[i->second];
			}
		}

		// Infer missing values in the cell line features
		if(mpi_rank == 0){
			cerr << "Inferring missing training feature values ... ";
		}

		infer_missing_values(train_cell_features);
		infer_missing_values(train_drug_features);

		if(mpi_rank == 0){
			cerr << "done." << endl;
		}

		vector<LabeledData> train_data;

		train_data.reserve( 10*train.size() );
		
		for(deque< pair<CellID, DrugID> >::const_iterator i = train.begin();
			i != train.end();++i){

			if(i->first >= num_cell){
				throw __FILE__ ":main: Cell index out of bounds";
			}

			MAP<DrugID, DoseResponse>::const_iterator drug_iter = 
				dose_response[i->first].find(i->second);

			if( drug_iter == dose_response[i->first].end() ){
				throw __FILE__ ":main: Unable to find drug id in dose response";
			}

			for(DoseResponse::const_iterator j = drug_iter->second.begin();
				j != drug_iter->second.end();++j){

				train_data.push_back( LabeledData() );

				LabeledData &ref = train_data.back();

				ref.cell_id = i->first;
				ref.drug_id = i->second;

				ref.dose = j->first;
				ref.response = j->second;
			}
		}

		if(mpi_rank == 0){
			cerr << "Building random forest ";
		}

		profile = time(NULL);

		// Each rank builds a subset of the trees in the forest 
		// (but not all ranks build the same number of trees since the
		// total number of trees is independent of the number of ranks).
		const size_t local_forest_size = opt.forest_size/mpi_numtasks + 
			(mpi_rank < int(opt.forest_size%mpi_numtasks) ? 1 : 0);

		RandomForest rf(local_forest_size,
			opt.forest_min_leaf,
			opt.forest_bag_fraction,
			&rand_seed);

		rf.build(train_data, 
			train_cell_features, 
			train_drug_features);

		profile = time(NULL) - profile;

		if(mpi_rank == 0){
			
			cerr << " done." << endl;
			cerr << "Random forest construction took " << profile << " sec" << endl;
			cerr << "Predicting test values" << endl;
		}
		
		// Evaluate the random forest on the specified testing set(s)
		for(deque<string>::const_iterator t = opt.testing_datasets.begin();
			t != opt.testing_datasets.end();++t){

			if(mpi_rank == 0){

				cerr << "Making predictions for test set: " << *t << endl;
				fout << "Making predictions for test set: " << *t << endl;
			}
	
			deque< pair<CellID, DrugID> > test;
			
			// Work with a copy of the training features so we can interpolate missing
			// values for each test set separately (from the other test sets)
			vector< vector<float> > test_cell_features(train_cell_features);
			vector< vector<float> > test_drug_features(train_drug_features);
		
			MAP< string, SET< pair<CellID, DrugID> > >::const_iterator iter = source.find(*t);

			if( iter == source.end() ){
				throw __FILE__ ":main: Unable to look up testing source";
			}

			for(SET< pair<CellID, DrugID> >::const_iterator i = iter->second.begin();
				i != iter->second.end();++i){
				
				test.push_back(*i);
			}

			make_set(test);
			
			if(mpi_rank == 0){

				cerr << "\tFound " << test.size() << " test values to predict" << endl;
				fout << "\tFound " << test.size() << " test values to predict" << endl;
			}
			
			// In order to evaluate the test data, we will need to add the test features
			// to the training features and re-interpolate the missing data
			for(deque< pair<CellID, DrugID> >::const_iterator i = test.begin();i != test.end();++i){

				if( (i->first >= num_cell) || (i->second >= num_drug) ){
					throw __FILE__ ":main: Index out of bounds";
				}

				if( test_cell_features[i->first].empty() ){

					// Only copy the feature vectors that have not yet been copied
					test_cell_features[i->first] = cell_features[i->first];
				}

				if( test_drug_features[i->second].empty() ){

					// Only copy the feature vectors that have not yet been copied
					test_drug_features[i->second] = drug_features[i->second];
				}
			}

			// Infer missing values in the cell line and drug features (for the combined
			// test and training data).
			if(mpi_rank == 0){
				cerr << "\tInferring missing training *and* test feature values ... ";
			}

			infer_missing_values(test_cell_features);
			infer_missing_values(test_drug_features);

			if(mpi_rank == 0){
				cerr << "done." << endl;
			}
			
			// Determine which test drugs and cell lines are also in the training set. This will
			// let use bin the results into one of four categories:
			// 	- Disjoint drugs & disjoint cell lines
			// 	- Disjoint drugs & overlapping cell lines
			// 	- Overlapping drugs & disjoint cell lines
			// 	- Overlapping drugs & overlapping cell lines
			
			MAP<CellID, bool> overlapping_cell;
			MAP<DrugID, bool> overlapping_drug;
			
			for(deque< pair<CellID, DrugID> >::const_iterator i = test.begin();i != test.end();++i){
				
				// By default, there is no overlap
				overlapping_cell[i->first] = false;
				overlapping_drug[i->second] = false;
			}
			
			size_t num_overlapping_cell = 0;
			size_t num_overlapping_drug = 0;
			
			for(MAP<CellID, bool>::iterator i = overlapping_cell.begin();i != overlapping_cell.end();++i){
				
				// Can we find a match to this cell line in the training set?
				bool match = false;
				
				for(size_t j = 0;(j < num_cell) && !match;++j){
					
					if( train_cell_features[j].empty() ){
						continue;
					}
	
					if( is_matching_cell_name(cell_names[i->first], cell_names[j]) ){
						match = true;
					}
				}
				
				i->second = match;
				
				num_overlapping_cell += match ? 1 : 0;
			}
			
			for(MAP<CellID, bool>::iterator i = overlapping_drug.begin();i != overlapping_drug.end();++i){
				
				// Can we find a match to this drug in the training set?
				bool match = false;
				
				for(size_t j = 0;(j < num_drug) && !match;++j){
					
					if( train_drug_features[j].empty() ){
						continue;
					}
	
					// Match drugs by features, not names
					if( is_matching_drug(test_drug_features[i->first], train_drug_features[j]) ){
						match = true;
					}
				}
				
				i->second = match;
				
				num_overlapping_drug += match ? 1 : 0;
			}
			
			// Compute the per-dose average test response (needed to compute the scikit R2 score).
			float ave_response = 0.0;
			size_t num_response = 0;

			#define		NUM_BIN		4
			
			vector<float> binned_ave_response(NUM_BIN);
			vector<size_t> binned_num_response(NUM_BIN);
			vector<size_t> binned_num_pair(NUM_BIN);
			
			vector< SET<CellID> > binned_cells(NUM_BIN);
			vector< SET<DrugID> > binned_drugs(NUM_BIN);
			
			for(deque< pair<CellID, DrugID> >::const_iterator i = test.begin();i != test.end();++i){

				const size_t cell_id = i->first;
				const size_t drug_id = i->second;

				// Look up the dose response data for this pair
				if( (cell_id >= num_cell) || (drug_id >= num_drug) ){
					throw __FILE__ ":main: Index out of bounds";
				}

				MAP<DrugID, DoseResponse>::const_iterator drug_iter = 
					dose_response[cell_id].find(drug_id);

				if( drug_iter == dose_response[cell_id].end() ){
					throw __FILE__ ":main: Unable to find drug id in dose response";
				}
				
				CrossValidationStratgy bin;
				
				if(overlapping_cell[cell_id]){
					
					if(overlapping_drug[drug_id]){
						bin = OVERLAPPING;
					}
					else{
						bin = DISJOINT_DRUG;
					}
				}
				else{
					if(overlapping_drug[drug_id]){
						bin = DISJOINT_CELL;
					}
					else{
						bin = DISJOINT;
					}
				}
				
				binned_cells[bin].insert(cell_id);
				binned_drugs[bin].insert(drug_id);
				
				for(DoseResponse::const_iterator j = drug_iter->second.begin();
					j != drug_iter->second.end();++j){

					ave_response += j->second;
					++num_response;
					
					binned_ave_response[bin] += j->second;
					++binned_num_response[bin];
				}
			}

			// Normalize the per-dose average test response
			if(num_response > 0){
				ave_response /= num_response;
			}

			for(size_t i = 0;i < NUM_BIN;++i){
				
				if(binned_num_response[i] > 0){
					binned_ave_response[i] /= binned_num_response[i];
				}
			}
			
			if(mpi_rank == 0){
				cerr << "\tComputed per-dose, average response for test set" << endl;
			}
			
			// The test set mean squared error
			float mse = 0.0;
			vector<float> binned_mse(NUM_BIN);
			
			// The test set mean absolute error
			float mae = 0.0;
			vector<float> binned_mae(NUM_BIN);
			
			float mse_of_ave = 0.0;
			size_t norm = 0;
			
			vector<float> binned_mse_of_ave(NUM_BIN);
			vector<size_t> binned_norm(NUM_BIN);
			
			deque< pair<float, float> > xy;
			
			for(deque< pair<CellID, DrugID> >::const_iterator i = test.begin();i != test.end();++i){

				const size_t cell_id = i->first;
				const size_t drug_id = i->second;

				// Look up the dose response data for this pair
				if( (cell_id >= num_cell) || (drug_id >= num_drug) ){
					throw __FILE__ ":main: Index out of bounds";
				}

				MAP<DrugID, DoseResponse>::const_iterator drug_iter = 
					dose_response[cell_id].find(drug_id);

				if( drug_iter == dose_response[cell_id].end() ){
					throw __FILE__ ":main: Unable to find drug id in dose response";
				}
				
				CrossValidationStratgy bin;
				
				if(overlapping_cell[cell_id]){
					
					if(overlapping_drug[drug_id]){
						bin = OVERLAPPING;
					}
					else{
						bin = DISJOINT_DRUG;
					}
				}
				else{
					if(overlapping_drug[drug_id]){
						bin = DISJOINT_CELL;
					}
					else{
						bin = DISJOINT;
					}
				}
				
				++binned_num_pair[bin];
				
				for(DoseResponse::const_iterator j = drug_iter->second.begin();
					j != drug_iter->second.end();++j){

					const float pred_y = rf.predict(j->first, 
						test_cell_features[cell_id],
						test_drug_features[drug_id]);
					
					if( (mpi_rank == 0) && fpredict.is_open() ){
						fpredict << pred_y << '\t' << j->second << endl;
					}
			
					xy.push_back( make_pair(pred_y, j->second) );
					
					float delta = pred_y - j->second;
					float delta2 = delta*delta;

					mse += delta2;
					binned_mse[bin] += delta2;
					
					mae += fabs(delta);
					binned_mae[bin] += fabs(delta);

					++norm;
					++binned_norm[bin];
					
					// MSE of the average
					delta = ave_response - j->second;
					delta2 = delta*delta;

					mse_of_ave += delta2;
					binned_mse_of_ave[bin] += delta2;
				}
			}

			if(norm > 0){
			
				mse /= norm;
				mae /= norm;

				mse_of_ave /= norm;
			}

			for(size_t i = 0;i < NUM_BIN;++i){
				
				if(binned_norm[i] > 0){
				
					binned_mae[i] /= binned_norm[i];
					binned_mse[i] /= binned_norm[i];
					binned_mse_of_ave[i] /= binned_norm[i];
				}
			}
			
			if(mpi_rank == 0){

				cerr << "\tMSE = " << mse << endl;
				fout << "\tMSE = " << mse << endl;

				cerr << "\tMAE = " << mae << endl;
				fout << "\tMAE = " << mae << endl;
				
				const float r2 = (mse_of_ave - mse)/mse_of_ave;
				
				cerr << "\tR2 = " << r2 << endl;
				fout << "\tR2 = " << r2 << endl;
				
				const float pearson = pearson_correlation(xy);
				const float spearman = spearman_correlation(xy);
				
				cerr << "\tpearson r = " << pearson << endl;
				fout << "\tpearson r = " << pearson << endl;
				
				cerr << "\tspearman r = " << spearman << endl;
				fout << "\tspearman r = " << spearman << endl;
				
				cerr << "\tNumber overlapping cell lines = " << num_overlapping_cell << endl;
				fout << "\tNumber overlapping cell lines = " << num_overlapping_cell << endl;
				
				cerr << "\tNumber overlapping drugs = " << num_overlapping_drug << endl;
				fout << "\tNumber overlapping drugs = " << num_overlapping_drug << endl;
				
				for(int i = OVERLAPPING;i <= DISJOINT;++i){
					
					switch(i){
						case OVERLAPPING:
							cerr << "\tOverlapping cell and overlapping drug" << endl;
							fout << "\tOverlapping cell and overlapping drug" << endl;
							break;
						case DISJOINT_CELL:
							cerr << "\tDisjoint cell and overlapping drug" << endl;
							fout << "\tDisjoint cell and overlapping drug" << endl;
							break;
						case DISJOINT_DRUG:
							cerr << "\tOverlapping cell and disjoint drug" << endl;
							fout << "\tOverlapping cell and disjoint drug" << endl;
							break;
						case DISJOINT:
							cerr << "\tDisjoint cell and disjoint drug" << endl;
							fout << "\tDisjoint cell and disjoint drug" << endl;
							break;
						default:
							throw __FILE__ ":main: Unknown overlap state!";
					};

					cerr << "\t\tNum response = " << binned_norm[i] << " (" 
						<< (100.0*binned_norm[i])/norm << "%)" << endl;
					fout << "\t\tNum response = " << binned_norm[i] << " (" 
						<< (100.0*binned_norm[i])/norm << "%)" << endl;
					
					cerr << "\t\tNum pair = " << binned_num_pair[i] << " (" 
						<< (100.0*binned_num_pair[i])/test.size() << "%)" << endl;
					fout << "\t\tNum pair = " << binned_num_pair[i] << " (" 
						<< (100.0*binned_num_pair[i])/test.size() << "%)" << endl;
					
					cerr << "\t\tNum cell = " << binned_cells[i].size() << endl;
					fout << "\t\tNum cell = " << binned_cells[i].size() << endl;
					
					cerr << "\t\tNum drug = " << binned_drugs[i].size() << endl;
					fout << "\t\tNum drug = " << binned_drugs[i].size() << endl;
					
					cerr << "\t\tMSE = " << binned_mse[i] << endl;
					fout << "\t\tMSE = " << binned_mse[i] << endl;

					cerr << "\t\tMAE = " << binned_mae[i] << endl;
					fout << "\t\tMAE = " << binned_mae[i] << endl;

					const float r2 = (binned_mse_of_ave[i] - binned_mse[i])/binned_mse_of_ave[i];

					cerr << "\t\tR2 = " << r2 << endl;
					fout << "\t\tR2 = " << r2 << endl;
				}
				
				// Add a separator between the transfer learning results for different test sets
				if( fpredict.is_open() ){
					fpredict <<'\t' << endl;
				}
			}
		}
		
		if( !pdm_doubling_time.empty() ){

			if(mpi_rank == 0){

				cerr << "Making predictions for PDM doubling time data" << endl;
				fout << "Making predictions for PDM doubling time data" << endl;
			}
	
			deque< pair<CellID, DrugID> > test;
			
			// Work with a copy of the training features so we can interpolate missing
			// values for each test set separately (from the other test sets)
			vector< vector<float> > test_cell_features(train_cell_features);
			vector< vector<float> > test_drug_features(train_drug_features);
		
			MAP< string, SET< pair<CellID, DrugID> > >::const_iterator iter 
				= source.find("NCIPDM");

			if( iter == source.end() ){
				throw __FILE__ ":main: Unable to look up NCIPDM source for testing";
			}

			for(SET< pair<CellID, DrugID> >::const_iterator i = iter->second.begin();
				i != iter->second.end();++i){
				
				test.push_back(*i);
			}

			make_set(test);
			
			if(mpi_rank == 0){

				cerr << "\tFound " << test.size() << " test values to predict" << endl;
				fout << "\tFound " << test.size() << " test values to predict" << endl;
			}
			
			// In order to evaluate the test data, we will need to add the test features
			// to the training features and re-interpolate the missing data
			for(deque< pair<CellID, DrugID> >::const_iterator i = test.begin();i != test.end();++i){

				if( (i->first >= num_cell) || (i->second >= num_drug) ){
					throw __FILE__ ":main: Index out of bounds";
				}

				if( test_cell_features[i->first].empty() ){

					// Only copy the feature vectors that have not yet been copied
					test_cell_features[i->first] = cell_features[i->first];
				}

				if( test_drug_features[i->second].empty() ){

					// Only copy the feature vectors that have not yet been copied
					test_drug_features[i->second] = drug_features[i->second];
				}
			}

			// Infer missing values in the cell line and drug features (for the combined
			// test and training data).
			if(mpi_rank == 0){
				cerr << "\tInferring missing training *and* test feature values ... ";
			}

			infer_missing_values(test_cell_features);
			infer_missing_values(test_drug_features);

			if(mpi_rank == 0){
				cerr << "done." << endl;
			}
			
			// Compute the pearson and spearman correlations between the observed
			// doubling time values and the predicted GI50 (determined from the
			// machine learning model). Use permutation testing to access the significance
			const size_t num_permutations = 101;
			
			float pearson_auc = 0.0;
			deque<float> pearson_auc_permute;
			
			float spearman_auc = 0.0;
			deque<float> spearman_auc_permute;
			
			float pearson_gi50 = 0.0;
			deque<float> pearson_gi50_permute;
			
			float spearman_gi50 = 0.0;
			deque<float> spearman_gi50_permute;
			
			deque< pair<CellID, DrugID> > curr_test(test);
			vector< MAP<DrugID, float> > curr_pdm_doubling_time(pdm_doubling_time);
				
			for(size_t p = 0;p < num_permutations;++p){
			
				// DEBUG
				if(mpi_rank == 0){
					cerr << "permutation " << p << endl;
				}
				
				double debug_profile = MPI_Wtime();
				
				deque< pair<float, float> > xy_auc;
				deque< pair<float, float> > xy_gi50;

				for(deque< pair<CellID, DrugID> >::const_iterator i = curr_test.begin();i != curr_test.end();++i){

					const size_t cell_id = i->first;
					const size_t drug_id = i->second;

					// Look up the doubling time data for this pair
					if( (cell_id >= num_cell) || (drug_id >= num_drug) ){
						throw __FILE__ ":main: Index out of bounds";
					}

					MAP<DrugID, float>::const_iterator drug_iter = 
						curr_pdm_doubling_time[cell_id].find(drug_id);

					if( drug_iter == curr_pdm_doubling_time[cell_id].end() ){
						throw __FILE__ ":main: Unable to find drug id in the doubling time data";
					}

					// Estimate the GI50 value using root-finding:
					// y(GI50) - 50.0 = 0

					// A potential pit-fall is that the machine learning model (like the data it
					// is trained on) may not be a monotonic function of dose. This can result in
					// multiple values that satisfy: y(GI50) - 50.0 = 0. This simple root finding
					// algorithm *assumes* that the model is monotonic.
					const float gi50 = find_root(rf, 
						test_cell_features[cell_id], 
						test_drug_features[drug_id],
						50.0);

					const float auc = AUC(rf, 
						test_cell_features[cell_id], 
						test_drug_features[drug_id]);

					xy_gi50.push_back( make_pair(gi50, drug_iter->second) );
					xy_auc.push_back( make_pair(auc, drug_iter->second) );

					if( (p == 0) && fpredict.is_open() ){
						fpredict << cell_names[cell_id] << '\t' 
							<< drug_names[drug_id] << '\t' 
							<< gi50 << '\t' 
							<< auc << '\t' 
							<< drug_iter->second << endl;
					}
				}
				
				const float local_pearson_auc = pearson_correlation(xy_auc);
				const float local_spearman_auc = spearman_correlation(xy_auc);
				const float local_pearson_gi50 = pearson_correlation(xy_gi50);
				const float local_spearman_gi50 = spearman_correlation(xy_gi50);
				
				// DEBUG
				debug_profile = MPI_Wtime() - debug_profile;
				
				// DEBUG
				if(mpi_rank == 0){
				
					cerr << "debug_profile (" << curr_test.size() 
						<< ") = " << debug_profile << " sec" << endl;
					
					cerr << "pearson AUC = " << local_pearson_auc << endl;
					cerr << "spearman AUC = " << local_spearman_auc << endl;
					cerr << "pearson GI50 = " << local_pearson_gi50 << endl;
					cerr << "spearman GI50 = " << local_spearman_gi50 << endl;
				}

				if(p == 0){
					
					pearson_auc = local_pearson_auc;
					spearman_auc = local_spearman_auc;

					pearson_gi50 = local_pearson_gi50;
					spearman_gi50 = local_spearman_gi50;
				}
				else{
					
					pearson_auc_permute.push_back(local_pearson_auc);
					spearman_auc_permute.push_back(local_spearman_auc);

					pearson_gi50_permute.push_back(local_pearson_gi50);
					spearman_gi50_permute.push_back(local_spearman_gi50);
				}
				
				// DEBUG
				if(mpi_rank == 0){
					cerr << "permuting values for " << p << endl;
				}
				
				// Permute test and pdm_doubling_time
				const deque<string> src_prefix = keys(pdm_prefix_to_cell);
				
				deque<string> dst_prefix = src_prefix;
				
				randomize(dst_prefix.begin(), dst_prefix.end(), &rand_seed);
				
				// Use the random PDM prefix order provided by rank 0
				broadcast(dst_prefix, mpi_rank == 0);
				
				deque<DrugID> src_drug;
				
				for(deque< pair<CellID, DrugID> >::const_iterator i = test.begin();i != test.end();++i){
					src_drug.push_back(i->second);
				}
				
				make_set(src_drug);
				
				deque<DrugID> dst_drug = src_drug;
				
				randomize(dst_drug.begin(), dst_drug.end(), &rand_seed);
				
				// Use the random drug order provided by rank 0
				broadcast(dst_drug, mpi_rank == 0);
				
				const size_t num_pdm_prefix = src_prefix.size();
				const size_t num_pdm_drug = src_drug.size();
				
				// Generate a new test and pdm_doubling time set for permutation testing
				curr_test.clear();
				
				for(vector< MAP<DrugID, float> >::iterator i = curr_pdm_doubling_time.begin();
					i != curr_pdm_doubling_time.end();++i){
					
					// Clear each element of curr_pdm_doubling_time, but keep the vector size
					// curr_pdm_doubling_time unchanged.
					i->clear();
				}
				
				for(size_t i = 0;i < num_pdm_prefix;++i){
					
					typedef MULTIMAP<string, CellID>::const_iterator I;
						
					// Look up a single representative cell belonging to the source
					// PDM prefix
					const I src_iter = pdm_prefix_to_cell.find(src_prefix[i]);

					if( src_iter == pdm_prefix_to_cell.end() ){
						throw __FILE__ ":main: Unable to find src prefix range";
					}
					
					// DEBUG
					//if(mpi_rank == 0){
					//	cerr << "\tsrc-prefix[" << i << "] = " << src_prefix[i] << endl;
					//	cerr << "\tdst-prefix[" << i << "] = " << dst_prefix[i] << endl;
					//}
					
					for(size_t j = 0;j < num_pdm_drug;++j){
						
						// DEBUG
						//if(mpi_rank == 0){
						//	cerr << "\tsrc-drug[" << j << "] = " << src_drug[j] << endl;
						//	cerr << "\tsrc-drug[" << i << "] = " << dst_drug[j] << endl;
						//}
					
						MAP<DrugID, float>::const_iterator drug_iter = 
							pdm_doubling_time[src_iter->second].find(src_drug[j]);
							
						if( drug_iter == pdm_doubling_time[src_iter->second].end() ){
						
							// DEBUG
							//if(mpi_rank == 0){
							//	cerr << "\t\tmissing data" << endl;
							//}
						
							// This combination is missing data
							continue;
						}
						
						// DEBUG
						//if(mpi_rank == 0){
						//	cerr << "\t\tsource doubling time = " << drug_iter->second << endl;
						//}

						const pair<I, I> dst_range = pdm_prefix_to_cell.equal_range(dst_prefix[i]);
						
						if(dst_range.first == dst_range.second){
							throw __FILE__ ":main: Unable to find dst prefix range";
						}
						
						for(I k = dst_range.first;k != dst_range.second;++k){
						
							curr_test.push_back( make_pair(k->second, dst_drug[j]) );
							curr_pdm_doubling_time[k->second][ dst_drug[j] ] = drug_iter->second;
						}
					}
				}				
			}
			
			if(mpi_rank == 0){
				
				cerr << "\tAUC pearson r = " << pearson_auc << endl;
				cerr << "\tAUC pearson r p-value = " 
					<< p_value(pearson_auc, pearson_auc_permute) << endl;
				
				fout << "\tAUC pearson r = " << pearson_auc << endl;
				fout << "\tAUC pearson r p-value = " 
					<< p_value(pearson_auc, pearson_auc_permute) << endl;
				
				cerr << "\tAUC spearman r = " << spearman_auc << endl;
				cerr << "\tAUC spearman r p-value = " 
					<< p_value(spearman_auc, spearman_auc_permute) << endl;
				
				fout << "\tAUC spearman r = " << spearman_auc << endl;
				fout << "\tAUC spearman r p-value = " 
					<< p_value(spearman_auc, spearman_auc_permute) << endl;
				
				cerr << "\tGI50 pearson r = " << pearson_gi50 << endl;
				cerr << "\tGI50 pearson r p-value = " 
					<< p_value(pearson_gi50, pearson_gi50_permute) << endl;
				
				fout << "\tGI50 pearson r = " << pearson_gi50 << endl;
				fout << "\tGI50 pearson r p-value = " 
					<< p_value(pearson_gi50, pearson_gi50_permute) << endl;
				
				cerr << "\tGI50 spearman r = " << spearman_gi50 << endl;
				cerr << "\tGI50 spearman r p-value = " 
					<< p_value(spearman_gi50, spearman_gi50_permute) << endl;
				
				fout << "\tGI50 spearman r = " << spearman_gi50 << endl;
				fout << "\tGI50 spearman r p-value = " 
					<< p_value(spearman_gi50, spearman_gi50_permute) << endl;
			}
		}
		
		if(mpi_rank == 0){
			cerr << report_run_time() << endl;
		}
		
		MPI_Finalize();
	}
	catch(const char *error){
	
		cerr << "Caught the error " << error << endl;
		return EXIT_FAILURE;
	}
	catch(const string error){
	
		cerr << "Caught the error " << error << endl;
		return EXIT_FAILURE;
	}
	catch(...){
	
		cerr << "Caught an unhandled error" << endl;
		return EXIT_FAILURE;
	}
	
	return EXIT_SUCCESS;
}


// Report our rank, the signal we caught and the time spent running
void terminate_program(int m_sig)
{
	cerr << "[" << mpi_rank << "] caught signal " << m_sig << endl;
	cerr << report_run_time() << endl;
	
	MPI_Abort(MPI_COMM_WORLD, 0);
}

// Run time computes the total run time. The results are formatted as a string.
string report_run_time()
{
	double elapsed_time = MPI_Wtime() - start_time; // In sec
	
	const double elapsed_sec = fmod(elapsed_time, 60.0);
	
	elapsed_time = (elapsed_time - elapsed_sec)/60.0; // In min
	
	const double elapsed_min = fmod(elapsed_time, 60.0);
	elapsed_time = (elapsed_time - elapsed_min)/60.0; // In hour
	
	const double elapsed_hour = fmod(elapsed_time, 24.0);
	elapsed_time = (elapsed_time - elapsed_hour)/24.0; // In day
	
	stringstream sout;
	
	sout << "Run time is " 
		<< elapsed_time 
		<< " days, "
		<< elapsed_hour
		<< " hours, "
		<< elapsed_min
		<< " min and "
		<< elapsed_sec
		<< " sec";
	
	return sout.str();
}

MAP< string, vector<float> > merge_features(const MAP< string, vector<float> > &m_features)
{
	MAP< string, vector<float> > ret(m_features);
	
	for(int i = 0;i < mpi_numtasks;++i){
		
		// Each rank broadcast its features to all other ranks
		size_t buffer_size = (i == mpi_rank) ? mpi_size(m_features) : 0;
		
		MPI_Bcast( &buffer_size, sizeof(buffer_size), MPI_BYTE, i, MPI_COMM_WORLD );
		
		unsigned char *buffer = new unsigned char[buffer_size];
		
		if(buffer == NULL){
			throw __FILE__ ":merge_features: Unable to allocate buffer";
		}
		
		if(i == mpi_rank){
			mpi_pack(buffer, m_features);
		}

		MPI_Bcast( buffer, buffer_size, MPI_BYTE, i, MPI_COMM_WORLD );
		
		if(i != mpi_rank){
			
			MAP< string, vector<float> > local;
			
			mpi_unpack(buffer, local);
			
			for(MAP< string, vector<float> >::const_iterator j = local.begin();j != local.end();++j){
			
				if( ret.find(j->first) != ret.end() ){
					throw __FILE__ ":merge_features: Duplicate feature detected!";
				}
				
				ret[j->first] = j->second;
			}
		}
		
		delete [] buffer;
	}
	
	return ret;
}

vector< MAP<DrugID, DoseResponse> > merge_dose_response(const vector< MAP<DrugID, DoseResponse> > &m_data)
{
	const size_t num_data = m_data.size();
	
	vector< MAP<DrugID, DoseResponse> > ret(m_data);
	
	for(int i = 0;i < mpi_numtasks;++i){
		
		// Each rank broadcast its dose response data to all other ranks
		size_t buffer_size = (i == mpi_rank) ? mpi_size(m_data) : 0;
		
		MPI_Bcast( &buffer_size, sizeof(buffer_size), MPI_BYTE, i, MPI_COMM_WORLD );
		
		unsigned char *buffer = new unsigned char[buffer_size];
		
		if(buffer == NULL){
			throw __FILE__ ":merge_dose_response: Unable to allocate buffer";
		}
		
		if(i == mpi_rank){
			mpi_pack(buffer, m_data);
		}

		MPI_Bcast( buffer, buffer_size, MPI_BYTE, i, MPI_COMM_WORLD );
		
		if(i != mpi_rank){
			
			vector< MAP<DrugID, DoseResponse> > local;
			
			mpi_unpack(buffer, local);
			
			for(size_t j = 0;j < num_data;++j){
				
				MAP<DrugID, DoseResponse> &ref = ret[j];
				
				for(MAP<DrugID, DoseResponse>::const_iterator k = local[j].begin();
					k != local[j].end();++k){
					
					// Is there already a record for this cell/drug combination?
					MAP<DrugID, DoseResponse>::iterator iter = ref.find(k->first);
					
					if( iter == ref.end() ){
						
						// A simple copy will do
						ref[k->first] = k->second;
					}
					else{
						
						// Merge the dose response values
						for(DoseResponse::const_iterator m = k->second.begin();
							m != k->second.end();++m){
							
							iter->second.push_back(*m);
						}
					}
				}
			}
		}
		
		delete [] buffer;
	}
	
	// The final dose response data must be sorted by dose so that all ranks make predictions on the same
	// dose values at the same time -- this is important since the random forest is distributed!.
	for(vector< MAP<DrugID, DoseResponse> >::iterator i = ret.begin();i != ret.end();++i){
		
		for(MAP<DrugID, DoseResponse>::iterator j = i->begin();j != i->end();++j){
			sort( j->second.begin(), j->second.end() );
		}
	}
	
	return ret;
}

vector< MAP<DrugID, float> > merge_doubling_time(const vector< MAP<DrugID, float> > &m_data)
{
	const size_t num_data = m_data.size();
	
	vector< MAP<DrugID, float> > ret(m_data);
	
	for(int i = 0;i < mpi_numtasks;++i){
		
		// Each rank broadcast its dose response data to all other ranks
		size_t buffer_size = (i == mpi_rank) ? mpi_size(m_data) : 0;
		
		MPI_Bcast( &buffer_size, sizeof(buffer_size), MPI_BYTE, i, MPI_COMM_WORLD );
		
		unsigned char *buffer = new unsigned char[buffer_size];
		
		if(buffer == NULL){
			throw __FILE__ ":merge_doubling_time: Unable to allocate buffer";
		}
		
		if(i == mpi_rank){
			mpi_pack(buffer, m_data);
		}

		MPI_Bcast( buffer, buffer_size, MPI_BYTE, i, MPI_COMM_WORLD );
		
		if(i != mpi_rank){
			
			vector< MAP<DrugID, float> > local;
			
			mpi_unpack(buffer, local);
			
			for(size_t j = 0;j < num_data;++j){
				
				if( local[j].empty() ){
					continue;
				}
				
				// There should *not* be an existing record for this cell/drug combination!
				// The replicate data for the records with replicates:
				//	NCIPDM.CN0428~F1126
				//	NCIPDM.349418~098-R
				// have been manually averaged into
				// a single record
				if( !ret[j].empty() ){
					throw __FILE__ ":merge_doubling_time: Duplicate record!";
				}
				
				ret[j] = local[j];
			}
		}
		
		delete [] buffer;
	}
	
	return ret;
}

MAP< string, SET< pair<CellID, DrugID> > > merge_source(const MAP< string, SET< pair<CellID, DrugID> > > &m_data)
{
	MAP< string, SET< pair<CellID, DrugID> > > ret(m_data);
	
	for(int i = 0;i < mpi_numtasks;++i){
		
		// Each rank broadcast its dose response data to all other ranks
		size_t buffer_size = (i == mpi_rank) ? mpi_size(m_data) : 0;
		
		MPI_Bcast( &buffer_size, sizeof(buffer_size), MPI_BYTE, i, MPI_COMM_WORLD );
		
		unsigned char *buffer = new unsigned char[buffer_size];
		
		if(buffer == NULL){
			throw __FILE__ ":merge_dose_response: Unable to allocate buffer";
		}
		
		if(i == mpi_rank){
			mpi_pack(buffer, m_data);
		}

		MPI_Bcast( buffer, buffer_size, MPI_BYTE, i, MPI_COMM_WORLD );
		
		if(i != mpi_rank){
			
			MAP< string, SET< pair<CellID, DrugID> > > local;
			
			mpi_unpack(buffer, local);
			
			typedef MAP< string, SET< pair<CellID, DrugID> > >::const_iterator I;
			
			for(I j = local.begin();j != local.end();++j){
				
				SET< pair<CellID, DrugID> > &ref = ret[j->first];
				
				for(SET< pair<CellID, DrugID> >::const_iterator k = j->second.begin();
					k != j->second.end();++k){
					
					ref.insert(*k);
				}
			}
		}
		
		delete [] buffer;
	}
	
	return ret;
}

float AUC(const RandomForest &m_rf, const vector<float> &m_cell_features, 
	const vector<float> &m_drug_features)
{
	float c_left = -15.0;
	float c_right = -0.5;
	
	const size_t N = 100;
	
	const float delta = (c_right - c_left)/N;
	
	float ret = 0.0;
	
	float last = m_rf.predict(c_left, m_cell_features, m_drug_features);;
	
	// Use the trapezoid method for now!
	for(size_t i = 1;i <= N;++i){
		
		float next = m_rf.predict(c_left + i*delta, m_cell_features, m_drug_features);
		
		ret += next + last;
		
		last = next;
	}
	
	ret *= 0.5*delta;
	
	return ret;
}
	
// Assume that the function we're root finding on is a monotonically decreasing
// function of concentration!
float find_root(const RandomForest &m_rf, const vector<float> &m_cell_features, 
	const vector<float> &m_drug_features, const float &m_y)
{

	float c_left = -15.0;
	float c_right = -0.5;
	
	// DEBUG	
	//for(float c = c_left;c <= c_right;c += 0.5){
	//
	//	const float y = m_rf.predict(c, m_cell_features, m_drug_features);
	//
	//	if(mpi_rank == 0){
	//		cout << "FUNC\t" << c << '\t' << y << endl;
	//	}
	//}

	//if(mpi_rank == 0){
	//	cout << endl;
	//}
	
	float y_left = m_rf.predict(c_left, m_cell_features, m_drug_features);
	float y_right = m_rf.predict(c_right, m_cell_features, m_drug_features);
	
	if(y_right > m_y){
		return c_right;
	}
	
	if(y_left < m_y){
		return c_left;
	}
	
	const float eps = 1.0e-3;
	const size_t max_iteration = 100;
	size_t iteration = 0;
	
	while(iteration < max_iteration){
		
		float c_mid = (c_left + c_right)*0.5;
	
		MPI_Bcast( &c_mid, 1, MPI_FLOAT, 0, MPI_COMM_WORLD );
		
		// Make sure that all ranks use the extact same stopping criteria
		unsigned char done = fabs(c_left - c_right) <= eps;
		
		MPI_Bcast( &done, 1, MPI_BYTE, 0, MPI_COMM_WORLD );
		
		if(done){
			return c_mid;
		}
		
		const float y_mid = m_rf.predict(c_mid, m_cell_features, m_drug_features);
		
		// DEBUG
		//if(mpi_rank == 0){
		//	cout << "ROOT\t" << c_mid << '\t' << y_mid << endl;
		//}
		
		done = fabs(y_mid - m_y) <= eps;
		
		MPI_Bcast( &done, 1, MPI_BYTE, 0, MPI_COMM_WORLD );
		
		if(done){
			return c_mid;
		}
		
		// Updating the bounds assumes that y(c) is a monotonically decreasing function of
		// concentration. While this is not true in practice (due to noise, experimental artifacts,
		// biology, ...), it captures the traditional definition of GI50 that is obtained by curve 
		// fitting a monotonically decreasing function of concetration (i.e. Hill equation, logistic 
		// function, etc.).
		if(y_mid > m_y){
			c_left = c_mid;
		}
		else{
			c_right = c_mid;
		}
		
		++iteration;
	}
	
	// We have failed to converge!
	throw __FILE__ ":find_root: Failed to converge!";
	return 0.0;
}

float p_value(const float &m_val, const deque<float> &m_dist)
{
	size_t num_greater = 0;
	const size_t total = m_dist.size();
	
	if(total == 0){
		throw __FILE__ ":p_value: |m_dist| == 0";
	}
	
	for(deque<float>::const_iterator i = m_dist.begin();i != m_dist.end();++i){
		
		if( fabs(m_val) <= fabs(*i) ){
			++num_greater;
		}
	}

	return double(num_greater)/total;
}


bool is_matching_cell_name(const string &m_a, const string &m_b)
{
	return mangle_cell_name(m_a) == mangle_cell_name(m_b);
}

string mangle_cell_name(const string &m_name)
{
	string ret;
	
	const string::size_type loc = m_name.find('.');
	
	if(loc == string::npos){
		throw __FILE__ ":mangle_cell_name: Unable to find '.' source name separator";
	}
	
	const size_t len = m_name.size();
	
	for(size_t i = loc + 1;i < len;++i){
		
		// Skip any non-alpha numeric character
		if( isalnum(m_name[i]) ){
			ret.push_back( toupper(m_name[i]) );
		}
	}
	
	return ret;
}

bool is_matching_drug(const vector<float> &m_a, const vector<float> &m_b)
{
	const double drug_match_threshold = 1.0e-3;
	
	const size_t num_feature = m_a.size();
	
	if( num_feature != m_b.size() ){
		throw __FILE__ ":is_matching_drug: Unequal number of drug features";
	}
	
	double d = 0.0;
	
	for(size_t i = 0;i < num_feature;++i){
		
		const float delta = m_a[i] - m_b[i];
		
		d += delta*delta;
	}
	
	d = sqrt(d/num_feature);
	
	return (d < drug_match_threshold) ? true : false;
}
