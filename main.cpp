// Dose-response regression test bed for NCI-60, GDSC, CCLE and other similar datasets
// J. D. Gans
// Bioscience Division, B-11
// Los Alamos National Laboratory
// Fri May 20 15:21:31 2016

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

#include "cancer.h"
#include "shuffle.h"
#include "deque_set.h"
#include "mpi_util.h"
#include "dose_response_function.h"
#include "multivariate_tree_model.h"
#include "elastic_net.h"

// Random number generator
#include <gsl/gsl_rng.h>

using namespace std;

// For making a histogram of the frequency of the number of drug replicates. Sort by #replicate.
struct sort_by_num_replicate
{
	inline bool operator()(const pair<size_t/*#replicate*/, size_t/*count*/> &m_a,
		const pair<size_t/*#replicate*/, size_t/*count*/> &m_b) const
	{
		return m_a.first < m_b.first;
	};
};

// Global variables for MPI
int mpi_numtasks;
int mpi_rank;
double start_time;

void terminate_program(int m_sig);
string report_run_time();
deque<string> validate_cell_line_names(const deque<string> &m_a, const deque<string> &m_b);
deque<float> extract_activity(const MULTIMAP<NSCID, DrugInfo>::const_iterator &m_begin, 
	const MULTIMAP<NSCID, DrugInfo>::const_iterator &m_end, const string &m_cell_line);
string extract_tissue_prefix(const string &m_cell_line_name);
string get_processor_name();

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
		
		Options opt(argc, argv);
		
		if(opt.print_usage){
		
			MPI_Finalize();
			return EXIT_SUCCESS;
		}
		
		ofstream fout;
		
		if(mpi_rank == 0){
			
			cerr << "Running with " << mpi_numtasks << " MPI rank(s)" << endl;
			
			fout.open( opt.filename_output.c_str() );
			
			if(!fout){
				throw __FILE__ ":main: Unable to open output file for writing";
			}
		
			fout << "# Crabby version " << CRABBY_VERSION << endl;
		
			fout << '#';
		
			for(int i = 0;i < argc;++i){
				fout << ' ' << argv[i];
			}
		
			fout << endl;
		
			fout << "# Random number seed = " << opt.seed << endl;
			fout << "# Performing " << opt.fold << "-fold cross validation" << endl;
			fout << "# Random forest parameters:" << endl;
			fout << "#\tNumber of trees = " << opt.forest_num_param << endl;
			fout << "#\tMinimum leaf size = " << opt.forest_min_leaf << endl;
			fout << "#\tBagging fraction = " << opt.forest_bag_fraction << endl;
			fout << "#\tMinimum variance = " << opt.forest_min_variance << endl;
			fout << "# Elastic net parameters:" << endl;
			fout << "#\tLambda1 = " << opt.elastic_lambda_1 << endl;
			fout << "#\tLambda2 = " << opt.elastic_lambda_2 << endl;
			
			fout << "# Clustering parameters:" << endl;
			switch(opt.cluster){
				case CLUSTER_BY_GENE:
					fout << "#\tClustering by gene" << endl;
					break;
				case CLUSTER_BY_SINGLE_LINKAGE:
					fout << "#\tClustering by single linkage (threshold = " 
						<< opt.cluster_threshold << ")" << endl;
					break;
				case CLUSTER_BY_IDENTITY:
					fout << "#\tClustering categorical features by identity (clusters must contain at least "
						<< opt.min_features_per_cluster << " features)" << endl;
					break;
				case CLUSTER_BY_NONE:
					fout << "#\tNo clustering" << endl;
					break;
				default:
					throw __FILE__ ":main: Unknown clustering option";
			};
			
			for(deque<string>::const_iterator i = opt.notes.begin();i != opt.notes.end();++i){
				fout << "# " << *i << endl;
			}
		}

		// Initialize the random number generator for each rank. Note that if we need to create multiple
		// threads, then we will need to create additional thread safe random number generators.

		gsl_rng *rand_gen = gsl_rng_alloc(gsl_rng_ranlux389);
		
		if(rand_gen == NULL){
			throw __FILE__ ":main: Unable to initialize random number generator";
		}
		
		// Make sure that each rank receives a different seed
		gsl_rng_set(rand_gen, opt.seed + mpi_rank);
		
		// The dose response data is indexed as [drug][cell line] -> experiment -> vector of (concentration, %GI) values
		vector<DrugData> dose_response_data;
		
		deque<Feature> independent_variables;
		
		deque<NSCID> drug_id;
		deque<string> cell_line_id;
		vector< vector<string> > drug_experiment_id;
		vector<size_t> random_cell_line_index;
		
		MAP<NSCID, DrugDescriptor> drug_descriptor;
		
		time_t profile = time(NULL);
		
		// Each MPI rank tracks the time required to compute all drug
		// dose response regressions for all folds.
		MAP<string /*host name*/, double /*time in sec*/> total_regression_profile;
		MAP<string /*host name*/, size_t> total_regression_profile_count;
		const string hostname = get_processor_name();
		
		total_regression_profile[hostname] = 0.0;
		total_regression_profile_count[hostname] = 0;
		
		if(mpi_rank == 0){
		
			deque<NSCID> include_drug_id;
			
			if( !opt.filename_drug_target.empty() ){

				include_drug_id = parse_drug_id(opt.filename_drug_target);

				make_set(include_drug_id);
			}
			
			if( !opt.exclude_cell.empty() ){
				cerr << "Found " << opt.exclude_cell.size() << " cell lines to *exclude*" << endl;
			}
			
			if( !include_drug_id.empty() ){
				cerr << "Found " << include_drug_id.size() << " drugs to *include*" << endl;
			}
			
			parse_dose_response(opt.filename_dose_response, 
				drug_id, cell_line_id, drug_experiment_id,
				dose_response_data,
				include_drug_id, opt.exclude_cell);
			
			const string summary_info = dose_response_summary(dose_response_data, drug_id);
			
			cerr << summary_info << endl;
			fout << summary_info << endl;
			
			//////////////////////////////////////////////////////////////////////////////////////////////
			// Begin DEBUG
			//////////////////////////////////////////////////////////////////////////////////////////////
			#ifdef DRUG_CELL_MATRIX
			const size_t debug_num_cell = cell_line_id.size();
			const size_t debug_num_drug = dose_response_data.size();
			
			for(size_t i = 0;i < debug_num_cell;++i){
				cout << '\t' << cell_line_id[i];
			}
			
			cout << endl;
			
			for(size_t d = 0;d < debug_num_drug;++d){

				cout << drug_id[d];
				
				const size_t num_exp = dose_response_data[d].size();

				if(num_exp > 1){
					throw "DEBUG -- Found a drug with replicates!!";
				}

				const ExperimentData &data = dose_response_data[d][0];

				const size_t num_conc = data[0].size();
				
				for(size_t i = 0;i < debug_num_cell;++i){

					// Compute the average response for all cell lines *except* cell line i
					CellLineData ave_response;
					
					for(size_t j = 0;j < debug_num_cell;++j){

						if(i == j){
							continue;
						}
						
						if( ave_response.empty() ){
							ave_response = data[j];
						}
						else{
							for(size_t k = 0;k < num_conc;++k){
								ave_response[k].response += data[j][k].response;
							}
						}
					}
					
					for(size_t k = 0;k < num_conc;++k){
						ave_response[k].response /= debug_num_cell - 1;
					}
					
					// Compute the MSE from the response for cell line i to the average
					// response over the remaining cell lines.
					
					double mse = 0.0;
					
					for(size_t k = 0;k < num_conc;++k){
					
						const double delta = ave_response[k].response - data[i][k].response;
						
						mse += delta*delta/num_conc;
					}
					
					cout << '\t' << mse;
				}
				
				cout << endl;
			}
			
			throw "DEBUG";
			#endif // DRUG_CELL_MATRIX
			//////////////////////////////////////////////////////////////////////////////////////////////
			// End DEBUG
			//////////////////////////////////////////////////////////////////////////////////////////////
			
			int current_feature_file = 0;
			
			if(opt.skip_any_missing_independent){
			
				cerr << "Skipping independent variables with *any* missing values" << endl;
				fout << "# Skipping independent variables with *any* missing values" << endl;
			}
			
			for(deque<string>::const_iterator i = opt.filename_independent_real.begin();
				i != opt.filename_independent_real.end();++i){
				
				parse_real(*i, current_feature_file, independent_variables, 
					cell_line_id, opt.skip_any_missing_independent);
				++current_feature_file;
			}
			
			for(deque<string>::const_iterator i = opt.filename_independent_categorical.begin();
				i != opt.filename_independent_categorical.end();++i){
				
				// Pass the minimum number of samples in a leaf node to the parsing function
				// allows to filter out catergorical features that will not be useful
				// (i.e. all samples belong to the same category, or the miniority category
				// has < opt.forest_min_leaf number of samples).
				parse_categorical(*i, current_feature_file, independent_variables, 
					cell_line_id, opt.forest_min_leaf, opt.skip_any_missing_independent);
				++current_feature_file;
			}
			
			// Make sure that the feature type has been set for all independent variables
			for(deque<Feature>::const_iterator i = independent_variables.begin();
				i != independent_variables.end();++i){
				if(i->type == Feature::UNKNOWN){
					throw __FILE__ ":main: At least one Feature does not have a valid type (real or categorical)";
				}
			}
			
			// DEBUG
			//cerr << "!!!! Truncating independent variables !!!" << endl;
			//fout << "!!!! Truncating independent variables !!!" << endl;
			//independent_variables.resize(100);
			
			const size_t num_drug = drug_id.size();
			const size_t num_cell_line = cell_line_id.size();
			
			// DEBUG -- which drugs have experiment dependent doses?
			//for(size_t i = 0;i < num_drug;++i){
			//
			//	const DrugData &curr_drug = dose_response_data[i];
			//	bool valid = true;
			//	
			//	deque<Concentration> dose_set;
			//	
			//	for(DrugData::const_iterator j = curr_drug.begin();j != curr_drug.end();++j){
			//		
			//		deque<Concentration> local_doses;
			//		
			//		for(ExperimentData::const_iterator k = j->begin();k != j->end();++k){
			//			
			//			for(CellLineData::const_iterator m = k->begin();m != k->end();++m){
			//				
			//				local_doses.push_back(m->dose);
			//			}
			//		}
			//		
			//		make_set(local_doses);
			//		
			//		if( dose_set.empty() ){
			//			dose_set = local_doses;
			//		}
			//		else{
			//			if(dose_set != local_doses){
			//				valid = false;
			//			}
			//		}
			//	}
			//	
			//	if(valid){
			//		cout << drug_id[i] << endl;
			//	}
			//	else{
			//		cerr << drug_id[i] << " is **invalid**" << endl;
			//	}
			//}
			//throw "DEBUG";
			
			cerr << "Found " << num_drug << " drugs, " << num_cell_line 
				<< " cell-lines and " << independent_variables.size() << " features" << endl;

			fout << "# Found " << num_drug << " drugs, " << num_cell_line 
				<< " cell-lines and " << independent_variables.size() << " features" << endl;
			
			if(opt.randomize_independent){
			
				cerr << "Randomizing independent variables!" << endl;
				fout << "# Randomizing independent variables!" << endl;

				// Randomize the cell lines of the independent variables (that is, per-cell line values are only
				// shuffled *within* a given feature (to preserve location and other feature-specific 
				// metadata)
				for(deque<Feature>::iterator i = independent_variables.begin();
					i != independent_variables.end();++i){

					i->shuffle(rand_gen);
				}
			}

			//#define RANDOM_RESPONSE
			#ifdef RANDOM_RESPONSE
			
			cerr << "** Randomizing dose response **" << endl;
			fout << "** Randomizing dose response **" << endl;
			
			for(size_t drug = 0;drug < num_drug;++drug){
				
				DrugData& curr_drug = dose_response_data[drug];
				
				// The experiment to randomize
				const size_t x = 1;

				if(curr_drug.size() <= x){
					continue;
				}

				for(size_t i = 0;i < num_cell_line;++i){

                                	CellLineData& curr_cell = curr_drug[x][i];

                                	const size_t num_dose = curr_cell.size();

					for(size_t j = 0;j < num_dose;++j){
						
						// Randomly generate a number between -100 and 100
						curr_cell[j].response = 200.0*(gsl_rng_uniform(rand_gen) - 0.5);	
					}
				}
			}
			#endif // RANDOM_RESPONSE

			//#define INJECT_SIGNAL
			#ifdef INJECT_SIGNAL
			
			cerr << "<--- Injecting signal --->" << endl;
			fout << "# !!! Injecting signal !!!" << endl;
			
			for(size_t drug = 0;drug < num_drug;++drug){
				
				enum {SYNTHETIC_ALPHA, SYNTHETIC_BETA, NUM_SYNTHETIC};
				
				// The set of features that will be used to determine the dose response curves
				vector<size_t> synthetic_features(NUM_SYNTHETIC);
				vector< pair<float, float> > synthetic_scaling(NUM_SYNTHETIC);

				for(size_t s = 0;s < NUM_SYNTHETIC;++s){

					float min_x = 0.0;
					float max_x = 0.0;
					
					// Randomly select the feature
					while(true){
					
						synthetic_features[s] = gsl_rng_uniform_int( rand_gen, independent_variables.size() );
						
						for(size_t i = 0;i < num_cell_line;++i){

							if(independent_variables[ synthetic_features[s] ].missing[i] == true){
								throw __FILE__ ":main: synthetic feature has missing data!";
							}

							const float x = independent_variables[ synthetic_features[s] ].val[i].r;

							if(i == 0){
								min_x = max_x = x;
							}
							else{
								min_x = min(min_x, x);
								max_x = max(max_x, x);
							}
						}
						
						// Only select feature that have some variability ...
						if( (max_x - min_x) > 0.5 ){
							break;
						}
					}
					
					cerr << "[drug " << drug << "] Synthetic feature " << s << ": " 
						<< synthetic_features[s] << " (" << independent_variables[ synthetic_features[s] ].name 
						<< ") -> [" << min_x << ", " << max_x << "]" << endl;

					synthetic_scaling[s] = make_pair(min_x, max_x);
				}

				DrugData& curr_drug = dose_response_data[drug];
				
				for(DrugData::iterator x = curr_drug.begin();x != curr_drug.end();++x){
					
					for(size_t i = 0;i < num_cell_line;++i){

						CellLineData& curr_cell = (*x)[i];
						
						const size_t num_dose = curr_cell.size();
						
						const float alpha = 2.0 + 10.0*(independent_variables[ synthetic_features[SYNTHETIC_ALPHA] ].val[i].r - 
							synthetic_scaling[SYNTHETIC_ALPHA].first)/
							(synthetic_scaling[SYNTHETIC_ALPHA].second - synthetic_scaling[SYNTHETIC_ALPHA].first);
						
						//const float alpha = 5.0;
						
						const float beta = float(num_dose)*(independent_variables[ synthetic_features[SYNTHETIC_BETA] ].val[i].r - 
							synthetic_scaling[SYNTHETIC_BETA].first)/
							(synthetic_scaling[SYNTHETIC_BETA].second - synthetic_scaling[SYNTHETIC_BETA].first);
						
						//cout << "# [drug " << drug << "] cell line " << i << ", beta = " << beta << ", alpha = " << alpha 
						//	<< ", x = " << independent_variables[ synthetic_features[SYNTHETIC_BETA] ].val[i].r << endl;
						
						for(size_t j = 0;j < num_dose;++j){
							
							//const float x = curr_cell[j].dose.value;
							const float x = float(j);
							
							curr_cell[j].response = 200.0/( 1.0 + exp(alpha*(x - beta)) ) - 100.0;
							
						//	cout << x << '\t' << curr_cell[j].response << endl;
						}
						
						//cout << '\n' << endl;
					}
				}
			}
			
			#endif // INJECT_SIGNAL

			// Find the drug and cell line combination with the largest number of
			// number of dose measurements.
			size_t max_dose = 0;
			NSCID max_dose_drug = 0;
			string max_dose_cell_line = "";

			// For each drug ...
			for(size_t i = 0;i < num_drug;++i){
				
				// For each experiment ...
				for(DrugData::const_iterator x = dose_response_data[i].begin();
					x != dose_response_data[i].end();++x){
					
					// For each cell line ...
					for(size_t j = 0;j < num_cell_line;++j){

						const size_t num = (*x)[j].size();

						if(num > max_dose){

							max_dose = num;
							max_dose_drug = drug_id[i];
							max_dose_cell_line = cell_line_id[j];
						}
					}
				}
			}

			cerr << "NSC" << max_dose_drug << " / " << max_dose_cell_line 
				<< " has the largest number of dose measurements = " << max_dose << endl;

			if( !opt.filename_tree.empty() ){

				cerr << "Writing a dose-response tree in newick format to: " << opt.filename_tree << endl;

				ofstream ftree( opt.filename_tree.c_str() );

				if(!ftree){
					throw __FILE__ ":main: Unable to open dose response tree file for writing";
				}

				// Compute the gene expression-based distance tree for cell-lines (this needs
				// a stand-alone command line option).
				//const string tree = expression_tree(independent_variables, cell_line_id);

				// Compute the dose response-based distance tree for cell-lines
				const string tree = dose_response_tree(dose_response_data, 
					drug_id, cell_line_id);

				ftree << tree << ';' << endl;
			}
			
			if( !opt.filename_drug_descriptor.empty() ){
				
				cerr << "Reading drug descriptor data from " 
					<< opt.filename_drug_descriptor << " ... ";
					
				parse_CACTVS_drug_descriptor(opt.filename_drug_descriptor, drug_descriptor);
				
				cerr << "done." << endl;
				
				cerr << "\t" << drug_descriptor.size() << "/" << num_drug 
					<< " drugs (" << ( 100.0*drug_descriptor.size() )/num_drug 
					<< "%) have descriptors" << endl;
				
				fout << "\t" << drug_descriptor.size() << "/" << num_drug 
					<< " drugs (" << ( 100.0*drug_descriptor.size() )/num_drug 
					<< "%) have descriptors" << endl;
					
				//#define RANDOMIZE_DRUG_DESCRIPTOR
				#ifdef RANDOMIZE_DRUG_DESCRIPTOR
				
				cerr << "!!!!!Randomizing drug descriptors!!!!!" << endl;
				fout << "# !!!!!Randomizing drug descriptors!!!!!" << endl;
				
				// Randomize the assignment of descriptors to drugs rather than
				// the descriptors themselves -- this maintains the distibution of pair-
				// wise similarity scores.
				const deque<NSCID> local_id = keys(drug_descriptor);
				
				const size_t num_descriptor = local_id.size();
				
				for(MAP<NSCID, DrugDescriptor>::iterator i = drug_descriptor.begin();
					i != drug_descriptor.end();++i){
					
					// Swap the descriptor of the current drug with that of a randomly
					// selected drug
					const size_t index = gsl_rng_uniform_int(rand_gen, num_descriptor);
					
					MAP<NSCID, DrugDescriptor>::iterator iter = drug_descriptor.find(local_id[index]);
					
					if( iter == drug_descriptor.end() ){
						throw __FILE__ ":main: Unable to find descriptor to swap";
					}
					
					if(i->first != iter->first){
						swap(i->second, iter->second);
					}
				}
				
				#endif // RANDOMIZE_DRUG_DESCRIPTOR
				
				if( !opt.filename_drug_tree.empty() ){
				
					cerr << "Writing a drug tree in newick format to: " << opt.filename_drug_tree << endl;

					ofstream ftree( opt.filename_drug_tree.c_str() );

					if(!ftree){
						throw __FILE__ ":main: Unable to open a drug tree file for writing";
					}

					// Compute the descriptor-based distance tree for drug
					const string tree = drug_descriptor_tree(drug_descriptor);

					ftree << tree << ';' << endl;
				}
			}
			
			cerr << "Using " << opt.fold << "-fold cross validation to assess regression model" << endl;

			// Randomize the cell-line ids prior to partitioning into test and training
			random_cell_line_index.resize(num_cell_line);
			
			for(size_t i = 0;i < num_cell_line;++i){
				random_cell_line_index[i] = i;
			}
			
			randomize(random_cell_line_index.begin(), random_cell_line_index.end(), rand_gen);
		}

		broadcast(drug_id);
		broadcast(cell_line_id);
				
		// All ranks need the same randomized cell indicies to perform cross-validation
		broadcast(random_cell_line_index);
		
		broadcast(dose_response_data);
		
		broadcast(drug_experiment_id);
		
		broadcast(independent_variables);
		
		broadcast(drug_descriptor);
		
		const size_t num_drug = drug_id.size();
		const size_t num_cell_line = cell_line_id.size();
		
		matrix<DoseResponseFunction> drug_curve_fit(num_drug, num_cell_line);
		
		// For computing activity-based similarity between drugs, we will fit sigmoidal functions
		// to *all* dose-response curves.
		if(opt.drug_similarity_threshold <= 1.0){
			
			// Compute the curve fits for the drugs that are local to this rank
			for(size_t i = 0;i < num_drug;++i){
				
				if( i%mpi_numtasks == size_t(mpi_rank) ){
					
					for(size_t j = 0;j < num_cell_line;++j){
						drug_curve_fit(i, j) = fit_dose_response_data(dose_response_data[i], j);
					}
				}
			}
			
			// Send the curve fit results from each rank back to the master
			accumulate(drug_curve_fit);
			
			// Distribute all of the curve fit results from the rank 0 node to all other ranks
			broadcast(drug_curve_fit);
		}
		
        	///////////////////////////////////////////////////////////////////////////
		// The per-cell line score
		MULTIMAP<string, double> score_by_cell_line;

        	///////////////////////////////////////////////////////////////////////////
		// The per-drug score
		MULTIMAP<NSCID, double> score_by_drug;

		///////////////////////////////////////////////////////////////////////////
		// The per-concentration score
		MULTIMAP<Concentration, double> score_by_dose;
		
		///////////////////////////////////////////////////////////////////////////
		// The per-experiment score
		MULTIMAP<string, double> score_by_experiment;
		
        	///////////////////////////////////////////////////////////////////////////
        	// The subset of per-fold scores computed by predicting the average value
        	// for data with replicate values
		vector< deque<double> > score_by_fold(opt.fold);

		///////////////////////////////////////////////////////////////////////////
		// The number (from all folds) of "similar" drugs, for each target drug. 
		// Acitivity data from similar drugs is leveraged using multi-task learning.
		MAP<NSCID, size_t> similar_drugs;
		
		//#define PRINT_SIMILARITY_SCATTER
		#ifdef PRINT_SIMILARITY_SCATTER
		for(size_t i = 0;i < num_drug;++i){
			
			const deque< pair<float, NSCID> > activity_distance = 
				find_similar_drugs(drug_id[i], drug_id, drug_curve_fit);
			
			//for(deque< pair<float, NSCID> >::const_iterator a = activity_distance.begin();a != activity_distance.end();++a){
			//	
			//	if(mpi_rank == 0){
			//		cout << a->first << endl;
			//	}
			//}
			
			//continue;
			
			const deque< pair<float, NSCID> > descriptor_distance = 
				find_similar_drugs(drug_id[i], drug_id, drug_descriptor);
			
			// Skip drugs that do not have a valid descriptor
			if( descriptor_distance.empty() ){
				continue;
			}

			MAP<NSCID, float> activity_map;
			
			for(deque< pair<float, NSCID> >::const_iterator a = activity_distance.begin();a != activity_distance.end();++a){
				activity_map.insert( make_pair(a->second, a->first) );
			}
			
			for(deque< pair<float, NSCID> >::const_iterator d = descriptor_distance.begin();d != descriptor_distance.end();++d){
				
				MAP<NSCID, float>::const_iterator iter = activity_map.find(d->second);
				
				if( iter == activity_map.end() ){
					continue;
				}
				
				if(mpi_rank == 0){
					cout << iter->second << '\t' << d->first << " #" << drug_id[i]
						<< "," << d->second << endl;
				}
			}
		}
		
		MPI_Barrier(MPI_COMM_WORLD);
			
		throw "DEBUG - similarity distribution";
		#endif // PRINT_SIMILARITY_SCATTER
		
        	///////////////////////////////////////////////////////////////////////////
        	// Drugs are uniformly partitioned among the available MPI ranks
		deque<NSCID> local_drug_id;

		for(size_t i = 0;i < num_drug;++i){
			
			if( i%mpi_numtasks == size_t(mpi_rank) ){
				local_drug_id.push_back(drug_id[i]);
			}
		}
		
		const size_t local_num_drug = local_drug_id.size();
		
		// Collect statistics on the frequency of selecting a given cluster representation operation
		vector<float> cluster_op_freq(Feature::NUM_OP);
		MAP<string, float> cluster_name_freq;
		size_t num_real_features_used = 0;
		size_t num_categorical_features_used = 0;
		
		// Use cross-validation to assess our ability to predict dose response for a novel cell line
		for(int f = 0;f < opt.fold;++f){
			
			deque<size_t> test_cell_line_index;
			deque<size_t> train_cell_line_index;

			for(size_t i = 0;i < num_cell_line;++i){

				if(i%opt.fold == (size_t)f){
					test_cell_line_index.push_back(random_cell_line_index[i]);
				}
				else{
					train_cell_line_index.push_back(random_cell_line_index[i]);
				}
			}
			
			// Setting opt.fold is a special case -- *no* cross validation is performed!
			// Train and test on the same set of cell lines!
			if(opt.fold == 1){
				
				test_cell_line_index.assign( random_cell_line_index.begin(), random_cell_line_index.end() );
				train_cell_line_index.assign( random_cell_line_index.begin(), random_cell_line_index.end() );
			}
			
			const size_t num_train = train_cell_line_index.size();
			const size_t num_test = test_cell_line_index.size();

			if(mpi_rank == 0){
			
				cerr << "Fold: " << (f + 1) << endl;
				cerr << "\t|training cell-lines| = " << num_train << endl;
				cerr << "\t|testing cell-lines| = " << num_test << endl;
				
				fout << "# Fold: " << (f + 1) << endl;
				fout << "#\t|training cell-lines| = " << num_train << endl;
				fout << "#\t|testing cell-lines| = " << num_test << endl;
			}

			size_t num_feature = independent_variables.size();

			vector<Feature> training_values;
			vector<Feature> testing_values;

			if(mpi_rank == 0){
			
				training_values.resize(num_feature);
				testing_values.resize(num_feature);
			
				for(size_t i = 0;i < num_feature;++i){

					Feature &src = independent_variables[i];
					Feature &train = training_values[i];
					Feature &test = testing_values[i];
					
					// Copy the feature metadata, the actual values and missing
					// value status will be handeled below.
					train = test = src;
					
					train.resize(num_train);
					test.resize(num_test);
					
					for(size_t j = 0;j < num_train;++j){						
						
						train.val[j] = src.val[ train_cell_line_index[j] ];
						train.missing[j] = src.missing[ train_cell_line_index[j] ];
					}

					for(size_t j = 0;j < num_test;++j){

						test.val[j] = src.val[ test_cell_line_index[j] ];
						test.missing[j] = src.missing[ test_cell_line_index[j] ];
					}
				}
			}
			
			////////////////////////////////////////////////////////////////////////////////////////////////////
			// The training and testing matricies are shared to all ranks by the infer_missing_training_values()
			// and infer_missing_testing_values() functions
			////////////////////////////////////////////////////////////////////////////////////////////////////
			
			time_t profile_infer = time(NULL);
			
			// DEBUG
			//cerr << "[" << mpi_rank << "] About to infer missing training values" << endl;
			
			// Infer missing expression values using an iterative approach that fills in 
			// (via linear extrapolation/interpolation) missing values from the most similar 
			// expression value (using a linear correlation to identify similar).
			const size_t num_missing_training = infer_missing_training_values(training_values) ;
			
			profile_infer = time(NULL) - profile_infer;
			
			if(mpi_rank == 0){
			
				cerr << "\tInferred " << num_missing_training << " missing training values in " 
					<< profile_infer << " sec" << endl;
				fout << "#\tInferred " << num_missing_training << " missing training values for fold " << f 
					<< " in " << profile_infer << " sec" << endl;
			}
			
			profile_infer = time(NULL);
			
			const size_t num_missing_testing = infer_missing_testing_values(training_values, testing_values) ;
			
			profile_infer = time(NULL) - profile_infer;
			
			if(mpi_rank == 0){
			
				cerr << "\tInferred " << num_missing_testing << " missing testing values in " 
					<< profile_infer << " sec" << endl;
				fout << "#\tInferred " << num_missing_testing << " missing testing values for fold " << f 
					<< " in " << profile_infer << " sec" << endl;
			}
			
			// If requested, cluster the input features using only the data provided in the training set.
			vector< deque<size_t> > clusters = 
				cluster_features(
					opt.cluster, 
					opt.cluster_size, 
					opt.cluster_threshold, 
					training_values,
					opt.min_features_per_cluster,
					rand_gen);

			if(mpi_rank == 0){

				cerr << "\tIdentified " << clusters.size() << " feature clusters" << endl;
				fout << "#\tIdentified " << clusters.size() << " feature clusters" << endl;
			}

			// Using the clusters defined for the training values, create derived sets of training and testing
			// feature values.
			training_values = apply_clusters(training_values, clusters);
			testing_values = apply_clusters(testing_values, clusters);

			if( training_values.size() != testing_values.size() ){
				throw __FILE__ ":main: |clustered training| != |clustered testing|";
			}

			// Recompute the number of features to account for features that were removed or added by the feature
			// clustering
			num_feature = training_values.size();
			
			// Build and test a model for each drug.
			for(size_t d = 0;d < local_num_drug;++d){

				const size_t drug_index = get_index(drug_id, local_drug_id[d]);
				
				const size_t num_experiment = dose_response_data[drug_index].size();
								
				DrugData test_dose_response(num_experiment);
				DrugData train_dose_response(num_experiment);
				
				for(size_t x = 0;x < num_experiment;++x){
					
					test_dose_response[x].resize(num_test);
					train_dose_response[x].resize(num_train);

					for(size_t i = 0;i < num_train;++i){					
						train_dose_response[x][i] = dose_response_data[drug_index][x][ train_cell_line_index[i] ];
					}

					for(size_t i = 0;i < num_test;++i){
						test_dose_response[x][i] = dose_response_data[drug_index][x][ test_cell_line_index[i] ];
					}
				}
				
				// Are we using multi-task learning between drugs?
				if(opt.drug_similarity_threshold <= 1.0){
					
					// Extract the curve fits for all drugs vs the training cell lines
					matrix<DoseResponseFunction> train_drug_curve_fit(num_drug, num_train);
					
					for(size_t i = 0;i < num_drug;++i){
						for(size_t j = 0;j < num_train;++j){
							train_drug_curve_fit(i, j) = drug_curve_fit(i, train_cell_line_index[j]);							
						}
					}
					
					// If the user has provided drug descriptor information, we will use it to identify similar drugs
					// that can be used for multitask learning. Compute the weights and ids of similar drugs. Only drugs
					// whose weight (i.e. similarity score) is *greater* than or equal to the opt.drug_similarity_threshold
					// will be included. Still need to account for a drug weight that is a function of the training dose
					// response data (this will require "slicing" the dose response data into a training set
					// for *all* drugs, not just the current target drug).
					//
					// If the use has *not* provided dose response data, then use the a similarity metric that
					// is based on the dose response data
					deque< pair<float, NSCID> > weights = 
						find_similar_drugs(local_drug_id[d], drug_id, drug_descriptor);
					
					// Use the training activity data to define pair-wise drug similarity
					//deque< pair<float, NSCID> > weights = 
					//	find_similar_drugs(local_drug_id[d], drug_id, train_drug_curve_fit);
					
					for(deque< pair<float, NSCID> >::const_iterator w = weights.begin();w != weights.end();++w){
						
						// Only include drugs that have weights greater than the user-defined 
						// similarity threshold
						if(w->first < opt.drug_similarity_threshold){
							continue;
						}
						
						// Count the number of similar drugs (to report back to the user)
						++similar_drugs[ local_drug_id[d] ];
						
						const size_t other_drug_index = get_index(drug_id, w->second);
				
						const size_t other_num_experiment = dose_response_data[other_drug_index].size();
						
						// The data from other, "similar" drugs are treated as virtual experimental replicates
						for(size_t x = 0;x < other_num_experiment;++x){
							
							train_dose_response.push_back( ExperimentData(num_train) );
							
							ExperimentData &other_dose_response = train_dose_response.back();
							
							for(size_t i = 0;i < num_train;++i){
							
								other_dose_response[i] = 
									dose_response_data[other_drug_index][x][ train_cell_line_index[i] ];
									
								
								// DEBUG -- no scaling (or top hat scaling)	
								// Scale this virual experiment by the weight factor
								//for(CellLineData::iterator j = other_dose_response[i].begin();
								//	j != other_dose_response[i].end();++j){
									
								//	j->response *= w->first;
									//j->response *= w->first*w->first;
								//}
							}
						}
					}
				}
									
				vector<double> test_score(num_test);
				vector< MAP<Concentration, double> > test_score_by_dose(num_test);
				
				const double init_regression_profile = MPI_Wtime();
				
				//#define	USE_ELASTIC_NET
				#define	USE_RANDOM_FOREST
				
				#ifdef USE_RANDOM_FOREST
				// Build a model that uses the *entire* dose response curve for each cell line in
				// the training set and outputs the predicted response as a function of dose
				MultivariateTreeModel model;
				
				model.build(train_dose_response, 	// Dependent variables
					training_values, 		// Independent variables
					opt.forest_num_param, 		// Number of trees
					opt.forest_min_variance, 	// Min variance stopping criteria
					opt.forest_min_leaf,    	// Min leaf occupancy
					opt.forest_bag_fraction,	// Fraction of features to use when bagging a tree
					rand_gen);
				#endif // USE_RANDOM_FOREST
				
				#ifdef USE_ELASTIC_NET
				// Build a model that uses the *entire* dose response curve for each cell line in
				// the training set and outputs the predicted response as a function of dose
				ElasticNetModel model;
				
				//const pair<float, float> lambda = model.lambda_grid_search(train_dose_response,
				//	training_values, rand_gen);
				
				const pair<float, float> lambda = make_pair(opt.elastic_lambda_1, opt.elastic_lambda_2);

				model.build(train_dose_response, 	// Dependent variables
					training_values, 		// Independent variables
					lambda.first,			// Lambda 1
					lambda.second);			// Lambda 2
				#endif // USE_ELASTIC_NET
				
				total_regression_profile[hostname] += MPI_Wtime() - init_regression_profile;
				total_regression_profile_count[hostname] ++;
				
				// Use the test set data to evaluate the predictions
				for(size_t x = 0;x < num_experiment;++x){
					
					for(size_t i = 0;i < num_test;++i){
						
						// Predict the response of the cell line from its gene expression data.
						//	1) Note that the prediction depends on both the cell line-dependent 
						//	   gene expression *and* the current experiment!
						const CellLineData predicted_response = 
							model.predict(x, 	// Experiment
								testing_values, // Independent variables
								i);		// Sample (cell line) index
								
						const CellLineData &actual_response = test_dose_response[x][i];
						
						// Make sure that we can compare predictions to the available test data
						const size_t num_dose = predicted_response.size();
						
						if( num_dose != actual_response.size() ){
							throw __FILE__ ":main: |predicted_response| != |actual_response|";
						}
						
						double local_score = 0.0;
						
						//#define SHOW_PREDICTIONS
						#ifdef SHOW_PREDICTIONS
						cout << "# NSC " << local_drug_id[d] << "; "
							<< cell_line_id[ test_cell_line_index[i] ]
							<< "; experiment " << x << endl;
						#endif // SHOW_PREDICTIONS
						
						for(size_t j = 0;j < num_dose;++j){
							
							// Make sure that we are comparing responses at the same doses
							if(predicted_response[j].dose != actual_response[j].dose){
								throw __FILE__ ":main: predicted and actual doses don't agree";
							}
							
							const float delta = 
								predicted_response[j].response - actual_response[j].response;
							
							const float delta2 = delta*delta;
							
							local_score += delta2;
							test_score_by_dose[i][predicted_response[j].dose] += delta2/num_experiment;
							
							#ifdef SHOW_PREDICTIONS
							cout << predicted_response[j].dose.value << '\t' 
								<< actual_response[j].response << '\t'
								<< predicted_response[j].response << '\t'
								<< predicted_response[j].stdev << endl;
							#endif // SHOW_PREDICTIONS
						}
						
						#ifdef SHOW_PREDICTIONS
						cout << endl;
						#endif // SHOW_PREDICTIONS
						
						if(num_dose == 0){
							throw __FILE__ ":main: Unable to normalize local_score";
						}
						
						// Normalize local_score in two steps. First by num_dose ...
						local_score /= num_dose;
						
						score_by_experiment.insert( make_pair(drug_experiment_id[drug_index][x], local_score) );
						
						// ... then by num_experiment.
						local_score /= num_experiment;
						
						test_score[i] += local_score;
					}
				}
				
				// Store the scores
				for(size_t i = 0;i < num_test;++i){
					
					score_by_cell_line.insert( make_pair(cell_line_id[ test_cell_line_index[i] ], test_score[i]) );
					score_by_drug.insert( make_pair(local_drug_id[d], test_score[i]) );
					score_by_fold[f].push_back(test_score[i]);
					
					// Accumulate the multimap score_by_dose from the map test_score_by_dose to enable proper
					// normalization by the number of experiments.
					for(MAP<Concentration, double>::const_iterator j = test_score_by_dose[i].begin();
						j != test_score_by_dose[i].end();++j){
						
						score_by_dose.insert(*j);
					}
				}
				
				// Measure the frequency of each cluster operation
				const deque< pair<unsigned int, float> > model_features = model.extract_features();
				
				for(deque< pair<unsigned int, float> >::const_iterator i = model_features.begin();i != model_features.end();++i){
					
					cluster_op_freq[training_values[i->first].cluster_op] += i->second;
					
					if(opt.dump_features){
						cluster_name_freq[training_values[i->first].name] += i->second;
					}
					
					num_real_features_used += (training_values[i->first].type == Feature::REAL) ? 1 : 0;
					num_categorical_features_used += (training_values[i->first].type == Feature::CATEGORICAL) ? 1 : 0;
				}
			}
		}
		
		if(mpi_rank == 0){
			cerr << "Finished cross-validation. Accumulating results ... ";
		}
		
		// All MPI ranks != 0 need to send all of the results back to MPI rank 0		
		accumulate(score_by_cell_line);
		accumulate(score_by_drug);
		accumulate(score_by_fold);
		accumulate(score_by_dose);
		accumulate(score_by_experiment);
		
		accumulate(cluster_op_freq);
		accumulate(num_real_features_used);
		accumulate(num_categorical_features_used);
		
		accumulate(similar_drugs);
			
		if(opt.dump_features){
			accumulate(cluster_name_freq);
		}
		
		if(opt.profile){
			
			accumulate(total_regression_profile);
			accumulate(total_regression_profile_count);
		}
		
		// All of the MPI ranks != 0 are now finished!
		if(mpi_rank != 0){
		
			// Deallocate the random number generator
			gsl_rng_free(rand_gen);
			
			MPI_Finalize();
			return EXIT_SUCCESS;
		}
		else{
			cerr << "done." << endl;
		}
		
		// Write the profile times
		if(opt.profile){
		
			cerr << "Profiling: system names, average regression times and number of drugs evaluated" << endl;
			
			deque< pair<double, string> > local;
			
			for(MAP<string, double>::const_iterator i = total_regression_profile.begin();
				i != total_regression_profile.end();++i){
				
				local.push_back( make_pair(i->second, i->first) );
			}
			
			// Print the computation times in ascending order
			sort( local.begin(), local.end() );
			
			for(deque< pair<double, string> >::const_iterator i = local.begin();i != local.end();++i){

				const MAP<string, size_t>::const_iterator iter = 
					total_regression_profile_count.find(i->second);

				if( iter == total_regression_profile_count.end() ){
					cerr << "Unable to find regression normalization for: " << i->second << endl;
				}
				else{
					cerr << '\t' << i->second << '\t' 
						<< ( (iter->second > 0) ? i->first/iter->second : 0 )
						<< '\t' << iter->second << endl;
				}
			}
		}
		
		//////////////////////////////////////////////////////////////////////////////
		// Output the results
		//////////////////////////////////////////////////////////////////////////////
		
		fout << "# Finished calculation in " << (time(NULL) - profile) 
			<< " sec" << endl;
		
		double total_score = 0.0;
		size_t num_drugs_tested = 0;
		
		for(deque<NSCID>::const_iterator i = drug_id.begin();i != drug_id.end();++i){
			
			typedef MULTIMAP<NSCID, double>::const_iterator I;
			
			pair<I, I> range = score_by_drug.equal_range(*i);
			
			size_t local_count = 0;
			double local_score = 0.0;
			
			for(I j = range.first;j != range.second;++j){
			
				local_score += j->second;
				++local_count;
			}
			
			// Normalize by the number of cell lines tested for each drug
			if(local_count > 0){
				
				local_score /= local_count;
				++num_drugs_tested;
			}
			
			total_score += local_score;
		}
		
		// Normalize by the number of drugs tested
		if(num_drugs_tested > 0){
			total_score /= num_drugs_tested;
		}
        
		cerr << "Total score = " << total_score << endl;
		fout << "\n# Total score = " << total_score << endl;
        
		fout << "\n# Number of dose response predicitions = " << score_by_cell_line.size() << endl;

		// Score values by fold
		fout << "\n# Score values by fold" << endl;

		for(int f = 0;f < opt.fold;++f){
			
			////////////////////////////////
			// Score
			////////////////////////////////
			double fold_score = 0.0;

			for(deque<double>::const_iterator i = score_by_fold[f].begin();i != score_by_fold[f].end();++i){
				fold_score += *i;
			}

			fold_score /= score_by_fold[f].size();

			fout << f << '\t' << fold_score << endl;
		}
		
		// Avoid 0/0 when regularization prevents any features from being selected
		const float total_op_count = max(accumulate(cluster_op_freq.begin(), cluster_op_freq.end(), 0.0f), 1.0f);
		
		// Avoid 0/0 when regularization prevents any features from being selected
		const size_t total_feature_count = max( num_real_features_used + num_categorical_features_used, size_t(1) );
		
		fout << "\n# Fraction of real vs categorical variables used" << endl;
		fout << (100.0*num_real_features_used)/total_feature_count << "% real-valued" << endl;
		fout << (100.0*num_categorical_features_used)/total_feature_count << "% categorical" << endl;
		
		// Frequency of operator use for cluster representation
		fout << "\n# Importance of cluster reduction operators:" << endl;
		
		fout << "RAW accounts for " << (cluster_op_freq[Feature::RAW])/total_op_count 
			<< " of cluster operations" << endl;
		
		fout << "RANDOM accounts for " << (cluster_op_freq[Feature::RANDOM])/total_op_count 
			<< " of cluster operations" << endl;
			
		fout << "MIN_VALUE accounts for " << (cluster_op_freq[Feature::MIN_VALUE])/total_op_count 
			<< " of cluster operations" << endl;

		fout << "MAX_VALUE accounts for " << (cluster_op_freq[Feature::MAX_VALUE])/total_op_count 
			<< " of cluster operations" << endl;

		fout << "MEDIAN_VALUE accounts for " << (cluster_op_freq[Feature::MEDIAN_VALUE])/total_op_count 
			<< " of cluster operations" << endl;

		fout << "AVERAGE_VALUE accounts for " << (cluster_op_freq[Feature::AVERAGE_VALUE])/total_op_count 
			<< " of cluster operations" << endl;

		fout << "MIN_ENTROPY accounts for " << (cluster_op_freq[Feature::MIN_ENTROPY])/total_op_count 
			<< " of cluster operations" << endl;

		fout << "MAX_ENTROPY accounts for " << (cluster_op_freq[Feature::MAX_ENTROPY])/total_op_count 
			<< " of cluster operations" << endl;

		fout << "MEDIAN_ENTROPY accounts for " << (cluster_op_freq[Feature::MEDIAN_ENTROPY])/total_op_count 
			<< " of cluster operations" << endl;

		fout << "UNKNOWN_OP accounts for " << (cluster_op_freq[Feature::UNKNOWN_OP])/total_op_count 
			<< " of cluster operations" << endl;

		// Score by dose
		fout << "\n# Score by dose concentration" << endl;
		const deque<Concentration> conc = keys(score_by_dose);
		
		for(deque<Concentration>::const_iterator i = conc.begin();i != conc.end();++i){
			
			////////////////////////////////
			// Score
			////////////////////////////////
			typedef MULTIMAP<Concentration, double>::const_iterator I;

			const pair<I, I> range = score_by_dose.equal_range(*i);
			
			double dose_score = 0.0;
			size_t count = 0;

			for(I j = range.first;j != range.second;++j){

				dose_score += j->second;
				++count;
			}

			// Skip doses that have no predicitions
			if(count == 0){
			
				fout << i->value << i->units << "\tNA" << endl;
			}
			else{
			
				dose_score /= count;
				fout << i->value << i->units << '\t' << dose_score << endl;
			}
		}
		
		// Score by cell-line
		fout << "\n# Score by cell-line" << endl;

		for(deque<string>::const_iterator i = cell_line_id.begin();i != cell_line_id.end();++i){

			////////////////////////////////
			// Score
			////////////////////////////////
			typedef MULTIMAP<string, double>::const_iterator I;

			const pair<I, I> range = score_by_cell_line.equal_range(*i);

			double cell_score = 0.0;
			size_t count = 0;

			for(I j = range.first;j != range.second;++j){

				cell_score += j->second;
				++count;
			}

			// Skip cell lines that have no predicitions
			if(count == 0){
			
				fout << *i << "\tNA" << endl;
			}
			else{
			
				cell_score /= count;
				fout << *i << '\t' << cell_score << endl;
			}
		}

		// Score by experiment id
		fout << "\n# Score by experiment" << endl;
		const deque<string> experiment_id = keys(score_by_experiment);
		
		for(deque<string>::const_iterator i = experiment_id.begin();i != experiment_id.end();++i){

			////////////////////////////////
			// Score
			////////////////////////////////
			typedef MULTIMAP<string, double>::const_iterator I;

			const pair<I, I> range = score_by_experiment.equal_range(*i);

			double experiment_score = 0.0;
			size_t count = 0;

			for(I j = range.first;j != range.second;++j){

				experiment_score += j->second;
				++count;
			}

			if(count == 0){
				throw __FILE__ ":main: Unable to normalize score by experiment!";
			}
			
			experiment_score /= count;
			fout << *i << '\t' << experiment_score << endl;
		}
		
		const bool has_similar_drugs = !similar_drugs.empty();
		
		// Score by drug
		if(has_similar_drugs){
			fout << "\n# [Score by drug][MTL average number (over fold) of similar drugs]" << endl;
		}
		else{
			fout << "\n# Score by drug" << endl;
		}
		
		for(deque<NSCID>::const_iterator i = drug_id.begin();i != drug_id.end();++i){

			////////////////////////////////
			// Score
			////////////////////////////////
			typedef MULTIMAP<NSCID, double>::const_iterator I;

			const pair<I, I> range = score_by_drug.equal_range(*i);

			double drug_score = 0.0;
			size_t count = 0;

			for(I j = range.first;j != range.second;++j){
			
				drug_score += j->second;
				++count;
			}
			
			if(count > 0){
				drug_score /= count;
			}
			
			
			// Write the drug ID
			fout << *i << '\t';
			
			// Write the drug score
			if(count == 0){
				fout << "NA";
			}
			else{
				fout << drug_score;
			}
			
			if(has_similar_drugs){ // Write the number of MTL similar_drugs for each target drug
			
				MAP<NSCID, size_t>::const_iterator iter = similar_drugs.find(*i);

				if( iter == similar_drugs.end() ){
					fout << "\t0" << endl;
				}
				else{
					fout << '\t' << float(iter->second)/opt.fold << endl;
				}
			}
			else{ 
				fout << endl;
			}
		}
		
		if(opt.dump_features){
			
			// Write the feature usage frequency (sorted in order of decreasing frequency)
			deque< pair<float, string> > local;
			double total = 0.0;
			
			for(MAP<string, float>::const_iterator i = cluster_name_freq.begin();
				i != cluster_name_freq.end();++i){
				
				total += i->second;
				local.push_back( make_pair(i->second, i->first) );
			}
			
			const double norm = (total <= 0.0) ? 1.0 : 1.0/total;
			
			// Sort in ascending order
			sort( local.begin(), local.end() );
			
			fout << "\n#[feature name][feature importance]"<< endl;
			
			// Iterate from the back of the deque to write the results in descending order
			for(deque< pair<float, string> >::const_reverse_iterator i = local.rbegin();
				i != local.rend();++i){
				
				fout << i->second << '\t' << i->first*norm << endl;
			}
		}
		
		// Deallocate the random number generator
		gsl_rng_free(rand_gen);
		
		profile = time(NULL) - profile;
		
		cerr << "Completed analysis in " << profile << " sec" << endl;
		
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

deque<string> validate_cell_line_names(const deque<string> &m_a, const deque<string> &m_b)
{
	deque<string> ret;
	
	deque<string>::const_iterator a = m_a.begin();
	deque<string>::const_iterator b = m_b.begin();
	
	while(true){
		
		if( a == m_a.end() ){
			
			if( b == m_b.end() ){
				break;
			}
			
			cerr << "Warning: cell-line " << *b << " is in second set, but not first" << endl;
			++b;
		}
		else{
			if( b == m_b.end() ){
				
				cerr << "Warning: cell-line " << *a << " is in first set, but not second" << endl;
				++a;
			}
			else{
				if(*a > *b){
				
					cerr << "Warning: cell-line " << *b << " is in second set, but not first" << endl;
					++b;
				}
				else{
					if(*a < *b){
					
						cerr << "Warning: cell-line " << *a << " is in first set, but not second" << endl;
						++a;
					}
					else{ // *a == *b
					
						ret.push_back(*a);
						++a;
						++b;
					}
				}
			}
		}
	}
	
	return ret;
}

deque<float> extract_activity(const MULTIMAP<NSCID, DrugInfo>::const_iterator &m_begin, 
	const MULTIMAP<NSCID, DrugInfo>::const_iterator &m_end, const string &m_cell_line)
{
	deque<float> ret;

	for(MULTIMAP<NSCID, DrugInfo>::const_iterator i = m_begin;i != m_end;++i){
		
		const MAP<string, float>::const_iterator iter = 
			i->second.activity.find(m_cell_line);

		if( iter == i->second.activity.end() ){
			
			// Missing activity data is ignored
			continue;
		}
		
		ret.push_back(iter->second);
	}
	
	return ret;
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

// Extract the tissue type prefix from the cell line name.
// Names must be in the form:
//	<tissue type>:<celline identifier>
string extract_tissue_prefix(const string &m_cell_line_name)
{
	string::size_type loc = m_cell_line_name.find(':');
	
	if(loc == string::npos){
		throw __FILE__ ":extract_tissue_prefix: Unable to find prefix delimeter";
	}
	
	return m_cell_line_name.substr(0, loc);
}

string get_processor_name()
{
	int buffer_size;
	char name[MPI_MAX_PROCESSOR_NAME];
	
	MPI_Get_processor_name(name, &buffer_size);
	
	return string(name);
}
