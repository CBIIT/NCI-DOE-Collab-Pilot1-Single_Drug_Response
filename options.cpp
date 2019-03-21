#include "xTx.h"
#include "deque_set.h"
#include "mpi_util.h"
#include <iostream>
#include <getopt.h>
#include <mpi.h>

using namespace std;

#define		DEFAULT_FOLD			5
#define		DEFAULT_BAG_FRACTION		0.3
#define		DEFAULT_NUM_TREE		500
#define		DEFAULT_MIN_LEAF		3
#define		DEFAULT_STRATEGY		DISJOINT
#define		DEFAULT_RESPONSE_SAMPLE		32
#define		DEFAULT_MIN_RESPONSE		-20.0
#define		DEFAULT_MAX_RESPONSE		0.0

void Options::load(int argc, char* argv[])
{
	// Command line options:
	// --dose-response <Dose response file>
	// --cell <independent cell line variables (i.e. gene expression)>
	// --drug <independent drug variables (i.e. fingerprints/descriptors)>
	// --train <dataset to train on> (can be repeated)
	// --test <dataset to test on> (can be repeated)
	// [--test.pdm <PDM dataset to test on>]
	// -o <output file>
	// [--o.raw <output file of predicted vs observed>]
	// [--cv.fold <folds for cross validation > (default is 5)]
	// [--cv.disjoint | --cv.disjoint-cell | --cv.disjoint-drug | --cv.overlap] (cross validation strategy)
	// [-s <random number seed> (default is time-based seed)]
	// [--forest.bag <fraction of variables to bag with>]
	// [--forest.leaf <min leaf size for random forest>]
	// [--forest.size <number of trees for random forest>]
	const char* options = "o:s:?h";
	int config_opt = 0;
	int long_index = 0;

	struct option long_opts[] = {
		{"dose-response", true, &config_opt, 1},
		{"cell", true, &config_opt, 2},
		{"drug", true, &config_opt, 3},
		{"cv.fold", true, &config_opt, 4},
		{"forest.bag", true, &config_opt, 5},
		{"forest.leaf", true, &config_opt, 6},
		{"forest.size", true, &config_opt, 7},
		{"train", true, &config_opt, 8},
		{"test", true, &config_opt, 9},
		{"cv.disjoint", false, &config_opt, 10},
		{"cv.disjoint-cell", false, &config_opt, 11},
		{"cv.disjoint-drug", false, &config_opt, 12},
		{"cv.overlap", false, &config_opt, 13},
		{"o.raw", true, &config_opt, 14},
		{"test.pdm", true, &config_opt, 15},
		{0,0,0,0} // Terminate options list
	};

	int opt_code;
	opterr = 0;

	print_usage = (argc == 1);

	fold = DEFAULT_FOLD;
	cv_strategy = DEFAULT_STRATEGY;

	seed = time(NULL);

	forest_bag_fraction = DEFAULT_BAG_FRACTION;
	forest_min_leaf = DEFAULT_MIN_LEAF;
	forest_size = DEFAULT_NUM_TREE;

	num_response_sample = DEFAULT_RESPONSE_SAMPLE;
	min_response_sample = DEFAULT_MIN_RESPONSE;
	max_response_sample = DEFAULT_MAX_RESPONSE;
	
	while( (opt_code = getopt_long( argc, argv, options, long_opts, &long_index) ) != EOF ){

		switch( opt_code ){
			case 0:

				if(config_opt == 1){ // dose-response

					dose_response_file = optarg;
					break;
				}

				if(config_opt == 2){ // cell

					cell_feature_file = optarg;
					break;
				}

				if(config_opt == 3){ // drug

					drug_feature_file = optarg;
					break;
				}

				if(config_opt == 4){ // cv.fold

					fold = abs( atoi(optarg) );
					break;
				}

				if(config_opt == 5){ // forest.bag

					forest_bag_fraction = atof(optarg);
					break;
				}

				if(config_opt == 6){ // forest.leaf

					forest_min_leaf = abs( atoi(optarg) );
					break;
				}

				if(config_opt == 7){ // forest.size

					forest_size = abs( atoi(optarg) );
					break;
				}

				if(config_opt == 8){ // train

					training_datasets.push_back(optarg);
					break;
				}

				if(config_opt == 9){ // test

					testing_datasets.push_back(optarg);
					break;
				}

				if(config_opt == 10){ // cv.disjoint

					cv_strategy = DISJOINT;
					break;
				}

				if(config_opt == 11){ // cv.disjoint-cell

					cv_strategy = DISJOINT_CELL;
					break;
				}

				if(config_opt == 12){ // cv.disjoint-drug

					cv_strategy = DISJOINT_DRUG;
					break;
				}

				if(config_opt == 13){ // cv.disjoint-drug

					cv_strategy = OVERLAPPING;
					break;
				}

				if(config_opt == 14){ // o.raw

					prediction_file = optarg;
					break;
				}
				
				if(config_opt == 15){ // test.pdm

					pdm_file = optarg;
					break;
				}
				
				cerr << "Unknown flag!" << endl;
				break;
			case 's':
				seed = abs( atoi(optarg) );
				break;
			case 'o':
				output_file = optarg;
				break;
			case 'h':
			case '?':
				print_usage = true;
				break;
			default:
				cerr << '\"' << (char)opt_code << "\" is not a valid option!" << endl;
				break;
		};
	}

	if(print_usage){

		cerr << "Usage (version " CROSS_TRAIN_VERSION "):" << endl;
		cerr << "\t--dose-response <Dose response file>" << endl;
		cerr << "\t--cell <independent cell line variables (i.e. gene expression)>" << endl;
		cerr << "\t--drug <independent drug variables (i.e. fingerprints/descriptors)>" << endl;
		cerr << "\t--train <dataset to train on> (can be repeated)" << endl;
		cerr << "\t--test <dataset to test on> (can be repeated)" << endl;
		cerr << "\t[--test.pdm <PDM dataset to test on>]" << endl;
		cerr << "\t-o <output file>" << endl;
		cerr << "\t[--o.raw <output file of predicted vs observed>]" << endl;
		cerr << "\t[--cv.fold <folds for cross validation > (default is 5)]" << endl;
		cerr << "\tCross validation strategy" << endl;
		cerr << "\t[--cv.disjoint | --cv.disjoint-cell | --cv.disjoint-drug | --cv.overlap] "
			<< "(cross validation strategy; default is cv.overlap)" << endl;
		cerr << "\t[-s <random number seed> (default is time-based seed)]" << endl;
		cerr << "\t[--forest.bag <fraction of variables to bag with>] (default is "
			<< DEFAULT_BAG_FRACTION << ")" << endl;
		cerr << "\t[--forest.leaf <min leaf size for random forest>] (default is "
			<< DEFAULT_MIN_LEAF << ")" << endl;
		cerr << "\t[--forest.size <number of trees for random forest>] (default is "
			<< DEFAULT_NUM_TREE << ")" << endl;
	}
	else{

		if( output_file.empty() ){

			cerr << "Please specify a -o output file" << endl;
			print_usage = true;
		}

		if( dose_response_file.empty() ){

			cerr << "Please specify a --dose-response file" << endl;
			print_usage = true;
		}

		if( cell_feature_file.empty() ){

			cerr << "Please specify a --cell file" << endl;
			print_usage = true;
		}

		if( drug_feature_file.empty() ){

			cerr << "Please specify a --drug file" << endl;
			print_usage = true;
		}

		if(forest_size < 1){

			cerr << "Invalid number of parameters (--forest.size), please use a value > 0" << endl;
			print_usage = true;
		}

		if( (forest_bag_fraction <= 0.0) || (forest_bag_fraction > 1.0) ){

			cerr << "Invalid bagging feature fraction (--forest.bag), please use a value (0, 1]" << endl;
			print_usage = true;
		}

		if(forest_min_leaf <= 0){

			cerr << "Invalid minimum leaf size (--forest.leaf) for random forest, please use a value > 0" << endl;
			print_usage = true;
		}

		if(num_response_sample <= 1){
		
			cerr << "Invalid number of response samples (must be > 1)" << endl;
			print_usage = true;
		}
		
		if(max_response_sample <= min_response_sample){
		
			cerr << "Invalid number of response sample range" << endl;
			print_usage = true;
		}
		
		if( training_datasets.empty() ){

			cerr << "Please specify one or more --train datasets to train with" << endl;
			print_usage = true;
		}
		
		make_set(training_datasets);
		make_set(testing_datasets);
	}
}

