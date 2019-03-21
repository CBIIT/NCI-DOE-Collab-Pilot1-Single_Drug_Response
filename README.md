# Using the Random Forest machine learning algorithm to predict the concentration-, cell line- and drug-dependent response function.

Given a cell line feature vector (currently gene expression) and a drug feature vector (currently molecular fingerprints or molecular descriptors), can we predict the dose-dependent response of a cell line to a given drug using machine learning? Furthermore, is a machine learning model trained on one data set able to make accurate predictions of dose response for a different data set (i.e. is the machine learning "generalizable")? Due to batch effects arising from different experimental protocols, it is all to common for machine learning algorithms *not* to be generalizable.

In an attempt to answer these questions, we have developed response prediction software, called "xTx", that uses the popular Random Forest machine learning algorithm to train and test predictions of dose response. The software offers the option to train on dose response values from one data set and then test on the response data from a number of other data sets.

If you encounter any problems installing or running this code, please let me (Jason Gans; jgans@lanl.gov) know.

## Installing the code
The xTx program is written in C++ and must be compiled prior to use. In addition, the machine-learning training and testing is performed in parallel using OpenMP (for optional thread-level parallelism) and MPI ("Message Passing Interface"; for compute-level parallelism). As a result, MPI must first be installed. While there are a number of freely-available MPI implmentation, the xTx code has been developed and tested using [OpenMPI](https://www.open-mpi.org/). The use of a C++ compiler that supports OpenMP is recommended, but not required (the current C++ clang compiler provided with Mac OS X does not support OpenMP).

After installing MPI, run the `make` command in the `LANL_SingleDrug_RF` directory to build the xTx application.

If the compilation was successful, you should now have an executable file, called `xTx`, located in the `LANL_SingleDrug_RF` director. Running the `xtx` program should produce a list of valid command line arguments:

```
$ ./xTx
Usage (version 0.1):
	--dose-response <Dose response file>
	--cell <independent cell line variables (i.e. gene expression)>
	--drug <independent drug variables (i.e. fingerprints/descriptors)>
	--train <dataset to train on> (can be repeated)
	--test <dataset to test on> (can be repeated)
	[--test.pdm <PDM dataset to test on>]
	-o <output file>
	[--o.raw <output file of predicted vs observed>]
	[--cv.fold <folds for cross validation > (default is 5)]
	Cross validation strategy
	[--cv.disjoint | --cv.disjoint-cell | --cv.disjoint-drug | --cv.overlap] (cross validation strategy; default is cv.overlap)
	[-s <random number seed> (default is time-based seed)]
	[--forest.bag <fraction of variables to bag with>] (default is 0.3)
	[--forest.leaf <min leaf size for random forest>] (default is 3)
	[--forest.size <number of trees for random forest>] (default is 500)
```

## Required inputs
A number of inputs are used to predict the dose-dependent response of a cell line to a given drug:

* Cell line (or other cancer model; i.e. PDX) features
* Drug features
* Dose-response

### Cell line features
The features for each cell line are the expression values of a set of genes. All of the gene expression values for all cell lines must be stored in a single, *tab-delimited* text file. The expected format is cell lines (rows) by genes (columns), i.e.:

|Sample|AARS|ABCB6|ABCC5|ABCF1|ABCF3|...|
|------|----|-----|-----|-----|-----|---|
|CCLE.22RV1|0.6436|1.666|-0.003286|-1.612|0.4407|...|
|CCLE.2313287|1.465|1.039|-0.3093|0.02362|0.3325|...|
|CCLE.253J|-0.3083|1.205|-0.5625|-1.101|-0.1426|...|
|CCLE.253JBV|-0.0345|0.8306|-0.07715|-0.249|0.2896|...|
|CCLE.42MGBA|0.03072|1.342|-1.091|0.569|-0.877|...|
|CCLE.5637|-2.2|-1.106|-1.133|-0.0275|-0.7905|...|
|CCLE.59M|0.9175|0.679|-0.953|0.177|-0.8335|...|
|CCLE.639V|-0.8687|-0.732|0.155|0.3984|0.635|...|
|...|...|...|...|...|...|...|

The first row contains the header information; the word `Sample` followed by gene names.
The first column in each row contains the cell line name, i.e. `CCL.22rV1`. The expected format of the cell line names is:

`<data source prefix>.<cell line name suffix>`

Data source prefixes are used by the xTx software to identify training and testing data. Currently allowed data source prefixes (corresponding to large, previously published, cell line-based dose response characterization efforts) are:
* [CCLE](https://portals.broadinstitute.org/ccle)
* [NCI60](https://discover.nci.nih.gov/cellminer/)
* [CTRP](https://portals.broadinstitute.org/ctrp/)
* [GDSC](https://www.cancerrxgene.org/)
* [gCSI](https://pharmacodb.pmgenomics.ca/datasets/4)

The gene expression values are real numbers. Missing values are allowed, and should be left blank/empty in the gene expression file.

### Drug features
The features for each drug are stored in a single, *tab-delimited* text file. The expected format is drugs (rows) by features (columns), i.e.:

|Drug|feature 1|feature 2|feature 3|feature 4|feature 5|...|
|----|---------|---------|---------|---------|---------|---|
|CCLE.10|1|0|0|0|1|...|
|CCLE.11|1|1|0|0|1|...|
|CCLE.12|1|1|0|0|0|...|
|CCLE.13|1|1|1|1|1|...|
|CCLE.14|1|1|1|1|1|...|
|CCLE.15|1|1|1|1|0|...|
|CCLE.16|1|0|0|1|1|...|
|CCLE.17|0|1|0|0|1|...|
|...|...|...|...|...|...|...|

The first row contains (arbitrary) header information and is *completely ignored!*
The first column of each row contains the drug name, i.e. `CCLE.10`. The expected format of the drug names is:

`<data source prefix>.<drug id suffix>`

While the example above shows numeric values for the drug id suffix, any string (without tabs) is valid. 
You need to have a separate entry for each drug/data source combination, even for the common case of the *same* drug appearing in different data sources (typically with different IDs; i.e. NSC, CID, ...).

The feature values are allowed to be real numbers in order to allow both molecular fingerprints (as shown in the example above) and molecular discriptors (not shown; these real-valued features typically encode chemical properties like molecular volume, charge distribution, etc.).

### Dose response data
There are many ways to characterize the response of a cell line (or other cancer model) to different concentrations of a given drug. In this application, the expected form of the dose-dependent response data is in a 7 column *tab-delimited* text file:

|SOURCE|DRUG_ID|CELLNAME|CONCUNIT|LOG_CONCENTRATION|EXPID|GROWTH|
|------|-------|--------|--------|-----------------|-----|------|
|CCLE|CCLE.1|CCLE.1321N1|M|-8.60205999132796|fake_exp|117.34|
|CCLE|CCLE.1|CCLE.1321N1|M|-8.09691001300806|fake_exp|122|
|CCLE|CCLE.1|CCLE.1321N1|M|-7.60205999132796|fake_exp|104.32|
|CCLE|CCLE.1|CCLE.1321N1|M|-7.09691001300806|fake_exp|100.54|
|CCLE|CCLE.1|CCLE.1321N1|M|-6.60205999132796|fake_exp|80|
|CCLE|CCLE.1|CCLE.1321N1|M|-6.09691001300806|fake_exp|74|
|CCLE|CCLE.1|CCLE.1321N1|M|-5.59687947882418|fake_exp|48|
|CCLE|CCLE.1|CCLE.1321N1|M|-5.09691001300806|fake_exp|14|
|....|......|...........|...|...............|........|...|

The first row is the file header. The columns are defined as:

* The first column contains the data source.
* The second column contains the drug name. Each drug name (in `<data source prefix>.<drug id suffix>` format) must exactly match a drug name entry stored in the first column of a row in the drug feature file.
* The third column contains the cell line name. Each cell line name (in `<data source prefix>.<cell line name suffix>` format) must exactly match a cell line name stored in the first column of a row in the cell line feature file.
* The fourth column is the units of the drug concentration. Currently, only `M` (Molarity) is supported.
* The fifth column is the base 10 log of the concentration.
* The sixth column is the experiment id, which is not used. While some data sources provide an experiment id (i.e. NCI60) that is useful for disentangling replicate measurements, most data sources do not.
* The seventh column is the experimentally measured growth. Growth values are expected to be in the range \[-100, 100\] (following the %GI convention introduced by the NCI60). A value of -100 corresponds to "all treated cells died", while a value of 100 corresponds to "treated cells grew the same as untreated cells". Treated cells that grew *faster* than untreated cells are allowed to have growth values that are greater than 100 (these values will be clamped to 100).

## Running the code

The `xTx` program is designed to be run in parallel on a cluster computer using MPI. Jobs are typically submitted to a cluster computer via scheduling software (i.e. torque, grid engine, ...) that vary by cluster. However, most schedulers expect a script file as input. 

An example shell script file for running the `xTx` program is provided below. This script trains on the dose-response data provided by the CCLE project and tests the resulting random forest model on data provided by the NCI60, GDSC, CTRP and gCSI projects.

```
#! /bin/sh

# The number of trees in the random forest
NUM_TREE=500

# The fraction of variables that will be randomly sampled from cell line feature vectors and from drug feature vectors to construct each tree in the random forest.
BAG=0.3

# The minimum number of samples that should be contained in each decision tree leaf (i.e. samples are not split after this limit is reached).
LEAF=3

# The number of nodes that job has access to. Typically this can be obtained from the scheduler via an environment variable.
# For the Torque scheduler, it is in the variable $PBS_NUM_NODES
NUM_NODES=128

# Start in the directory of your choice
cd /your/home/directory/path/to/xtx/project

# Use mpirun to run xTx on the requested number of nodes. When running with the OpenMPI variant of MPI, 
# the "--bind-to none" flag allows OpenMP to use multiple threads.
mpirun -np $PBS_NUM_NODES --bind-to none ./xTx \
	--cv.fold 5 \
	--forest.size $NUM_TREE \
	--forest.bag $BAG \
	--forest.leaf $LEAF \
	--dose-response data/response/rescaled_combined_single_drug_growth \
	--cell data/cell_features/combined_rnaseq_data_lincs1000_source_scale_median_PDM \
	--drug data/drug_features/pan_drugs_dragon7_PFP.tsv \
	-o output/CCLE.out \
	--test NCI60 \
	--test CTRP \
	--train CCLE \
	--test gCSI \
	--test GDSC
```

Please note the following:

* The `--cv.fold` flag controls the number of cross validation folds that will be used to evaluate the prediction accuracy when both training and testing on the data source indicated by `--train`. A value of 0 for `--cv.fold` disables cross validation on the input training set.
* The `--dose-response` flag specifies the input file that contains the dose response data. The example shown above assumes the existence of a subdirectory call `data`. If your dose response data is in a different location, please change this path.
* The `--cell` flag specifies the input file that contains the cell line feature data. The example shown above assumes the existence of a subdirectory call `data`. If your gene expression data is in a different location, please change this path.
* The `--drug` flag specifies the input file that contains the drug feature data. The example shown above assumes the existence of a subdirectory call `data`. If your drug feature data is in a different location, please change this path.
* The `-o` flag specifies the output file that will be created to store the software output. The example shown above assumes the existence of an output directory call `output`. Change this path to the location that you would like the results written to.
* The `--train` flag specifies a data source prefix that will be used for training. This flag can be repeated multiple times to combine multiple data sources in a single training set.
* The `--test` flag specifies a data sorce prefix that will be used from testing. This flag can be repeated multiple times to evaluate the performance of the same random forest algorithm on data from multiple sources.
