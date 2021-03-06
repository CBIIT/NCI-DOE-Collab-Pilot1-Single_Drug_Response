#! /bin/sh

NUM_TREE=500
BAG=0.3
LEAF=3

cd /home/jgans/Cancer/src/cross_train

export OMP_NUM_THREADS=24

#mpirun -np $PBS_NUM_NODES --bind-to none ./xTx \
#	--cv.fold 0 \
#	--forest.size $NUM_TREE \
#	--forest.bag $BAG \
#	--forest.leaf $LEAF \
#	--dose-response data/response/rescaled_combined_single_drug_growth \
#	--cell data/cell_features/combined_rnaseq_data_lincs1000_source_scale_median_PDM \
#	--drug data/drug_features/pan_drugs_dragon7_PFP.tsv \
#	-o output/NCI60.pdm.median.out \
#	--o.raw output/NCI60.pdm.median.dat \
#	--test.pdm data/PDM/PDM-NCI.July2018.csv \
#	--train NCI60
#
#exit

mpirun -np $PBS_NUM_NODES --bind-to none ./xTx \
	--cv.fold 0 \
	--forest.size $NUM_TREE \
	--forest.bag $BAG \
	--forest.leaf $LEAF \
	--dose-response data/response/rescaled_combined_single_drug_growth \
	--cell data/cell_features/combined_rnaseq_data_lincs1000_source_scale_median_PDM \
	--drug data/drug_features/pan_drugs_dragon7_PFP.tsv \
	-o output/NCI60.out \
	--test gCSI \
	--train NCI60 \
	--test CTRP \
	--test CCLE \
	--test GDSC
