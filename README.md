# WarpLDA: Cache Efficient Implementation of Latent Dirichlet Allocation

## Introduction

WarpLDA is a cache efficient implementation of Latent Dirichlet Allocation, which samples each token in O(1).

## Installation
Prerequisites:

	* GCC (>=4.8.5)
	* CMake (>=2.8.12)

Clone this project

	git clone https://github.com/cjf00000/warplda

Install third-party dependency

	./get_gflags.sh

Download some data

	mkdir data
	cd data
	wget blah

Compile the project

	./build.sh
	cd release/src
	make -j

## Quick-start

Format the data

	./transform -input ../../data/nips.txt -method text2bin -skip 2	# Yahoo! LDA format
	# TODO libsvm format

Train the model

	./warplda --k 100 --niter 100

Check the result. Each line is a topic, its id, number of tokens assigned to it, and ten most frequent words with their probabilities.

	vim prefix.info.full.txt

Infer latent topics of some testing data.

	# TODO

## Data format

The data format is identical to Yahoo! LDA. The input data is a text file with a number of lines, where each line is a document. The format of each line is

	id1	id2 word1 word2 word3 ...

id1, id2 are two string document identifiers, and each word is a string, separated by white space.

## Output format

## License

MIT

## Reference

Please cite WarpLDA if you find it is useful!

	@inproceedings{chen2016warplda,
	  title={WarpLDA: a Cache Efficient O(1) Algorithm for Latent Dirichlet Allocation},
	  author={Chen, Jianfei and Li, Kaiwei and Zhu, Jun and Chen, Wenguang},
	  booktitle={VLDB},
	  year={2016}
	}
