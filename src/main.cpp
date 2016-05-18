#define STRIP_FLAG_HELP 0
#include <gflags/gflags.h>
#include "Bigraph.hpp"
#include "Utils.hpp"
#include "warplda.hpp"

DEFINE_string(prefix, "./prefix", "prefix of result files");
DEFINE_int32(niter, 10, "number of iterations");
DEFINE_int32(k, 1000, "number of topics");
DEFINE_double(alpha, 50, "sum of alpha");
DEFINE_double(beta, 0.01, "beta");
DEFINE_int32(mh, 1, "number of Metropolis-Hastings steps");
DEFINE_int32(ntop, 10, "num top words per each topic");
DEFINE_string(bin, "", "binary file");
DEFINE_string(model, "", "model file");
DEFINE_string(info, "", "info");
DEFINE_string(vocab, "", "vocabulary file");
DEFINE_string(topics, "", "topic assignment file");
DEFINE_bool(estimate, false, "estimate model parameters");
DEFINE_bool(inference, false, "inference latent topic assignments");
DEFINE_bool(writeinfo, true, "write info");
DEFINE_bool(dumpmodel, true, "dump model");

int main(int argc, char** argv)
{
    gflags::SetUsageMessage("Usage : ./warplda [ flags... ]");
	gflags::ParseCommandLineFlags(&argc, &argv, true);

    SetIfEmpty(FLAGS_bin, FLAGS_prefix + ".bin");
    SetIfEmpty(FLAGS_model, FLAGS_prefix + ".model");
    SetIfEmpty(FLAGS_info, FLAGS_prefix + ".info");
    SetIfEmpty(FLAGS_vocab, FLAGS_prefix + ".vocab");
    SetIfEmpty(FLAGS_topics, FLAGS_prefix + ".topics");

    if ((FLAGS_inference || FLAGS_estimate) == false)
        FLAGS_estimate = true;

    LDA *lda = new WarpLDA<1>();
    lda->loadBinary(FLAGS_bin);
    if (FLAGS_estimate)
    {
        lda->estimate(FLAGS_k, FLAGS_alpha / FLAGS_k, FLAGS_beta, FLAGS_niter);
        if (FLAGS_dumpmodel)
            lda->storeModel(FLAGS_model);
        if (FLAGS_writeinfo)
            lda->writeInfo(FLAGS_vocab, FLAGS_info, FLAGS_ntop);
    }
    else if(FLAGS_inference)
    {
        lda->loadModel(FLAGS_model);
        lda->inference(FLAGS_niter);
	//	lda->dumpZ(FLAGS_topics);
    }
	return 0;
}
