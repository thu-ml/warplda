#define STRIP_FLAG_HELP 0
#include <gflags/gflags.h>
#include "Bigraph.hpp"
#include "Utils.hpp"
#include "warplda.hpp"

DEFINE_string(prefix, "./prefix", "prefix of temporary files");
DEFINE_int32(niter, 10, "Num of iteration");
DEFINE_int32(k, 1000, "Num of topics");
DEFINE_double(alpha, 50, "alpha");
DEFINE_double(beta, 0.01, "beta");
DEFINE_int32(mh, 1, "mh steps");
DEFINE_string(bin, "", "binary file");
DEFINE_string(model, "", "model file");
DEFINE_string(info, "", "info");
DEFINE_string(vocab, "", "vocabulary file");
DEFINE_bool(estimate, true, "estimate");
DEFINE_bool(inference, false, "inference");
DEFINE_bool(writeinfo, false, "write info");

int main(int argc, char** argv)
{
    gflags::SetUsageMessage("Usage : ./warplda [ flags... ]");
	gflags::ParseCommandLineFlags(&argc, &argv, true);

    SetIfEmpty(FLAGS_bin, FLAGS_prefix + ".bin");
    SetIfEmpty(FLAGS_model, FLAGS_prefix + ".model");
    SetIfEmpty(FLAGS_info, FLAGS_prefix + ".info");
    SetIfEmpty(FLAGS_vocab, FLAGS_prefix + ".vocab");

    LDA *lda = new WarpLDA<1>();
    lda->loadBinary(FLAGS_bin);
    if (FLAGS_estimate)
        lda->estimate(FLAGS_k, FLAGS_alpha / FLAGS_k, FLAGS_beta, FLAGS_niter);
    else if(FLAGS_inference)
    {
        lda->loadModel(FLAGS_model);
        lda->inference(FLAGS_k, FLAGS_alpha / FLAGS_k, FLAGS_beta, FLAGS_niter);
    }
    if (FLAGS_writeinfo)
        lda->writeInfo(FLAGS_vocab, FLAGS_info);
	return 0;
}
