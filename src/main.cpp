#include <gflags/gflags.h>

DEFINE_string(input, "", "Input file prefix");

int main(int argc, char** argv)
{
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	printf("input %s\n", FLAGS_input.c_str());
	return 0;
}
