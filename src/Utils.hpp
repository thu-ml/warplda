#pragma once

//#include <mpi.h>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
//#include <gzstream.h>
#include <unistd.h>

inline uint32_t logceil(uint32_t n)
{
	for (uint32_t i = 0; i < 31; i++)
		if ((1U << i) >= n)
			return i;
	return -1;
}

inline unsigned int Divup(unsigned int a, unsigned int b)
{
	return (a + b - 1) / b;
}

inline std::vector<std::string> ReadlinesFromFile(std::string fname)
{
	std::vector<std::string> ret;
	std::ifstream fin(fname);
	if (!fin)
	{
		std::cerr << " : ReadlinesFromFile " << fname << " Failed" << std::endl;
		abort();
	}
	std::string line;
	while (std::getline(fin, line))
	{
		ret.push_back(line);
	}
	return ret;
}

inline void SetIfEmpty(std::string &s, std::string t)
{
	if (s.empty())
		s = t;
}

template <class Function>
bool ForEachLinesInFile(std::string fname, Function f)
{
    std::ifstream fin(fname);
    if (!fin) return false;
    std::string line;
    while (std::getline(fin, line))
    {
		f(line);
	}
	return true;
}
#if 0
template <class Function>
void ForEachLinesInFile(std::string fname, Function f)
{
	if (fname.substr(fname.size() - 3) == ".gz")
	{
		igzstream fin(fname.c_str());
		if (!fin)
		{
			std::cerr << " : Open " << fname << " Failed" << std::endl;
			abort();
		}

		std::string line;
		while (std::getline(fin, line))
		{
			std::istringstream sin(line);
			f(sin);
		}
	}else
	{
		std::ifstream fin(fname);
		if (!fin)
		{
			std::cerr << " : Open " << fname << " Failed" << std::endl;
			abort();
		}

		std::string line;
		while (std::getline(fin, line))
		{
			std::istringstream sin(line);
			f(sin);
		}
	}
}
#endif

inline std::string operator+(std::string str, int x)
{
	std::ostringstream ss;
	ss << str << x;
	return ss.str();
}

inline ssize_t Filesize(std::istream &fin)
{
	auto pos = fin.tellg();
	fin.seekg(0, std::ios_base::end);
	auto sz = fin.tellg();
	fin.seekg(pos);
	return sz;
}

/*
template <class Function>
void MyMPISerialize(Function f, int mpi_size, int mpi_rank)
{
	MPI_Barrier(MPI_COMM_WORLD);
	for (int i = 0 ; i < mpi_size; i++)
	{
		if (i == mpi_rank)
		{
			std::cout << "Serialize rank " << i << std::endl;
			f();
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
}

template <typename T>
struct MyMPIDataType
{
	static MPI_Datatype get_type();
};

template <>
struct MyMPIDataType<int>
{
	static MPI_Datatype get_type(){ return MPI_INT; }
};

template <>
struct MyMPIDataType<uint32_t>
{
	static MPI_Datatype get_type(){ return MPI_UNSIGNED; }
};

template <>
struct MyMPIDataType<unsigned long>
{
	static MPI_Datatype get_type(){ return MPI_UNSIGNED_LONG; }
};
*/

inline void Memoryinfo(double &tot, double &used)
{
  long phypz = sysconf(_SC_PHYS_PAGES);
  long psize = sysconf(_SC_PAGE_SIZE);
  long avphys = sysconf(_SC_AVPHYS_PAGES);
  tot = 1.0/(1L<<30)*psize*phypz;
  used = 1.0/(1L<<30)*avphys*psize;
}
static size_t filesize(std::ifstream &fs)
{
	size_t last = fs.tellg();
	fs.seekg(0, std::ios::end);
	size_t ret = fs.tellg();
	fs.seekg(last, std::ios::beg);
	return ret;

}

template <class T>
bool ReadVector(T & vec, std::string fname)
{
	std::ifstream f(fname, std::ios::binary);
	if (!f)
		return false;
	size_t sz = filesize(f);
	vec.resize(sz / sizeof(T::value_type));
	if (!f.read((char*)vec.data(), sz))
		return false;
	f.close();
	return true;
}

template <class T>
bool WriteVector(T & vec, std::string fname)
{
	std::ofstream f(fname, std::ios::binary);
	if (!f)
		return false;
	if (!f.write((char*)vec.data(), sizeof(T::value_type) * vec.size()))
		return false;
	f.close();
	return true;
}
