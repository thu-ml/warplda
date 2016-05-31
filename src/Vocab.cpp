#include "Vocab.hpp"
#include "Utils.hpp"

#include <fstream>
#include <sstream>

Vocab::Vocab()
{
}

Vocab::~Vocab()
{
}

void Vocab::clear()
{
    dict.clear();
    words.clear();
}

bool Vocab::load(std::string fname)
{
    clear();
    int id = 0;
    bool success = ForEachLinesInFile(fname, [&](std::string line)
    {
        std::istringstream sin(line);
        std::string word;
        sin >> word;
        dict[word] = id++;
        words.push_back(word);
    });
    return success;
}

bool Vocab::store(std::string fname)
{
    std::ofstream fou(fname);
    if (!fou) return false;
    std::string line;
    for (unsigned i = 0; i < words.size(); i++)
    {
        fou << words[i] << std::endl;
    }
    fou.close();
    return true;
}

int Vocab::addWord(std::string w)
{
    auto it = dict.find(w);
    if (it == dict.end())
    {
        dict[w] = words.size();
        words.push_back(w);
        return dict[w];
    }else
        return it->second;
}

int Vocab::getIdByWord(std::string w) const
{
    auto it = dict.find(w);
    if (it == dict.end())
        return -1;
    else
        return it->second;
}

std::string Vocab::getWordById(int id) const
{
    if (id < 0 || id >= (int)words.size())
        return "";
    else
        return words[id];
}

int Vocab::nWords() const
{
    return words.size();
}

void Vocab::RearrangeId(const unsigned int* new_id)
{
    for (auto &e : this->dict)
        e.second = new_id[e.second];
    decltype(words) new_words(words.size());
    for (unsigned i = 0; i < words.size(); i++)
        new_words[new_id[i]] = words[i];
    words = std::move(new_words);
}
