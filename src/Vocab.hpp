#pragma once

#include <string>
#include <vector>
#include <unordered_map>

class Vocab
{
private:
    std::unordered_map<std::string, int> dict;
    std::vector<std::string> words;
public:
    Vocab();
    ~Vocab();
    bool load(std::string fname);
    bool store(std::string fname);
    int addWord(std::string w);
    void clear();
    int getIdByWord(std::string w) const;
    std::string getWordById(int id) const;
    void RearrangeId(const unsigned int* new_id);
    int nWords() const;
    int operator[](std::string w) const { return getIdByWord(w); }
    std::string operator[](int id) const { return getWordById(id); }
};
