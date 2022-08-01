#include <string>
#include <array>
#include <vector>
#include <iostream>

using std::string;

class Solution {
public:
    using StringMap = std::array<int, 26>;
    StringMap string2Map(string const& str)
    {
        StringMap char2num{};
        for (char c : str)
        {
            char2num[c - 'a']++;
        }
        return char2num;
    }
    int canBeFormed(string const& str, StringMap const& strMap)
    {
        StringMap m = string2Map(str);

        for (int32_t i = 0; i < 26; ++i)
        {
            if (m[i] > strMap[i])
            {
                return 0;
            }
        }
        return str.size();
    }
    int countCharacters(std::vector<string> const& words, string chars) {
        StringMap charsMap = string2Map(chars);
        int result = 0;
        for (auto const& word: words)
        {
            result += canBeFormed(word, charsMap);
        }
        return result;
    }
};

int main()
{
    Solution s;
    std::cout << s.countCharacters(std::vector<std::string>{"cat","bt","hat","tree"}, "atach") << std::endl;
}
