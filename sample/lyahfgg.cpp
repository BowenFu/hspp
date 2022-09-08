// Learn you a hspp for great good.

#include "hspp.h"

auto expectTrue(bool x)
{
    if (!x)
    {
        throw std::runtime_error{"False in expectedTrue!"};
    }
}

template <typename T>
auto expectEq(T const& l, T const& r)
{
    if (l != r)
    {
        std::stringstream ss;
        ss << l << " != " << r;
        throw std::runtime_error{ss.str()};
    }
}

using namespace hspp;
using namespace hspp::doN;
using namespace hspp::data;
using namespace std::literals;



void example1()
{
#if 0
    // haskell version
    doubleMe x = x + x
#endif // 0
    constexpr auto doubleMe = toGFunc<1> | [](auto x)
    {
        return x + x;
    };
    expectEq(doubleMe | 9, 18);
    expectEq(doubleMe | 8.3, 16.6);
}

int main()
{
    example1();
    return 0;
}