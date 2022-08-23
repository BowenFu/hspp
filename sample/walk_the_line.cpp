#include "hspp.h"
#include <memory>
#include <variant>
#include <cassert>

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
    if (!(l == r))
    {
        throw std::runtime_error{"Values are not equal!"};
    }
}

using namespace hspp;
using namespace hspp::doN;
using namespace hspp::data;

// Based on haskell version from Learn You a Haskell for Great Good!


void walkTheLine()
{
    using Birds = int;
    using Pole = std::pair<Birds, Birds>;
    constexpr auto landLeft = toFunc<>([](Birds n, Pole p)
    {
        auto [left, right] = p;
        if (std::abs((left + n) - right) < 4) 
        {
            return Maybe<Pole>{Pole{left + n, right}};
        }
        return Maybe<Pole>{};
    });
    constexpr auto landRight = toFunc<>([](Birds n, Pole p)
    {
        auto [left, right] = p;
        if (std::abs((right + n) - left) < 4) 
        {
            return Maybe<Pole>{Pole{left, right + n}};
        }
        return Maybe<Pole>{};
    });
    auto const result = (((return_ | Pole{0,0}) >>= (landRight | 2)) >>= (landLeft | 2)) >>= (landRight | 2);
    expectEq(result, just(Pole(2, 4)));

    auto const result2 = ((((return_ | Pole{0,0}) >>= (landLeft | 1)) >>= (landRight | 4)) >>= (landLeft | -1)) >>= (landRight | -2);
    expectEq<decltype(result2)>(result2, nothing);

    constexpr auto banana = toFunc<>([](Pole)
    {
        return Maybe<Pole>{};
    });
    auto const result3 = (((return_ | Pole{0,0}) >>= (landLeft | 1)) >>= banana) >>= (landRight | 1);
    expectEq<decltype(result3)>(result3, nothing);
}

int main()
{
    walkTheLine();
    return 0;
}
