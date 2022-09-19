// Learn you a hspp for great good.
// The samples are originated from Learn you a Haskell for great good.

#include "hspp.h"
#include "common.h"
#include <string>

using namespace hspp;
using namespace hspp::doN;
using namespace hspp::data;
using namespace std::literals;

void believeTheType0()
{
#if 0
    // haskell version
    addThree :: Int -> Int -> Int -> Int
    addThree x y z = x + y + z
#endif // 0

    constexpr auto addThree = toFunc<> | [](int x, int y, int z) -> int
    {
        return x + y + z;
    };
    expectEq(addThree | 1 | 2 | 3, 6);
}


void believeTheType1()
{
#if 0
    // haskell version
    circumference :: Float -> Float
    circumference r = 2 * pi * r
    ghci> circumference 4.0
    25.132742
#endif // 0

    constexpr auto circumference = toFunc<> | [](float r) -> float
    {
        constexpr float kPI = 3.14159265F;
        return 2 * kPI * r;
    };
    expectEq(circumference | 4, 25.13274F);
}

void typeclasses101_0()
{
#if 0
    // haskell version
    ghci> "Abrakadabra" < "Zebra"
    True
    ghci> "Abrakadabra" `compare` "Zebra"
    LT
#endif // 0

    using hspp::operators::operator <;
    expectTrue("Abrakadabra"s < "Zebra"s);
    expectEq("Abrakadabra"s <compare> "Zebra"s, Ordering::kLT);
}


void typeclasses101_1()
{
#if 0
    // haskell version
    ghci> show 3
    "3"
    ghci> show 5.334 "5.334"
    ghci> show True
    "True"
#endif // 0

    expectEq(show | 3, "3");
    expectEq(show | 5.334, "5.334");
    expectEq(show | true, "true");
}

void typeclasses101_2()
{
#if 0
    // haskell version
    ghci> read "True" || False
    True
    ghci> read "8.2" + 3.8
    12.0
    ghci> read "5" - 2
    3
    ghci> read "[1,2,3,4]" ++ [3]
    [1,2,3,4,3]
#endif // 0

    expectEq(read<bool>("true"), true);
    expectEq(read<double>("8.2") + 3.8, 12.0);
    expectEq(read<int>("5") - 2, 3);
    // read<list> has not been implemented yet.
}


int main()
{
    believeTheType0();
    believeTheType1();
    typeclasses101_0();
    typeclasses101_1();
    typeclasses101_2();
    return 0;
}