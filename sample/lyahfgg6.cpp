// Learn you a hspp for great good.
// Chapter 6 Higher order functions
// The samples are originated from Learn you a Haskell for great good.

#include "hspp.h"
#include "common.h"
#include <string>

using namespace hspp;
using namespace hspp::doN;
using namespace hspp::data;
using namespace std::literals;

void curriedFunctions0()
{
#if 0
    // haskell version
    ghci> max 4 5
    5
    ghci> (max 4) 5
    5
#endif // 0

    constexpr auto max = toFunc<> | [](int a, int b) { return std::max(a, b); };

    expectEq(max | 4 | 5 , 5);
    expectEq(max(4)(5), 5);
}

void curriedFunctions1()
{
#if 0
    // haskell version
    multThree :: (Num a) => a -> a -> a -> a
    multThree x y z = x * y * z
    ghci> let multTwoWithNine = multThree 9
    ghci> multTwoWithNine 2 3
    54
    ghci> let multWithEighteen = multTwoWithNine 2 ghci> multWithEighteen 10
    180
#endif // 0

    constexpr auto multThree = toFunc<> | [](int x, int y, int z) { return x * y * z; };
    constexpr auto multTwoWithNine = multThree | 9;
    expectEq(multTwoWithNine | 2 | 3 , 54);

    constexpr auto multWithEighteen = multTwoWithNine | 2;
    expectEq(multWithEighteen | 10, 180);
}

void curriedFunctions2()
{
#if 0
    // haskell version
    compareWithHundred :: (Num a, Ord a) => a -> Ordering
    compareWithHundred x = compare 100 x

    compareWithHundred :: (Num a, Ord a) => a -> Ordering
    compareWithHundred = compare 100

#endif // 0

    constexpr auto compareWithHundred0 = toFunc<> | [](int x) { return compare | 100 | x; };
    constexpr auto compareWithHundred1 = compare | 100;

    expectEq(compareWithHundred0 | 99, compareWithHundred1 | 99);
}

void curriedFunctions3()
{
#if 0
    // haskell version
    divideByTen :: (Floating a) => a -> a
    divideByTen = (/10)

    isUpperAlphanum :: Char -> Bool
    isUpperAlphanum = (`elem` ['A'..'Z'])

#endif // 0

    constexpr auto divide = toFunc<> | std::divides<float>{} ;
    constexpr auto divideByTen = flip | divide | 100.F;
    constexpr auto isUpperAlphanum = flip | elem | within('A', 'Z');

    expectEq(divideByTen | 111.F, 1.11F);
    expectEq(isUpperAlphanum | 's', false);
    expectEq(isUpperAlphanum | 'S', true);
}


void someHigherOrderismIsInOrder0()
{
#if 0
    // haskell version
    applyTwice :: (a -> a) -> a -> a
    applyTwice f x = f (f x);

    ghci> applyTwice (+3) 10
    16
    ghci> applyTwice (++ " HAHA") "HEY"
    "HEY HAHA HAHA"
    ghci> applyTwice ("HAHA " ++) "HEY"
    "HAHA HAHA HEY"
    ghci> applyTwice (multThree 2 2) 9
    144
    ghci> applyTwice (3:) [1]
    [3,3,1]
#endif // 0

    constexpr auto applyTwice = toGFunc<2> | [](auto f, auto x) { return f(f(x)); };
    constexpr auto multThree = toFunc<> | [](int x, int y, int z) { return x * y * z; };

    constexpr auto add = toGFunc<2> | std::plus<>{} ;
    expectEq(applyTwice | (add | 3) | 10, 16);
    expectEq(applyTwice | (chain | "HAHA "s) | "HEY"s, "HAHA HAHA HEY");
    expectEq(applyTwice | (multThree | 2 | 2) | 9, 144);
    auto result = applyTwice | (cons | 3) | std::vector{1};
    expectEq(result, std::vector{3, 3, 1});
}

int main()
{
    curriedFunctions0();
    curriedFunctions1();
    curriedFunctions2();
    curriedFunctions3();
    someHigherOrderismIsInOrder0();
    return 0;
}