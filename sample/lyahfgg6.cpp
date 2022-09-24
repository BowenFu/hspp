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

void someHigherOrderismIsInOrder1()
{
#if 0
    // haskell version
    ghci> zipWith (+) [4,2,5,6] [2,6,2,3]
    [6,8,7,9]
    ghci> zipWith max [6,3,2,1] [7,3,1,5]
    [7,3,2,5]
    ghci> zipWith (++) ["foo ", "bar ", "baz "] ["fighters", "hoppers", "aldrin"]
    ["foo fighters","bar hoppers","baz aldrin"]
    ghci> zipWith (*) (replicate 5 2) [1..]
    [2,4,6,8,10]
    ghci> zipWith (zipWith (*)) [[1,2,3],[3,5,6],[2,3,4]] [[3,2,2],[3,4,5],[5,4,3]]
    [[3,4,6],[9,20,30],[10,12,12]]
#endif // 0

    auto result0 = zipWith | std::plus<>{} | std::vector{4, 2, 5, 6} | std::vector{2, 6, 2, 3};
    expectEq(to<std::vector> | result0, std::vector{6, 8, 7, 9});

    constexpr auto max = toFunc<> | [](int a, int b) { return std::max(a, b); };
    auto result1 = zipWith | max | std::vector{6, 3, 2, 1} | std::vector{7, 3, 1, 5};
    expectEq(to<std::vector> | result1, std::vector{7, 3, 2, 5});

    auto result2 = zipWith | chain | std::vector{"foo "s, "bar "s, "baz "s} | std::vector{"fighters"s, "hoppers"s, "aldrin"s};
    expectEq(to<std::vector> | result2, std::vector{"foo fighters"s, "bar hoppers"s, "baz aldrin"s});

    auto result3 = zipWith | std::multiplies<>{} | replicate(5U, 2) | enumFrom(1);
    expectEq(to<std::vector> | result3, std::vector{2, 4, 6, 8, 10});

    auto result4 = zipWith | (zipWith | std::multiplies<>{})
        | std::vector{std::vector{1, 2, 3}, std::vector{3, 5, 6}, std::vector{2, 3, 4}}
        | std::vector{std::vector{3, 2, 2}, std::vector{3, 4, 5}, std::vector{5, 4, 3}};
    expectEq(to<std::vector> || to<std::vector> <fmap> result4, std::vector{std::vector{3, 4, 6}, std::vector{9, 20, 30}, std::vector{10, 12, 12}});
}

void someHigherOrderismIsInOrder2()
{
#if 0
    // haskell version
    ghci> flip zip [1,2,3,4,5] "hello"
    [('h',1),('e',2),('l',3),('l',4),('o',5)]
    ghci> zipWith (flip div) [2,2..] [10,8,6,4,2]
    [5,4,3,2,1]
#endif // 0

    auto const result0 = flip | zip | within(1, 5) | "hello"s;
    expectEq(to<std::vector> | result0, std::vector{std::tuple{'h', 1}, std::tuple{'e', 2}, std::tuple{'l', 3}, std::tuple{'l', 4}, std::tuple{'o', 5}});

    auto const result1 = zipWith | (flip | (toGFunc<2> | std::divides<>{})) | repeat(2) | within_(10, 8, 2);
    expectEq(to<std::vector> | result1, std::vector{5, 4, 3, 2, 1});
}

void mapsAndFilters0()
{
#if 0
    // haskell version
    ghci> map (+3) [1,5,3,1,6]
    [4,8,6,4,9]
    ghci> map (++ "!") ["BIFF", "BANG", "POW"]
    ["BIFF!","BANG!","POW!"]
    ghci> map (replicate 3) [3..6]
    [[3,3,3],[4,4,4],[5,5,5],[6,6,6]]
    ghci> map (map (^2)) [[1,2],[3,4,5,6],[7,8]]
    [[1,4],[9,16,25,36],[49,64]]
    ghci> map fst [(1,2),(3,5),(6,3),(2,6),(2,5)]
    [1,3,6,2,2]
#endif // 0

    auto const result0 = map | (toGFunc<2> | std::plus<>{} | 3) | std::vector{1, 5, 3, 1, 6};
    expectEq(to<std::vector> | result0, std::vector{4, 8, 6, 4, 9});

    auto const result1 = map | (flip | chain | "!"s) | std::vector{"BIFF"s, "BANG"s, "POW"s};
    expectEq(to<std::vector> | result1, std::vector{"BIFF!"s, "BANG!"s, "POW!"s});

    auto const result2 = map | (replicate | 3U) | within(3, 6);
    expectEq(to<std::vector> || to<std::vector> <fmap> result2, std::vector{std::vector{3, 3, 3}, std::vector{4, 4, 4}, std::vector{5, 5, 5}, std::vector{6, 6, 6}});

    auto const result3 = map | (map | [](auto x){ return x*x; }) | std::vector{std::vector{1, 2}, std::vector{3, 4, 5, 6}, std::vector{7, 8}};
    expectEq(to<std::vector> || to<std::vector> <fmap> result3, std::vector{std::vector{1, 4}, std::vector{9, 16, 25, 36}, std::vector{49, 64}});

    auto const result4 = map | fst | std::vector{std::tuple{1, 2}, std::tuple{3, 5}, std::tuple{6, 3}, std::tuple{2, 6}, std::tuple{2, 5}};
    expectEq(to<std::vector> | result4, std::vector{1, 3, 6, 2, 2});
}

void mapsAndFilters1()
{
#if 0
    // haskell version
    ghci> map (+3) [1,5,3,1,6]
    [4,8,6,4,9]
    ghci> map (++ "!") ["BIFF", "BANG", "POW"]
    ["BIFF!","BANG!","POW!"]
    ghci> map (replicate 3) [3..6]
    [[3,3,3],[4,4,4],[5,5,5],[6,6,6]]
    ghci> map (map (^2)) [[1,2],[3,4,5,6],[7,8]]
    [[1,4],[9,16,25,36],[49,64]]
    ghci> map fst [(1,2),(3,5),(6,3),(2,6),(2,5)]
    [1,3,6,2,2]
#endif // 0

    auto const result0 = map | (toGFunc<2> | std::plus<>{} | 3) | std::vector{1, 5, 3, 1, 6};
    expectEq(to<std::vector> | result0, std::vector{4, 8, 6, 4, 9});

    auto const result1 = map | (flip | chain | "!"s) | std::vector{"BIFF"s, "BANG"s, "POW"s};
    expectEq(to<std::vector> | result1, std::vector{"BIFF!"s, "BANG!"s, "POW!"s});

    auto const result2 = map | (replicate | 3U) | within(3, 6);
    expectEq(to<std::vector> || to<std::vector> <fmap> result2, std::vector{std::vector{3, 3, 3}, std::vector{4, 4, 4}, std::vector{5, 5, 5}, std::vector{6, 6, 6}});

    auto const result3 = map | (map | [](auto x){ return x*x; }) | std::vector{std::vector{1, 2}, std::vector{3, 4, 5, 6}, std::vector{7, 8}};
    expectEq(to<std::vector> || to<std::vector> <fmap> result3, std::vector{std::vector{1, 4}, std::vector{9, 16, 25, 36}, std::vector{49, 64}});

    auto const result4 = map | fst | std::vector{std::tuple{1, 2}, std::tuple{3, 5}, std::tuple{6, 3}, std::tuple{2, 6}, std::tuple{2, 5}};
    expectEq(to<std::vector> | result4, std::vector{1, 3, 6, 2, 2});
}

int main()
{
    curriedFunctions0();
    curriedFunctions1();
    curriedFunctions2();
    curriedFunctions3();
    someHigherOrderismIsInOrder0();
    someHigherOrderismIsInOrder1();
    someHigherOrderismIsInOrder2();
    mapsAndFilters0();
    mapsAndFilters1();
    return 0;
}