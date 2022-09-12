// Learn you a hspp for great good.
// The samples are originated from Learn you a Haskell for great good.

#include "hspp.h"
#include "common.h"
#include <string>

using namespace hspp;
using namespace hspp::doN;
using namespace hspp::data;
using namespace std::literals;


// Baby's first functions
void babysFirstFunctions()
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

#if 0
    // haskell version
    doubleUs x y = doubleMe x + doubleMe y
#endif // 0

    constexpr auto doubleUs = toGFunc<2> | [=](auto x, auto y)
    {
        return (doubleMe | x) + (doubleMe | y);
    };
    expectEq(doubleUs | 3 | 4, 14);

#if 0
    // haskell version
    doubleSmallNumber x = if x > 100 then x else x*2
#endif // 0

    constexpr auto doubleSmallNumber = toGFunc<1> | [](auto x)
    {
        return x > 100 ? x : x*2;
    };
    expectEq(doubleSmallNumber | 30, 60);
    expectEq(doubleSmallNumber | 130, 130);
}

constexpr auto toVector = data::to<std::vector>;
constexpr auto toString = data::to<std::basic_string>;

void anIntroToLists1()
{
#if 0
    // haskell version
    ghci> [1,2,3,4] ++ [9,10,11,12]
    [1,2,3,4,9,10,11,12]
#endif // 0

    auto const result = chain | within(1, 4) | within(9, 12);
    auto const expected = std::vector{1, 2, 3, 4, 9, 10, 11, 12};
    expectEq(toVector | result, expected);
}

void anIntroToLists2()
{
#if 0
    // haskell version
    ghci> "hello" ++ " " ++ "world"
    "hello world"
#endif // 0

    auto const result = "hello"s <chain> " "s <chain> "world"s;
    auto const expected = "hello world"s;
    expectEq(toString | result, expected);
}

void anIntroToLists3()
{
#if 0
    // haskell version
    ghci> 'A':" SMALL CAT"
    "A SMALL CAT"
#endif // 0

    auto const result = 'A' <cons> " SMALL CAT"s;
    auto const expected = "A SMALL CAT"s;
    expectEq(toString | result, expected);
}

void anIntroToLists4()
{
#if 0
    // haskell version
    ghci> 5:[1,2,3,4,5]
    [5,1,2,3,4,5]
#endif // 0

    auto const result = 5 <cons> within(1, 5);
    auto const expected = std::vector{5, 1, 2, 3, 4, 5};
    expectEq(toVector | result, expected);
}

void anIntroToLists5()
{
#if 0
    // haskell version
    ghci> head [5,4,3,2,1]
    5
    ghci> tail [5,4,3,2,1]
    [4,3,2,1]
    ghci> last [5,4,3,2,1]
    1
    ghci> init [5,4,3,2,1]
    [5,4,3,2]
    ghci> length [5,4,3,2,1]
    5
    ghci> null [1,2,3]
    False
    ghci> null []
    True
#endif // 0

    auto const result0 = head || within_(5, 4, 1);
    auto const expected0 = 5;
    expectEq(result0, expected0);

    auto const result1 = tail || within_(5, 4, 1);
    auto const expected1 = std::vector{4, 3, 2, 1};
    expectEq(toVector | result1, expected1);

    auto const result2 = last || within_(5, 4, 1);
    auto const expected2 = 1;
    expectEq(result2, expected2);

    auto const result3 = init || within_(5, 4, 1);
    auto const expected3 = std::vector{5, 4, 3, 2};
    expectEq(toVector | result3, expected3);

    auto const result4 = length || within_(5, 4, 1);
    auto const expected4 = 5U;
    expectEq(result4, expected4);

    auto const result5 = null || std::vector{1, 2, 3};
    auto const expected5 = false;
    expectEq(result5, expected5);

    auto const result6 = null || std::vector<int>{};
    auto const expected6 = true;
    expectEq(result6, expected6);
}

void anIntroToLists6()
{
#if 0
    // haskell version
    ghci> take 3 [5,4,3,2,1]
    [5,4,3]
    ghci> take 1 [3,9,3]
    [3]
    ghci> take 5 [1,2]
    [1,2]
    ghci> take 0 [6,6,6]
    []
#endif // 0

    auto const result0 = take | 3U | within_(5, 4, 1);
    auto const expected0 = std::vector{5, 4, 3};
    expectEq(toVector | result0, expected0);

    auto const result1 = take | 1U | std::vector{3, 9, 3};
    auto const expected1 = std::vector{3};
    expectEq(toVector | result1, expected1);

    auto const result2 = take | 5U | std::vector{1, 2};
    auto const expected2 = std::vector{1, 2};
    expectEq(toVector | result2, expected2);

    auto const result3 = take | 0U | std::vector{6, 6, 6};
    auto const expected3 = std::vector<int>{};
    expectEq(toVector | result3, expected3);
}

void anIntroToLists7()
{
#if 0
    // haskell version
    ghci> drop 3 [8,4,2,1,5,6]
    [1,5,6]
    ghci> drop 0 [1,2,3,4]
    [1,2,3,4]
    ghci> drop 100 [1,2,3,4]
    []
#endif // 0

    auto const result0 = drop | 3U | std::vector{8, 4, 2, 1, 5, 6};
    auto const expected0 = std::vector{1, 5, 6};
    expectEq(toVector | result0, expected0);

    auto const result1 = drop | 0U | within(1, 4);
    auto const expected1 = std::vector{1, 2, 3, 4};
    expectEq(toVector | result1, expected1);

    auto const result2 = drop | 100U | within(1, 4);
    auto const expected2 = std::vector<int>{};
    expectEq(toVector | result2, expected2);
}

void anIntroToLists8()
{
#if 0
    // haskell version
    ghci> minimum [8,4,2,1,5,6]
    1
    ghci> maximum [1,9,2,3,4]
    9
#endif // 0

    auto const result0 = minimum | std::vector{8, 4, 2, 1, 5, 6};
    auto const expected0 = 1;
    expectEq(result0, expected0);

    auto const result1 = maximum | std::vector{1, 9, 2, 3, 4};
    auto const expected1 = 9;
    expectEq(result1, expected1);
}

void anIntroToLists9()
{
#if 0
    // haskell version
    ghci> sum [5,2,1,6,3,2,5,7]
    31
    ghci> product [6,2,1,2]
    24
    ghci> product [1,2,5,6,7,9,2,0]
    0
#endif // 0

    auto const result0 = sum | std::vector{5, 2, 1, 6, 3, 2, 5, 7};
    auto const expected0 = 31;
    expectEq(result0, expected0);

    auto const result1 = product | std::vector{6, 2, 1, 2};
    auto const expected1 = 24;
    expectEq(result1, expected1);

    auto const result2 = product | std::vector{1, 2, 5, 6, 7, 9, 2, 0};
    auto const expected2 = 0;
    expectEq(result2, expected2);
}

void texasRanges()
{
#if 0
    // haskell version
    [1..20]
    ['a'..'z']
#endif // 0

    auto result0 = toVector || within(1, 20);
    auto result1 = toString || within('a', 'z');

    auto const expected0 = std::vector{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    auto const expected1 = "abcdefghijklmnopqrstuvwxyz";
    expectEq(result0, expected0);
    expectEq(result1, expected1);

#if 0
    // haskell version
    [2,4..20]
    [3,6..20]
#endif // 0

    auto result2 = toVector || within_ | 2 | 4 | 20;
    auto result3 = toVector || within_ | 3 | 6 | 20;

    auto const expected2 = std::vector{2, 4, 6, 8, 10, 12, 14, 16, 18, 20};
    auto const expected3 = std::vector{3, 6, 9, 12, 15, 18};
    expectEq(result2, expected2);
    expectEq(result3, expected3);
}

void imAListComprehension1()
{
#if 0
    // haskell version
    [x*2 | x <- [1..10]]
    [x*2 | x <- [1..10], x*2 >= 12]
#endif // 0
    Id<int> x;
    auto const result = _(
        x * 2,
        x <= within(1, 10),
        if_ | (x*2 >= 12)
    );
    auto const expected = std::vector{12, 14, 16, 18, 20};
    expectEq(toVector(result), expected);
}

void imAListComprehension2()
{
#if 0
    // haskell version
    [ x | x <- [50..100], x `mod` 7 == 3]
#endif // 0

    Id<int> x;
    auto const result = _(
        x,
        x <= within(50, 100),
        if_ | (x%7 == 3)
    );
    auto const expected = std::vector{52, 59, 66, 73, 80, 87, 94};
    expectEq(toVector(result), expected);
}

void imAListComprehension3()
{
#if 0
    // haskell version
    boomBangs xs = [ if x < 10 then "BOOM!" else "BANG!" | x <- xs, odd x]
#endif // 0
    constexpr auto boomBangs = toGFunc<1> | [](auto xs)
    {
        Id<int> x;
        return _(
            ifThenElse || x < 10 || "BOOM!"sv || "BANG!"sv,
            x <= xs,
            if_ || odd | x
        );
    };
    auto const result = toVector(boomBangs || within(7, 13));
    auto const expected = std::vector{"BOOM!"sv, "BOOM!"sv,"BANG!"sv,"BANG!"sv};
    expectEq(result, expected);
}

void imAListComprehension4()
{
#if 0
    // haskell version
    [ x | x <- [10..20], x /= 13, x /= 15, x /= 19] [10,11,12,14,16,17,18,20]
#endif // 0
    Id<int> x;
    auto const rng = _(
        x,
        x <= within(10, 20),
        if_ || x != 13,
        if_ || x != 15,
        if_ || x != 19
    );
    auto const result = toVector(rng);
    auto const expected = std::vector{10, 11, 12, 14, 16, 17, 18, 20};
    expectEq(result, expected);
}

void imAListComprehension5()
{
#if 0
    // haskell version
    [ x*y | x <- [2,5,10], y <- [8,10,11]]
    [ x*y | x <- [2,5,10], y <- [8,10,11], x*y > 50]
#endif // 0
    Id<int> x, y;
    auto const rng = _(
        x*y,
        x <= std::vector{2, 5, 10},
        y <= std::vector{8, 10, 11},
        if_ || x*y > 50
    );
    auto const result = toVector(rng);
    auto const expected = std::vector{55, 80, 100, 110};
    expectEq(result, expected);
}

void imAListComprehension6()
{
#if 0
    // haskell version
    ghci> let nouns = ["hobo","frog","pope"]
    ghci> let adjectives = ["lazy","grouchy","scheming"]
    ghci> [adjective ++ " " ++ noun | adjective <- adjectives, noun <- nouns]
    ["lazy hobo","lazy frog","lazy pope","grouchy hobo","grouchy frog",
    "grouchy pope","scheming hobo","scheming frog","scheming pope"]
#endif // 0

    auto const nouns = std::vector{"hobo", "frog", "pope"};
    auto const adjectives = std::vector{"lazy", "grouchy", "scheming"};
    Id<char const*> adjective, noun;
    auto const rng = _(
        (cast<std::string> | adjective) + " " + noun,
        adjective <= adjectives,
        noun <= nouns
    );
    auto const result = toVector(rng);
    auto const expected = std::vector{"lazy hobo"s, "lazy frog"s, "lazy pope"s, "grouchy hobo"s, "grouchy frog"s, "grouchy pope"s, "scheming hobo"s, "scheming frog"s, "scheming pope"s};
    expectEq(result, expected);
}

void imAListComprehension7()
{
#if 0
    // haskell version
    removeNonUppercase st = [ c | c <- st, c `elem` ['A'..'Z']]
    ghci> removeNonUppercase "Hahaha! Ahahaha!"
    "HA"
    ghci> removeNonUppercase "IdontLIKEFROGS"
    "ILIKEFROGS"
#endif // 0

    constexpr auto removeNonUppercase = toGFunc<1> | [](auto st)
    {
        Id<char> c;
        return _(
            c,
            c <= st,
            if_ || (c >= 'A' && c <= 'Z')
        );
    };
    auto const result1 = removeNonUppercase | "Hahaha! Ahahaha!"s;
    auto const expected1 = "HA"sv;
    expectEq(result1, expected1);

    auto const result2 = removeNonUppercase | "IdontLIKEFROGS"s;
    auto const expected2 = "ILIKEFROGS"sv;
    expectEq(result2, expected2);
}

int main()
{
    babysFirstFunctions();
    anIntroToLists1();
    anIntroToLists2();
    anIntroToLists3();
    anIntroToLists4();
    anIntroToLists5();
    anIntroToLists6();
    anIntroToLists7();
    anIntroToLists8();
    anIntroToLists9();
    texasRanges();
    imAListComprehension1();
    imAListComprehension2();
    imAListComprehension3();
    imAListComprehension4();
    imAListComprehension5();
    imAListComprehension6();
    imAListComprehension7();
    return 0;
}