// Learn you a hspp for great good.
// Chapter 7 Modules
// The samples are originated from Learn you a Haskell for great good.

#include "hspp.h"
#include "common.h"
#include <string>
#include <cmath>

using namespace hspp;
using namespace hspp::doN;
using namespace hspp::data;
using namespace std::literals;

void dataList0()
{
#if 0
    // haskell version
    ghci> intersperse '.' "MONKEY"
    "M.O.N.K.E.Y"
    ghci> intersperse 0 [1,2,3,4,5,6]
    [1,0,2,0,3,0,4,0,5,0,6]

    ghci> intercalate " " ["hey","there","guys"]
    "hey there guys"
    ghci> intercalate [0,0,0] [[1,2,3],[4,5,6],[7,8,9]]
    [1,2,3,0,0,0,4,5,6,0,0,0,7,8,9]

    ghci> concat ["foo","bar","car"]
    "foobarcar"
    ghci> concat [[3,4,5],[2,3,4],[2,1,1]]
    [3,4,5,2,3,4,2,1,1]

    ghci> concatMap (replicate 4) [1..3]
    [1,1,1,1,2,2,2,2,3,3,3,3]
#endif // 0

    auto const result0 = concat | std::vector{"foo"s, "bar"s, "car"s};
    expectEq(to<std::basic_string> | result0, "foobarcar");

    auto const result1 = concat | std::vector{std::vector{3, 4, 5}, std::vector{2, 3, 4}, std::vector{2, 1, 1}};
    expectEq(to<std::vector> | result1, std::vector{3, 4, 5, 2, 3, 4, 2, 1, 1});
}

void dataList1()
{
#if 0
    // haskell version
    ghci> and $ map (>4) [5,6,7,8]
    True
    ghci> and $ map (==4) [4,4,4,3,4]
    False

    ghci> or $ map (==4) [2,3,4,5,6,1]
    True
    ghci> or $ map (>4) [1,2,3]
    False

    ghci> any (==4) [2,3,5,6,1,4]
    True
    ghci> all (>4) [6,9,10]
    True
    ghci> all (`elem` ['A'..'Z']) "HEYGUYSwhatsup"
    False
    ghci> any (`elem` ['A'..'Z']) "HEYGUYSwhatsup"
    True
#endif // 0

    auto const result0 = and_ || map | [](auto x) { return x > 4; } | within(5, 8);
    expectEq(result0, true);

    auto const result1 = and_ || map | [](auto x) { return x == 4; } | std::vector{4, 4, 4, 3, 4};
    expectEq(result1, false);

    auto const result2 = or_ || map | [](auto x) { return x == 4; } | std::vector{2, 3, 4, 5, 6, 1};
    expectEq(result2, true);

    auto const result3 = or_ || map | [](auto x) { return x > 4; } | within(1, 3);
    expectEq(result3, false);

    auto const result4 = any | [](auto x) { return x == 4; } | std::vector{2, 3, 5, 6, 1, 4};
    expectEq(result4, true);

    auto const result5 = all | [](auto x) { return x > 4; } | std::vector{6, 9, 10};
    expectEq(result5, true);

    auto const result6 = all | [](auto x) { return x >= 'A' && x <= 'Z'; } | "HEYGUYSwhatsup"s;
    expectEq(result6, false);

    auto const result7 = any | [](auto x) { return x >= 'A' && x <= 'Z'; } | "HEYGUYSwhatsup"s;
    expectEq(result7, true);
}

void dataList2()
{
#if 0
    ghci> take 10 $ iterate (*2) 1
    [1,2,4,8,16,32,64,128,256,512]

    ghci> take 3 $ iterate (++ "haha") "haha"
    ["haha","hahahaha","hahahahahaha"]

    ghci> splitAt 3 "heyman"
    ("hey","man")
    ghci> splitAt 100 "heyman"
    ("heyman","")
    ghci> splitAt (-3) "heyman"
    ("","heyman")
    ghci> let (a,b) = splitAt 3 "foobar" in b ++ a "barfoo"
#endif // 0

    auto const result0 = take | 10U || iterate | [](auto x) { return x*2; } | 1;
    expectEq(to<std::vector> | result0, std::vector{1, 2, 4, 8, 16, 32, 64, 128, 256, 512});

    auto const result1 = take | 3U || iterate | [](auto x) { return x + "haha"s; } | "haha"s;
    expectEq(to<std::vector> | result1, std::vector{"haha"s, "hahahaha"s, "hahahahahaha"s});

    auto const result2 = splitAt | 3 | "heyman"s;
    expectEq(to<std::basic_string>(result2.first), "hey"s);
    expectEq(to<std::basic_string>(result2.second), "man"s);

    auto const result3 = splitAt | 100 | "heyman"s;
    expectEq(to<std::basic_string>(result3.first), "heyman"s);
    expectEq(to<std::basic_string>(result3.second), ""s);

    auto const result4 = splitAt | -3 | "heyman"s;
    expectEq(to<std::basic_string>(result4.first), ""s);
    expectEq(to<std::basic_string>(result4.second), "heyman"s);
}

int main()
{
    dataList0();
    dataList1();
    dataList2();
    return 0;
}