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
    // haskell version
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

void dataList3()
{
#if 0
    // haskell version
    ghci> takeWhile (>3) [6,5,4,3,2,1,2,3,4,5,4,3,2,1]
    [6,5,4]
    ghci> takeWhile (/=' ') "This is a sentence"
    "This"
    ghci> sum $ takeWhile (<10000) $ map (^3) [1..]
    53361

    ghci> dropWhile (/=' ') "This is a sentence"
    " is a sentence"
    ghci> dropWhile (<3) [1,2,2,2,3,4,5,4,3,2,1]
    [3,4,5,4,3,2,1]

    ghci> let stock = [(994.4,2008,9,1), (995.2,2008,9,2), (999.2,2008,9,3), (1001.4,2008,9,4), (998.3,2008,9,5)]
    ghci> head (dropWhile (\(val,y,m,d) -> val < 1000) stock)
    (1001.4,2008,9,4)
#endif

    auto const result0 = takeWhile | [](auto x) { return x > 3; } | std::vector{6, 5, 4, 3, 2, 1 ,2 ,3 ,4 ,5 ,4 ,3 ,2 ,1};
    expectEq(to<std::vector> | result0, std::vector{6, 5, 4});

    auto const result1 = takeWhile | [](auto x) { return x != ' '; } | "This is a sentence"s;
    expectEq(to<std::basic_string> | result1, "This"s);

    auto const result2 = sum || (takeWhile | [](auto x) { return x < 10000; }) | (map | [](auto x) { return x*x*x; } | enumFrom(1));
    expectEq(result2, 53361);

    auto const result3 = dropWhile | [](auto x) { return x != ' '; } | "This is a sentence"s;
    expectEq(to<std::basic_string> | result3, " is a sentence"s);

    auto const result4 = dropWhile | [](auto x) { return x < 3; } | std::vector{1, 2, 2, 2, 3, 4, 5, 4, 3, 2, 1};
    expectEq(to<std::vector> | result4, std::vector{3, 4, 5, 4, 3, 2, 1});

    auto const stock = std::vector{std::tuple{994.4,2008, 9, 1}, std::tuple{995.2, 2008, 9, 2}, std::tuple{999.2, 2008, 9, 3}, std::tuple{1001.4, 2008, 9, 4}, std::tuple{998.3, 2008, 9, 5}};
    auto const result5 = head || dropWhile | [](auto t) { return std::get<0>(t) < 1000; } | stock;
    expectEq(result5, std::tuple{1001.4, 2008, 9, 4});
}

int main()
{
    dataList0();
    dataList1();
    dataList2();
    dataList3();
    return 0;
}