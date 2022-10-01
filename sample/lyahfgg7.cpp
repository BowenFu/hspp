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

    ghci> let (fw, rest) = span (/=' ') "This is a sentence" in "First word: " ++ fw ++ ", the rest:" ++ rest
    "First word: This, the rest: is a sentence"

    ghci> break (==4) [1,2,3,4,5,6,7]
    ([1,2,3],[4,5,6,7])
    ghci> span (/=4) [1,2,3,4,5,6,7]
    ([1,2,3],[4,5,6,7])
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

    auto const [fw, rest] = span | [](auto x) { return x != ' '; } | "This is a sentence"s;
    auto const result6 = "First word: "s + to<std::basic_string>(fw) + ", the rest:" + to<std::basic_string>(rest);
    expectEq(result6, "First word: This, the rest: is a sentence"s);

    auto const [result7a, result7b] =  break_ | equalTo(4) | within(1, 7);
    expectEq(to<std::vector> | result7b, std::vector{4, 5, 6, 7});
    expectEq(to<std::vector> | result7a, std::vector{1, 2, 3});

    auto const [result8a, result8b] =  span | [](auto x){ return x != 4; } | within(1, 7);
    expectEq(to<std::vector> | result8a, std::vector{1, 2, 3});
    expectEq(to<std::vector> | result8b, std::vector{4, 5, 6, 7});
}

void dataList4()
{
#if 0
    // haskell version
    ghci> group [1,1,1,1,2,2,2,2,3,3,2,2,2,5,6,7]
    [[1,1,1,1],[2,2,2,2],[3,3],[2,2,2],[5],[6],[7]]

    ghci> map (\l@(x:xs) -> (x,length l)) . group . sort $ [1,1,1,1, 2,2,2,2,3,3,2,2,2,5,6,7]
    [(1,4),(2,7),(3,2),(5,1),(6,1),(7,1)]

    ghci> let values = [-4.3, -2.4, -1.2, 0.4, 2.3, 5.9, 10.5, 29.1, 5.3, -2.4, -14.5, 2.9, 2.3]
    ghci> groupBy (\x y -> (x > 0) == (y > 0)) values
    [[-4.3,-2.4,-1.2],[0.4,2.3,5.9,10.5,29.1,5.3],[-2.4,-14.5],[2.9,2.3]]

    on :: (b -> b -> c) -> (a -> b) -> a -> a -> c
    f `on` g = \x y -> f (g x) (g y)

    ghci> groupBy ((==) `on` (> 0)) values
    [[-4.3,-2.4,-1.2],[0.4,2.3,5.9,10.5,29.1,5.3],[-2.4,-14.5],[2.9,2.3]]
#endif

    // store in a separate variable to prolong its life so that result0 is still valid in expectEq.
    auto const vec = std::vector{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 2, 2, 2, 5, 6, 7};
    auto const result0 = group | nonOwnedRange(vec);
    auto const resultIVec = to<std::vector>(result0);
    auto const resultVec = to<std::vector> <fmap>  resultIVec;
    expectEq(resultVec, std::vector{
            std::vector{1, 1, 1, 1}, std::vector{2, 2, 2, 2},
            std::vector{3, 3}, std::vector{2, 2, 2},
            std::vector{5}, std::vector{6}, std::vector{7}}
        );
    auto const vec2 = std::vector{1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 5, 6, 7};
    auto const result1 = (map | [](auto l) { return std::make_tuple(head | l, length | l); }) <o> group || nonOwnedRange(vec2);
    expectEq(to<std::vector> | result1, std::vector{std::tuple{1, 4U}, std::tuple{2, 7U}, std::tuple{3, 2U}, std::tuple{5, 1U}, std::tuple{6, 1U}, std::tuple{7, 1U}});

    auto const values = std::vector{-4.3, -2.4, -1.2, 0.4, 2.3, 5.9, 10.5, 29.1, 5.3, -2.4, -14.5, 2.9, 2.3};
    auto const result2 = groupBy | [](auto x, auto y) { return (x > 0) == (y > 0); } | nonOwnedRange(values);
    expectEq(to<std::vector> <fmap> to<std::vector>(result2), std::vector{
            std::vector{-4.3, -2.4, -1.2}, std::vector{0.4, 2.3, 5.9, 10.5, 29.1, 5.3},
            std::vector{-2.4, -14.5}, std::vector{2.9, 2.3}}
        );

    auto const result3 = groupBy | (equalTo  <on> [](auto x) { return x > 0 ; }) | nonOwnedRange(values);
    expectEq(to<std::vector> <fmap> to<std::vector>(result3), std::vector{
            std::vector{-4.3, -2.4, -1.2}, std::vector{0.4, 2.3, 5.9, 10.5, 29.1, 5.3},
            std::vector{-2.4, -14.5}, std::vector{2.9, 2.3}}
        );
}

void dataList5()
{
#if 0
    // haskell version
    ghci> inits "w00t"
    ["","w","w0","w00","w00t"]
    ghci> tails "w00t"
    ["w00t","00t","0t","t",""]
    ghci> let w = "w00t" in zip (inits w) (tails w)
    [("","w00t"),("w","00t"),("w0","0t"),("w00","t"),("w00t","")]

    search :: (Eq a) => [a] -> [a] -> Bool search needle haystack =
    let nlen = length needle
        in foldl (\acc x -> if take nlen x == needle then True else acc) False (tails haystack)

    ghci> "cat" `isInfixOf` "im a cat burglar"
    True
    ghci> "Cat" `isInfixOf` "im a cat burglar"
    False
    ghci> "cats" `isInfixOf` "im a cat burglar"
    False

    ghci> "hey" `isPrefixOf` "hey there!"
    True
    ghci> "hey" `isPrefixOf` "oh hey there!"
    False
    ghci "there!" `isSuffixOf` "oh hey there!">
    True
    ghci> "there!" `isSuffixOf` "oh hey there"
    False


    ghci> find (>4) [1,2,3,4,5,6]
    Just 5
    ghci> find (>9) [1,2,3,4,5,6]
    Nothing
    ghci> :t find
    find :: (a -> Bool) -> [a] -> Maybe a

    ghci> :t elemIndex
    elemIndex :: (Eq a) => a -> [a] -> Maybe Int
    ghci> 4 `elemIndex` [1,2,3,4,5,6]
    Just 3
    ghci> 10 `elemIndex` [1,2,3,4,5,6]
    Nothing

    ghci> ' ' `elemIndices` "Where are the spaces?"
    [5,9,13]

    ghci> findIndex (==4) [5,3,2,1,6,4] Just 5
    ghci> findIndex (==7) [5,3,2,1,6,4] Nothing
    ghci> findIndices (`elem` ['A'..'Z']) "Where Are The Caps?" [0,6,10,14]

    ghci> zipWith3 (\x y z -> x + y + z) [1,2,3] [4,5,2,2] [2,2,3]
    [7,9,8]
    ghci> zip4 [2,3,3] [2,2,2] [5,5,3] [2,2,2]
    [(2,2,5,2),(3,2,5,2),(3,2,3,2)]

    ghci> lines "first line\nsecond line\nthird line"
    ["first line","second line","third line"]

    ghci> unlines ["first line", "second line", "third line"]
    "first line\nsecond line\nthird line\n"

    ghci> words "hey these are the words in this sentence"
    ["hey","these","are","the","words","in","this","sentence"]
    ghci> words "hey these           are    the words in this\nsentence"
    ["hey","these","are","the","words","in","this","sentence"]
    ghci> unwords ["hey","there","mate"]
    "hey there mate"

    ghci> nub [1,2,3,4,3,2,1,2,3,4,3,2,1]
    [1,2,3,4]
    ghci> nub "Lots of words and stuff"
    "Lots fwrdanu"

    ghci> delete 'h' "hey there ghang!"
    "ey there ghang!"
    ghci> delete 'h' . delete 'h' $ "hey there ghang!"
    "ey tere ghang!"
    ghci> delete 'h' . delete 'h' . delete 'h' $ "hey there ghang!" "ey tere gang!"

    ghci> [1..10] \\ [2,5,9] [1,3,4,6,7,8,10]
    ghci> "Im a big baby" \\ "big" "Im a baby"
    Doing [1..10] \\ [2,5,9] is like doing delete 2 . delete 5 . delete 9 $ [1..10].

    ghci> "hey man" `union` "man what's up" "hey manwt'sup"
    ghci> [1..7] `union` [5..10] [1,2,3,4,5,6,7,8,9,10]

    ghci> [1..7] `intersect` [5..10]
    [5,6,7]

    ghci> insert 4 [3,5,1,2,8,2]
    [3,4,5,1,2,8,2]
    ghci> insert 4 [1,3,4,4,1]
    [1,3,4,4,4,1]

    ghci> insert 4 [1,2,3,5,6,7]
    [1,2,3,4,5,6,7]
    ghci> insert 'g' $ ['a'..'f'] ++ ['h'..'z']
    "abcdefghijklmnopqrstuvwxyz"
    ghci> insert 3 [1,2,4,3,2,1]
    [1,2,3,4,3,2,1]

    ghci> let xs = [[5,4,5,4,4],[1,2,3],[3,5,4,3],[],[2],[2,2]]
    ghci> sortBy (compare `on` length) xs
    [[],[2],[2,2],[1,2,3],[3,5,4,3],[5,4,5,4,4]]
#endif
}

void dataList6()
{
#if 0
    // haskell version
    ghci> partition (`elem` ['A'..'Z']) "BOBsidneyMORGANeddy"
    ("BOBMORGAN","sidneyeddy")
    ghci> partition (>3) [1,3,5,6,3,2,1,0,3,7]
    ([5,6,7],[1,3,3,2,1,0,3])

    ghci> span (`elem` ['A'..'Z']) "BOBsidneyMORGANeddy"
    ("BOB","sidneyMORGANeddy")
#endif

    auto const [result00, result01] = partition | (flip | elem | within('A', 'Z')) | "BOBsidneyMORGANeddy"s;
    expectEq(to<std::basic_string> | result00, "BOBMORGAN"s);
    expectEq(to<std::basic_string> | result01, "sidneyeddy"s);
}

int main()
{
    dataList0();
    dataList1();
    dataList2();
    dataList3();
    dataList4();
    dataList5();
    dataList6();
    return 0;
}