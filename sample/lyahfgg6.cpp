// Learn you a hspp for great good.
// Chapter 6 Higher order functions
// The samples are originated from Learn you a Haskell for great good.

#include "hspp.h"
#include "common.h"
#include <string>
#include <cmath>

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

    constexpr auto add = toGFunc<2> | std::plus<>{};
    expectEq(applyTwice | (add | 3) | 10, 16);
    expectEq(applyTwice | (plus | "HAHA "s) | "HEY"s, "HAHA HAHA HEY");
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

    auto result2 = zipWith | plus | std::vector{"foo "s, "bar "s, "baz "s} | std::vector{"fighters"s, "hoppers"s, "aldrin"s};
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

    constexpr auto div = (toGFunc<2> | std::divides<>{});
    auto const result1 = zipWith | (flip | div) | repeat(2) | within_(10, 8, 2);
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

    auto const result1 = map | (flip | plus | "!"s) | std::vector{"BIFF"s, "BANG"s, "POW"s};
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
    ghci> filter (>3) [1,5,3,2,1,6,4,3,2,1]
    [5,6,4]
    ghci> filter (==3) [1,2,3,4,5]
    [3]
    ghci> filter even [1..10]
    [2,4,6,8,10]
    ghci> let notNull x = not (null x) in filter notNull [[1,2,3], [], [3,4,5], [2,2], [], [], []]
    [[1,2,3],[3,4,5],[2,2]]
    ghci> filter (`elem` ['a'..'z']) "u LaUgH aT mE BeCaUsE I aM diFfeRent"
    "uagameasadifeent"
    ghci> filter (`elem` ['A'..'Z']) "i lauGh At You BecAuse u r aLL the Same"
    "GAYBALLS"
#endif // 0

    auto const result0 = filter | [](auto x){ return x > 3; } | std::vector{1, 5, 3, 2, 1, 6, 4, 3, 2, 1};
    expectEq(to<std::vector> | result0, std::vector{5, 6, 4});

    auto const result1 = filter | (equalTo | 3) | within(1, 5);
    expectEq(to<std::vector> | result1, std::vector{3});

    auto const result2 = filter | even | within(1, 10);
    expectEq(to<std::vector> | result2, std::vector{2, 4, 6, 8, 10});

    auto const result3 = filter | [](auto const& x){ return !null(x); } | std::vector{std::vector{1, 2, 3}, std::vector<int>{}, std::vector{3, 4, 5}, std::vector{2, 2}, std::vector<int>{}, std::vector<int>{}, std::vector<int>{}};
    expectEq(to<std::vector> | result3, std::vector{std::vector{1, 2, 3}, std::vector{3, 4, 5}, std::vector{2, 2}});

    auto const result4 = filter | (flip | elem | within('a', 'z')) | "u LaUgH aT mE BeCaUsE I aM diFfeRent"s;
    expectEq(to<std::basic_string> | result4, "uagameasadifeent");

    auto const result5 = filter | (flip | elem | within('A', 'Z')) | "i lauGh At You BecAuse u r aLL the Same"s;
    expectEq(to<std::basic_string> | result5, "GAYBALLS");
}

void mapsAndFilters2()
{
#if 0
    // haskell version
    largestDivisible :: (Integral a) => a
    largestDivisible = head (filter p [100000,99999..])
        where p x = x `mod` 3829 == 0
    ghci> largestDivisible
    99554

    ghci> sum (takeWhile (<10000) (filter odd (map (^2) [1..])))
    166650

    ghci> sum (takeWhile (<10000) [n^2 | n <- [1..], odd (n^2)])
    166650

    ghci> let listOfFuns = map (*) [0..]
    ghci> (listOfFuns !! 4) 5
    20
#endif // 0

    auto const largestDivisible = head || filter | [](auto x){ return x % 3824 == 0; } | within_(1000000, 99999, 0);
    expectEq(largestDivisible, 99554);

    auto const result1 = sum | (takeWhile | [](int x){ return x < 10000; } | (filter | odd | (map | [](auto x){ return x*x; } | enumFrom(1))));
    expectEq(result1, 166650);

    Id<int> n;
    auto const view = (takeWhile | [](int x){ return x < 10000; } | _(n*n, n <= enumFrom(1), if_ || odd | (n*n)));
    auto const result2 = sum | to<std::vector>(view);
    expectEq(result2, 166650);

    auto const result3 = filter | [](auto const& x){ return !null(x); } | std::vector{std::vector{1, 2, 3}, std::vector<int>{}, std::vector{3, 4, 5}, std::vector{2, 2}, std::vector<int>{}, std::vector<int>{}, std::vector<int>{}};
    expectEq(to<std::vector> | result3, std::vector{std::vector{1, 2, 3}, std::vector{3, 4, 5}, std::vector{2, 2}});

    auto const listOfFuns = map | (toGFunc<2> | std::multiplies<>{}) | enumFrom(0);
    auto const result6 = (listOfFuns <idx> 4U) | 5;
    expectEq(result6, 20);
}

void lambdas()
{
#if 0
    // haskell version
    ghci> zipWith (\a b -> (a * 30 + 3) / b) [5,4,3,2,1] [1,2,3,4,5]
    [153.0,61.5,31.0,15.75,6.6]

    ghci> map (\(a,b) -> a + b) [(1,2),(3,5),(6,3),(2,6),(2,5)]
    [3,8,9,8,7]

    addThree :: (Num a) => a -> a -> a -> a
    addThree x y z = x + y + z
    addThree :: (Num a) => a -> a -> a -> a
    addThree = \x -> \y -> \z -> x + y + z

    flip :: (a -> b -> c) -> b -> a -> c
    flip f = \x y -> f y x
#endif // 0

    auto const result0 = zipWith | [](auto a, auto b){ return (a * 30. + 3.) / b; } | within_(5, 4, 1) | within(1, 5);
    expectEq(to<std::vector> | result0, std::vector{153.0, 61.5, 31.0, 15.75, 6.6});

    auto const result1 = map | [](auto ab){ auto [a, b] = ab; return a + b; } | std::vector{std::tuple{1, 2}, std::tuple{3, 5}, std::tuple{6, 3}, std::tuple{2, 6}, std::tuple{2, 5}};
    expectEq(to<std::vector> | result1, std::vector{3, 8, 9, 8, 7});
}

void onlyFoldsAndHorses0()
{
#if 0
    // haskell version
    sum :: (Num a) => [a] -> a
    sum xs = foldl (\acc x -> acc + x) 0 xs

    ghci> sum [3,5,2,1]
    11

    sum :: (Num a) => [a] -> a
    sum = foldl (+) 0

    elem :: (Eq a) => a -> [a] -> Bool
    elem y ys = foldl (\acc x -> if x == y then True else acc) False ys

    map :: (a -> b) -> [a] -> [b]
    map f xs = foldr (\x acc -> f x : acc) [] xs
#endif // 0

    constexpr auto sum = foldl | std::plus<>{} | 0;
    expectEq(sum | std::vector{5, 3, 2}, 10);

    constexpr auto elem = toGFunc<2> | [](auto y, auto ys) { return foldl | [y](auto acc, auto x) { return x == y ? true : acc; } | false  | ys; };
    expectEq(elem | 3 | std::vector{5, 3, 2}, true);

    // only works for container types, not ranges.
    constexpr auto map = toGFunc<2> | [](auto f, auto xs) { return foldr | [f](auto x, auto acc) { return f(x) <cons> acc; } | decltype(xs){}  | xs; };
    expectEq(map | std::negate<>{} | std::vector{5, 3, 2}, std::vector{-5, -3, -2});
}

void onlyFoldsAndHorses1()
{
#if 0
    // haskell version
    maximum :: (Ord a) => [a] -> a
    maximum = foldr1 (\x acc -> if x > acc then x else acc)

    reverse :: [a] -> [a]
    reverse = foldl (\acc x -> x : acc) []

    product :: (Num a) => [a] -> a
    product = foldr1 (*)

    filter :: (a -> Bool) -> [a] -> [a]
    filter p = foldr (\x acc -> if p x then x : acc else acc) []

    head :: [a] -> a
    head = foldr1 (\x _ -> x)

    last :: [a] -> a
    last = foldl1 (\_ x -> x)
#endif // 0

    constexpr auto maximum = foldr1 | (toGFunc<2> | [](auto x, auto acc){ return x > acc ? x : acc; });
    expectEq(maximum | std::vector{4, 2, 6}, 6);

    constexpr auto reverse = toGFunc<1> | [](auto xs) { return foldl | (toGFunc<2> | [](auto acc, auto x){ return x <cons> acc; }) | decltype(xs){} | xs; };
    expectEq(reverse | std::vector{4, 2, 6}, std::vector{6, 2, 4});

    constexpr auto product = foldr1 || toGFunc<2> | std::multiplies<>{};
    expectEq(product | std::vector{4, 2, 6}, 48);

    constexpr auto filter = toGFunc<2> | [](auto p, auto xs) { return foldr | (toGFunc<2> | [p](auto x, auto acc) { return p(x) ? x <cons> acc : acc; }) | decltype(xs){} | xs; };
    expectEq(filter | even | std::vector{4, 5, 2}, std::vector{4, 2});

    constexpr auto head = foldr1 || toGFunc<2> | [](auto x, auto){ return x; };
    expectEq(head | std::vector{4, 5, 2}, 4);

    constexpr auto last = foldl1 || toGFunc<2> | [](auto, auto x){ return x; };
    expectEq(last | std::vector{4, 5, 2}, 2);
}

void functionApplicationWithS()
{
#if 0
    // haskell version
    ($) :: (a -> b) -> a -> b
    f $ x = f x

    ghci> map ($ 3) [(4+), (10*), (^2), sqrt]
    [7.0,30.0,9.0,1.7320508075688772]
#endif // 0

    // FIX ME build failure
    // auto const result = map | [](auto f) { return f || 3; } |
    //     std::vector<TEFunction<double, double>>{
    //         toTEFunc<double, double> | [](double x) { return 4 + x; },
    //         toTEFunc<double, double> | [](double x) { return 10 * x; },
    //         toTEFunc<double, double> | [](double x) { return x * x; },
    //         toTEFunc<double, double> | [](double x) { return std::sqrt(x); }
    //     };
    // expectEq(to<std::vector> | result, std::vector{7.0, 30.0, 9.0, 1.7320508075688772});
}

void functionComposition()
{
#if 0
    // haskell version
    (.) :: (b -> c) -> (a -> b) -> a -> c
    f . g = \x -> f (g x)

    ghci> map (\x -> negate (abs x)) [5,-3,-6,7,-3,2,-19,24]
    [-5,-3,-6,-7,-3,-2,-19,-24]

    ghci> map (negate . abs) [5,-3,-6,7,-3,2,-19,24]
    [-5,-3,-6,-7,-3,-2,-19,-24]

    ghci> map (\xs -> negate (sum (tail xs))) [[1..5],[3..6],[1..7]]
    [-14,-15,-27]

    ghci> map (negate . sum . tail) [[1..5],[3..6],[1..7]]
    [-14,-15,-27]

    oddSquareSum :: Integer
    oddSquareSum = sum (takeWhile (<10000) (filter odd (map (^2) [1..])))

    oddSquareSum :: Integer
    oddSquareSum = sum . takeWhile (<10000) . filter odd . map (^2) $ [1..]

    oddSquareSum :: Integer oddSquareSum =
        let oddSquares = filter odd $ map (^2) [1..]
            belowLimit = takeWhile (<10000) oddSquares
        in  sum belowLimit
#endif

    auto const negate = toGFunc<1> | std::negate<>{};
    auto const abs = toGFunc<1> | [](auto x) { return std::abs(x); };
    auto const result0 = map | (negate <o> abs) | std::vector{5, -3, -6, 7, -3, 2, -19, 24};
    auto const expected0 = std::vector{-5, -3, -6, -7, -3, -2, -19, -24};
    expectEq(to<std::vector> | result0, expected0);

    auto const result1 = map | (negate <o> sum <o> tail) | std::vector{within(1, 5), within(3, 6), within(1, 7)};
    auto const expected1 = std::vector{-14, -15, -27};
    expectEq(to<std::vector> | result1, expected1);

    auto const oddSquareSum = sum <o> (takeWhile | [](auto x){ return x < 10000; }) <o> (filter | odd) <o> (map | [](auto x){ return x*x; }) | enumFrom(1);
    expectEq(oddSquareSum, 166650);

    auto const oddSquares = filter | odd || map | [](auto x){return x*x; } | enumFrom(1);
    auto const belowLimit = takeWhile | [](auto x) { return x < 10000; } | oddSquares;
    auto const oddSquareSum0 = sum | belowLimit;
    expectEq(oddSquareSum0, 166650);
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
    lambdas();
    onlyFoldsAndHorses0();
    onlyFoldsAndHorses1();
    functionApplicationWithS();
    functionComposition();
    return 0;
}