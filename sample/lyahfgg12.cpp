// Learn you a hspp for great good.
// Chapter 12 A fistful of Monads
// The samples are originated from Learn you a Haskell for great good.

#include "hspp.h"
#include "common.h"
#include <string>
#include <cmath>
#include <cctype>

using namespace hspp;
using namespace hspp::doN;
using namespace hspp::data;
using namespace std::literals;

void doNotation()
{
#if 0
    // haskell version
    foo :: Maybe String
    foo = Just 3 >>= (\x ->
          Just "!" >>= (\y ->
          Just (show x ++ y)))

    foo :: Maybe String
    foo = do
        x <- Just 3
        y <- Just "!"
        Just (show x ++ y)
#endif

    Id<int> x;
    Id<std::string> y;
    auto const foo = do_(
        x <= (just | 3),
        y <= (just | "!"s),
        just || (show | x) <plus> y
    );
    expectEq(foo, just | "3!"s);
}

void doNotation1()
{
#if 0
    listOfTuples :: [(Int,Char)]
    listOfTuples = do
        n <- [1,2]
        ch <- ['a','b']
        return (n,ch)

    ghci> [ (n,ch) | n <- [1,2], ch <- ['a','b'] ]
    [(1,'a'),(1,'b'),(2,'a'),(2,'b')]
#endif

    Id<int> n;
    Id<char> ch;
    auto const listOfTuples0 = do_(
        n <= std::vector{1, 2},
        ch <= std::vector{'a', 'b'},
        return_ | (makeTuple<2> | n | ch)
    );

    auto const expected = std::vector<std::tuple<int, char>>{{1, 'a'}, {1, 'b'}, {2, 'a'}, {2, 'b'}};
    expectEq(to<std::vector>(listOfTuples0), expected);

    auto const listOfTuples1 = _(makeTuple<2> | n | ch, n <= std::vector{1, 2}, ch <= std::vector{'a', 'b'});
    expectEq(to<std::vector>(listOfTuples1), expected);
}

int main()
{
    doNotation();
    doNotation1();
    return 0;
}
