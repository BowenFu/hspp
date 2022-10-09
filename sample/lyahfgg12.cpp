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

int main()
{
    doNotation();
    return 0;
}
