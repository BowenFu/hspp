#include "hspp.h"
#include <cassert>
#include <memory>
#include <variant>

using namespace hspp;
using namespace hspp::doN;
using namespace hspp::data;
using namespace std::literals;

auto expectTrue(bool x)
{
    if (!x)
    {
        throw std::runtime_error{"False in expectedTrue!"};
    }
}

template <typename T>
auto expectEq(T const& l, T const& r)
{
    if (l != r)
    {
        std::stringstream ss;
        ss << l << " != " << r;
        throw std::runtime_error{ss.str()};
    }
}

// Monadic operations for std::optional p0798r0

void sample1()
{
    // std::optional<std::string> opt_string = "123"s;
    // std::optional<std::size_t> s = opt_string.map(&std::string::size);

    Maybe<std::string> optString = "123"s;

    Maybe<std::size_t> s0 = &std::string::size <fmap> optString;
    expectTrue(s0.hasValue());
    expectEq(s0.value(), 3UL);

    auto f = toFunc<> | &std::string::size;

    Id<std::string> idS;
    Maybe<std::size_t> s1 = do_(
        idS <= optString,
        return_ || f | idS
    );
    expectTrue(s1.hasValue());
    expectEq(s1.value(), 3UL);
}

int main()
{
    return 0;
}