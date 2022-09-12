// Learn you a hspp for great good.

#include "hspp.h"
#include "common.h"

using namespace hspp;
using namespace hspp::doN;
using namespace hspp::data;
using namespace std::literals;


// Baby's first functions
void example1()
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

void example2()
{
#if 0
    // haskell version
    [x*2 | x <- [1..10], x*2 >= 12]
#endif // 0
    Id<int> x;
    auto const result = _(
        x * 2,
        x <= (within | 1 | 10),
        if_ | (x*2 >= 12)
    );
    auto const expected = std::vector{12, 14, 16, 18, 20};
    expectEq(toVector(result), expected);
}

void example3()
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
    auto const result = toVector(boomBangs || (within | 7 | 13));
    auto const expected = std::vector{"BOOM!"sv, "BOOM!"sv,"BANG!"sv,"BANG!"sv};
    expectEq(result, expected);
}

void example4()
{
#if 0
    // haskell version
    [ x | x <- [10..20], x /= 13, x /= 15, x /= 19] [10,11,12,14,16,17,18,20]
#endif // 0
    Id<int> x;
    const auto rng = _(
        x,
        x <= (within | 10 | 20),
        if_ || x != 13,
        if_ || x != 15,
        if_ || x != 19
    );
    auto const result = toVector(rng);
    auto const expected = std::vector{10, 11, 12, 14, 16, 17, 18, 20};
    expectEq(result, expected);
}

void example5()
{
#if 0
    // haskell version
    [ x*y | x <- [2,5,10], y <- [8,10,11], x*y > 50]
#endif // 0
    Id<int> x, y;
    const auto rng = _(
        x*y,
        x <= std::vector{2, 5, 10},
        y <= std::vector{8, 10, 11},
        if_ || x*y > 50
    );
    auto const result = toVector(rng);
    auto const expected = std::vector{55, 80, 100, 110};
    expectEq(result, expected);
}

int main()
{
    example1();
    example2();
    example3();
    example4();
    example5();
    return 0;
}