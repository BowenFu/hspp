#include <gtest/gtest.h>
#include "hspp.h"
#include "parser.h"
#include <vector>
#include <list>
#include <cmath>
#include <cctype>

using namespace hspp;
using namespace hspp::data;
using namespace hspp::parser;
using namespace std::literals;

constexpr auto toVector = data::to<std::vector>;

TEST(Range, 1)
{
    auto vv = MapView{ProductView{SingleView{42}},
        toFunc<>([](std::tuple<int> a) {
            return std::get<0>(a);
        })};
    auto v = toVector(vv);
    auto result = {42};
    EXPECT_TRUE(std::equal(v.begin(), v.end(), result.begin()));
}

TEST(SingleView, 1)
{
    auto single = 42;
    auto v = SingleView{single};
    auto r = toVector(v);
    auto result = {42};
    EXPECT_TRUE(std::equal(r.begin(), r.end(), result.begin()));
}

TEST(SingleView, 2)
{
    auto v = SingleView{42};
    auto r = toVector(v);
    auto result = {42};
    EXPECT_TRUE(std::equal(r.begin(), r.end(), result.begin()));
}

TEST(RefView, 1)
{
    auto single = toVector | SingleView{42};
    auto v = toVector | RefView{single};
    EXPECT_TRUE(std::equal(v.begin(), v.end(), single.begin()));
}

TEST(MapView, 1)
{
    auto single = SingleView{42};
    auto v = MapView{single, std::negate<>{}};
    auto const result = toVector(v);
    auto expected = {-42};
    EXPECT_TRUE(std::equal(result.begin(), result.end(), expected.begin()));
}

TEST(JoinView, 1)
{
    std::list<std::list<int>> nested = {{1, 2}, {3, 4}};
    auto v = JoinView{nested};
    auto const result = toVector(v);
    auto const expected = {1, 2, 3, 4};
    EXPECT_TRUE(std::equal(result.begin(), result.end(), expected.begin()));
}

TEST(JoinMapView, 1)
{
    auto single = SingleView{42};
    auto posAndNeg = [](auto e){ return std::vector{e, -e}; };
    auto v = JoinView{MapView{single, posAndNeg}};
    auto const result = toVector(v);
    auto const expected = {42, -42};
    EXPECT_TRUE(std::equal(result.begin(), result.end(), expected.begin()));
}

TEST(JoinMapView, 2)
{
    auto const test = [=](auto filterEven)
    {
        const std::list<int> x = {3, 4, 5, 6};
        auto const z = JoinView{MapView{RefView{x}, filterEven}};
        auto const result = toVector(z);
        auto const u = {4, 6};
        EXPECT_TRUE(std::equal(result.begin(), result.end(), u.begin()));
    };
    auto f = [](int x)-> std::list<int> { return x%2==0 ? std::list<int>{x} : std::list<int>{}; };
    auto const filterEven1 = toTEFunc<>(f);
    test(filterEven1);
    auto const filterEven2 = toFunc<>(f);
    test(filterEven2);
}

TEST(ProductView, 1)
{
    std::list<int> a = {1, 2};
    std::list<int> b = {3, 4};
    auto v = ProductView{RefView{a}, RefView{b}};
    std::vector<std::tuple<int, int>> result = toVector(v);
    std::vector<std::tuple<int, int>> expected = {{1, 3,}, {1, 4}, {2, 3}, {2, 4}};
    EXPECT_TRUE(std::equal(result.begin(), result.end(), expected.begin()));
}

TEST(ProductView, 2)
{
    std::list<int> a = {1, 2};
    std::list<int> b = {3, 4};
    std::list<int> c = {5, 6};
    auto v = ProductView{RefView{a}, RefView{b}, RefView{c}};
    std::vector<std::tuple<int, int, int>> result = toVector(v);
    std::vector<std::tuple<int, int, int>> expected = {{1, 3, 5}, {1, 3, 6}, {1, 4, 5}, {1, 4, 6}, {2, 3, 5}, {2, 3, 6}, {2, 4, 5}, {2, 4, 6}} ;
    EXPECT_TRUE(std::equal(result.begin(), result.end(), expected.begin()));
}

TEST(ZipWithView, 1)
{
    std::list<int> a = {1, 2};
    std::list<int> b = {3, 4, 5};
    auto v = zip | RefView{a} | RefView{b};
    std::vector<std::tuple<int, int>> result = toVector(v);
    std::vector<std::tuple<int, int>> expected = {{1, 3,}, {2, 4}};
    EXPECT_TRUE(std::equal(result.begin(), result.end(), expected.begin()));
}

TEST(zipWith, 1)
{
    std::vector<int> a = {1, 2};
    std::list<int> b = {3, 4, 5};
    auto v = zipWith | toGFunc<2>(std::plus<>{}) | RefView{a} | RefView{b};
    std::vector<int> result = toVector | v;
    std::vector<int> expected = {4, 6};
    EXPECT_EQ(result, expected);
}

TEST(RepeatView, 1)
{
    std::list<int> b = {3, 4, 5};
    auto v = zip(RepeatView{1}, RefView{b});
    std::vector<std::tuple<int, int>> result = toVector(v);
    std::vector<std::tuple<int, int>> expected = {{1, 3,}, {1, 4}, {1, 5}};
    EXPECT_TRUE(std::equal(result.begin(), result.end(), expected.begin()));
}

TEST(CycleView, 1)
{
    std::list<int> b = {3, 4, 5};
    auto v = take | 5U || cycle | RefView{b};
    auto result = toVector(v);
    auto expected = std::vector{3, 4, 5, 3, 4};
    EXPECT_TRUE(std::equal(result.begin(), result.end(), expected.begin()));
}

TEST(CycleView, 2)
{
    std::list<int> b = {};
    auto v = take | 5U || cycle | RefView{b};
    auto result = toVector(v);
    auto expected = std::vector<int>{};
    EXPECT_EQ(result, expected);
}

TEST(TakeView, 1)
{
    auto v = TakeView{RepeatView{"3"}, 4U};
    auto const result = toVector(v);
    auto const expected = {"3", "3", "3", "3"};
    EXPECT_TRUE(std::equal(result.begin(), result.end(), expected.begin()));
}

TEST(IotaView, 1)
{
    auto v = IotaView<int>{3, 4};
    auto const result = toVector(v);
    auto const expected = {3, 4};
    EXPECT_TRUE(std::equal(result.begin(), result.end(), expected.begin()));
}

TEST(TakeView, 2)
{
    auto v = take | 4U | IotaView<int>{3, 4};
    auto const result = toVector(v);
    auto const expected = {3, 4};
    EXPECT_TRUE(std::equal(result.begin(), result.end(), expected.begin()));
}

TEST(DropView, 1)
{
    auto v = drop | 2 | IotaView<int>{3, 6};
    auto const result = toVector(v);
    auto const expected = {5, 6};
    EXPECT_TRUE(std::equal(result.begin(), result.end(), expected.begin()));
}

TEST(splitAt, 1)
{
    auto [p1, p2] = splitAt | 2U | IotaView<int>{3, 7};
    auto const result1 = toVector(p1);
    auto const result2 = toVector(p2);
    auto const expected1 = {3, 4};
    auto const expected2 = {5, 6, 7};

    EXPECT_TRUE(std::equal(result1.begin(), result1.end(), expected1.begin()));
    EXPECT_TRUE(std::equal(result2.begin(), result2.end(), expected2.begin()));
}

TEST(ConcatView, 1)
{
    auto const a = std::vector{1, 2};
    auto const b = std::vector{3, 4};
    auto v = ConcatView{nonOwnedRange(a), nonOwnedRange(b)};
    auto const result = toVector(v);
    auto const expected = {1, 2, 3, 4};
    EXPECT_TRUE(std::equal(result.begin(), result.end(), expected.begin()));
}

TEST(TEFunction, 2)
{
    auto const g = [](auto x, auto y){ return x - y;};
    const TEFunction<int, int, int> f = g;
    auto x = f(1);
    auto y = x(2);
    EXPECT_EQ(y, -1);
}

TEST(TEFunction, 3)
{
    const TEFunction<int, int, int, int> f = [](auto x, auto y, auto z){ return x + y + z;};
    auto x = f(1);
    auto y = x(2);
    auto z = y(3);
    EXPECT_EQ(z, 6);
}

TEST(Function, 2)
{
    auto const f = toFunc<int, int, int>([](auto x, auto y){ return x - y;});
    auto x = f(1);
    auto y = x(2);
    EXPECT_EQ(y, -1);
}

TEST(Function, 3)
{
    auto const f = toFunc<int, int, int, int>([](auto x, auto y, auto z){ return x + y + z;});
    auto x = f(1);
    auto y = x(2);
    auto z = y(3);
    EXPECT_EQ(z, 6);
}

TEST(GenericFunction, 3)
{
    auto const f = toGFunc<3>([](auto x, auto y, auto z){ return x + y + z;});
    auto x = f(1);
    auto y = x(2);
    auto z = y(3);
    EXPECT_EQ(z, 6);
}

TEST(Functor, vector)
{
    auto const test = [=](auto f)
    {
        const std::vector<int> x = {3};
        auto const y = f <fmap> x;
        EXPECT_EQ(y.size(), 1);
        EXPECT_EQ(y.front(), -3);
    };
    const TEFunction<int, int> g = std::negate<>{};
    test(g);
    auto const h = toFunc<>(std::negate<int>{});
    test(h);
}

TEST(Functor, Range)
{
    std::vector<int> const vi{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto const g = toFunc<>(std::negate<int>{});
    auto h = g <fmap> nonOwnedRange(vi);
    std::vector<int> const result = {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10};
    auto hV = toVector(h);
    EXPECT_EQ(result.size(), hV.size());
    EXPECT_TRUE(std::equal(hV.begin(), hV.end(), result.begin()));
}

TEST(Functor, tuple)
{
    auto const tu = std::make_tuple(_o_, std::list{5.6}, "2", true);
    auto const result = show <fmap> tu;
    auto const expected = std::make_tuple(_o_, std::list{5.6}, "2", "true");
    EXPECT_EQ(result, expected);
}

TEST(Functor, Maybe)
{
    auto const test = [=](auto f)
    {
        const Maybe<int> x = just(1);
        auto const y = f <fmap> x;
        EXPECT_EQ(just(-1), y);
    };
    const TEFunction<int, int> g = std::negate<>{};
    test(g);
    auto const h = toFunc<>(std::negate<int>{});
    test(h);
}

TEST(Functor, IO)
{
    auto const test = [=](auto f)
    {
        auto const x = io([]{ return 42;});
        auto const y = f <fmap> x;
        EXPECT_EQ(-42, y.run());
    };
    const TEFunction<int, int> g = std::negate<>{};
    test(g);
    auto const h = toFunc<>(std::negate<int>{});
    test(h);
}

TEST(Functor, GenericFunction)
{
    constexpr auto g = toGFunc<1>([](auto e) { return -e; });
    auto const y = show <fmap> g;
    EXPECT_EQ(y(12), "-12");
}

TEST(Applicative, vector)
{
    auto const test = [=](auto f)
    {
        const std::vector<int> x = {3};
        auto const y = pure(f) <ap> x;
        EXPECT_EQ(y.size(), 1);
        EXPECT_EQ(y.front(), -3);
    };
    const TEFunction<int, int> g = std::negate<>{};
    test(g);
    auto const h = toFunc<>(std::negate<int>{});
    test(h);
}

TEST(Applicative, list)
{
    auto const test = [=](auto f)
    {
        const std::list<int> x = {3, 4};
        const int y = 5;
        auto const z = pure(f) <ap> x <ap> pure(y);
        EXPECT_EQ(z.size(), 2);
        auto const u = {15, 20};
        EXPECT_TRUE(std::equal(z.begin(), z.end(), u.begin()));
    };
    const TEFunction<int, int, int> g = std::multiplies<>{};
    test(g);
    auto const h = toFunc<int, int, int>(std::multiplies<>{});
    test(h);
}

TEST(Applicative, list2)
{
    const std::list<int> x = {3, 4};
    const std::list<bool> y = {true, false};
    auto const z = pure(makeTuple<>) <ap> x <ap> y;
    auto const u = std::list{std::make_tuple(3, true),std::make_tuple(3, false), std::make_tuple(4, true),std::make_tuple(4, false)};
    EXPECT_EQ(z, u);
}

TEST(Applicative, range)
{
    auto const test = [=](auto f)
    {
        const std::list<int> x = {3, 4};
        auto const z = pure(f) <ap> nonOwnedRange(x);
        std::vector<int> zz = toVector(z);
        auto const u = {-3, -4};
        EXPECT_EQ(zz.size(), u.size());
        EXPECT_TRUE(std::equal(zz.begin(), zz.end(), u.begin()));
    };
    const TEFunction<int, int> g = std::negate<>{};
    test(g);
    auto const h = toFunc<int, int>(std::negate<>{});
    test(h);
}

TEST(Applicative, tuple)
{
    auto const x = std::make_tuple(toAll | true, 4);
    auto const result = pure(show) <ap> x;
    auto const expected = std::make_tuple(toAll | true, "4");
    EXPECT_EQ(result, expected);
}

TEST(Applicative, range1)
{
    auto const test = [=](auto f)
    {
        const std::list<int> x = {3, 4};
        const std::list<int> y = {5, 6};
        auto const z0 = pure(f) <ap> nonOwnedRange(x);
        auto result0 = toVector(z0);
        EXPECT_EQ(result0.size(), 2);
        auto const z = z0 <ap> nonOwnedRange(y);
        auto result = toVector(z);
        EXPECT_EQ(result.size(), 4);
        auto const u = {15, 18, 20, 24};
        EXPECT_TRUE(std::equal(result.begin(), result.end(), u.begin()));
    };
    const TEFunction<int, int, int> g = std::multiplies<>{};
    test(g);
    auto const h = toFunc<int, int, int>(std::multiplies<>{});
    test(h);
}

TEST(Applicative, range2)
{
    auto const test = [=](auto f)
    {
        const std::list<size_t> x = {1U, 2U};
        const std::list<int> y = {3, 4};
        auto const z0 = pure(f) <ap> nonOwnedRange(x);
        auto result0 = toVector(z0);
        EXPECT_EQ(result0.size(), 2);
        auto const z = z0 <ap> nonOwnedRange(y);
        auto result = toVector(toVector <fmap> z);
        const std::vector<std::vector<int>> expected = {{3}, {4}, {3, 3}, {4, 4}};
        EXPECT_EQ(result, expected);
    };
    test(replicate);
}

TEST(Applicative, Maybe)
{
    auto const test = [=](auto f)
    {
        const Maybe<int> x = {1};
        auto const y = pure(f) <ap> x;
        EXPECT_EQ(just(-1), y);
    };
    const TEFunction<int, int> g = std::negate<>{};
    test(g);
    auto const h = toFunc<>(std::negate<int>{});
    test(h);
}

TEST(Applicative, IO)
{
    auto const test = [=](auto f)
    {
        auto const x = ioData(42);
        auto const y = pure(f) <ap> x;
        EXPECT_EQ(-42, y.run());
    };
    const TEFunction<int, int> g = std::negate<>{};
    test(g);
    auto const h = toFunc<>(std::negate<int>{});
    test(h);
}

TEST(Applicative, TEFunction)
{
    const TEFunction<int, std::string> u = [](std::string const& str) { return str.size(); };
    const TEFunction<std::string, bool> f = [](bool x) { return x ? "true" : "false"; };
    auto const y = pure(u) <ap> f;
    EXPECT_EQ(y(true), 4);
    EXPECT_EQ(y(false), 5);
}

TEST(Applicative, GenericFunction)
{
    auto const u = toGFunc<1>([](std::string const& str) { return str.size(); });
    auto const f = toGFunc<1>([](auto x) { return x ? "true" : "false"; });
    auto const y = pure(u) <ap> f;
    EXPECT_EQ(y(true), 4);
    EXPECT_EQ(y(false), 5);
}

TEST(Compose, x)
{
    constexpr auto neg = toFunc<>(std::negate<int>{});
    auto const pos = neg <o> neg;
    EXPECT_EQ(pos(1), 1);
}

TEST(elem, x)
{
    auto const v = std::vector{2, 1};
    EXPECT_TRUE(elem | 1 | v);
    EXPECT_FALSE(elem | 3 | v);
}

TEST(Monad, list)
{
    auto const test = [=](auto filterEven)
    {
        const std::list<int> x = {3, 4, 5, 6};
        auto const z = x >>= filterEven;
        EXPECT_EQ(z.size(), 2);
        auto const u = {4, 6};
        EXPECT_TRUE(std::equal(z.begin(), z.end(), u.begin()));
    };
    auto f = [](int x)-> std::list<int> { return x%2==0 ? std::list<int>{x} : std::list<int>{}; };
    const TEFunction<std::list<int>, int> filterEven1 = f;
    test(filterEven1);
    auto const filterEven2 = toFunc<>(f);
    test(filterEven2);
}

TEST(Monad, list2)
{
    auto const test = [=](auto thisAndNeg)
    {
        const std::list<int> x = {3, 4, 5, 6};
        auto const z = x >>= thisAndNeg;
        EXPECT_EQ(z.size(), 8);
        auto const u = {3, -3, 4, -4, 5, -5, 6, -6};
        EXPECT_TRUE(std::equal(z.begin(), z.end(), u.begin()));
    };
    auto const thisAndNeg = [](int x) { return std::list<int>{x, -x}; };
    const TEFunction<std::list<int>, int> thisAndNeg1 = thisAndNeg;
    test(thisAndNeg1);
    auto const thisAndNeg2 = toFunc<>(thisAndNeg);
    test(thisAndNeg2);
}

TEST(Monad, Range)
{
    auto const test = [=](auto thisAndNeg)
    {
        auto const x = std::list<int>{3, 4, 5, 6};
        auto const z = nonOwnedRange(x) >>= thisAndNeg;
        auto const result = toVector(z);
        auto const u = {3, -3, 4, -4, 5, -5, 6, -6};
        EXPECT_TRUE(std::equal(result.begin(), result.end(), u.begin()));
    };
    auto const thisAndNeg = [](int x) { return ownedRange(ConcatView{SingleView{x}, SingleView{-x}}); };
    auto const thisAndNeg2 = toFunc<>(thisAndNeg);
    test(thisAndNeg2);
}

TEST(Monad, tuple)
{
    auto const thisAndRepr = [](auto x) { return std::make_tuple(toAll | true, show | x); };
    auto const x = std::make_tuple(toAll | false, true);
    auto const result = x >>= thisAndRepr;
    auto const expected = std::make_tuple(toAll | false, "true"s);
    EXPECT_EQ(result, expected);
}

TEST(Monad, Maybe)
{
    auto const test = [=](auto filterEven)
    {
        auto const x = return_(3);
        auto const y = return_(4);
        auto const z = x >>= filterEven;
        auto const u = y >>= filterEven;
        EXPECT_EQ(z, nothing<int>);
        EXPECT_EQ(u, just(4));
    };
    auto const filterEven = [](int x)-> Maybe<int> { return x%2==0 ? Maybe<int>{x} : Maybe<int>{}; };
    const TEFunction<Maybe<int>, int> filterEven1 = filterEven;
    auto const filterEven2 = toFunc<>(filterEven);
    test(filterEven1);
    test(filterEven2);
}

TEST(doNotation, Maybe)
{
    auto const test = [=](auto filterEven)
    {
        auto const x = return_(3);
        auto const y = return_(4);

        auto const z = x >>= filterEven;
        auto const u = y >>= filterEven;
        EXPECT_EQ(z, nothing<int>);
        EXPECT_EQ(u, just(4));
    };
    auto const filterEven = [](int x)-> Maybe<int> { return x%2==0 ? Maybe<int>{x} : Maybe<int>{}; };
    const TEFunction<Maybe<int>, int> filterEven1 = filterEven;
    auto const filterEven2 = toFunc<>(filterEven);
    test(filterEven1);
    test(filterEven2);
}

TEST(Monad, Maybe2)
{
    auto const result = Maybe<int>{} >> Maybe<int>{3};
    EXPECT_EQ(result, nothing<int>);
}

TEST(Monad, IO)
{
    testing::internal::CaptureStdout();

    auto const x = return_("3");
    auto const y = return_("4");

    auto const z = x >>= putStrLn;
    auto const u = y >>= putStrLn;

    auto const v = ioData("5") >> ioData("6");

    std::string output0 = testing::internal::GetCapturedStdout();
    EXPECT_EQ(output0, "");

    testing::internal::CaptureStdout();
    EXPECT_EQ(z.run(), _o_);
    std::string output1 = testing::internal::GetCapturedStdout();
    EXPECT_EQ(output1, "3\n");

    testing::internal::CaptureStdout();
    EXPECT_EQ(u.run(), _o_);
    std::string output2 = testing::internal::GetCapturedStdout();
    EXPECT_EQ(output2, "4\n");

    testing::internal::CaptureStdout();
    EXPECT_EQ(v.run(), "6");
    std::string output3 = testing::internal::GetCapturedStdout();
    EXPECT_EQ(output3, "");
}

TEST(Monad, Function)
{
    constexpr auto u = toFunc<>([](std::string const& str) { return str.size(); });
    constexpr auto f = toFunc<>([](size_t i, std::string const& str) { return !str.empty() && i > 0; });
    constexpr auto y = u >>= f;
    EXPECT_EQ(y(""), false);
    EXPECT_EQ(y("true"), true);
}

TEST(Monad, GenericFunction)
{
    constexpr auto f = toGFunc<2>([](std::string const& str, size_t i) { return equalTo | str.size() | i; });
    constexpr auto y = show >>= f;
    EXPECT_EQ(y(1U), true);
    EXPECT_EQ(y(3U), false);
}

TEST(Monad, Function2)
{
    constexpr auto f = toFunc<>([](std::string const& str, size_t i) { return str.size() == i; });
    constexpr auto y = toFunc<std::string, size_t>(show) >>= f;
    EXPECT_EQ(y(1U), true);
    EXPECT_EQ(y(3U), false);
}

TEST(Monad, return_)
{
    constexpr auto f = toFunc<>([](int i, std::string const& str) { return !str.empty() && i > 0; });
    constexpr auto x = Monad<Function, const std::string&>::return_(5);
    EXPECT_EQ(x(""), 5);
    constexpr auto y = x >>= f;
    EXPECT_EQ(y(""), false);
    EXPECT_EQ(y("true"), true);
}

TEST(FunctionOp, pipe)
{
    constexpr auto f = toFunc<>(std::multiplies<int>{});
    constexpr auto r = f | 2 | 5;
    EXPECT_EQ(r, 10);
}

TEST(FunctionOp, pipes)
{
    constexpr auto f = toFunc<>(std::multiplies<int>{});
    constexpr auto g = toFunc<>(std::negate<int>{});
    constexpr auto r = g || f | 2 | 5;
    EXPECT_EQ(r, -10);
}

TEST(FunctionOp, flipPipe)
{
    constexpr auto f = toFunc<>(std::minus<int>{});
    constexpr auto r = 5 & (2 & f);
    EXPECT_EQ(r, -3);
}

TEST(FunctionOp, flipPipe2)
{
    constexpr auto f = toFunc<>(std::negate<int>{});
    constexpr auto g = toFunc<>([](int x) { return x + 1; });
    constexpr auto r = 5 & f & g;
    EXPECT_EQ(r, -4);
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wenum-compare"
TEST(on, 1)
{
    constexpr auto f = toFunc<>(std::equal_to<int>{});
    constexpr auto g = toFunc<>([](int e){ return e > 0; });
    constexpr auto r = f <on> g | 2 | -4;
    EXPECT_EQ(r, false);
    constexpr auto s = f <on> g | 2 | 4;
    EXPECT_EQ(s, true);
}
#pragma GCC diagnostic pop

TEST(IO, myAction)
{
    constexpr auto f = toFunc<>(
        [](std::string l, std::string r)
        {
            return l + r;
        }
    );
    auto const myAction = f <fmap> getLine <ap> getLine;
    (void)myAction;
}

TEST(Function, myFunc)
{
    constexpr auto myFunc = toFunc<>(std::plus<int>{})
                     <fmap> toFunc<>([](int x) { return x + 3;})
                      <ap> toFunc<>([](int x) { return x * 100; });
    EXPECT_EQ(myFunc(5), 508);
}

TEST(Function, myFunc2)
{
    constexpr auto myFunc = toFunc<>([](float x, float y, float z) { return std::vector{x, y, z}; })
                     <fmap> toFunc<>([](float x) { return x + 3;})
                      <ap> toFunc<>([](float x) { return x * 2; })
                      <ap> toFunc<>([](float x) { return x / 2; });
    auto const result = myFunc(5);
    auto const expected = {8.0,10.0,2.5};
    EXPECT_TRUE(std::equal(result.begin(), result.end(), expected.begin()));
}

TEST(Fmap, lift)
{
    constexpr auto f= [](int x) { return x * 2; };
    auto u = fmap | f;

    {
        auto const result = u(Maybe<int>{5});
        auto const expected = Maybe<int>{10};
        EXPECT_EQ(result, expected);
    }

    {
        auto const result = u(std::vector{5, 10});
        auto const expected = {10, 20};
        EXPECT_TRUE(std::equal(result.begin(), result.end(), expected.begin()));
    }
}

TEST(filter, 1)
{
    auto const x = filter | [](auto n) { return n%2 == 0; };
    auto y = x | std::vector{1,2,3,4};
    auto const z = toVector(y);
    auto result = {2, 4};
    EXPECT_TRUE(std::equal(z.begin(), z.end(), result.begin()));
}

TEST(map, 1)
{
    auto const x = map | [](auto n) { return n%2 == 0; };
    auto y = x | std::vector{1,2,3,4};
    auto const z = toVector(y);
    auto result = {false, true, false, true};
    EXPECT_TRUE(std::equal(z.begin(), z.end(), result.begin()));
}

TEST(replicate, 1)
{
    auto const str = "3";
    auto const x = replicate | 4U | str;
    auto const z = toVector(x);
    for (auto i : z)
    {
        std::cout << i << std::endl;
    }
    auto const result = {"3", "3", "3", "3"};
    EXPECT_TRUE(std::equal(z.begin(), z.end(), result.begin(), [](auto x, auto y) { return std::string_view{x} == std::string_view{y};}));
}

TEST(id, 1)
{
    auto const x = id | 4U;
    EXPECT_EQ(x, 4U);
}

TEST(unCurry, 1)
{
    constexpr auto f = toFunc<>([](std::vector<std::string> acc, std::string l)
    {
        acc.push_back(l);
        return acc;
    });
    auto result = unCurry(f)(std::vector{std::string{"123"}}, "456");
    EXPECT_EQ(result[0], "123");
    EXPECT_EQ(result[1], "456");
}

TEST(foldl, 1)
{
    constexpr auto f = toFunc<>([](std::string acc, std::string l)
    {
        return acc + l;
    });
    auto const x = foldl | f | std::string{};
    auto const l = std::vector{"1", "2", "3"};
    auto const result = x | l;
    EXPECT_EQ(result, "123");
}

TEST(foldl, 2)
{
    constexpr auto f = toFunc<>([](std::vector<std::string> acc, std::string l)
    {
        acc.push_back(l);
        return acc;
    });
    auto const x = foldl | f | std::vector<std::string>{};
    auto const l = std::vector<std::string>{"1", "2", "3"};
    auto const result = x | l;
    EXPECT_TRUE(std::equal(result.begin(), result.end(), l.begin()));
}

TEST(foldr, 1)
{
    constexpr auto f = toFunc<>([](std::string l, std::string acc)
    {
        return l + acc;
    });
    auto const x = hspp::foldr | f | std::string{};
    auto const l = std::vector{"1", "2", "3"};
    auto const result = x | l;
    EXPECT_EQ(result, "123");
}

TEST(fold, 1)
{
    constexpr auto pow = toGFunc<2>([](auto x, auto y) { return std::pow(x, y); });
    auto const r1 = foldr | pow | 2.f | ownedRange(IotaView{1.f, 4.f});
    EXPECT_EQ(r1, 1);
    auto const r2 = foldl | pow | 2.f | ownedRange(IotaView{1.f, 4.f});
    EXPECT_EQ(r2, 64);
}

TEST(fold, 2)
{
    // cons results are of different types, so cannot reassign to init.
    // Use vector instead.
    auto const r1 = foldr | cons | std::vector<float>{} || toVector | IotaView{1.f, 4.f};
    auto const e1 = std::vector{1.f, 2.f, 3.f};
    EXPECT_EQ(r1, e1);

    auto const r2 = foldl | (flip | cons) | std::vector<float>{} || toVector | IotaView{1.f, 4.f};
    auto const e2 = std::vector{3.f, 2.f, 1.f};
    EXPECT_EQ(r2, e2);
}

TEST(cons, range)
{
    // foldr does not support Range, thus converting to Vector.
    auto const r1 = toVector || cons | 2.f | ownedRange(IotaView{1.f, 4.f});
    auto const e1 = std::vector{2.f, 1.f, 2.f, 3.f};
    EXPECT_EQ(r1, e1);
}

TEST(flip, x)
{
    constexpr auto gt = flip | toGFunc<2>(std::less<>{});
    EXPECT_TRUE(gt | 3 | 2);
}

TEST(equalTo, x)
{
    EXPECT_TRUE(equalTo | 2 | 2);
    EXPECT_FALSE(equalTo | 3 | 2);
}

TEST(equalTo, 2)
{
    constexpr auto f = length <o> (filter || equalTo | 'a');

    EXPECT_EQ(f | "abracadabra"sv, 5);
}

TEST(print, 1)
{
    auto const io = print | 3;
    testing::internal::CaptureStdout();
    io.run();
    std::string output1 = testing::internal::GetCapturedStdout();
    EXPECT_EQ(output1, "3\n");
}

TEST(read, 1)
{
    auto const i = hspp::read<int> | "3";
    EXPECT_EQ(i, 3);
}

TEST(Monoid, vector)
{
    auto const l = std::vector{1, 2, 3};
    auto const r = std::vector{4, 5, 6};
    auto const result = l <mappend> r;
    auto const expected = {1, 2, 3, 4, 5, 6};
    EXPECT_TRUE(std::equal(result.begin(), result.end(), expected.begin()));
}

TEST(Monoid, string)
{
    auto const l = std::string{"123"};
    auto const r = std::string{"456"};
    auto const result = l <mappend> r;
    auto const expected = "123456";
    EXPECT_EQ(result, expected);
}

TEST(Monoid, list)
{
    std::list<std::list<int>> nested = {{1, 2}, {3, 4}};
    auto v = mconcat | nested;
    auto const result = toVector(v);
    auto const expected = {1, 2, 3, 4};
    EXPECT_TRUE(std::equal(result.begin(), result.end(), expected.begin()));
}

TEST(Monoid, ZipList)
{
    auto const nested = std::list{toZipList || toSum <fmap> std::list{1, 2}, toZipList || toSum <fmap> std::list{3, 4}};
    auto v = mconcat | nested;
    auto const result = toVector(v);
    auto const expected = toSum <fmap> std::vector{4, 6};
    EXPECT_EQ(result, expected);
}

TEST(Monoid, range)
{
    auto const l = std::vector{1, 2, 3};
    auto const r = std::vector{4, 5, 6};
    auto const result_ = nonOwnedRange(l) <mappend> nonOwnedRange(r);
    auto const result = toVector(result_);
    auto const expected = {1, 2, 3, 4, 5, 6};
    EXPECT_TRUE(std::equal(result.begin(), result.end(), expected.begin()));
}

TEST(Monoid, range2)
{
    auto const a = std::vector{1, 2};
    auto const b = std::vector{3, 4};
    auto nested = std::tuple{nonOwnedRange(a), nonOwnedRange(b)};
    auto v = mconcat | nested;
    auto const result = toVector(v);
    auto const expected = {1, 2, 3, 4};
    EXPECT_TRUE(std::equal(result.begin(), result.end(), expected.begin()));
}

TEST(Monoid, range3)
{
    auto const a = std::vector{1, 2};
    auto const b = std::vector{3, 4};
    auto nested = std::vector{nonOwnedRange(a), nonOwnedRange(b)};
    auto v = mconcat | nested;
    auto const result = toVector(v);
    auto const expected = {1, 2, 3, 4};
    EXPECT_TRUE(std::equal(result.begin(), result.end(), expected.begin()));
}

TEST(Monoid, maybe)
{
    auto const nested = std::list{just(toProduct(2)), nothing<Product<int>>, just(toProduct(3))};
    auto const result = mconcat | nested;
    auto const expected = just(toProduct(6));
    EXPECT_EQ(result, expected);
}

TEST(Monoid, toProduct)
{
    auto const a = toProduct | 1.5;
    auto const b = toProduct | 2.0;
    auto nested = std::vector{a, b};
    auto v = mconcat | nested;
    auto const result = v;
    auto const expected = 3.0;
    EXPECT_EQ(expected, result.get());

    auto const result2 = getProduct || (toProduct | 3) <mappend> (toProduct | 4) <mappend> (toProduct | 2);
    EXPECT_EQ(result2, 24);

    auto const result3 = getProduct <o> mconcat || fmap | toProduct | std::vector{3, 4, 2};
    EXPECT_EQ(result3, 24);
}

TEST(Monoid, product2)
{
    auto const result = getProduct <o> mconcat <o> (map | toProduct) || std::vector{3, 4, 2};
    EXPECT_EQ(result, 24);
}

TEST(Monoid, toSum)
{
    auto const result = getSum <o> mconcat <o> (map | toSum) || std::vector{3, 4, 2};
    EXPECT_EQ(result, 9);
}

TEST(Monoid, toAll)
{
    auto const result = getAll <o> mconcat <o> (map | toAll) || std::list{true, false, true};
    EXPECT_EQ(result, false);
    auto const result2 = all | id | std::list{true, false, true};
    EXPECT_EQ(result2, false);
}

TEST(Monoid, any)
{
    auto const result = getAny <o> mconcat <o> (map | toAny) || std::list{true, false, true};
    EXPECT_EQ(result, true);
    auto const result2 = any | id | std::list{true, false, true};
    EXPECT_EQ(result2, true);
}

TEST(Monoid, first)
{
    auto const result = getFirst <o> mconcat <o> (map | toFirst) || std::list<Maybe<int>>{nothing<int>, just(2), just(3)};
    EXPECT_EQ(result, just(2));
}

TEST(Monoid, toLast)
{
    auto const result = getLast <o> mconcat <o> (map | toLast) || std::list<Maybe<int>>{nothing<int>, just(2), just(3)};
    EXPECT_EQ(result, just(3));
}

TEST(Monoid, Ordering)
{
    constexpr auto length = toFunc<>([](std::string const& x)
    {
        return x.length();
    });
    auto const lengthCompare = toFunc<>([=](std::string const& x, std::string const& y)
    {
        return ((length | x) <compare> (length | y)) <mappend> (x <compare> y);
    });
    auto const result1 = lengthCompare | "zen" | "ants";
    EXPECT_EQ(result1, Ordering::kLT);
    auto const result2 = lengthCompare | "zen" | "ant";
    EXPECT_EQ(result2, Ordering::kGT);
}

TEST(Monoid, GenericFunction)
{
    auto const f = toGFunc<1>([](auto u){ return toProduct | (u+1); });
    auto const g = toGFunc<1>([](auto u){ return toProduct | (u*2); });
    auto const result = mappend | f | g;
    EXPECT_EQ(result(24).get(), 1200);
}

TEST(Monoid, Function)
{
    auto const f = toFunc<>([](int u){ return toProduct | (u+1); });
    auto const g = toFunc<>([](int u){ return toProduct | (u*2); });
    auto const result = mappend | f | g;
    EXPECT_EQ(result(24).get(), 1200);
}

TEST(Monoid, Tuple)
{
    auto const x = std::make_tuple(toSum | 1, std::string{"123"});
    auto const y = std::make_tuple(toSum | 0, std::string{"23"});
    auto const z = x <mappend> y;
    EXPECT_EQ(z, std::make_tuple(toSum | 1, std::string("12323")));

    auto const result = mconcat | std::list{x, y};
    EXPECT_EQ(result, std::make_tuple(toSum | 1, std::string("12323")));
}

TEST(Monoid, toEndo)
{
    auto const result = appEndo || (toEndo | id) <mappend> (toEndo | show);
    EXPECT_EQ(result | 2, "2");
}

TEST(Foldable, list)
{
    std::list<std::list<int>> nested = {{1, 2}, {3, 4}};
    auto v = fold | nested;
    auto const result = toVector(v);
    auto const expected = std::vector{1, 2, 3, 4};
    EXPECT_EQ(result, expected);
}

TEST(Foldable, list2)
{
    auto const nested = std::vector{3, 4};
    auto const result = foldMap | toSum | nested;
    auto const expected = toSum | 7;
    EXPECT_EQ(result, expected);
}

TEST(Foldable, list3)
{
    std::list<std::tuple<Sum<int>, All>> nested = {{1, true}, {3, false}};
    auto result = fold | nested;
    std::tuple<Sum<int>, All> const expected = {4, false};
    EXPECT_EQ(result, expected);
}

TEST(Foldable, maybe)
{
    auto const nested = just | 3;
    auto const result = foldMap | toSum | nested;
    auto const expected = toSum | 3;
    EXPECT_EQ(result, expected);
}

TEST(Foldable, tuple)
{
    auto const t = std::tuple{3, false};
    auto const result = foldMap | show | t;
    auto const result2 = foldMap | id | t;
    auto const result3 = fold | t;
    auto const expected = "false";
    EXPECT_EQ(result, expected);
    EXPECT_EQ(result2, false);
    EXPECT_EQ(result3, false);
}

TEST(Traversable, tuple)
{
    auto const t = std::tuple{3, just | false};
    auto const result = sequenceA | t;
    auto const expected = just | std::tuple{3, false};
    EXPECT_EQ(result, expected);
}

TEST(Traversable, maybe)
{
    auto const result = sequenceA | just(std::vector{4});
    auto const expected = std::vector{just | 4};
    EXPECT_EQ(result, expected);
}

TEST(Traversable, vector)
{
    auto const result = sequenceA || fmap | just | std::vector{1, 2, 3};
    auto const expected = just | std::vector{1, 2, 3};
    EXPECT_EQ(result, expected);
    auto const result2 = sequenceA <o> (fmap | just) | std::vector{1, 2, 3};
    EXPECT_EQ(result2, expected);
}

TEST(Maybe, 1)
{
    auto result = just("andy"s) <mappend> nothing<std::string>;
    EXPECT_EQ(result, just("andy"s));
    result = just("andy"s) <mappend> just("123"s);
    EXPECT_EQ(result, just("andy123"s));
    result = nothing<std::string> <mappend> nothing<std::string>;
    EXPECT_EQ(result, nothing<std::string>);

    result = mconcat | std::vector{just("123"s), just("xxx"s)};
    EXPECT_EQ(result, just("123xxx"s));
}

TEST(Maybe, fromOptional)
{
    auto fromRValue = Maybe<std::string>::fromOptional(std::make_optional("andy"s));
    EXPECT_EQ(fromRValue, just("andy"s));

    auto value = std::make_optional("andy"s);
    auto fromLValue = Maybe<std::string>::fromOptional(value);
    EXPECT_EQ(fromLValue, just("andy"s));
}

TEST(Maybe, toOptional)
{
    auto value = just | "andy"s;
    std::optional<std::string> fromLvalue = value;
    EXPECT_EQ(fromLvalue, std::make_optional("andy"s));

    std::optional<std::string> fromRvalue = just | "andy"s;
    EXPECT_EQ(fromRvalue, std::make_optional("andy"s));
}

TEST(Monad, vec)
{
    auto const result = std::vector{1,2}
                    >>= toFunc<>([](int n)
                    {
                        return std::vector{'a','b'}
                            >>= [=](char ch)
                            {
                                return return_(std::make_pair(n,ch));
                            };
                    });
    auto const expected = std::vector<std::pair<int, char>>{
        std::make_pair(1,'a'),
        std::make_pair(1,'b'),
        std::make_pair(2,'a'),
        std::make_pair(2,'b')};
    EXPECT_EQ(result, expected);
}

TEST(Applicative, vec)
{
    auto const result = toFunc<>([=](int n, char ch)
    {
        return std::make_pair(n,ch);
    })
    <fmap> std::vector{1,2}
    <ap> std::vector{'a','b'};

    auto const expected = std::vector<std::pair<int, char>>{
        std::make_pair(1,'a'),
        std::make_pair(1,'b'),
        std::make_pair(2,'a'),
        std::make_pair(2,'b')};
    EXPECT_EQ(result, expected);
}

TEST(Copy, View)
{
    int const x = true;
    auto const alwaysTrue = [x](auto) { return x;};
    auto const id = [](auto x) { return SingleView{x};};
    auto const v= JoinView{MapView{FilterView{alwaysTrue, IotaView{1}}, id}};
    auto const y = v;
    (void)y;
}

TEST(MonadPlus, guard)
{
    auto const result = guard(5 > 2) >> just(1);
    EXPECT_EQ(result, just(1));

    auto const result1 = guard(1 > 2) >> just(1);
    EXPECT_EQ(result1, nothing<int>);

    auto const result3 = guard(5 > 2) >> std::vector{2};
    EXPECT_EQ(result3, std::vector{2});

    auto const result4 = guard(1 > 2) >> std::vector{2};
    EXPECT_TRUE(result4.empty());

    auto const result5 = guard(1 > 2) >> ownedRange(SingleView{4});
    EXPECT_TRUE(toVector(result5).empty());

    auto const result6 = (guard | ('7' <elem> (show | 567 ))) >> ownedRange(SingleView{4});
    EXPECT_EQ(toVector(result6).at(0), 4);

    auto const func = toGFunc<1>([](auto x)
    {
        return ownedRange(SingleView{4})
            >> (guard || '7' <elem> (show | x))
            >> (return_ | x);
    });
    auto const l = std::vector{7, 9, 17, 22};
    auto const result7 = func <fmap> nonOwnedRange(l);
    auto const expected7 = std::vector{7, 17};
    EXPECT_EQ(toVector(JoinView{result7}), expected7);

    auto const tmp = ownedRange(IotaView{1, 50}) >>= func;
    auto const result8 = toVector || tmp;
    auto const expected = std::vector{7,17,27,37,47};
    EXPECT_EQ(result8, expected);
}

TEST(const_, x)
{
    EXPECT_EQ(const_ | 1 | 2, 1);
    auto const v = std::vector{1, 2};
    EXPECT_EQ(foldl | const_ | 0 | v, 0);
    EXPECT_EQ(foldr | const_ | 0 | v, 1);
}


TEST(Parser, item)
{
    constexpr auto p =
        item >>= toFunc<> | [](char c) { return
            item >>
            item >>= toFunc<> | [c](char d) {
                return return_ | std::make_tuple(c, d);
            };
        };

    auto const result = runParser | p | "123";
    auto const expected = std::make_tuple('1', '3');
    EXPECT_EQ(std::get<0>(result.front()), expected);
}

TEST(Parser, string)
{
    auto const result = runParser | (string | ""s) | "123";
    auto const expected = ""s;
    EXPECT_EQ(std::get<0>(result.at(0)), expected);

    auto const result2 = runParser | (string | "13") | "123";
    EXPECT_TRUE(result2.empty());

    auto const result3 = runParser | (string | "12") | "123";
    auto const expected3 = "12"s;
    EXPECT_EQ(std::get<0>(result3.at(0)), expected3);

    auto const result4 = runParser | (string | "123") | "12";
    EXPECT_TRUE(result4.empty());
}

TEST(Parser, many)
{
    auto const result = runParser || many | (string | "12"s) || "12123";
    auto const expected = std::vector{"12"s, "12"s};
    EXPECT_EQ(std::get<0>(result.at(0)), expected);
}

TEST(Parser, seqBy)
{
    auto const result = runParser | (sepBy | (string | "1"s) | (char_ | '2')) || "12123";
    auto const expected = std::vector{"1"s, "1"s};
    EXPECT_EQ(std::get<0>(result.at(0)), expected);
}

TEST(Parser, space)
{
    auto const result = runParser || space || "  12123";
    auto const expected = std::vector{' ', ' '};
    EXPECT_EQ(std::get<0>(result.at(0)), expected);
}

TEST(Parser, token)
{
    auto const result = runParser || token | (string | "12") || "12 123";
    auto const expected = "12"s;
    EXPECT_EQ(std::get<0>(result.at(0)), expected);
}

TEST(Parser, symb)
{
    auto const result = runParser || symb | "12" || "12 123";
    auto const expected = "12"s;
    EXPECT_EQ(std::get<0>(result.at(0)), expected);
}

TEST(Parser, apply)
{
    auto const result = apply || symb | "12" || " 12 123";
    auto const expected = "12"s;
    EXPECT_EQ(std::get<0>(result.at(0)), expected);
}

TEST(do_, x)
{
    auto const result = doN::do_(
        just | 1,
        return_ | 2
    );
    auto const expected = just(2);
    EXPECT_EQ(result, expected);
}

TEST(do_, y)
{
    doN::Id<int> i;
    auto const result = doN::do_(
        i <= (just | 1),
        return_ | i
    );
    auto const expected = just(1);
    EXPECT_EQ(result, expected);
}

TEST(do_, z)
{
    doN::Id<int> i;
    doN::Id<int> j;
    auto const result = doN::do_(
        j <= (just | 2),
        just(4),
        i <= (just | 1),
        return_ | j
    );
    auto const expected = just(2);
    EXPECT_EQ(result, expected);
}

TEST(do_, range)
{
    doN::Id<int> i;
    auto const result = toVector | doN::do_(
        i <= ownedRange(SingleView{1}),
        return_ | i
    );
    auto const expected = std::vector{1};
    EXPECT_EQ(result, expected);
}

TEST(do_, vector)
{
    doN::Id<int> i;
    auto const result = doN::do_(
        i <= std::vector{1, 2, 3, 4},
        return_ | i
    );
    auto const expected = std::vector{1, 2, 3, 4};
    EXPECT_EQ(result, expected);
}

TEST(do_, list)
{
    doN::Id<int> i;
    auto const result = doN::do_(
        i <= std::list{1, 2, 3, 4},
        return_ | i
    );
    auto const expected = std::list{1, 2, 3, 4};
    EXPECT_EQ(result, expected);
}

TEST(do_, vector0)
{
    using namespace hspp::doN;
    Id<int> i;
    auto const result = do_(
        i <= std::vector{1, 2, 3, 4},
        guard | (i % 2 == 0),
        return_ | i
    );
    auto const expected = std::vector{2, 4};
    EXPECT_EQ(result, expected);
}

TEST(do_, vector1)
{
    using namespace hspp::doN;
    Id<int> i;
    auto const result = do_(
        i <= std::vector{1, 2, 3, 4},
        guard | (i % 2 == 0),
        return_ | (i * 3)
    );
    auto const expected = std::vector{6, 12};
    EXPECT_EQ(result, expected);
}

TEST(do_, vector2)
{
    using namespace hspp::doN;
    Id<int> i;
    Id<int> j;
    auto const result = do_(
        i <= std::vector{1, 2},
        j <= std::vector{3, 4},
        guard | (i + j == 5),
        return_ | (i * j)
    );
    auto const expected = std::vector{4, 6};
    EXPECT_EQ(result, expected);
}

TEST(do_, comprehension)
{
    using namespace hspp::doN;
    Id<int> i;
    Id<int> j;
    auto const result =
        _(i * j, i <= std::vector{1, 2}, j <= std::vector{3, 4}, if_ | (i + j == 5));
    auto const expected = std::vector{4, 6};
    EXPECT_EQ(result, expected);
}

TEST(do_, comprehension2)
{
    doN::Id<int> i;
    doN::Id<int> j;
    auto const result = toVector | doN::_(i * j, i <= ownedRange(IotaView{1, 10}), j <= ownedRange(IotaView{1, 10}), guard | (i + j == 5));
    auto const expected = std::vector{4, 6, 6, 4};
    EXPECT_EQ(result, expected);
}

TEST(do_, comprehension3)
{
    using namespace hspp::doN;
    Id<int> i, j, k;
    auto const rng = _(
        makeTuple<3> | i | j | k,
        k <= (enumFrom | 1),
        i <= (iota | 1 | k),
        j <= (iota | i | k),
        if_ || (i*i + j*j == k*k)
    );
    auto const result = toVector || take | 5U | rng;
    auto const expected = std::vector<std::tuple<int, int, int>>{ { 3, 4, 5 }, { 6, 8, 10 }, { 5, 12, 13 }, { 9, 12, 15 }, { 8, 15, 17 } };
    EXPECT_EQ(result, expected);
}

TEST(MapM_, IO)
{
    testing::internal::CaptureStdout();

    auto const lst = std::vector{"3"s, "4"s};
    auto const func = putStrLn;
    auto const mapM_result = mapM_ | func | lst;

    std::string output0 = testing::internal::GetCapturedStdout();
    EXPECT_EQ(output0, "");

    testing::internal::CaptureStdout();
    EXPECT_EQ(mapM_result.run(), _o_);
    std::string output1 = testing::internal::GetCapturedStdout();
    EXPECT_EQ(output1, "3\n4\n");
}

TEST(MapM, IO)
{
    auto const lst = std::vector{"3"s, "4"s};
    auto const func = hspp::Monad<IO>::return_ <o> hspp::read<int>;
    auto const mapMResult = mapM | func | lst;

    auto const expected = std::vector{3, 4};
    EXPECT_EQ(mapMResult.run(), expected);
}

TEST(catch_, 1)
{
    auto io_ = io([]{
        throw std::runtime_error{"Some error"};
        return 1;
    });
    auto handler = [](std::runtime_error const& re)
    {
        return io([=]{
            std::cout << re.what() << std::endl;
            return 2;
        });
    };
    auto newIo = io_ <catch_> handler;
    testing::internal::CaptureStdout();
    EXPECT_EQ(newIo.run(), 2);
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_EQ(output, "Some error\n");

    auto newIo2 = handler <handle> io_ ;
    testing::internal::CaptureStdout();
    EXPECT_EQ(newIo2.run(), 2);
    std::string output2 = testing::internal::GetCapturedStdout();
    EXPECT_EQ(output2, "Some error\n");
}

TEST(even, 1)
{
    EXPECT_FALSE(even(3));
    EXPECT_TRUE(even(4L));
}

TEST(doNotation, func)
{
    auto plus1 = toFunc<> | [](int x){ return 1+x; };
    auto showStr = toFunc<> | [](int x){ return show | x; };

    using namespace hspp::doN;
    Id<int> x;
    Id<std::string> y;
    auto go = do_(
        x <= plus1,
        y <= showStr,
        return_ || makeTuple<2> | x | y
    );
    auto result = go | 3;
    EXPECT_EQ(result, std::make_tuple(4, "3"));
}

TEST(liftM, lst)
{
    auto negate = liftM | std::negate<>{};
    auto result = negate | std::vector{1, 2, 3};
    auto expected = std::vector{-1, -2, -3};
    EXPECT_EQ(result, expected);
}

