#include "hspp.h"
#include <vector>
#include <list>
#include <gtest/gtest.h>

TEST(Range, 1)
{
    auto vv = MapView{ProductView{SingleView{42}},
        function([](std::tuple<int> a) {
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
    auto single = SingleView{42};
    auto v = RefView{single};
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
        const auto z = JoinView{MapView{RefView{x}, filterEven}};
        auto const result = toVector(z);
        auto const u = {4, 6};
        EXPECT_TRUE(std::equal(result.begin(), result.end(), u.begin()));
    };
    auto f = [](int x)-> std::list<int> { return x%2==0 ? std::list<int>{x} : std::list<int>{}; };
    const TEFunction<std::list<int>, int> filterEven1 = f;
    test(filterEven1);
    const auto filterEven2 = function(f);
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

TEST(ZipView, 1)
{
    std::list<int> a = {1, 2};
    std::list<int> b = {3, 4, 5};
    auto v = ZipView{RefView{a}, RefView{b}};
    std::vector<std::tuple<int, int>> result = toVector(v);
    std::vector<std::tuple<int, int>> expected = {{1, 3,}, {2, 4}};
    EXPECT_TRUE(std::equal(result.begin(), result.end(), expected.begin()));
}

TEST(RepeatView, 1)
{
    std::list<int> b = {3, 4, 5};
    auto v = ZipView{RepeatView{1}, RefView{b}};
    std::vector<std::tuple<int, int>> result = toVector(v);
    std::vector<std::tuple<int, int>> expected = {{1, 3,}, {1, 4}, {1, 5}};
    EXPECT_TRUE(std::equal(result.begin(), result.end(), expected.begin()));
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
    auto v = TakeView{IotaView<int>{3}, 4U};
    auto const result = toVector(v);
    auto const expected = {3, 4, 5, 6};
    EXPECT_TRUE(std::equal(result.begin(), result.end(), expected.begin()));
}

TEST(ChainView, 1)
{
    auto const a = std::vector{1, 2};
    auto const b = std::vector{3, 4};
    auto v = ChainView{nonOwnedRange(a), nonOwnedRange(b)};
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
    const auto f = function<int, int, int>([](auto x, auto y){ return x - y;});
    auto x = f(1);
    auto y = x(2);
    EXPECT_EQ(y, -1);
}

TEST(Function, 3)
{
    const auto f = function<int, int, int, int>([](auto x, auto y, auto z){ return x + y + z;});
    auto x = f(1);
    auto y = x(2);
    auto z = y(3);
    EXPECT_EQ(z, 6);
}

TEST(GenericFunction, 3)
{
    const auto f = genericFunction<3>([](auto x, auto y, auto z){ return x + y + z;});
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
        const auto y = f <fmap> x;
        EXPECT_EQ(y.size(), 1);
        EXPECT_EQ(y.front(), -3);
    };
    const TEFunction<int, int> g = std::negate<>{};
    test(g);
    const auto h = function(std::negate<int>{});
    test(h);
}

TEST(Functor, Range)
{
    std::vector<int> const vi{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    const auto g = function(std::negate<int>{});
    auto h = g <fmap> nonOwnedRange(vi);
    std::vector<int> const result = {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10};
    auto hV = toVector(h);
    EXPECT_EQ(result.size(), hV.size());
    EXPECT_TRUE(std::equal(hV.begin(), hV.end(), result.begin()));
}

TEST(Functor, tuple)
{
    auto const tu = std::make_tuple(1, 5.6, "2", true);
    auto const result = show <fmap> tu;
    auto const expected = std::make_tuple("1", "5.6", "2", "true");
    EXPECT_EQ(result, expected);
}

TEST(Functor, Maybe)
{
    auto const test = [=](auto f)
    {
        const Maybe<int> x = Just{1};
        const auto y = f <fmap> x;
        EXPECT_EQ(Just{-1}, y);
    };
    const TEFunction<int, int> g = std::negate<>{};
    test(g);
    const auto h = function(std::negate<int>{});
    test(h);
}

TEST(Functor, IO)
{
    auto const test = [=](auto f)
    {
        const auto x = io([]{ return 42;});
        const auto y = f <fmap> x;
        EXPECT_EQ(-42, y.run());
    };
    const TEFunction<int, int> g = std::negate<>{};
    test(g);
    const auto h = function(std::negate<int>{});
    test(h);
}

TEST(Functor, TEFunction)
{
    const TEFunction<Maybe<double>, double> f = Applicative<Maybe>::pure<double>;
    const TEFunction<double, int> g = [](auto e) { return -e; };
    const auto y = f <fmap> g;
    EXPECT_EQ(y(1), Just{-1.});
}

TEST(Functor, GenericFunction)
{
    constexpr auto g = genericFunction<1>([](auto e) { return -e; });
    const auto y = show <fmap> g;
    EXPECT_EQ(y(12), "-12");
}

TEST(Applicative, vector)
{
    auto const test = [=](auto f)
    {
        const std::vector<int> x = {3};
        const auto y = pure(f) <app> x;
        EXPECT_EQ(y.size(), 1);
        EXPECT_EQ(y.front(), -3);
    };
    const TEFunction<int, int> g = std::negate<>{};
    test(g);
    const auto h = function(std::negate<int>{});
    test(h);
}

TEST(Applicative, list)
{
    auto const test = [=](auto f)
    {
        const std::list<int> x = {3, 4};
        const int y = 5;
        const auto z = pure(f) <app> x <app> pure(y);
        EXPECT_EQ(z.size(), 2);
        const auto u = {15, 20};
        EXPECT_TRUE(std::equal(z.begin(), z.end(), u.begin()));
    };
    const TEFunction<int, int, int> g = std::multiplies<>{};
    test(g);
    const auto h = function<int, int, int>(std::multiplies<>{});
    test(h);
}

TEST(Applicative, range)
{
    auto const test = [=](auto f)
    {
        const std::list<int> x = {3, 4};
        const auto z = pure(f) <app> nonOwnedRange(x);
        std::vector<int> zz = toVector(z);
        const auto u = {-3, -4};
        EXPECT_EQ(zz.size(), u.size());
        EXPECT_TRUE(std::equal(zz.begin(), zz.end(), u.begin()));
    };
    const TEFunction<int, int> g = std::negate<>{};
    test(g);
    const auto h = function<int, int>(std::negate<>{});
    test(h);
}

TEST(Applicative, tuple)
{
    const auto x = std::make_tuple(3, "4");
    const auto result = pure(show) <app> x;
    const auto expected = std::make_tuple("3", "4");
    EXPECT_EQ(result, expected);
}

TEST(Applicative, range1)
{
    auto const test = [=](auto f)
    {
        const std::list<int> x = {3, 4};
        const std::list<int> y = {5, 6};
        const auto z0 = pure(f) <app> nonOwnedRange(x);
        auto result0 = toVector(z0);
        EXPECT_EQ(result0.size(), 2);
        const auto z = z0 <app> nonOwnedRange(y);
        auto result = toVector(z);
        EXPECT_EQ(result.size(), 4);
        const auto u = {15, 18, 20, 24};
        EXPECT_TRUE(std::equal(result.begin(), result.end(), u.begin()));
    };
    const TEFunction<int, int, int> g = std::multiplies<>{};
    test(g);
    const auto h = function<int, int, int>(std::multiplies<>{});
    test(h);
}

TEST(Applicative, range2)
{
    auto const test = [=](auto f)
    {
        const std::list<int> x = {3, 4};
        const std::list<size_t> y = {1U, 2U};
        const auto z0 = pure(f) <app> nonOwnedRange(x);
        auto result0 = toVector(z0);
        EXPECT_EQ(result0.size(), 2);
        const auto z = z0 <app> nonOwnedRange(y);
        auto result = toVector(toVector <fmap> z);
        EXPECT_EQ(result.size(), 4);
        const std::vector<std::vector<int>> u = {{3}, {3, 3}, {4}, {4, 4}};
        EXPECT_TRUE(std::equal(result.begin(), result.end(), u.begin()));
    };
    test(replicate);
}

TEST(Applicative, Maybe)
{
    auto const test = [=](auto f)
    {
        const Maybe<int> x = {1};
        const auto y = pure(f) <app> x;
        EXPECT_EQ(Just{-1}, y);
    };
    const TEFunction<int, int> g = std::negate<>{};
    test(g);
    const auto h = function(std::negate<int>{});
    test(h);
}

TEST(Applicative, IO)
{
    auto const test = [=](auto f)
    {
        const auto x = ioData(42);
        const auto y = pure(f) <app> x;
        EXPECT_EQ(-42, y.run());
    };
    const TEFunction<int, int> g = std::negate<>{};
    test(g);
    const auto h = function(std::negate<int>{});
    test(h);
}

TEST(Applicative, TEFunction)
{
    const TEFunction<int, std::string> u = [](std::string const& str) { return str.size(); };
    const TEFunction<std::string, bool> f = [](bool x) { return x ? "true" : "false"; };
    const auto y = pure(u) <app> f;
    EXPECT_EQ(y(true), 4);
    EXPECT_EQ(y(false), 5);
}

TEST(Applicative, GenericFunction)
{
    const auto u = genericFunction<1>([](std::string const& str) { return str.size(); });
    const auto f = genericFunction<1>([](auto x) { return x ? "true" : "false"; });
    const auto y = pure(u) <app> f;
    EXPECT_EQ(y(true), 4);
    EXPECT_EQ(y(false), 5);
}

TEST(Compose, x)
{
    constexpr auto neg = function(std::negate<int>{});
    const auto pos = neg <o> neg;
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
        const auto z = x >>= filterEven;
        EXPECT_EQ(z.size(), 2);
        const auto u = {4, 6};
        EXPECT_TRUE(std::equal(z.begin(), z.end(), u.begin()));
    };
    auto f = [](int x)-> std::list<int> { return x%2==0 ? std::list<int>{x} : std::list<int>{}; };
    const TEFunction<std::list<int>, int> filterEven1 = f;
    test(filterEven1);
    const auto filterEven2 = function(f);
    test(filterEven2);
}

TEST(Monad, list2)
{
    auto const test = [=](auto thisAndNeg)
    {
        const std::list<int> x = {3, 4, 5, 6};
        const auto z = x >>= thisAndNeg;
        EXPECT_EQ(z.size(), 8);
        const auto u = {3, -3, 4, -4, 5, -5, 6, -6};
        EXPECT_TRUE(std::equal(z.begin(), z.end(), u.begin()));
    };
    auto const thisAndNeg = [](int x) { return std::list<int>{x, -x}; };
    const TEFunction<std::list<int>, int> thisAndNeg1 = thisAndNeg;
    test(thisAndNeg1);
    const auto thisAndNeg2 = function(thisAndNeg);
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
    auto const thisAndNeg = [](int x) { return std::list<int>{x, -x}; };
    const TEFunction<std::list<int>, int> thisAndNeg1 = thisAndNeg;
    test(thisAndNeg1);
    const auto thisAndNeg2 = function(thisAndNeg);
    test(thisAndNeg2);
}

TEST(Monad, tuple)
{
    auto const thisAndRepr = [](auto x) { return std::make_tuple(x, show | x); };
    auto const x = std::make_tuple(3, true);
    auto const result = x >>= thisAndRepr;
    const auto expected = std::make_tuple(3, "3", true, "true");
    EXPECT_EQ(result, expected);
}

TEST(Monad, Maybe)
{
    auto const test = [=](auto filterEven)
    {
        const auto x = return_(3);
        const auto y = return_(4);
        const auto z = x >>= filterEven;
        const auto u = y >>= filterEven;
        EXPECT_EQ(z, Nothing{});
        EXPECT_EQ(u, Just{4});
    };
    auto const filterEven = [](int x)-> Maybe<int> { return x%2==0 ? Maybe<int>{x} : Maybe<int>{}; };
    const TEFunction<Maybe<int>, int> filterEven1 = filterEven;
    const auto filterEven2 = function(filterEven);
    test(filterEven1);
    test(filterEven2);
}

TEST(Monad, Maybe2)
{
    auto const result = Maybe<int>{} >> Maybe<int>{3};
    EXPECT_EQ(result, Nothing{});
}

TEST(Monad, IO)
{
    testing::internal::CaptureStdout();

    using namespace std::literals;
    const auto x = return_("3");
    const auto y = return_("4");

    const auto z = x >>= putStrLn;
    const auto u = y >>= putStrLn;

    const auto v = ioData("5") >> ioData("6");

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
    constexpr auto u = function([](std::string const& str) { return str.size(); });
    constexpr auto f = function([](size_t i, std::string const& str) { return !str.empty() && i > 0; });
    constexpr auto y = u >>= f;
    EXPECT_EQ(y(""), false);
    EXPECT_EQ(y("true"), true);
}

TEST(Monad, GenericFunction)
{
    constexpr auto f = genericFunction<2>([](std::string const& str, size_t i) { return equalTo | str.size() | i; });
    constexpr auto y = show >>= f;
    EXPECT_EQ(y(1U), true);
    EXPECT_EQ(y(3U), false);
}

TEST(Monad, Function2)
{
    constexpr auto f = function([](std::string const& str, size_t i) { return str.size() == i; });
    constexpr auto y = function<std::string, size_t>(show) >>= f;
    EXPECT_EQ(y(1U), true);
    EXPECT_EQ(y(3U), false);
}

TEST(Monad, return_)
{
    constexpr auto f = function([](int i, std::string const& str) { return !str.empty() && i > 0; });
    constexpr auto x = Monad<Function, const std::string&>::return_(5);
    EXPECT_EQ(x(""), 5);
    constexpr auto y = x >>= f;
    EXPECT_EQ(y(""), false);
    EXPECT_EQ(y("true"), true);
}

TEST(FunctionOp, bar)
{
    constexpr auto f = function(std::multiplies<int>{});
    constexpr auto r = f | 2 | 5;
    EXPECT_EQ(r, 10);
}

TEST(FunctionOp, bars)
{
    constexpr auto f = function(std::multiplies<int>{});
    constexpr auto g = function(std::negate<int>{});
    constexpr auto r = g || f | 2 | 5;
    EXPECT_EQ(r, -10);
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wenum-compare"
TEST(on, 1)
{
    constexpr auto f = function(std::equal_to<int>{});
    constexpr auto g = function([](int e){ return e > 0; });
    constexpr auto r = f <on> g | 2 | -4;
    EXPECT_EQ(r, false);
    constexpr auto s = f <on> g | 2 | 4;
    EXPECT_EQ(s, true);
}
#pragma GCC diagnostic pop

TEST(IO, myAction)
{
    constexpr auto f = function(
        [](std::string l, std::string r)
        {
            return l + r;
        }
    );
    auto const myAction = f <fmap> getLine <app> getLine;
    (void)myAction;
}

TEST(Function, myFunc)
{
    constexpr auto myFunc = function(std::plus<int>{})
                     <fmap> function([](int x) { return x + 3;})
                      <app> function([](int x) { return x * 100; });
    EXPECT_EQ(myFunc(5), 508);
}

TEST(Function, myFunc2)
{
    constexpr auto myFunc = function([](float x, float y, float z) { return std::vector{x, y, z}; })
                     <fmap> function([](float x) { return x + 3;})
                      <app> function([](float x) { return x * 2; })
                      <app> function([](float x) { return x / 2; });
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
    using namespace std::literals;
    auto const str = "3";
    auto const x = replicate | str | 4U;
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
    constexpr auto f = function([](std::vector<std::string> acc, std::string l)
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
    constexpr auto f = function([](std::string acc, std::string l)
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
    constexpr auto f = function([](std::vector<std::string> acc, std::string l)
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
    constexpr auto f = function([](std::string l, std::string acc)
    {
        return l + acc;
    });
    auto const x = foldr | f | std::string{};
    auto const l = std::vector{"1", "2", "3"};
    auto const result = x | l;
    EXPECT_EQ(result, "123");
}

TEST(flip, x)
{
    constexpr auto gt = flip | genericFunction<2>(std::less<>{});
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
    using namespace std::literals;

    EXPECT_EQ(f | "abracadabra"sv, 5);
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

TEST(Monoid, range)
{
    auto const l = std::vector{1, 2, 3};
    auto const r = std::vector{4, 5, 6};
    auto const result_ = nonOwnedRange(l) <mappend> nonOwnedRange(r);
    auto const result = toVector(result_);
    auto const expected = {1, 2, 3, 4, 5, 6};
    EXPECT_TRUE(std::equal(result.begin(), result.end(), expected.begin()));
}

TEST(Monoid, list)
{
    std::list<std::list<int>> nested = {{1, 2}, {3, 4}};
    auto v = mconcat | nested;
    auto const result = toVector(v);
    std::decay_t<decltype(result)> x;
    auto const expected = {1, 2, 3, 4};
    EXPECT_TRUE(std::equal(result.begin(), result.end(), expected.begin()));
}

TEST(Monoid, range2)
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
    auto const nested = std::list{Maybe{Product{2}}, Maybe<Product<int>>{}, Maybe{Product{3}}};
    auto const result = mconcat | nested;
    auto const expected = Maybe{Product{6}};
    EXPECT_EQ(result, expected);
}

TEST(Monoid, product)
{
    auto const a = Product{1.5};
    auto const b = Product{2.0};
    auto nested = std::vector{a, b};
    auto v = mconcat | nested;
    auto const result = v;
    auto const expected = 3.0;
    EXPECT_EQ(expected, result.get());

    auto const result2 = getProduct || (product | 3) <mappend> (product | 4) <mappend> (product | 2);
    EXPECT_EQ(result2, 24);

    auto const result3 = getProduct <o> mconcat | std::vector{Product{3}, Product{4}, Product{2}};
    EXPECT_EQ(result3, 24);
}

TEST(Monoid, product2)
{
    auto const result = getProduct <o> mconcat <o> (map | product) || std::vector{3, 4, 2};
    EXPECT_EQ(result, 24);
}

TEST(Monoid, Ordering)
{
    constexpr auto length = function([](std::string const& x)
    {
        return x.length();
    });
    auto const lengthCompare = function([=](std::string const& x, std::string const& y)
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
    auto const f = genericFunction<1>([](auto u){ return product | (u+1); });
    auto const g = genericFunction<1>([](auto u){ return product | (u*2); });
    auto const result = mappend | f | g;
    EXPECT_EQ(result(24).get(), 1200);
}

TEST(Monoid, Function)
{
    auto const f = function([](int u){ return product | (u+1); });
    auto const g = function([](int u){ return product | (u*2); });
    auto const result = mappend | f | g;
    EXPECT_EQ(result(24).get(), 1200);
}

TEST(Monoid, Tuple)
{
    auto const x = std::make_tuple(1, std::string{"123"});
    auto const y = std::make_tuple(true);
    auto const z = x <mappend> y <mappend> _o_;
    EXPECT_EQ(z, std::make_tuple(1, std::string("123"), true));

    auto const result = mconcat | std::tuple{y, std::make_tuple(_o_), x};
    EXPECT_EQ(result, std::make_tuple(true, _o_, 1, std::string("123")));
}

TEST(Maybe, 1)
{
    auto result = Maybe<std::string>{"andy"} <mappend> Maybe<std::string>{};
    EXPECT_EQ(result, Just<std::string>{"andy"});
    result = Maybe<std::string>{"andy"} <mappend> Maybe<std::string>{"123"};
    EXPECT_EQ(result, Just<std::string>{"andy123"});
    result = Maybe<std::string>{} <mappend> Maybe<std::string>{};
    EXPECT_EQ(result, Nothing{});

    result = mconcat | std::vector{Maybe<std::string>{"123"}, Maybe<std::string>{"xxx"}};
    EXPECT_EQ(result, Just<std::string>{"123xxx"});
}

TEST(Monad, WalkTheLine)
{
    using Birds = int;
    using Pole = std::pair<Birds, Birds>;
    constexpr auto landLeft = function([](Birds n, Pole p)
    {
        auto [left, right] = p;
        if (std::abs((left + n) - right) < 4) 
        {
            return Maybe<Pole>{Pole{left + n, right}};
        }
        return Maybe<Pole>{};
    });
    constexpr auto landRight = function([](Birds n, Pole p)
    {
        auto [left, right] = p;
        if (std::abs((right + n) - left) < 4) 
        {
            return Maybe<Pole>{Pole{left, right + n}};
        }
        return Maybe<Pole>{};
    });
    auto const result = (((return_ | Pole{0,0}) >>= (landRight | 2)) >>= (landLeft | 2)) >>= (landRight | 2);
    EXPECT_EQ(result, Just(Pole(2, 4)));

    auto const result2 = ((((return_ | Pole{0,0}) >>= (landLeft | 1)) >>= (landRight | 4)) >>= (landLeft | -1)) >>= (landRight | -2);
    EXPECT_EQ(result2, Nothing{});

    constexpr auto banana = function([](Pole)
    {
        return Maybe<Pole>{};
    });
    auto const result3 = (((return_ | Pole{0,0}) >>= (landLeft | 1)) >>= banana) >>= (landRight | 1);
    EXPECT_EQ(result3, Nothing{});
}

TEST(Monad, vec)
{
    auto const result = std::vector{1,2}
                    >>= function([](int n)
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
    auto const result = function([=](int n, char ch)
    {
        return std::make_pair(n,ch);
    })
    <fmap> std::vector{1,2}
    <app> std::vector{'a','b'};

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
    auto const v= JoinView{MapView{FilterView{IotaView{1}, alwaysTrue}, id}};
    auto const y = v;
    (void)y;
}

TEST(MonadPlus, guard)
{
    auto const result = guard<Maybe>(5 > 2);
    EXPECT_EQ(result, Maybe{_o_});

    auto const result1 = guard<Maybe>(1 > 2);
    EXPECT_EQ(result1, Nothing{});

    auto const result3 = guard<std::vector>(5 > 2);
    EXPECT_EQ(result3, std::vector{_o_});

    auto const result4 = guard<std::vector>(1 > 2);
    EXPECT_TRUE(result4.empty());

    auto const result5 = guard<Range>(1 > 2);
    EXPECT_TRUE(toVector(result5).empty());

    auto const result6 = guard<Range> | ('7' <elem> (show | 567 ));
    EXPECT_EQ(toVector(result6).at(0), _o_);

    auto const func = genericFunction<1>([](auto x) { return (guard<Range> | ('7' <elem> (show | x))) >> (return_ | x); });
    auto const l = std::vector{7, 9, 17, 22};
    auto const result7 = func <fmap> nonOwnedRange(l);
    auto const expected7 = std::vector{7, 17};
    EXPECT_EQ(toVector(JoinView{result7}), expected7);

    auto const tmp = ownedRange(IotaView{1, 50}) >>= func;
    auto const result8 = toVector || tmp;
    auto const expected = std::vector{7,17,27,37,47};
    EXPECT_EQ(result8, expected);
}