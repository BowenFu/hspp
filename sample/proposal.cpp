#include "hspp.h"
#include <cassert>
#include <memory>
#include <variant>
#include "common.h"

using namespace hspp;
using namespace hspp::doN;
using namespace hspp::data;
using namespace std::literals;

// Monadic operations for std::optional p0798r0

void optionalMap()
{
    // proposal
    // std::optional<std::string> opt_string = "123"s;
    // std::optional<std::size_t> s = opt_string.map(&std::string::size);

    Maybe<std::string> optString = "123"s;

    // hspp without do notation
    Maybe<std::size_t> s0 = &std::string::size <fmap> optString;
    expectTrue(s0.hasValue());
    expectEq(s0.value(), 3UL);

    // hspp with do notation
    auto f = toFunc<> | &std::string::size;
    Id<std::string> idS;
    Maybe<std::size_t> s1 = do_(
        idS <= optString,
        return_ || f | idS
    );
    expectTrue(s1.hasValue());
    expectEq(s1.value(), 3UL);
}

void optionalAndThen()
{
    constexpr auto stoi = toFunc<> | [](std::string const& s) noexcept -> Maybe<int>
    {
        try
        {
            return std::stoi(s);
        }
        catch (...)
        {
            return {};
        }
    };

    // proposal
    // std::optional<std::string> opt_string = "123"s;
    // std::optional<int> i = opt_string.and_then(stoi);

    Maybe<std::string> optString = "123"s;

    // hspp without do notation
    Maybe<int> s0 = (optString >>= stoi);
    expectTrue(s0.hasValue());
    expectEq(s0.value(), 123);

    // hspp with do notation
    Id<std::string> idS;
    Maybe<int> s1 = do_(
        idS <= optString,
        stoi | idS
    );
    expectTrue(s1.hasValue());
    expectEq(s1.value(), 123);
}

// p0323r0 https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0323r0.pdf

enum class arithmetic_errc
{
    divide_by_zero, // 9/0 == ?
    not_integer_division // 5/2 == 2.5 (which is not an integer)
};

#if 0 // proposal p0323r0
expected<double,error_condition> safe_divide(double i, double j)
{
    if (j==0) return make_unexpected(arithmetic_errc::divide_by_zero); // (1)
    else return i / j; // (2)
}
#endif // proposal p0323r0

auto safe_divide(double i, double j) -> Either<arithmetic_errc, double>
{
    if (j == 0)
    {
        return toLeft | arithmetic_errc::divide_by_zero;
    }
    return toRight || i / j;
}

#if 0 // proposal p0323r0
// i + j / k
expected<double, error_condition> f1(double i, double j, double k)
{
    return safe_divide(j, k).map([&](double q)
    {
        return i + q;
    });
}
#endif // proposal p0323r0

auto f1(double i, double j, double k) -> Either<arithmetic_errc, double>
{
    return [&](double q) { return i + q; } <fmap> safe_divide(j, k);
}

auto f1_(double i, double j, double k) -> Either<arithmetic_errc, double>
{
    Id<double> q;
    return do_(
        q <= safe_divide(j, k),
        return_ || i + q
    );
}

void testSafeDivideDouble()
{
    auto result1 = f1(2, 4, 5);
    expectTrue(result1.isRight());
    expectEq(result1.right(), 2.8f);

    auto result1_ = f1_(2, 4, 5);
    expectTrue(result1_.isRight());
    expectEq(result1_.right(), 2.8f);

    auto result2 = f1(2, 4, 0);
    expectTrue(!result2.isRight());
    expectEq(result2.left(), arithmetic_errc::divide_by_zero);

    auto result2_ = f1_(2, 4, 0);
    expectTrue(!result2_.isRight());
    expectEq(result2_.left(), arithmetic_errc::divide_by_zero);
}

#if 0 // proposal p0323r0
expected<int, error_condition> safe_divide(int i, int j)
{
    if (j == 0) return make_unexpected(arithmetic_errc::divide_by_zero);
    if (i%j != 0) return make_unexpected(arithmetic_errc::not_integer_division);
    else return i / j;
}
#endif // proposal p0323r0

auto safe_divide(int i, int j) -> Either<arithmetic_errc, int>
{
    if (j == 0) return toLeft | arithmetic_errc::divide_by_zero;
    if (i%j != 0) return toLeft | arithmetic_errc::not_integer_division;
    else return toRight || i / j;
}

#if 0 // proposal p0323r0
// i / k + j / k
expected<int, error_condition> f(int i, int j, int k)
{
    return safe_divide(i, k).bind([=](int q1)
    {
        return safe_divide(j,k).bind([=](int q2)
        {
            return q1+q2;
        });
    });
}
#endif // proposal p0323r0

auto f2(int i, int j, int k) -> Either<arithmetic_errc, int>
{
    return safe_divide(i, k) >>= [=](int q1)
    {
        return [=](int q2)
        {
            return q1+q2;
        }
        <fmap>
        safe_divide(j,k);
    };
}

auto f2_(int i, int j, int k) -> Either<arithmetic_errc, int>
{
    Id<int> q1, q2;
    return do_(
        q1 <= safe_divide(i, k),
        q2 <= safe_divide(j, k),
        return_ || q1 + q2
    );
}

void testSafeDivideInt()
{
    auto result1 = f2(6, 4, 2);
    expectTrue(result1.isRight());
    expectEq(result1.right(), 5);

    auto result1_ = f2_(6, 4, 2);
    expectTrue(result1_.isRight());
    expectEq(result1_.right(), 5);

    auto result2 = f2(2, 4, 0);
    expectTrue(!result2.isRight());
    expectEq(result2.left(), arithmetic_errc::divide_by_zero);

    auto result2_ = f2_(2, 4, 0);
    expectTrue(!result2_.isRight());
    expectEq(result2_.left(), arithmetic_errc::divide_by_zero);

    auto result3 = f2(2, 4, 3);
    expectTrue(!result3.isRight());
    expectEq(result3.left(), arithmetic_errc::not_integer_division);

    auto result3_ = f2_(2, 4, 3);
    expectTrue(!result3_.isRight());
    expectEq(result3_.left(), arithmetic_errc::not_integer_division);
}

int main()
{
    optionalMap();
    optionalAndThen();
    testSafeDivideDouble();
    testSafeDivideInt();
    return 0;
}