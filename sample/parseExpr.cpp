#include "parser.h"
#include <cassert>

using namespace hspp::parser;
using namespace hspp;

enum class Op
{
    kADD,
    kSUB,
    kMUL,
    kDIV
};

class OpFunc
{
    Op mOp;
public:
    constexpr OpFunc(Op op)
    : mOp{op}
    {}
    template <typename T>
    constexpr auto operator()(T x, T y) const
    {
        switch (mOp)
        {
            case Op::kADD: return x + y;
            case Op::kSUB: return x - y;
            case Op::kMUL: return x * y;
            case Op::kDIV: return x / y;
        }
        throw std::runtime_error{"Never reach here!"};
    }
};

namespace op
{
constexpr auto add = toGFunc<2> | OpFunc{Op::kADD};
constexpr auto sub = toGFunc<2> | OpFunc{Op::kSUB};
constexpr auto mul = toGFunc<2> | OpFunc{Op::kMUL};
constexpr auto div = toGFunc<2> | OpFunc{Op::kDIV};

static_assert((add | 1 | 2) == 3);
static_assert((sub | 1 | 2) == -1);
static_assert((mul | 1 | 2) == 2);
static_assert((div | 4 | 2) == 2);

auto const addOp = ((symb | "+") >> (return_ | add)) <triPlus> ((symb | "-") >> (return_ | sub));
auto const mulOp = ((symb | "*") >> (return_ | mul)) <triPlus> ((symb | "/") >> (return_ | div));
} // namespace op

using op::addOp;
using op::mulOp;

constexpr auto isDigit = toFunc<> | [](char x)
{
    return isdigit(x);
};

extern TEParser<int> const expr;

auto const digit = (token || sat | isDigit)
                    >>= [](char x) { return
                    return_ | (x - '0');
                };

using namespace std::literals;
auto const factor =
    digit <triPlus>
        (((symb | "("s) >> expr) >>= [](auto n){ return
                (symb | ")"s) >>
                    (return_ | n);
    });

auto const term = factor <chainl1> mulOp;

extern TEParser<int> const expr = toTEParser || (term <chainl1> addOp);

int main()
{
    {
        auto const rawResult = apply || many | digit || " 1  2 34";
        auto const& result = std::get<0>(rawResult.at(0));
        auto const expected = std::vector{1, 2, 3, 4};
        assert(std::equal(result.begin(), result.end(), expected.begin()));
    }

    {
        auto const result = apply || addOp || " +  * /-";
        assert((std::get<0>(result.at(0))| 1 | 2) == 3);

        auto const result2 = apply || mulOp || "   * /-";
        assert((std::get<0>(result2.at(0))| 1 | 2) == 2);
    }

    {
        auto const p = digit <chainl1> addOp;
        auto const result = runParser | p | "1 + 2";
        auto const expected = 3;
        assert(std::get<0>(result.at(0)) == expected);
    }

    {
        auto const result = apply | expr | "1 - 2 * 3 + 4";
        auto const expected = -1;
        assert(std::get<0>(result.at(0)) == expected);
    }
    return 0;
}