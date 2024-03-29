/*
 *  Copyright (c) 2022 Bowen Fu
 *  Distributed Under The Apache-2.0 License
 */

#ifndef HSPP_PARSER_H
#define HSPP_PARSER_H

namespace hspp
{

namespace parser
{

// TODO: in param can be std::string const&
template <typename A, typename Repr>
class Parser : public data::DataHolder<data::Function<Repr, std::vector<std::tuple<A, std::string>>, std::string>>
{
public:
    using data::DataHolder<data::Function<Repr, std::vector<std::tuple<A, std::string>>, std::string>>::DataHolder;
};

template <typename A, typename Repr>
constexpr auto toParserImpl(data::Function<Repr, std::vector<std::tuple<A, std::string>>, std::string> func)
{
    return Parser<A, Repr>{std::move(func)};
}

constexpr auto toParser = data::toGFunc<1>([](auto func)
{
    return toParserImpl(std::move(func));
});

constexpr auto runParser = data::from;

template <typename A>
using TEParser = Parser<A, std::function<std::vector<std::tuple<A, std::string>>(std::string)>>;

template <typename A, typename Repr>
constexpr auto toTEParserImpl(Parser<A, Repr> p)
{
    return TEParser<A>{(runParser | p)};
}

constexpr auto toTEParser = toGFunc<1> | [](auto p)
{
    return toTEParserImpl(p);
};

} // namespace parser

template <typename A, typename Repr>
struct DataTrait<parser::Parser<A, Repr>>
{
    using Type = A;
};

template <template <template<typename...> typename Type, typename... Ts> class TypeClassT, typename A, typename Repr>
struct TypeClassTrait<TypeClassT, parser::Parser<A, Repr>>
{
    using Type = TypeClassT<parser::Parser>;
};

template <typename... Ts>
class Functor<parser::Parser, Ts...>
{};

template <>
class Applicative<parser::Parser> : public Functor<parser::Parser>
{
public:
    constexpr static auto pure = toGFunc<1> | [](auto a)
    {
        return parser::toParser || data::toFunc<> | [a=std::move(a)](std::string cs){ return std::vector{std::make_tuple(a, cs)}; };
    };
};

template <>
class MonadBase<parser::Parser>
{
public:
    template <typename A, typename Repr, typename Func>
    constexpr static auto bind(parser::Parser<A, Repr> const& p, Func f)
    {
        return parser::toParser | toFunc<>([=](std::string cs)
        {
            auto&& tempResult = parser::runParser | p | cs;
            auto const cont = toGFunc<1> | [f=std::move(f)](auto tu)
            {
                auto&& [a, cs] = tu;
                return return_ || parser::runParser | f(a) | cs;
            };
            return mconcat || (tempResult >>= cont);
        });
    }
};

template <typename A>
class MonadZero<parser::Parser, A>
{
public:
    constexpr static auto mzero = parser::toParser || toFunc<> | [](std::string)
    {
        return std::vector<std::tuple<A, std::string>>{};
    };
};

template <typename A>
class MonadPlus<parser::Parser, A>
{
public:
    constexpr static auto mplus = toGFunc<2> | [](auto p, auto q)
    {
        return parser::toParser || toFunc<> | [=](std::string cs)
        {
            return (parser::runParser | p | cs) <hspp::mplus> (parser::runParser | q | cs);
        };
    };
};

namespace parser
{

constexpr auto alt = toGFunc<2> | [](auto p, auto q)
{
    return parser::toParser <o> data::toFunc<> | [=](std::string cs)
    {
        auto const tmp = runParser | (p <mplus> q) | cs;
        if (tmp.empty())
        {
            return tmp;
        }
        std::remove_const_t<decltype(tmp)> tmp2;
        tmp2.push_back(tmp.front());
        return tmp2;
    };
};

constexpr auto item = parser::toParser | toFunc<>([](std::string cs) -> std::vector<std::tuple<char, std::string>>
{
    if (cs.empty())
    {
        return {};
    }
    return {{cs.front(), cs.substr(1)}};
});

constexpr auto sat = toGFunc<1> | [](auto p)
{
    return item >>= toFunc<> | [=](char c) { return
        parser::toParser || toFunc<> | [flag = p | c, posParser = Monad<Parser>::return_ | c, negParser = MonadZero<Parser, char>::mzero]
        (std::string cs) -> std::vector<std::tuple<char, std::string>>
        {
            return flag ? (runParser | posParser | cs) : (runParser | negParser | cs);
        };
    };
};

constexpr auto char_ = toFunc<> | [](char c)
{
    return sat | (data::equalTo | c);
};

class StringParser;

inline auto stringImpl(std::string const& cs)
    -> Parser<std::string, StringParser>;

class StringParser
{
    std::string mCs;
public:
    StringParser(std::string cs)
    : mCs{std::move(cs)}
    {}
    auto operator()(std::string str) const -> std::vector<std::tuple<std::string, std::string>>
    {
        if (mCs.empty())
        {
            auto const p1 = Monad<Parser>::return_ | std::string{""};
            return runParser | p1 | str;
        }
        auto const c = mCs.front();
        auto const cr = mCs.substr(1);
        auto const p2 =
            (char_ | c)
                >> stringImpl(cr)
                >> (return_ | mCs);
        return runParser | p2 | str;
    }
};

inline auto stringImpl(std::string const& cs)
    -> Parser<std::string, StringParser>
{
    return toParser || toFunc<> | StringParser{cs};
}

constexpr auto string = toFunc<> || [](std::string const& cs)
{
    return stringImpl(cs);
};

template <typename A, typename Repr>
constexpr auto manyImpl(Parser<A, Repr> p)
    -> TEParser<std::vector<A>>;

constexpr auto many1 = toGFunc<1> | [](auto p)
{
    return
    p >>= [=](auto a) { return
       manyImpl(p) >>= [=](auto as) { return
        return_ | (a <data::cons> as);
        };
       };
};

template <typename A, typename Repr>
constexpr auto manyImpl(Parser<A, Repr> p)
    -> TEParser<std::vector<A>>
{
    return toTEParser || alt | (many1 | p) | (Monad<Parser>::return_ | std::vector<A>{});
}

constexpr auto many = toGFunc<1> | [](auto p)
{
    return manyImpl(p);
};

constexpr auto sepBy1 = toGFunc<2> | [](auto p, auto sep)
{
    return p
    >>= [=](auto a) { return
        (many | (sep >> p))
        >>= [=](auto as) { return
            return_ || a <data::cons>  as;
        };
    };
};

template <typename A, typename Repr, typename B, typename Repr1>
constexpr auto sepByImpl(Parser<A, Repr> p, Parser<B, Repr1> sep)
{
    return (alt | (p <sepBy1> sep) | (Monad<Parser>::return_ | std::vector<A>{}));
}

constexpr auto sepBy = toGFunc<2> | [](auto p, auto sep)
{
    return sepByImpl(p, sep);
};

template <typename A, typename Repr, typename Op>
constexpr auto chainl1Impl(Parser<A, Repr> p, Op op)
{
    auto const rest = toGFunc<1> | yCombinator | [=](auto const& self, auto a) -> TEParser<A>
    {
        auto const lhs =
            op >>= [=](auto f) { return
                p >>= [=](auto b) { return
                    self(f | a | b);
                };
            };
        auto const rhs = Monad<Parser>::return_ | a;
        return toTEParser || (lhs <alt> rhs);
    };
    return (p >>= rest);
}

constexpr auto chainl1 = toGFunc<2> | [](auto p, auto op)
{
    return chainl1Impl(p, op);
};

constexpr auto chainl = toGFunc<3> | [](auto p, auto op, auto a)
{
    return (p <chainl1> op) <alt> (Monad<Parser>::return_ | a);
};

constexpr auto isSpace = toFunc<> | [](char c)
{
    return isspace(c);
};

inline auto const space = many || sat | isSpace;

// This will fail some tests.
constexpr auto token = toGFunc<1> | [](auto p)
{
    using A = DataType<std::decay_t<decltype(p)>>;
    doN::Id<A> a;
    return doN::do_(
        a <= p,
        space,
        return_ | a
    );
};

constexpr auto symb = token <o> string;

constexpr auto apply = toGFunc<1> | [](auto p)
{
    return runParser | (space >> p);
};

} // namespace parser

} // namespace hspp

#endif // HSPP_PARSER_H
