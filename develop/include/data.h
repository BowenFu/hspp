/*
 *  Copyright (c) 2022 Bowen Fu
 *  Distributed Under The Apache-2.0 License
 */

#ifndef HSPP_DATA_H
#define HSPP_DATA_H

#include <variant>
#include <string>
#include <vector>
#include <list>
#include <forward_list>
#include <tuple>
#include <iostream>
#include <functional>
#include <type_traits>

namespace hspp
{

template <typename... Ts>
class Overload : Ts...
{
public:
    constexpr Overload(Ts... ts)
    : Ts{ts}...
    {}
    using Ts::operator()...;
};

template <typename... Ts>
auto overload(Ts&&... ts)
{
    return Overload<Ts...>{std::forward<Ts>(ts)...};
}

namespace impl
{
    template <typename T>
    struct AddConstToPointer
    {
        using type = std::conditional_t<
            !std::is_pointer_v<T>, T,
            std::add_pointer_t<std::add_const_t<std::remove_pointer_t<T>>>>;
    };
    template <typename T>
    using AddConstToPointerT = typename AddConstToPointer<T>::type;

    static_assert(std::is_same_v<AddConstToPointerT<void *>, void const *>);
    static_assert(std::is_same_v<AddConstToPointerT<int32_t>, int32_t>);

    template <typename Data>
    using StoreT = AddConstToPointerT<std::decay_t<Data>>;
} // namespace impl

namespace detail
{
    template <std::size_t start, class Tuple, std::size_t... I>
    constexpr decltype(auto) subtupleImpl(Tuple &&t, std::index_sequence<I...>)
    {
        return std::make_tuple(std::get<start + I>(std::forward<Tuple>(t))...);
    }
} // namespace detail

// [start, end)
template <std::size_t start, std::size_t end, class Tuple>
constexpr decltype(auto) subtuple(Tuple &&t)
{
    constexpr auto tupleSize = std::tuple_size_v<std::remove_reference_t<Tuple>>;
    static_assert(start <= end);
    static_assert(end <= tupleSize);
    return detail::subtupleImpl<start>(std::forward<Tuple>(t),
                                        std::make_index_sequence<end - start>{});
}

template <std::size_t len, class Tuple>
constexpr decltype(auto) takeTuple(Tuple &&t)
{
    constexpr auto tupleSize = std::tuple_size_v<std::remove_reference_t<Tuple>>;
    static_assert(len <= tupleSize);
    return subtuple<0, len>(std::forward<Tuple>(t));
}

template <std::size_t len, class Tuple>
using TakeTupleType = std::decay_t<decltype(takeTuple<len>(std::declval<Tuple>()))>;


namespace data
{
template <typename Data>
class Maybe;

class Nothing final
{
};

constexpr inline Nothing nothing;

template <typename Data>
class Just final
{
public:
    constexpr Just(Data d)
    : data{std::move(d)}
    {}
    Data data;
};

template <typename Data>
using MaybeBase = std::variant<Nothing, Just<Data>>;

template <typename Data>
class Maybe : public MaybeBase<Data>
{
public:
    using std::variant<Nothing, Just<Data>>::variant;

    bool hasValue() const
    {
        return std::visit(overload(
            [](Nothing)
            {
                return false;
            },
            [](Just<Data> const&)
            {
                return true;
            }
        ), static_cast<MaybeBase<Data> const&>(*this));
    }

    auto const& value() const
    {
        return std::get<Just<Data>>(*this).data;
    }
};

template <typename T>
constexpr bool operator== (Maybe<T> const& lhs, Maybe<T> const& rhs)
{
    return std::visit(overload(
        [](Nothing, Nothing)
        {
            return true;
        },
        [](Just<T> const& l, Just<T> const& r)
        {
            return l.data == r.data;
        },
        [](auto, auto)
        {
            return false;
        }
    ),
    static_cast<MaybeBase<T> const&>(lhs),
    static_cast<MaybeBase<T> const&>(rhs));
}

template <typename T>
constexpr bool operator== (Maybe<T> const& lhs, Just<T> const& rhs)
{
    return lhs == Maybe<T>{rhs};
}

template <typename T>
constexpr bool operator== (Maybe<T> const& lhs, Nothing)
{
    return !lhs.hasValue();
}

template <typename T>
constexpr bool operator== (Just<T> const& lhs, Maybe<T> const& rhs)
{
    return Maybe<T>{lhs} == rhs;
}

template <typename T>
constexpr bool operator== (Nothing, Maybe<T> const& rhs)
{
    return !rhs.hasValue();
}

template <bool TE, typename Func>
class ToFunction;

template <typename Repr, typename Ret, typename Arg, typename... Rest>
class Function;

// type erased function.
template <typename Ret, typename... Args>
using TEFunction = Function<std::function<Ret(Args...)>, Ret, Args...>;

template <bool TE, typename Func, typename Ret, typename... Args>
class ToFunction<TE, Ret(Func::*)(Args...) const>
{
public:
    using Sig = Ret(Args...);
    static constexpr auto run(Func const& func)
    {
        if constexpr (!TE)
        {
            return Function<Func, Ret, Args...>{func};
        }
        else
        {
            return TEFunction<Ret, Args...>{func};
        }
    }
};

template <typename F>
class CallViaPipe
{
public:
    template <typename Arg>
    constexpr auto operator|(Arg&& arg) const
    {
        return static_cast<F const&>(*this).operator()(std::forward<Arg>(arg));
    }

    template <typename Arg>
    constexpr auto operator||(Arg&& arg) const
    {
        return operator|(arg);
    }
};
template <typename Arg, typename F>
constexpr auto operator&(Arg&& arg, CallViaPipe<F> const& func)
{
    return func | arg;
}

template <typename Repr, typename Ret, typename Arg, typename... Rest>
class Function : public CallViaPipe<Function<Repr, Ret, Arg, Rest...>>
{
public:
    template <typename F>
    constexpr Function(F func)
    : mFunc{std::move(func)}
    {
    }
    constexpr auto operator()(Arg const& arg) const
    {
        if constexpr (sizeof...(Rest) == 0)
        {
            return mFunc(arg);
        }
        else
        {
            auto lamb = [=, func=mFunc](Rest const&... rest){ return func(arg, rest...); };
            return Function<decltype(lamb), Ret , Rest...>{lamb};
        }
    }
private:
    Repr mFunc;
};

template <typename... Ts>
class IsFunction : public std::false_type
{};
template <typename... Ts>
class IsFunction<Function<Ts...>> : public std::true_type
{};
template <typename T>
constexpr static auto isFunctionV = IsFunction<std::decay_t<T>>::value;

template <typename...Args, typename Func>
constexpr auto toFuncImpl(Func const& func)
{
    if constexpr(sizeof...(Args) == 0)
    {
        return ToFunction<false, decltype(&Func::operator())>::run(func);
    }
    else
    {
        return Function<Func, Args...>{func};
    }
}

template <typename Func>
constexpr auto toTEFuncImpl(Func const& func)
{
    return ToFunction<true, decltype(&Func::operator())>::run(func);
}

template <typename Ret, typename... Args, typename Func>
constexpr auto toTEFuncImpl(Func const& func)
{
    return TEFunction<Ret, Args...>{func};
}

template <size_t nbArgs, typename Repr>
class GenericFunction : public CallViaPipe<GenericFunction<nbArgs, Repr>>
{
    static_assert(nbArgs > 0);
public:
    constexpr GenericFunction(Repr func)
    : mFunc{std::move(func)}
    {
    }
    template <typename Arg>
    constexpr auto operator()(Arg const& arg) const
    {
        if constexpr (nbArgs == 1)
        {
            return mFunc(arg);
        }
        else
        {
            auto lamb = [=, func=mFunc](auto const&... rest){ return func(arg, rest...); };
            return GenericFunction<nbArgs-1, std::decay_t<decltype(lamb)>>{std::move(lamb)};
        }
    }
private:
    Repr mFunc;
};

template <size_t nbArgs, typename Func>
constexpr auto toGFuncImpl(Func const& func)
{
    return GenericFunction<nbArgs, Func>{func};
}

template <size_t nbArgs>
constexpr inline auto toGFunc = toGFuncImpl<1>([](auto data)
{
    return toGFuncImpl<nbArgs>(std::move(data));
});

template <typename...Args>
constexpr inline auto toFunc = toGFunc<1>([](auto func)
{
    return toFuncImpl<Args...>(std::move(func));
});

template <typename...Args>
constexpr inline auto toTEFunc = toGFunc<1>([](auto func)
{
    return toTEFuncImpl<Args...>(std::move(func));
});

constexpr inline auto id = toGFunc<1>([](auto data)
{
    return data;
});

constexpr auto just = toGFunc<1>([](auto d)
{
    return Maybe<decltype(d)>{Just{std::move(d)}};
});

template <typename T>
class IsGenericFunction : public std::false_type
{};
template <size_t I, typename T>
class IsGenericFunction<GenericFunction<I, T>> : public std::true_type
{};
template <typename T>
constexpr static auto isGenericFunctionV = IsGenericFunction<std::decay_t<T>>::value;

constexpr auto unCurry = toGFunc<1>([](auto func)
{
    return [func=std::move(func)](auto&&... args)
    {
        return (func | ... | args);
    };
});

template <typename Func, typename Left>
class LeftClosedFunc
{
public:
    Func func;
    Left left;
};

template <typename Left, typename Func, typename = std::enable_if_t<(isFunctionV<Func> || isGenericFunctionV<Func>), bool>>
constexpr auto operator<(Left&& left, Func&& func)
{
    return LeftClosedFunc<std::decay_t<Func>, std::decay_t<Left>>{std::forward<Func>(func), std::forward<Left>(left)};
}

template <typename Left, typename Func, typename Right>
constexpr auto operator>(LeftClosedFunc<Func, Left> const& lcFunc, Right&& right)
{
    return lcFunc.func | lcFunc.left | right;
}

class Compose
{
public:
    template <typename Func, typename Repr, typename Ret, typename FirstArg, typename... Args>
    constexpr auto operator()(Func&& f, Function<Repr, Ret, FirstArg, Args...> const& g) const
    {
        return toFunc<> | [=](FirstArg x){ return f(g(x));};
    }
    template <typename F, size_t I, typename Repr>
    constexpr auto operator()(F&& f, GenericFunction<I, Repr> const& g) const
    {
        // return toGFunc<I>([=](auto&&... args){ return f((g | ... | args));});
        // static_assert(I==1);
        return toGFunc<1>([=](auto x){ return f(g(x));});
    }
    template <typename F, typename G>
    constexpr auto operator()(F&& f, G&&g) const
    {
        return toGFunc<1>([=](auto x){ return f(g(x));});
    }
};

constexpr inline auto o = toGFunc<2>(Compose{});

using _O_ = std::tuple<>;
constexpr inline _O_ _o_;

template <typename Data, typename Func = std::function<Data()>>
class IO
{
    static_assert(std::is_invocable_v<Func>);
    static_assert(std::is_same_v<std::invoke_result_t<Func>, Data>);
public:
    template <typename F>
    constexpr IO(F func)
    : mFunc{std::move(func)}
    {}
    Data run() const
    {
        return mFunc();
    }
private:
    Func mFunc;
};

template <typename... Ts>
class IsIO : public std::false_type
{};
template <typename... Ts>
class IsIO<IO<Ts...>> : public std::true_type
{};
template <typename T>
constexpr static auto isIOV = IsIO<std::decay_t<T>>::value;

template <typename Func>
constexpr auto io(Func func)
{
    using Data = std::invoke_result_t<Func>;
    return IO<Data, Func>{std::move(func)};
}

template <typename Data>
constexpr auto ioData(Data data)
{
    return io([data=std::move(data)] { return data; });
}

template <typename Data, typename Func>
constexpr auto toTEIOImpl(IO<Data, Func> const& p)
{
    return IO<Data>{[p]{
        return p.run();
    }};
}

constexpr auto toTEIO = toGFunc<1> | [](auto p)
{
    return toTEIOImpl(p);
};


constexpr auto putChar = toFunc<> | [](char c)
{
    return io(
        [c]
        {
            std::cout << c << std::flush;
            return _o_;
        }
    );
};

constexpr auto putStr = toFunc<> | [](std::string str)
{
    return io(
        [str=std::move(str)]
        {
            std::cout << str << std::flush;
            return _o_;
        }
    );
};

constexpr auto putStrLn = toFunc<> | [](std::string str)
{
    return io(
        [str=std::move(str)]
        {
            std::cout << str << std::endl;
            return _o_;
        }
    );
};

const inline auto getLine = io([]
    {
        std::string str;
        std::getline(std::cin, str);
        return str;
    }
);

namespace impl
{
    template <typename Value, typename = std::void_t<>>
    struct IsContainer : std::false_type
    {
    };

    template <typename Value>
    struct IsContainer<Value, std::void_t<decltype(std::begin(std::declval<Value>())),
                                        decltype(std::end(std::declval<Value>()))>>
        : std::true_type
    {
    };

    template <typename Cont>
    constexpr auto isContainerV = IsContainer<std::decay_t<Cont>>::value;

    static_assert(!isContainerV<std::pair<int32_t, char>>);
    static_assert(isContainerV<std::vector<int32_t>>);

    template <typename Value, typename = std::void_t<>>
    struct HasReverseIterators : std::false_type
    {
    };

    template <typename Value>
    struct HasReverseIterators<Value, std::void_t<decltype(std::rbegin(std::declval<Value>())),
                                        decltype(std::rend(std::declval<Value>()))>>
        : std::true_type
    {
    };

    template <typename Cont>
    constexpr auto hasReverseIteratorsV = HasReverseIterators<std::decay_t<Cont>>::value;

    static_assert(hasReverseIteratorsV<std::vector<int32_t>>);
    static_assert(!hasReverseIteratorsV<std::forward_list<int32_t>>);

    template <typename Value, typename = std::void_t<>>
    struct IsTupleLike : std::false_type
    {
    };

    template <typename Value>
    struct IsTupleLike<Value, std::void_t<decltype(std::tuple_size<Value>::value)>>
        : std::true_type
    {
    };

    template <typename ValueTuple>
    constexpr auto isTupleLikeV = IsTupleLike<std::decay_t<ValueTuple>>::value;

    static_assert(isTupleLikeV<std::pair<int32_t, char>>);
    static_assert(!isTupleLikeV<bool>);

    template <size_t nbArgs, typename Repr>
    constexpr auto flipImpl(GenericFunction<nbArgs, Repr> f)
    {
        return toGFunc<2>([f=std::move(f)](auto x, auto y){ return f | y | x; });
    }
    template <typename Repr, typename Ret, typename Arg1, typename Arg2, typename... Rest>
    constexpr auto flipImpl(Function<Repr, Ret, Arg1, Arg2, Rest...> f)
    {
        return toFunc<>([f=std::move(f)](Arg1 x, Arg2 y){ return f | y | x; });
    }
    template <typename Repr1, typename Repr2, typename Ret, typename Arg1, typename Arg2, typename... Rest>
    constexpr auto flipImpl(Function<Repr1, Function<Repr2, Ret, Arg2, Rest...>, Arg1> f)
    {
        return toFunc<>([f=std::move(f)](Arg2 x, Arg1 y){ return f | y | x; });
    }
} // namespace impl

using impl::isContainerV;
using impl::isTupleLikeV;
using impl::hasReverseIteratorsV;

constexpr auto flip = toGFunc<1>([](auto func)
{
    return impl::flipImpl(std::move(func));
});

// Note different iterator types for begin and end.
template <typename Iter1, typename Iter2, typename Init, typename Func>
constexpr auto accumulate(Iter1 begin, Iter2 end, Init init, Func func)
{
    for (auto iter = begin; iter != end; ++iter)
    {
        init = func(init, *iter);
    }
    return init;
}

template <typename Iter1, typename Iter2, typename Init, typename Func>
constexpr auto foldrRecur(Iter1 begin, Iter2 end, Init init, Func func) -> Init
{
    if (begin != end)
    {
        auto iter = begin;
        return func | *begin | foldrRecur(++iter, end, std::move(init), std::move(func));
    }
    return init;
}

constexpr auto listFoldr = toGFunc<3>([](auto func, auto init, auto const& list)
{
    if constexpr (data::hasReverseIteratorsV<decltype(list)>)
    {
        return accumulate(list.rbegin(), list.rend(), init, unCurry <o> flip | func);
    }
    else
    {
        return foldrRecur(list.begin(), list.end(), init, func);
    }
});

constexpr auto foldl = toGFunc<3>([](auto func, auto init, auto const& list)
{
    return data::accumulate(list.begin(), list.end(), init, unCurry | func);
});

constexpr auto equalTo = toGFunc<2>(std::equal_to<>{});

template <typename Repr1, typename Ret1, typename Arg1, typename... Rest1, typename Repr2, typename Ret2, typename Arg2, typename... Rest2>
constexpr auto onImpl(Function<Repr1, Ret1, Arg1, Rest1...> f, Function<Repr2, Ret2, Arg2, Rest2...> g)
{
    return toFunc<Ret1, Arg2, Arg2>([f=std::move(f), g=std::move(g)](Arg2 x, Arg2 y) { return f | g(x) | g(y); });
}

constexpr inline auto on = toGFunc<2>([](auto f, auto g)
{
    return onImpl(std::move(f), std::move(g));
});

template <template <typename...> class Class>
constexpr auto to = toGFunc<1>([](auto view)
{
    Class<std::decay_t<decltype(*view.begin())>> result;
    for (auto e: view)
    {
        result.push_back(std::move(e));
    }
    return result;
});

constexpr inline auto filter = toGFunc<2>([](auto pred, auto data)
{
    return ownedRange(FilterView{std::move(data), std::move(pred)});
});

constexpr inline auto map = toGFunc<2>([](auto func, auto data)
{
    return ownedRange(MapView{std::move(data), std::move(func)});
});

constexpr inline auto zip = toGFunc<2>([](auto lhs, auto rhs)
{
    return ownedRange(ZipView{std::move(lhs), std::move(rhs)});
});

constexpr inline auto zipWith = toGFunc<3>([](auto func, auto lhs, auto rhs)
{
    return ownedRange(MapView{ZipView{std::move(lhs), std::move(rhs)}, [func=std::move(func)](auto tu) { return func | std::get<0>(tu) | std::get<1>(tu);}});
});

constexpr inline auto repeat = toGFunc<1>([](auto data)
{
    return ownedRange(RepeatView{std::move(data)});
});

constexpr inline auto replicate = toGFunc<2>([](auto data, size_t times)
{
    return ownedRange(TakeView{RepeatView{std::move(data)}, times});
});

constexpr inline auto enumFrom = toGFunc<1>([](auto start)
{
    return ownedRange(IotaView{start});
});

constexpr inline auto iota = toGFunc<2>([](auto start, auto end)
{
    return ownedRange(IotaView{start, end});
});

constexpr inline auto take = toGFunc<2>([](auto r, size_t num)
{
    return ownedRange(TakeView{r, num});
});

constexpr inline auto drop = toGFunc<2>([](auto r, size_t num)
{
    return ownedRange(DropView{r, num});
});

constexpr inline auto splitAt = toGFunc<2>([](auto r, size_t num)
{
    return std::make_pair(ownedRange(TakeView{r, num}), ownedRange(DropView{r, num}));
});

constexpr inline auto const_ = toGFunc<2>([](auto r, auto)
{
    return r;
});

constexpr inline auto cons = toGFunc<2>([](auto e, auto l)
{
    if constexpr(isRangeV<decltype(l)>)
    {
        return ownedRange(ChainView{SingleView{std::move(e)}, std::move(l)});
    }
    else
    {
        l.insert(l.begin(), e);
        return l;
    }
});

template <size_t N=2>
constexpr inline auto makeTuple = toGFunc<N>([](auto e, auto... l)
{
    return std::make_tuple(std::move(e), std::move(l)...);
});

constexpr inline auto fst = toGFunc<1>([](auto e)
{
    return std::get<0>(e);
});

constexpr inline auto snd = toGFunc<1>([](auto e)
{
    return std::get<1>(e);
});

constexpr inline auto deref = toGFunc<1>([](auto e)
{
    return *e;
});

template <typename Data>
class DataHolder
{
    Data mData;
public:
    constexpr DataHolder(Data data)
    : mData{std::move(data)}
    {}
    auto const& get() const
    {
        return mData;
    }
};

template <typename T>
constexpr auto operator==(DataHolder<T> const& l, DataHolder<T> const& r)
{
    return l.get() == r.get();
}

template <template <typename...> typename Type>
constexpr auto toType = toGFunc<1>([](auto data)
{
    return Type<decltype(data)>{data};
});

constexpr auto from = toGFunc<1>([](auto data)
{
    return data.get();
});

template <typename R, typename A, typename Repr>
class Reader : public DataHolder<Function<Repr, A, R>>
{
public:
    using DataHolder<Function<Repr, A, R>>::DataHolder;
};

template <typename R, typename A, typename Repr>
constexpr auto toReaderImpl(Function<Repr, A, R> func)
{
    return Reader<R, A, Repr>{std::move(func)};
}

constexpr auto toReader = toGFunc<1>([](auto func)
{
    return toReaderImpl(std::move(func));
});

constexpr auto runReader = from;


template <typename S, typename A, typename Repr>
class State : public DataHolder<Function<Repr, std::tuple<A, S>, S>>
{
public:
    using DataHolder<Function<Repr, std::tuple<A, S>, S>>::DataHolder;
};

template <typename S, typename A, typename Repr>
constexpr auto toStateImpl(Function<Repr, std::tuple<A, S>, S> func)
{
    return State<S, A, Repr>{std::move(func)};
}

constexpr auto toState = toGFunc<1>([](auto func)
{
    return toStateImpl(std::move(func));
});

constexpr auto runState = from;

// TODO: in param can be std::string const&
template <typename A, typename Repr>
class Parser : public DataHolder<Function<Repr, std::vector<std::tuple<A, std::string>>, std::string>>
{
public:
    using DataHolder<Function<Repr, std::vector<std::tuple<A, std::string>>, std::string>>::DataHolder;
};

template <typename A, typename Repr>
constexpr auto toParserImpl(Function<Repr, std::vector<std::tuple<A, std::string>>, std::string> func)
{
    return Parser<A, Repr>{std::move(func)};
}

constexpr auto toParser = toGFunc<1>([](auto func)
{
    return toParserImpl(std::move(func));
});

constexpr auto runParser = from;

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

} // namespace data

using data::o;
using data::_o_;
using data::_O_;
using data::flip;
using data::putStrLn;
using data::toFunc;
using data::toGFunc;
using data::unCurry;
using data::id;

// For io
constexpr auto mapM_ = toGFunc<2> | [](auto func, auto lst)
{
    return data::io([=]
    {
        for (auto e : lst)
        {
            func(e).run();
        }
        return _o_;
    });
};

constexpr auto even = toGFunc<1> | [](auto n)
{
    static_assert(std::is_integral_v<decltype(n)>);
    return n % 2 == 0;
};

constexpr auto odd = toGFunc<1> | [](auto n)
{
    return !(even | n);
};

} // namespace hspp

#endif // HSPP_DATA_H