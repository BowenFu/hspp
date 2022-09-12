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
#include <optional>

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

template <typename Data>
class Maybe;

class Nothing final
{
public:
    template <typename Data>
    constexpr operator Maybe<Data> () const
    {
        return Maybe<Data>{};
    }
};

template <typename Data>
class Just final : public DataHolder<Data>
{
};

template <typename Data>
class Maybe
{
    std::optional<Data> mData;
    constexpr Maybe(std::optional<Data> data)
    : mData{std::move(data)}
    {}
public:
    constexpr Maybe()
    : mData{}
    {}
    constexpr Maybe(Data data)
    : mData{std::move(data)}
    {}

    static constexpr auto fromOptional(std::optional<Data> data)
    {
        return Maybe<Data>{std::move(data)};
    }

    constexpr operator std::optional<Data>&& () &&
    {
        return std::move(mData);
    }

    constexpr operator std::optional<Data> const& () const &
    {
        return mData;
    }

    bool hasValue() const
    {
        return mData.has_value();
    }

    auto const& value() const
    {
        return mData.value();
    }
};

template <typename T>
const inline Maybe<T> nothing;

template <typename T>
constexpr bool operator== (Maybe<T> const& lhs, Maybe<T> const& rhs)
{
    if (lhs.hasValue() && rhs.hasValue())
    {
        return lhs.value() == rhs.value();
    }
    return lhs.hasValue() == rhs.hasValue();
}

template <bool TE, typename Func>
class ToFunction;

template <typename Repr, typename Ret, typename Arg, typename... Rest>
class Function;

// type erased function.
template <typename Ret, typename... Args>
using TEFunction = Function<std::function<Ret(Args...)>, Ret, Args...>;

template <bool TE, typename Class, typename Ret, typename... Args>
class ToFunction<TE, Ret(Class::*)(Args...) const>
{
public:
    static constexpr auto run(Class const& func)
    {
        if constexpr (!TE)
        {
            return Function<Class, Ret, Args...>{func};
        }
        else
        {
            return TEFunction<Ret, Args...>{func};
        }
    }
    using MemFunc = Ret(Class::*)(Args...) const;
    static constexpr auto fromMemFunc(MemFunc const& func)
    {
        auto const f = [=](Class const& c, Args const&... args)
        {
            return std::invoke(func, c, args...);
        };
        if constexpr (!TE)
        {
            return Function<decltype(f), Ret, Class, Args...>{f};
        }
        else
        {
            return TEFunction<Ret, Class, Args...>{f};
        }
    }
};

template <bool TE, typename Class, typename Ret, typename... Args>
class ToFunction<TE, Ret(Class::*)(Args...) const noexcept> : public ToFunction<TE, Ret(Class::*)(Args...) const>
{
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
    template <typename... Ts>
    constexpr auto operator()(Arg const& arg, Ts&&... ts) const
    {
        if constexpr (sizeof...(Rest) == 0)
        {
            static_assert(sizeof...(Ts) == 0);
            return mFunc(arg);
        }
        else if constexpr (sizeof...(Rest) == sizeof...(Ts))
        {
            static_assert((std::is_same_v<Rest, Ts> && ...));
            return ((*this)(arg) | ... | std::forward<Ts>(ts));
        }
        else
        {
            static_assert(sizeof...(Ts) == 0);
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
        if constexpr(std::is_member_function_pointer_v<Func>)
        {
            return ToFunction<false, Func>::fromMemFunc(func);
        }
        else
        {
            return ToFunction<false, decltype(&Func::operator())>::run(func);
        }
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
    template <typename Arg, typename... Ts>
    constexpr auto operator()(Arg const& arg, Ts&&... ts) const
    {
        if constexpr (sizeof...(Ts) > 0)
        {
            static_assert(nbArgs == sizeof...(Ts) + 1);
            return ((*this)(arg) | ... | std::forward<Ts>(ts));
        }
        else
        {
            if constexpr (nbArgs == 1)
            {
                static_assert(sizeof...(Ts) == 0);
                return mFunc(arg);
            }
            else
            {
                auto lamb = [=, func=mFunc](auto const&... rest){ return func(arg, rest...); };
                return GenericFunction<nbArgs-1, std::decay_t<decltype(lamb)>>{std::move(lamb)};
            }
        }
    }
private:
    Repr mFunc;
};

#if 0
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
    constexpr auto operator()(Arg&& arg) const
    {
        if constexpr (nbArgs == 1)
        {
            return mFunc(std::forward<Arg>(arg));
        }
        else
        {
            auto lamb = [arg=std::forward<Arg>(arg), func=mFunc](auto&&... rest){ return func(arg, std::forward<decltype(rest)>(rest)...); };
            return GenericFunction<nbArgs-1, std::decay_t<decltype(lamb)>>{std::move(lamb)};
        }
    }
private:
    Repr mFunc;
};
#endif

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
    return Maybe<decltype(d)>{std::move(d)};
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
        return toGFunc<1>([=](auto x)
        {
            return f(g(x));
        });
    }
};

constexpr inline auto o = toGFunc<2>(Compose{});

class _O_ final{};
constexpr inline _O_ _o_;

constexpr inline bool operator== (_O_, _O_)
{
    return true;
}

constexpr inline bool operator!= (_O_, _O_)
{
    return false;
}

static_assert(std::is_standard_layout_v<_O_>);

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
    template <typename F>
    constexpr IO(IO<Data, F> other)
    : mFunc{std::move(other.mFunc)}
    {}
    Data run() const
    {
        return mFunc();
    }
private:
    template <typename, typename F>
    friend class IO;
    Func mFunc;
};

template <typename Data>
using TEIO = IO<Data, std::function<Data()>>;

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
    return TEIO<Data>{[p]{
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

constexpr inline auto cycle = toGFunc<1>([](auto data)
{
    return ownedRange(CycleView{std::move(data)});
});

#if 0
// TODO: implement CycleView when needed.
constexpr inline auto cycle = toGFunc<1>([](auto data)
{
    return ownedRange(CycleView{std::move(data)});
});
#endif // 0

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

constexpr inline auto within = toGFunc<2>([](auto start, auto end)
{
    return ownedRange(IotaView<decltype(start), /*includeUpperbound*/ true>{start, end});
});

constexpr inline auto within_ = toGFunc<3>([](auto start, auto next, auto end)
{
    return ownedRange(IotaView<decltype(start), /*includeUpperbound*/ true>{start, end, next - start});
});

constexpr inline auto take = toGFunc<2>([](size_t num, auto r)
{
    return ownedRange(TakeView{r, num});
});

constexpr inline auto drop = toGFunc<2>([](size_t num, auto r)
{
    return ownedRange(DropView{r, num});
});

constexpr inline auto splitAt = toGFunc<2>([](size_t num, auto r)
{
    return std::make_pair(ownedRange(TakeView{r, num}), ownedRange(DropView{r, num}));
});

constexpr inline auto const_ = toGFunc<2>([](auto r, auto)
{
    return r;
});

constexpr inline auto chain = toGFunc<2>([](auto l, auto r)
{
    if constexpr(isRangeV<decltype(l)>)
    {
        static_assert(std::is_same_v<decltype(*l.begin()), std::decay_t<decltype(*r.begin())>>);
        return ownedRange(ChainView{std::move(l), std::move(r)});
    }
    else
    {
        l.insert(l.end(), r.begin(), r.end());
        return l;
    }
});

constexpr inline auto cons = toGFunc<2>([](auto e, auto l)
{
    if constexpr(isRangeV<decltype(l)>)
    {
        static_assert(std::is_same_v<decltype(e), std::decay_t<decltype(*l.begin())>>);
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

template <typename L, typename R>
class Either;

template <typename Data>
class Left final : public DataHolder<Data>
{};

template <typename Data>
class Right final : public DataHolder<Data>
{};

template <typename LData, typename RData>
class Either
{
    using VT = std::variant<LData, RData>;
    VT mData;
public:
    constexpr Either(VT v)
    : mData{std::move(v)}
    {}
    constexpr Either(Left<LData> l)
    : mData{std::move(l.get())}
    {}
    constexpr Either(Right<RData> r)
    : mData{std::move(r.get())}
    {}

    static constexpr auto fromVariant(std::variant<LData, RData> v)
    {
        return Either{std::move(v)};
    }

    constexpr operator std::variant<LData, RData>&& () &&
    {
        return std::move(mData);
    }

    constexpr operator std::variant<LData, RData> const& () const &
    {
        return mData;
    }

    bool isRight() const
    {
        constexpr auto kRIGHT_INDEX = 1;
        return mData.index() == kRIGHT_INDEX;
    }

    auto const& left() const
    {
        return std::get<0>(mData);
    }

    auto const& right() const
    {
        return std::get<1>(mData);
    }
};

constexpr auto toLeft = data::toType<Left>;

constexpr auto toRight = data::toType<Right>;

template <typename T>
constexpr auto cast = toGFunc<1> | [](auto v)
{
    return static_cast<T>(v);
};

constexpr auto null = toGFunc<1> | [](auto v)
{
    return !(v.begin() != v.end());
};

constexpr auto head = toGFunc<1> | [](auto v)
{
    if (!(v.begin() != v.end()))
    {
        throw std::logic_error{"At least one element is needed!"};
    }
    return *v.begin();
};

constexpr auto tail = toGFunc<1> | [](auto v)
{
    if (!(v.begin() != v.end()))
    {
        throw std::logic_error{"At least one element is needed!"};
    }
    return data::drop | 1U | v;
};

constexpr auto last = toGFunc<1> | [](auto v)
{
    if (!(v.begin() != v.end()))
    {
        throw std::logic_error{"At least one element is needed!"};
    }
    auto result = *v.begin();
    for (auto const& e: v)
    {
        result = e;
    }
    return result;
};

constexpr auto init = toGFunc<1> | [](auto v)
{
    constexpr auto length = toGFunc<1> | [](auto v)
    {
        auto i = 0U;
        for (auto const& e: v)
        {
            static_cast<void>(e);
            ++i;
        }
        return i;
    };

    if (!(v.begin() != v.end()))
    {
        throw std::logic_error{"At least one element is needed!"};
    }

    return take | static_cast<size_t>((length | v) - 1) | v;
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

// For io
constexpr auto mapM = toGFunc<2> | [](auto func, auto lst)
{
    return data::io([=]
    {
        using Data = std::decay_t<decltype(func(lst.front()).run())>;
        std::vector<Data> result;
        for (auto e : lst)
        {
            result.emplace_back(func(e).run());
        }
        return result;
    });
};

template <typename Func>
class FunctionTrait;

template <typename Func, typename Ret, typename... Args>
class FunctionTrait<Ret(Func::*)(Args...) const>
{
public:
    using RetT = Ret;
    using ArgsT = std::tuple<Args...>;
    template <size_t I>
    using ArgT = std::tuple_element_t<I, ArgsT>;
    constexpr static auto nbArgsV = sizeof...(Args);
};

template <typename A, typename Func, typename Handler>
constexpr auto catchImpl(data::IO<A, Func> const& io_, Handler handler)
{
    // extract exception type from Hanlder arg
    using HandlerTrait = FunctionTrait<decltype(&Handler::operator())>;
    using ExceptionT = typename HandlerTrait::template ArgT<0>;
    return data::io([=]{
        try
        {
            return io_.run();
        }
        catch (ExceptionT const& e)
        {
            return handler(e).run();
        }
    });
}

constexpr auto catch_ = toGFunc<2> | [](auto io, auto handler)
{
    return catchImpl(io, handler);
};

constexpr auto handle = flip | catch_;

constexpr auto even = toGFunc<1> | [](auto n)
{
    static_assert(std::is_integral_v<decltype(n)>);
    return n % 2 == 0;
};

constexpr auto odd = toGFunc<1> | [](auto n)
{
    return !(even | n);
};

template <typename Func>
class YCombinator
{
    Func mFunc;
public:
    constexpr YCombinator(Func func)
    : mFunc{std::move(func)}
    {}
    template <typename... Args>
    constexpr auto operator()(Args&&... args) const
    {
        return mFunc(*this, args...);
    }
};

constexpr auto yCombinator = toGFunc<1> | [](auto func)
{
    return YCombinator<decltype(func)>{std::move(func)};
};

template <typename Data>
class Product : public data::DataHolder<Data>
{
public:
    using data::DataHolder<Data>::DataHolder;
};

template <typename Data>
class Sum : public data::DataHolder<Data>
{
public:
    using data::DataHolder<Data>::DataHolder;
};

template <typename Data>
class Max : public data::DataHolder<Data>
{
public:
    using data::DataHolder<Data>::DataHolder;
};

template <typename Data>
class Min : public data::DataHolder<Data>
{
public:
    using data::DataHolder<Data>::DataHolder;
};

template <typename Data>
class AllImpl : public data::DataHolder<Data>
{
public:
    using data::DataHolder<Data>::DataHolder;
};

template <typename Data>
class AnyImpl : public data::DataHolder<Data>
{
public:
    using data::DataHolder<Data>::DataHolder;
};

template <typename Data>
class First : public data::DataHolder<data::Maybe<Data>>
{
public:
    using data::DataHolder<data::Maybe<Data>>::DataHolder;
};

template <typename Data>
class Last : public data::DataHolder<data::Maybe<Data>>
{
public:
    using data::DataHolder<data::Maybe<Data>>::DataHolder;
};

template <typename Data>
class ZipList
{
    std::variant<Data, std::list<Data>> mData;
public:
    class Iter
    {
    public:
        constexpr Iter(ZipList const& zipList)
        : mZipList{zipList}
        , mBaseIter{}
        {
            std::visit(overload(
                [](Data) {},
                [this](std::list<Data> const& data)
                {
                    mBaseIter = data.begin();
                }
            ), mZipList.get().mData);
        }
        auto& operator++()
        {
            if (mBaseIter)
            {
                ++mBaseIter.value();
            }
            return *this;
        }
        auto operator*() const
        {
            if (mBaseIter)
            {
                return *mBaseIter.value();
            }
            return std::get<Data>(mZipList.get().mData);
        }
        bool hasValue() const
        {
            return std::visit(overload(
                [](Data const&) { return true; },
                [this](std::list<Data> const& data)
                {
                    return mBaseIter.value() != data.end();
                }
            ), mZipList.get().mData);
        }
    private:
        std::reference_wrapper<ZipList const> mZipList;
        std::optional<std::decay_t<decltype(std::get<1>(mZipList.get().mData).begin())>> mBaseIter;
    };
    class Sentinel
    {};
    friend bool operator!=(Iter const& iter, Sentinel const&)
    {
        return iter.hasValue();
    }
    ZipList(Data data)
    : mData{std::move(data)}
    {}
    ZipList(std::list<Data> data)
    : mData{std::move(data)}
    {}
    auto begin() const
    {
        return Iter(*this);
    }
    auto end() const
    {
        return Sentinel{};
    }
};

template <typename Func>
class Endo : public data::DataHolder<Func>
{
public:
    using data::DataHolder<Func>::DataHolder;
};

constexpr auto toProduct = data::toType<Product>;
constexpr auto toSum = data::toType<Sum>;
constexpr auto toMax = data::toType<Max>;
constexpr auto toMin = data::toType<Min>;
constexpr auto toAll = data::toType<AllImpl>;
constexpr auto toAny = data::toType<AnyImpl>;
constexpr auto toFirst = toGFunc<1>([](auto data)
{
    using Type = std::decay_t<decltype(data.value())>;
    return First<Type>{data};
});
constexpr auto toLast = toGFunc<1>([](auto data)
{
    using Type = std::decay_t<decltype(data.value())>;
    return Last<Type>{data};
});
constexpr auto toZipList = toGFunc<1>([](auto data)
{
    return ZipList{data};
});
constexpr auto toEndo = data::toType<Endo>;

constexpr auto getProduct = data::from;
constexpr auto getSum = data::from;
constexpr auto getMax = data::from;
constexpr auto getMin = data::from;
constexpr auto getAll = data::from;
constexpr auto getAny = data::from;
constexpr auto getFirst = data::from;
constexpr auto getLast = data::from;
constexpr auto getZipList = data::from;
constexpr auto appEndo = data::from;

using All = AllImpl<bool>;
using Any = AnyImpl<bool>;

} // namespace hspp

#endif // HSPP_DATA_H