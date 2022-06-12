/*
 *  Copyright (c) 2022 Bowen Fu
 *  Distributed Under The Apache-2.0 License
 */

#ifndef HSPP_H
#define HSPP_H

#include <optional>
#include <functional>
#include <algorithm>
#include <cassert>
#include <variant>
#include <iterator>
#include <string>
#include <iostream>
#include <sstream>
#include <numeric>
#include <limits>
#include <memory>

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

class Nothing final
{};

template <typename Data>
class Just final
{
public:
    Just(Data d)
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

    Maybe(Data data)
    : std::variant<Nothing, Just<Data>>{Just{std::move(data)}}
    {
    }

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
        return static_cast<F const*>(this)->operator()(std::forward<Arg>(arg));
    }
    template <typename Arg>
    constexpr auto operator||(Arg&& arg) const
    {
        return operator|(arg);
    }
};

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
constexpr auto function(Func const& func)
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
constexpr auto teFunction(Func const& func)
{
    return ToFunction<true, decltype(&Func::operator())>::run(func);
}

template <typename Ret, typename... Args, typename Func>
constexpr auto teFunction(Func const& func)
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
constexpr auto genericFunction(Func const& func)
{
    return GenericFunction<nbArgs, Func>{func};
}

constexpr inline auto id = genericFunction<1>([](auto data)
{
    return std::move(data);
});

template <typename T>
class IsGenericFunction : public std::false_type
{};
template <size_t I, typename T>
class IsGenericFunction<GenericFunction<I, T>> : public std::true_type
{};
template <typename T>
constexpr static auto isGenericFunctionV = IsGenericFunction<std::decay_t<T>>::value;

constexpr auto unCurry = genericFunction<1>([](auto func)
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

template <typename Left, typename Func, typename = std::enable_if_t<isFunctionV<Func> || isGenericFunctionV<Func>, bool>>
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
    template <typename Func, typename Repr, typename Ret, typename InnerArg, typename... Args>
    constexpr auto operator()(Func&& f, Function<Repr, Ret, InnerArg, Args...> const& g) const
    {
        return function([=](InnerArg x){ return f(g(x));});
    }
    template <typename F, typename G>
    constexpr auto operator()(F&& f, G&&g) const
    {
        return genericFunction<1>([=](auto x){ return f(g(x));});
    }
};

constexpr inline auto o = genericFunction<2>(Compose{});

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

template <typename Func>
auto io(Func func)
{
    using Data = std::invoke_result_t<Func>;
    return IO<Data, Func>{func};
}

template <typename Data>
auto ioData(Data data)
{
    return io([data=std::move(data)] { return data; });
}

constexpr auto putStrLn = function([](std::string str)
{
    return io(
        [str=std::move(str)]
        {
            std::cout << str << std::endl;
            return _o_;
        }
    );
});

const inline auto getLine = io([]
    {
        std::string str;
        std::getline(std::cin, str);
        return str;
    }
);

template <typename Data>
class EmptyView
{
public:
    class Iter
    {
    public:
        auto operator++()
        {
        }
        Data operator*() const
        {
            throw std::runtime_error{"Never reach here!"};
        }
        bool hasValue() const
        {
            return false;
        }
    };
    class Sentinel
    {};
    friend bool operator!=(Iter const& iter, Sentinel const&)
    {
        return iter.hasValue();
    }
    constexpr EmptyView() = default;
    auto begin() const
    {
        return Iter{};
    }
    auto end() const
    {
        return Sentinel{};
    }
};

template <typename Base>
class SingleView
{
public:
    constexpr SingleView(Base base)
    : mBase{std::move(base)}
    {}
    auto begin() const
    {
        return &mBase;
    }
    auto end() const
    {
        return begin() + 1;
    }
private:
    Base mBase;
};

template <typename Base>
class RepeatView
{
public:
    class Iter
    {
    public:
        constexpr Iter(RepeatView const& repeatView)
        : mView{repeatView}
        {}
        auto operator++()
        {
        }
        auto operator*() const
        {
            return mView.get().mBase;
        }
        bool hasValue() const
        {
            return true;
        }
    private:
        std::reference_wrapper<RepeatView const> mView;
    };
    class Sentinel
    {};
    friend bool operator!=(Iter const& iter, Sentinel const&)
    {
        return iter.hasValue();
    }
    constexpr RepeatView(Base base)
    : mBase{std::move(base)}
    {}
    auto begin() const
    {
        return Iter(*this);
    }
    auto end() const
    {
        return Sentinel{};
    }
private:
    Base mBase;
};

template <typename Num = int32_t>
class IotaView
{
public:
    class Iter
    {
    public:
        constexpr Iter(Num start, Num end)
        : mNum{start}
        , mBound{end}
        {}
        auto operator++()
        {
            ++mNum;
        }
        auto operator*() const
        {
            return mNum;
        }
        bool hasValue() const
        {
            return mNum < mBound;
        }
    private:
        Num mNum;
        Num mBound;
    };
    class Sentinel
    {};
    friend bool operator!=(Iter const& iter, Sentinel const&)
    {
        return iter.hasValue();
    }
    constexpr IotaView(Num begin, Num end)
    : mBegin{begin}
    , mEnd{end}
    {}
    constexpr IotaView(Num begin)
    : IotaView{begin, std::numeric_limits<Num>::max()}
    {}
    auto begin() const
    {
        return Iter(mBegin, mEnd);
    }
    auto end() const
    {
        return Sentinel{};
    }
private:
    Num mBegin;
    Num mEnd;
};

template <typename Base>
class RefView
{
public:
    constexpr RefView(Base const& base)
    : mBase{base}
    {}
    auto begin() const
    {
        return mBase.get().begin();
    }
    auto end() const
    {
        return mBase.get().end();
    }
private:
    std::reference_wrapper<Base const> mBase;
};

template <typename Base, typename Func>
class MapView
{
public:
    class Iter
    {
    public:
        constexpr Iter(MapView const& mapView)
        : mView{mapView}
        , mBaseIter{mView.get().mBase.begin()}
        {}
        auto operator++()
        {
            ++mBaseIter;
        }
        auto operator*() const
        {
            return mView.get().mFunc(*mBaseIter);
        }
        bool hasValue() const
        {
            return mBaseIter != mView.get().mBase.end();
        }
    private:
        std::reference_wrapper<MapView const> mView;
        std::decay_t<decltype(mView.get().mBase.begin())> mBaseIter;
    };
    class Sentinel
    {};
    friend bool operator!=(Iter const& iter, Sentinel const&)
    {
        return iter.hasValue();
    }
    constexpr MapView(Base base, Func func)
    : mBase{std::move(base)}
    , mFunc{std::move(func)}
    {}
    auto begin() const
    {
        return Iter(*this);
    }
    auto end() const
    {
        return Sentinel{};
    }
private:
    Base mBase;
    Func mFunc;
};

template <typename Base, typename Pred>
class FilterView
{
public:
    class Iter
    {
    private:
        void fixIter()
        {
            while (hasValue() && !mView.get().mPred(*mBaseIter))
            {
                ++mBaseIter;
            };
        }
    public:
        constexpr Iter(FilterView const& filterView)
        : mView{filterView}
        , mBaseIter{mView.get().mBase.begin()}
        {
            fixIter();
        }
        auto operator++()
        {
            ++mBaseIter;
            fixIter();
        }
        auto operator*() const
        {
            return *mBaseIter;
        }
        bool hasValue() const
        {
            return mBaseIter != mView.get().mBase.end();
        }
    private:
        std::reference_wrapper<FilterView const> mView;
        std::decay_t<decltype(mView.get().mBase.begin())> mBaseIter;
    };
    class Sentinel
    {};
    friend bool operator!=(Iter const& iter, Sentinel const&)
    {
        return iter.hasValue();
    }
    constexpr FilterView(Base base, Pred pred)
    : mBase{std::move(base)}
    , mPred{std::move(pred)}
    {}
    auto begin() const
    {
        return Iter(*this);
    }
    auto end() const
    {
        return Sentinel{};
    }
private:
    Base mBase;
    Pred mPred;
};

template <typename Base>
class TakeView
{
public:
    class Iter
    {
    public:
        constexpr Iter(TakeView const& takeView)
        : mView{takeView}
        , mBaseIter{mView.get().mBase.begin()}
        , mCount{}
        {
        }
        auto operator++()
        {
            ++mBaseIter;
            ++mCount;
        }
        auto operator*() const
        {
            return *mBaseIter;
        }
        bool hasValue() const
        {
            return mCount < mView.get().mLimit && mBaseIter != mView.get().mBase.end();
        }
    private:
        std::reference_wrapper<TakeView const> mView;
        std::decay_t<decltype(mView.get().mBase.begin())> mBaseIter;
        size_t mCount;
    };
    class Sentinel
    {};
    friend bool operator!=(Iter const& iter, Sentinel const&)
    {
        return iter.hasValue();
    }
    constexpr TakeView(Base base, size_t number)
    : mBase{std::move(base)}
    , mLimit{number}
    {}
    auto begin() const
    {
        return Iter(*this);
    }
    auto end() const
    {
        return Sentinel{};
    }
private:
    Base mBase;
    size_t mLimit;
};

template <typename Base>
class JoinView
{
public:
    class Iter
    {
    private:
        auto& fetch()
        {
            if (!mCache)
            {
                using T = std::decay_t<decltype(*mCache)>;
                mCache = std::make_unique<T>(std::move(*mBaseIter));
            }
            return *mCache;
        }
        void advanceBase()
        {
            ++mBaseIter;
            mCache.reset();
        }
        void fixIter()
        {
            while (!(mInnerIter != fetch().end()))
            {
                advanceBase();
                if (!hasValue())
                {
                    break;
                }
                mInnerIter = fetch().begin();
            }
        }

    public:
        constexpr Iter(JoinView const& view)
        : mView{view}
        , mBaseIter{mView.get().mBase.begin()}
        , mCache{}
        // This can be invalid if base view is empty, but the compiler requires initialization of mInnerIter.
        , mInnerIter{fetch().begin()}
        {
            fixIter();
        }
        auto operator++()
        {
            ++mInnerIter;
            fixIter();
        }
        auto operator*() const
        {
            return *mInnerIter;
        }
        bool hasValue() const
        {
            return mBaseIter != mView.get().mBase.end();
        }
    private:
        std::reference_wrapper<JoinView const> mView;
        std::decay_t<decltype(mView.get().mBase.begin())> mBaseIter;
        // not thread-safe
        // have to use std::unique_ptr instead of Maybe, because Maybe requrires copy assignments.
        // TODO: use placement new to avoid heap memory allocation.
        mutable std::unique_ptr<std::decay_t<decltype(*mBaseIter)>> mCache;
        std::decay_t<decltype(mCache->begin())> mInnerIter;
    };
    class Sentinel
    {};
    friend bool operator!=(Iter const& iter, Sentinel const&)
    {
        return iter.hasValue();
    }
    constexpr JoinView(Base base)
    : mBase{std::move(base)}
    {}
    auto begin() const
    {
        return Iter(*this);
    }
    auto end() const
    {
        return Sentinel{};
    }
private:
    Base mBase;
};

template <typename... Bases>
class ProductView
{
public:
    class Iter
    {
    public:
        constexpr Iter(ProductView const& view)
        : mView{view}
        , mIters{getBegins(mView.get())}
        {
        }
        auto operator++()
        {
            next();
        }
        auto operator*() const
        {
            return std::apply([](auto&&... iters)
            {
                return std::make_tuple(((*iters))...);
            }, mIters);
        }
        bool hasValue() const
        {
            return std::get<0>(mIters) != std::get<0>(mView.get().mBases).end();
        }
    private:
        static auto getBegins(ProductView const& view)
        {
            auto result = std::apply([](auto&&... views)
            {
                return std::make_tuple((views.begin())...);
            }, view.mBases);
            return result;
        }

        std::reference_wrapper<ProductView const> mView;
        std::decay_t<decltype(getBegins(mView.get()))> mIters;

        template <size_t I = std::tuple_size_v<std::decay_t<decltype(mIters)>> - 1>
        void next()
        {
            auto& iter = std::get<I>(mIters);
            ++iter;
            if constexpr (I != 0)
            {
                auto const view = std::get<I>(mView.get().mBases);
                if (iter == view.end())
                {
                    iter = view.begin();
                    next<I-1>();
                }
            }
        }
    };
    class Sentinel
    {};
    friend bool operator!=(Iter const& iter, Sentinel const&)
    {
        return iter.hasValue();
    }
    constexpr ProductView(Bases... bases)
    : mBases{std::make_tuple(std::move(bases)...)}
    {}
    auto begin() const
    {
        return Iter(*this);
    }
    auto end() const
    {
        return Sentinel{};
    }
private:
    std::tuple<std::decay_t<Bases>...> mBases;
};

template <typename... Bases>
class ZipView
{
public:
    class Iter
    {
    public:
        constexpr Iter(ZipView const& view)
        : mView{view}
        , mIters{getBegins(mView.get())}
        {
        }
        auto operator++()
        {
            std::apply([](auto&&... iters)
            {
                ((++iters), ...);
            }, mIters);
        }
        auto operator*() const
        {
            return std::apply([](auto&&... iters)
            {
                return std::make_tuple(((*iters))...);
            }, mIters);
        }
        bool hasValue() const
        {
            return hasValueImpl();
        }
    private:
        static auto getBegins(ZipView const& view)
        {
            auto result = std::apply([](auto&&... views)
            {
                return std::make_tuple((views.begin())...);
            }, view.mBases);
            return result;
        }

        template <size_t I = 0>
        bool hasValueImpl() const
        {
            if constexpr (I == sizeof...(Bases))
            {
                return true;
            }
            else
            {
                return std::get<I>(mIters) != std::get<I>(mView.get().mBases).end() && hasValueImpl<I+1>();
            }
        }

        std::reference_wrapper<ZipView const> mView;
        std::decay_t<decltype(getBegins(mView.get()))> mIters;
    };
    class Sentinel
    {};
    friend bool operator!=(Iter const& iter, Sentinel const&)
    {
        return iter.hasValue();
    }
    constexpr ZipView(Bases... bases)
    : mBases{std::make_tuple(std::move(bases)...)}
    {}
    auto begin() const
    {
        return Iter(*this);
    }
    auto end() const
    {
        return Sentinel{};
    }
private:
    std::tuple<std::decay_t<Bases>...> mBases;
};

template <typename... Bases>
class ChainView
{
public:
    class Iter
    {
    public:
        constexpr Iter(ChainView const& view)
        : mView{view}
        , mIter{std::get<0>(mView.get().mBases).begin()}
        {
            fixIter();
        }
        auto operator++()
        {
            ++mIter;
            fixIter();
        }
        auto operator*() const
        {
            return *mIter;
        }
        bool hasValue() const
        {
            return mIter != std::get<sizeof...(Bases) - 1>(mView.get().mBases).end();
        }
    private:
        template <size_t I=0>
        void fixIter()
        {
            if constexpr (I >= sizeof...(Bases) - 1)
            {
                return;
            }
            else
            {
                if (mIter == std::get<I>(mView.get().mBases).end())
                {
                    mIter = std::get<I+1>(mView.get().mBases).begin();
                    fixIter<I+1>();
                }
            }
        }

        std::reference_wrapper<ChainView const> mView;
        std::decay_t<decltype(std::get<0>(mView.get().mBases).begin())> mIter;
    };
    class Sentinel
    {};
    friend bool operator!=(Iter const& iter, Sentinel const&)
    {
        return iter.hasValue();
    }
    constexpr ChainView(Bases... bases)
    : mBases{std::make_tuple(std::move(bases)...)}
    {}
    auto begin() const
    {
        return Iter(*this);
    }
    auto end() const
    {
        return Sentinel{};
    }
private:
    std::tuple<std::decay_t<Bases>...> mBases;
};


template <typename Data, typename Repr>
class Range : public Repr
{
};

template <typename Repr>
constexpr auto ownedRange(Repr&& repr)
{
    return Range<std::decay_t<decltype(*repr.begin())>, std::decay_t<Repr>>{std::forward<Repr>(repr)};
}

template <typename Repr>
constexpr auto nonOwnedRange(Repr const& repr)
{
    return ownedRange(RefView(repr));
}

template <template <template<typename...> typename Type, typename... Ts> class TypeClassT,  typename T>
struct TypeClassTrait;

template <template <template<typename...> typename Type, typename... Ts> class TypeClassT, template <typename...> class C, typename... Args>
struct TypeClassTrait<TypeClassT, C<Args...>>
{
    using Type = TypeClassT<C>;
};

template <template <template<typename...> typename Type, typename... Ts> class TypeClassT, typename Repr, typename Ret, typename Arg, typename... Rest>
struct TypeClassTrait<TypeClassT, Function<Repr, Ret, Arg, Rest...>>
{
    using Type = TypeClassT<Function, Arg>;
};

template <typename>
class DummyTemplateClass{};

struct GenericFunctionTag{};

template <template <template<typename...> typename Type, typename... Ts> class TypeClassT, size_t nbArgs, typename Repr>
struct TypeClassTrait<TypeClassT, GenericFunction<nbArgs, Repr>>
{
    using Type = TypeClassT<DummyTemplateClass, GenericFunctionTag>;
};

constexpr auto flip = genericFunction<1>([](auto func)
{
    return genericFunction<2>([=](auto x, auto y){ return func | y |x; });
});

template <typename Iter1, typename Iter2, typename Init, typename Func>
constexpr auto accumulate(Iter1 begin, Iter2 end, Init init, Func func)
{
    for (auto iter = begin; iter != end; ++iter)
    {
        init = func(init, *iter);
    }
    return init;
}

constexpr auto foldr = genericFunction<3>([](auto func, auto init, auto const& list)
{
    return accumulate(list.rbegin(), list.rend(), init, unCurry <o> flip | func);
});

constexpr auto foldl = genericFunction<3>([](auto func, auto init, auto const& list)
{
    return accumulate(list.begin(), list.end(), init, unCurry | func);
});

/////////// Monoid ///////////

template <template<typename...> typename Type, typename... Args>
class Monoid;

template <typename MType>
class MonoidConcat
{
public:
    constexpr static auto mconcat = genericFunction<1>([](auto v) { return foldl | MType::mappend | MType::mempty | v; });
};

template <template<typename...> typename Type, typename... Args>
class MonoidBase
{
public:
    const static Type<Args...> mempty;

    constexpr static auto mappend = function([](Type<Args...> const& lhs, Type<Args...> const& rhs)
    {
        auto const r = ChainView{RefView{lhs}, RefView{rhs}};
        Type<Args...> result;
        for (auto e : r)
        {
            result.insert(result.end(), e);
        }
        return result;
    });
};

template <template<typename...> typename Type, typename... Args>
const Type<Args...> MonoidBase<Type, Args...>::mempty{};

template <typename Data>
class Product
{
    Data mData;
public:
    constexpr Product(Data data)
    : mData{data}
    {}
    auto get() const
    {
        return mData;
    }
};

template <typename T>
constexpr auto operator==(Product<T> l, Product<T> r) 
{
    return l.get() == r.get();
}

constexpr auto product = genericFunction<1>([](auto data)
{
    return Product{data};
});

constexpr auto getProduct = genericFunction<1>([](auto data)
{
    return data.get();
});

template <typename Data>
class MonoidBase<Product, Data>
{
public:
    constexpr static auto mempty = Product<Data>{1};

    constexpr static auto mappend = function([](Product<Data> const& lhs, Product<Data> const& rhs)
    {
        return Product<Data>{lhs.get() * rhs.get()};
    });
};

enum class Ordering
{
    kLT,
    kEQ,
    kGT
};

constexpr auto compare = genericFunction<2>([](auto lhs, auto rhs)
{
    if (lhs < rhs)
    {
        return Ordering::kLT;
    }
    if (lhs == rhs)
    {
        return Ordering::kEQ;
    }
    return Ordering::kGT;
});

template <typename T, typename Enable=void>
struct MonoidTrait;

template <typename T>
using MonoidType = typename MonoidTrait<T>::Type;

template <>
class MonoidBase<DummyTemplateClass, Ordering>
{
public:
    constexpr static auto mempty = Ordering::kEQ;

    constexpr static auto mappend = function([](Ordering lhs, Ordering rhs)
    {
        switch (lhs)
        {
            case Ordering::kLT:
                return Ordering::kLT;
            case Ordering::kEQ:
                return rhs;
            case Ordering::kGT:
                return Ordering::kGT;
        }
        return Ordering::kEQ;
    });
};

template <>
class MonoidBase<DummyTemplateClass, _O_>
{
public:
    constexpr static auto mempty = _o_;

    constexpr static auto mappend = function([](_O_, _O_)
    {
        return _o_;
    });
};

template <template<typename...> typename Type, typename... Args>
class Monoid : public MonoidBase<Type, Args...>, public MonoidConcat<MonoidBase<Type, Args...>>
{};

template <typename Data>
class Monoid<Range, Data>
{
public:
    constexpr static auto mempty = ownedRange(EmptyView<Data>{});

    constexpr static auto mappend = genericFunction<2>([](auto lhs, auto rhs)
    {
        return ownedRange(ChainView{lhs, rhs});
    });

    constexpr static auto mconcat = genericFunction<1>([](auto const& nested)
    {
        return ownedRange(JoinView{nested});
    });
};

template <>
class Monoid<DummyTemplateClass, GenericFunctionTag>
{
private:
    constexpr static auto mconcatTupleImpl()
    {
        return id;
    }
    template <typename Arg, typename... Rest>
    constexpr static auto mconcatTupleImpl(Arg&& arg, Rest&&... rest)
    {
        return arg <o> mconcatTupleImpl(rest...);
    }
    template <typename... Funcs>
    constexpr static auto mconcatImpl(std::tuple<Funcs...> const& nested)
    {
        return std::apply([](auto&&... funcs)
        {
            return mconcatTupleImpl(funcs...);
        }, nested);
    }
    template <typename Func>
    constexpr static auto mconcatImpl(Func&& nested)
    {
        return genericFunction<1>([nested=std::move(nested)](auto arg)
        {
            return nested | arg | arg;
        });
    }
public:
    constexpr static auto mempty = id;

    constexpr static auto mappend = o;

    constexpr static auto mconcat = genericFunction<1>([](auto&& nested)
    {
        return mconcatImpl(nested);
    });
};

template <typename InArg>
class Monoid<Function, InArg>
{
public:
    constexpr static auto mempty = function([](InArg data)
    {
        return std::move(data);
    });

    constexpr static auto mappend = o;

    constexpr static auto mconcat = genericFunction<1>([](auto&& nested)
    {
        return function([nested=std::move(nested)](InArg arg)
        {
            return nested | arg | arg;
        });
    });
};

template <>
class Monoid<std::tuple>
{
public:
    constexpr static auto mempty = _o_;

    constexpr static auto mappend = genericFunction<2>([](auto&& lhs, auto&& rhs)
    {
        return std::tuple_cat(lhs, rhs);
    });

    constexpr static auto mconcat = genericFunction<1>([](auto const& nested)
    {
        return std::apply([](auto&&... funcs)
        {
            return std::tuple_cat(funcs...);
        }
        , nested);
    });
};

template <typename Data>
class MonoidBase<Maybe, Data>
{
public:
    const static Maybe<Data> mempty;

    constexpr static auto mappend = function([](Maybe<Data> const& lhs, Maybe<Data> const& rhs)
    {
        return std::visit(overload(
            [](Just<Data> const& l, Just<Data> const& r) -> Maybe<Data>
            {
                using MType = MonoidType<Data>;
                return Just{MType::mappend | l.data | r.data};
            },
            [&](Nothing, Just<Data> const&)
            {
                return rhs;
            },
            [&](auto, Nothing)
            {
                return lhs;
            }
        ),
        static_cast<MaybeBase<Data> const&>(lhs),
        static_cast<MaybeBase<Data> const&>(rhs));
    });
};

template <typename Data>
const Maybe<Data> MonoidBase<Maybe, Data>::mempty = Nothing{};

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
} // namespace impl

template <template <typename...> class C, typename Data, typename... Rest>
struct MonoidTrait<C<Data, Rest...>, std::enable_if_t<!impl::isTupleLikeV<C<Data, Rest...>> && !isFunctionV<C<Data, Rest...>>, void>>
{
    using Type = Monoid<C, Data>;
};

template <typename... Args>
struct MonoidTrait<std::tuple<Args...>>
{
    using Type = Monoid<std::tuple>;
};

template <>
struct MonoidTrait<Ordering>
{
    using Type = Monoid<DummyTemplateClass, Ordering>;
};

template <>
struct MonoidTrait<_O_>
{
    using Type = Monoid<DummyTemplateClass, _O_>;
};

template <size_t nbArgs, typename Repr>
struct MonoidTrait<GenericFunction<nbArgs, Repr>>
{
    using Type = Monoid<DummyTemplateClass, GenericFunctionTag>;
};

template <typename Repr, typename Ret, typename InnerArg, typename... Rest>
struct MonoidTrait<Function<Repr, Ret, InnerArg, Rest...>>
{
    using Type = Monoid<Function, InnerArg>;
};

class Mappend
{
public:
    template <typename MData1, typename MData2>
    constexpr auto operator()(MData1 const& lhs, MData2 const& rhs) const
    {
        using MType = MonoidType<MData1>;
        return MType::mappend | lhs | rhs;
    }
};

constexpr inline auto mappend = genericFunction<2>(Mappend{});

class Mconcat
{
public:
    template <template <typename...> class C, typename MData, typename... Rest>
    constexpr auto operator()(C<MData, Rest...> const& data) const
    {
        using MType = MonoidType<MData>;
        return MType::mconcat | data;
    }
    template <typename Repr, typename Ret, typename InnerArg, typename... Rest>
    constexpr auto operator()(Function<Repr, Ret, InnerArg, Rest...> const& func) const
    {
        using MType = MonoidType<Function<Repr, Ret, InnerArg, Rest...>>;
        return MType::mconcat | func;
    }
};

constexpr inline auto mconcat = genericFunction<1>(Mconcat{});

/////////// Foldable ///////////

template <template<typename...> typename Type, typename... Args>
class Foldable
{};

/////////// Functor ///////////

template <template<typename...> typename Type, typename... Ts>
class Functor
{
public:
    template <typename Func, typename Arg, typename... Rest>
    constexpr static auto fmap(Func const& func, Type<Arg, Rest...> const& in)
    {
        using R = std::invoke_result_t<Func, Arg>;
        Type<R> result;
        std::transform(in.begin(), in.end(), std::back_inserter(result), [&](auto e){ return func(e); });
        return result;
    }
};

template <>
class Functor<Range>
{
public:
    template <typename Func, typename Arg, typename Repr>
    constexpr static auto fmap(Func const& func, Range<Arg, Repr> const& in)
    {
        return ownedRange(MapView{in, func});
    }
};

template <>
class Functor<std::tuple>
{
public:
    template <typename Func, typename... Args>
    constexpr static auto fmap(Func const& func, std::tuple<Args...> const& in)
    {
        return std::apply([&](auto&&... args)
        {
            return std::make_tuple(func(args)...);
        }, in);
    }
};

template <>
class Functor<Maybe>
{
public:
    template <typename Func, typename Arg>
    constexpr static auto fmap(Func const& func, Maybe<Arg> const& in)
    {
        using R = std::invoke_result_t<Func, Arg>;
        return std::visit(overload(
            [](Nothing) -> Maybe<R>
            {
                return Nothing{};
            },
            [func](Just<Arg> const& j) -> Maybe<R>
            {
                return Just<R>(func(j.data));
            }
        ), static_cast<MaybeBase<Arg>const &>(in));
    }
};

template <typename InnerArg>
class Functor<Function, InnerArg>
{
public:
    template <typename Func, typename Repr, typename Ret, typename... Args>
    constexpr static auto fmap(Func&& func, Function<Repr, Ret, InnerArg, Args...> const& in)
    {
        return o | func | in;
    }
};

template <>
class Functor<DummyTemplateClass, GenericFunctionTag>
{
public:
    template <typename Func, size_t nbArgs, typename Repr>
    constexpr static auto fmap(Func&& func, GenericFunction<nbArgs, Repr> const& in)
    {
        return o | func | in;
    }
};

template <>
class Functor<IO>
{
public:
    template <typename Func1, typename Data, typename Func2>
    constexpr static auto fmap(Func1 const& func, IO<Data, Func2> const& in)
    {
        return io([=]{ return func(in.run()); });
    }
};

template <typename T>
using FunctorType = typename TypeClassTrait<Functor, T>::Type;

class Fmap
{
public:
    template <typename Func, typename Data>
    constexpr auto operator()(Func const& func, Data const& data) const
    {
        using FType = FunctorType<Data>;
        return FType::fmap(func, data);
    }
};

constexpr inline auto fmap = genericFunction<2>(Fmap{});

template <template<typename...> class Type, typename... Ts>
class Applicative : public Functor<Type, Ts...>
{
public:
    template <typename In>
    constexpr static auto pure(In in)
    {
        return Type<std::decay_t<In>>{in};
    }
    template <typename Func, typename Arg>
    constexpr static auto app(Type<Func> const& func, Type<Arg> const& in)
    {
        using R = std::invoke_result_t<Func, Arg>;
        Type<R> result;
        for (auto const& f : func)
        {
            for (auto const& e : in)
            {
                result.push_back(f(e));
            }
        }
        return result;
    }
};

template <>
class Applicative<Range> : public Functor<Range>
{
public:
    template <typename In>
    constexpr static auto pure(In&& in)
    {
        return ownedRange(SingleView{std::forward<In>(in)});
    }
    template <typename Func, typename Arg, typename Repr1, typename Repr2>
    constexpr static auto app(Range<Func, Repr1> const& func, Range<Arg, Repr2> const& in)
    {
        auto view = MapView{ProductView{func, in}, [](auto&& tuple) { return std::get<0>(tuple)(std::get<1>(tuple)); }};
        return ownedRange(view);
    }
};

template <>
class Applicative<std::tuple> : public Functor<std::tuple>
{
public:
    template <typename In>
    constexpr static auto pure(In&& in)
    {
        return std::make_tuple(std::forward<In>(in));
    }
    template <typename... Funcs, typename... Args>
    constexpr static auto app(std::tuple<Funcs...> const& funcs, std::tuple<Args...> const& args)
    {
        return std::apply([&](auto&&... funcs_)
        {
            auto const f = [&](auto&& func)
            {
                return std::apply([&](auto&&... args_)
                {
                    return std::make_tuple(func(args_)...);
                }, args);
            };
            return std::tuple_cat(f(funcs_)...);
        }, funcs);
    }
};

template <>
class Applicative<Maybe> : public Functor<Maybe>
{
public:
    template <typename In>
    constexpr static auto pure(In in)
    {
        return Maybe<std::decay_t<In>>{Just{in}};
    }
    template <typename Func, typename Arg>
    constexpr static auto app(Maybe<Func> const& func, Maybe<Arg> const& in)
    {
        using R = std::invoke_result_t<Func, Arg>;
        return std::visit(overload(
            [](Just<Func> const& f, Just<Arg> const& a) -> Maybe<R>
            {
                return Just<R>{f.data(a.data)};
            },
            [](auto, auto) -> Maybe<R>
            {
                return Nothing{};
            }
        ),
        static_cast<MaybeBase<Func>const &>(func),
        static_cast<MaybeBase<Arg>const &>(in)
        );
    }
};

template <>
class Applicative<IO> : public Functor<IO>
{
public:
    template <typename In>
    constexpr static auto pure(In in)
    {
        return ioData(std::move(in));
    }
    template <typename Func, typename Arg, typename Func1, typename Func2>
    constexpr static auto app(IO<Func, Func1> const& func, IO<Arg, Func2> const& in)
    {
        return io([=]{ return func.run()(in.run()); });
    }
};

template <typename InnerArg>
class Applicative<Function, InnerArg> : public Functor<Function, InnerArg>
{
public:
    template <typename Ret>
    constexpr static auto pure(Ret ret)
    {
        return function([ret=std::move(ret)](InnerArg){ return ret; });
    }
    template <typename Func1, typename Func2>
    constexpr static auto app(Func1 func, Func2 in)
    {
        return function(
            [func=std::move(func), in=std::move(in)](InnerArg arg)
            {
                return func(arg)(in(arg));
            });
    }
};

template <>
class Applicative<DummyTemplateClass, GenericFunctionTag> : public Functor<DummyTemplateClass, GenericFunctionTag>
{
public:
    template <typename Ret>
    constexpr static auto pure(Ret ret)
    {
        return genericFunction<1>([=](auto){ return ret; });
    }
    template <typename Func1, typename Func2>
    constexpr static auto app(Func1 func, Func2 in)
    {
        return genericFunction<1>([f=std::move(func), g=std::move(in)](auto arg) {return f(arg)(g(arg)); });
    }
};

template <typename T>
using ApplicativeType = typename TypeClassTrait<Applicative, T>::Type;

template <typename Data>
class DeferredPure
{
public:
    Data mData;
};

template <typename Data>
constexpr auto pureImpl(Data const& data)
{
    return DeferredPure<impl::StoreT<Data>>{data};
}

constexpr auto pure = genericFunction<1>([](auto const& data)
{
    return pureImpl(data);
}
);

constexpr auto return_ = pure;

template <typename T>
class IsDeferredPure : public std::false_type
{};
template <typename T>
class IsDeferredPure<DeferredPure<T>> : public std::true_type
{};
template <typename T>
constexpr static auto isDeferredPureV = IsDeferredPure<T>::value;

class App
{
public:
    template <typename Func, typename Data>
    constexpr auto operator()(DeferredPure<Func> const& func, Data const& data) const
    {
        using AppType = ApplicativeType<Data>;
        return AppType::app(AppType::pure(func.mData), data);
    }
    template <typename Func, typename Data>
    constexpr auto operator()(Func const& func, DeferredPure<Data> const& in) const
    {
        using AppType = ApplicativeType<Func>;
        return AppType::app(func, AppType::pure(in.mData));
    }
    template <typename Func, typename Data>
    constexpr auto operator()(Func const& func, Data const& in) const
    {
        using AppType1 = ApplicativeType<Func>;
        using AppType2 = ApplicativeType<Data>;
        static_assert(std::is_same_v<AppType1, AppType2>);
        return AppType1::app(func, in);
    }
};

constexpr inline auto app = genericFunction<2>(App{});

/////////// Monad ///////////

template <template<typename...> class Type, typename... Ts>
class MonadBase;

template <typename MonadB>
class MonadRShift
{
public:
    constexpr static auto rshift = genericFunction<2>
    ([](auto x, auto y){
        return MonadB::bind(x, [y](auto) { return y; });
    });
};

// Works for Function as well.
template <template<typename...> class Type, typename... Ts>
class Monad : public Applicative<Type, Ts...>, public MonadBase<Type, Ts...>, public MonadRShift<MonadBase<Type, Ts...>>
{
public:
    template <typename Arg>
    constexpr static auto return_(Arg arg)
    {
        return Applicative<Type, Ts...>::pure(arg);
    }
};

template <template<typename...> class Type, typename... Ts>
class MonadBase
{
public:
    template <typename... Args, typename Func>
    constexpr static auto bind(Type<Args...> const& arg, Func const& func)
    {
        return mconcat || fmap | func | arg;
    }
};

template <>
class MonadBase<Maybe>
{
public:
    template <typename Arg, typename Func>
    constexpr static auto bind(Maybe<Arg> const& arg, Func const& func)
    {
        using R = std::invoke_result_t<Func, Arg>;
        return std::visit(overload(
            [](Nothing) -> R
            {
                return Nothing{};
            },
            [func](Just<Arg> const& j) -> R
            {
                return func(j.data);
            }
        ), static_cast<MaybeBase<Arg> const&>(arg));
    }
};

template <>
class MonadBase<IO>
{
public:
    template <typename Arg, typename Func1, typename Func>
    constexpr static auto bind(IO<Arg, Func1> const& arg, Func const& func)
    {
        return io([=]{ return func(arg.run()).run(); });
    }
};

template <>
class MonadBase<DummyTemplateClass, GenericFunctionTag>
{
public:
    template <size_t nbArgs, typename Repr, typename Func>
    constexpr static auto bind(GenericFunction<nbArgs, Repr> const& arg, Func func)
    {
        return mconcat || fmap | func | arg;
    }
};

template <typename T>
using MonadType = typename TypeClassTrait<Monad, T>::Type;

/////////// MonadPlus //////////

template <template<typename...> class Type, typename... Ts>
class MonadPlus : public Monoid<Type, Ts...>
{
public:
    const static decltype(Monoid<Type, Ts...>::mempty) mzero;
    constexpr static auto mplus = Monoid<Type, Ts...>::mappend;
};

template <template<typename...> typename Type, typename... Args>
const decltype(Monoid<Type, Args...>::mempty) MonadPlus<Type, Args...>::mzero = Monoid<Type, Args...>::mempty;

template <typename T>
struct MonadPlusTrait;

template <template <typename...> class C, typename Data, typename... Rest>
struct MonadPlusTrait<C<Data, Rest...>>
{
    using Type = MonadPlus<C, Data>;
};

template <typename T>
using MonadPlusType = typename MonadPlusTrait<T>::Type;

template <template <typename...> class ClassT>
constexpr auto guardImpl(bool b)
{
    return b ? MonadType<ClassT<_O_>>::return_( _o_) : MonadPlusType<ClassT<_O_>>::mzero;
}

template <>
constexpr auto guardImpl<Range>(bool b)
{
    return ownedRange(FilterView{Monad<Range>::return_(_o_), [b](auto){ return b; }});
}

template <template <typename...> class ClassT>
constexpr auto guard = function([](bool b)
{
    return guardImpl<ClassT>(b);
});

constexpr auto show = genericFunction<1>([](auto&& d)
{
    std::stringstream os;
    os << std::boolalpha << d;
    return os.str();
});

template <typename MType, typename T>
constexpr auto evalDeferredImpl(T&& t)
{
    return std::forward<T>(t);
}

template <typename MType, typename T>
constexpr auto evalDeferredImpl(DeferredPure<T> t)
{
    return MType::return_(t.mData);
}

template <typename MType>
constexpr auto evalDeferred = genericFunction<1>([](auto&& d)
{
    return evalDeferredImpl<MType>(d);
});

// >>= is right-assocative in C++, have to add some parens when chaining the calls.
template <typename Arg, typename Func, typename Ret = std::invoke_result_t<Func, Arg>>
constexpr auto operator>>=(DeferredPure<Arg> const& arg, Func const& func)
{
    using MType = MonadType<Ret>;
    return MType::bind(evalDeferred<MType> | arg, func);
}

template <typename MonadData, typename Func>
constexpr auto operator>>=(MonadData const& data, Func const& func)
{
    using MType = MonadType<MonadData>;
    return MType::bind(data, evalDeferred<MType> <o> func);
}

template <typename Arg, typename MonadData>
constexpr auto operator>>(DeferredPure<Arg> const& arg, MonadData const& data)
{
    using MType = MonadType<MonadData>;
    return MType::rshift(evalDeferred<MType> | arg, data);
}

template <typename MonadData1, typename MonadData2, typename MType = MonadType<MonadData1>>
constexpr auto operator>>(MonadData1 const& lhs, MonadData2 const& rhs)
{
    return MType::rshift | lhs || evalDeferred<MType> | rhs;
}

constexpr inline auto elem = genericFunction<2>([](auto t, auto const& c)
{
    return std::any_of(c.begin(), c.end(), [t=std::move(t)](auto const& e){return e == t;});
});

template <typename Repr1, typename Ret1, typename Arg1, typename... Rest1, typename Repr2, typename Ret2, typename Arg2, typename... Rest2>
constexpr auto onImpl(Function<Repr1, Ret1, Arg1, Rest1...> f, Function<Repr2, Ret2, Arg2, Rest2...> g)
{
    return function<Ret1, Arg2, Arg2>([f=std::move(f), g=std::move(g)](Arg2 x, Arg2 y) { return f | g(x) | g(y); });
}

constexpr inline auto on = genericFunction<2>([](auto f, auto g)
{
    return onImpl(std::move(f), std::move(g));
});

constexpr auto toVector = genericFunction<1>([](auto view)
{
    std::vector<std::decay_t<decltype(*view.begin())>> result;
    for (auto e: view)
    {
        result.push_back(std::move(e));
    }
    return result;
});

constexpr inline auto filter = genericFunction<2>([](auto pred, auto data)
{
    return ownedRange(FilterView{std::move(data), std::move(pred)});
});

constexpr inline auto map = genericFunction<2>([](auto func, auto data)
{
    return ownedRange(MapView{std::move(data), std::move(func)});
});

constexpr inline auto repeat = genericFunction<1>([](auto data)
{
    return ownedRange(RepeatView{std::move(data)});
});

constexpr inline auto replicate = genericFunction<2>([](auto data, size_t times)
{
    return ownedRange(TakeView{RepeatView{std::move(data)}, times});
});

#endif // HSPP_H

// =======
// tuple as Monad?