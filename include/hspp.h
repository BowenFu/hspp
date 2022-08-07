/*
 *  Copyright (c) 2022 Bowen Fu
 *  Distributed Under The Apache-2.0 License
 */

#ifndef HSPP_RANGE_H
#define HSPP_RANGE_H

#include <stdexcept>
#include <limits>
#include <memory>
#include <tuple>

namespace hspp
{
namespace data
{

template <typename Data>
class EmptyView
{
public:
    class Iter
    {
    public:
        auto& operator++()
        {
            return *this;
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
    class Iter
    {
    public:
        constexpr Iter(SingleView const& singleView)
        : mView{singleView}
        , mHasValue{true}
        {}
        auto& operator++()
        {
            mHasValue = false;
            return *this;
        }
        auto operator*() const
        {
            return mView.get().mBase;
        }
        bool hasValue() const
        {
            return mHasValue;
        }
    private:
        std::reference_wrapper<SingleView const> mView;
        bool mHasValue;
    };
    class Sentinel
    {};
    friend bool operator!=(Iter const& iter, Sentinel const&)
    {
        return iter.hasValue();
    }
    constexpr SingleView(Base base)
    : mBase{std::move(base)}
    {}
    auto begin() const
    {
        return Iter{*this};
    }
    auto end() const
    {
        return Sentinel{};
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
        auto& operator++()
        {
            return *this;
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
        auto& operator++()
        {
            ++mNum;
            return *this;
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
        auto& operator++()
        {
            ++mBaseIter;
            return *this;
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
        auto& operator++()
        {
            ++mBaseIter;
            fixIter();
            return *this;
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
        auto& operator++()
        {
            ++mBaseIter;
            ++mCount;
            return *this;
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
class DropView
{
public:
    class Iter
    {
    public:
        constexpr Iter(DropView const& dropView)
        : mView{dropView}
        , mBaseIter{mView.get().mBase.begin()}
        {
            for (size_t i = 0; i < mView.get().mNum; ++i)
            {
                if (!hasValue())
                {
                    break;
                }
                ++mBaseIter;
            }
        }
        auto& operator++()
        {
            ++mBaseIter;
            return *this;
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
        std::reference_wrapper<DropView const> mView;
        std::decay_t<decltype(mView.get().mBase.begin())> mBaseIter;
    };
    class Sentinel
    {};
    friend bool operator!=(Iter const& iter, Sentinel const&)
    {
        return iter.hasValue();
    }
    constexpr DropView(Base base, size_t number)
    : mBase{std::move(base)}
    , mNum{number}
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
    size_t mNum;
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
        auto& operator++()
        {
            ++mInnerIter;
            fixIter();
            return *this;
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

namespace impl
{
template <typename... Bases>
auto getBegins(std::tuple<Bases...> const& bases)
{
    auto result = std::apply([](auto&&... views)
    {
        return std::make_tuple((views.begin())...);
    }, bases);
    return result;
}
}

template <typename... Bases>
class ProductView
{
public:
    class Iter
    {
    public:
        constexpr Iter(ProductView const& view)
        : mView{view}
        , mIters{impl::getBegins(mView.get().mBases)}
        {
        }
        auto& operator++()
        {
            next();
            return *this;
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

        std::reference_wrapper<ProductView const> mView;
        std::decay_t<decltype(impl::getBegins(mView.get().mBases))> mIters;

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
        , mIters{impl::getBegins(mView.get().mBases)}
        {
        }
        auto& operator++()
        {
            std::apply([](auto&&... iters)
            {
                ((++iters), ...);
            }, mIters);
            return *this;
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
        std::decay_t<decltype(impl::getBegins(mView.get().mBases))> mIters;
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
        , mIters{impl::getBegins(mView.get().mBases)}
        {
        }
        auto& operator++()
        {
            next();
            return *this;
        }
        auto operator*() const
        {
            return deref();
        }
        bool hasValue() const
        {
            return hasValueImpl();
        }
    private:
        std::reference_wrapper<ChainView const> mView;
        std::decay_t<decltype(impl::getBegins(mView.get().mBases))> mIters;

        template <size_t I = 0>
        auto deref() const
        {
            auto& iter = std::get<I>(mIters);
            auto const& view = std::get<I>(mView.get().mBases);
            if (iter != view.end())
            {
                return *iter;
            }
            constexpr auto nbIters = std::tuple_size_v<std::decay_t<decltype(mIters)>>;
            if constexpr (I < nbIters-1)
            {
                return deref<I+1>();
            }
            throw std::runtime_error{"Never reach here!"};
        }

        template <size_t I = 0>
        bool hasValueImpl() const
        {
            if constexpr (I == sizeof...(Bases))
            {
                return false;
            }
            else
            {
                return std::get<I>(mIters) != std::get<I>(mView.get().mBases).end() || hasValueImpl<I+1>();
            }
        }

        template <size_t I = 0>
        void next()
        {
            auto& iter = std::get<I>(mIters);
            auto const view = std::get<I>(mView.get().mBases);
            if (iter != view.end())
            {
                ++iter;
                return;
            }
            constexpr auto nbIters = std::tuple_size_v<std::decay_t<decltype(mIters)>>;
            if constexpr (I < nbIters-1)
            {
                next<I+1>();
            }
        }
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

template <typename... Ts>
class IsRange : public std::false_type
{};
template <typename... Ts>
class IsRange<Range<Ts...>> : public std::true_type
{};
template <typename T>
constexpr static auto isRangeV = IsRange<std::decay_t<T>>::value;

static_assert(!isRangeV<SingleView<int32_t>>);
static_assert(isRangeV<Range<int32_t, SingleView<int32_t>>>);

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

} // namespace data
} // namespace hspp

#endif // HSPP_RANGE_H
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

} // namespace hspp

#endif // HSPP_DATA_H
/*
 *  Copyright (c) 2022 Bowen Fu
 *  Distributed Under The Apache-2.0 License
 */

#ifndef HSPP_TYPECLASS_H
#define HSPP_TYPECLASS_H

#include <optional>
#include <functional>
#include <algorithm>
#include <iterator>
#include <string>
#include <iostream>
#include <sstream>
#include <numeric>
#include <type_traits>

namespace hspp
{

/////////////// Data Trait //////////////////
template <typename T>
struct DataTrait;

template <template <typename...> class Class, typename Data, typename... Rest>
struct DataTrait<Class<Data, Rest...>>
{
    using Type = Data;
    template <typename DataT>
    using ReplaceDataTypeWith = Class<DataT>;
};

template <typename A, typename Repr>
struct DataTrait<data::Parser<A, Repr>>
{
    using Type = A;
};

template <typename A, typename Repr>
struct DataTrait<data::Range<A, Repr>>
{
    using Type = A;
};

template <typename T>
using DataType = typename DataTrait<std::decay_t<T>>::Type;

template <typename T, typename Data>
using ReplaceDataTypeWith = typename DataTrait<std::decay_t<T>>::template ReplaceDataTypeWith<Data>;

/////////////// TypeClass Traits //////////////////

template <template <template<typename...> typename Type, typename... Ts> class TypeClassT,  typename T>
struct TypeClassTrait;

template <template <template<typename...> typename Type, typename... Ts> class TypeClassT, typename... Args>
struct TypeClassTrait<TypeClassT, std::vector<Args...>>
{
    using Type = TypeClassT<std::vector>;
};

template <template <template<typename...> typename Type, typename... Ts> class TypeClassT, typename... Args>
struct TypeClassTrait<TypeClassT, std::list<Args...>>
{
    using Type = TypeClassT<std::list>;
};

template <template <template<typename...> typename Type, typename... Ts> class TypeClassT, typename... Args>
struct TypeClassTrait<TypeClassT, std::basic_string<Args...>>
{
    using Type = TypeClassT<std::basic_string>;
};

template <template <template<typename...> typename Type, typename... Ts> class TypeClassT, typename... Args>
struct TypeClassTrait<TypeClassT, data::Range<Args...>>
{
    using Type = TypeClassT<data::Range>;
};

template <template <template<typename...> typename Type, typename... Ts> class TypeClassT, typename... Args>
struct TypeClassTrait<TypeClassT, data::Maybe<Args...>>
{
    using Type = TypeClassT<data::Maybe>;
};

template <template <template<typename...> typename Type, typename... Ts> class TypeClassT, typename... Args>
struct TypeClassTrait<TypeClassT, data::IO<Args...>>
{
    using Type = TypeClassT<data::IO>;
};

template <template <template<typename...> typename Type, typename... Ts> class TypeClassT, typename Repr, typename Ret, typename Arg, typename... Rest>
struct TypeClassTrait<TypeClassT, data::Function<Repr, Ret, Arg, Rest...>>
{
    using Type = TypeClassT<data::Function, Arg>;
};

template <template <template<typename...> typename Type, typename... Ts> class TypeClassT, typename A, typename Repr>
struct TypeClassTrait<TypeClassT, data::Parser<A, Repr>>
{
    using Type = TypeClassT<data::Parser>;
};

template <typename>
class DummyTemplateClass{};

struct GenericFunctionTag{};

template <template <template<typename...> typename Type, typename... Ts> class TypeClassT, size_t nbArgs, typename Repr>
struct TypeClassTrait<TypeClassT, data::GenericFunction<nbArgs, Repr>>
{
    using Type = TypeClassT<DummyTemplateClass, GenericFunctionTag>;
};

namespace impl
{
template <template <template<typename...> typename Type, typename... Ts> class TypeClassT, typename>
struct TupleTypeClassTrait;

template <template <template<typename...> typename Type, typename... Ts> class TypeClassT, typename... Init>
struct TupleTypeClassTrait<TypeClassT, std::tuple<Init...>>
{
    using Type = TypeClassT<std::tuple, std::decay_t<Init>...>;
};
} // namespace impl

template <template <template<typename...> typename Type, typename... Ts> class TypeClassT, typename... Args>
struct TypeClassTrait<TypeClassT, std::tuple<Args...>>
{
    using Type = typename impl::TupleTypeClassTrait<TypeClassT, TakeTupleType<sizeof...(Args)-1, std::tuple<Args...>>>::Type;
};

/////////////// Functor Traits ///////////////

template <template<typename...> typename Type, typename... Ts>
class Functor;

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

constexpr inline auto fmap = toGFunc<2>(Fmap{});

template <template<typename...> typename Type, typename... Args>
class Foldable;

template <typename T>
using FoldableType = typename TypeClassTrait<Foldable, std::decay_t<T>>::Type;

class Fold
{
public:
    template <template <typename...> class T, typename MData, typename... Rest, typename FType = FoldableType<T<MData, Rest...>>>
    constexpr auto operator()(T<MData, Rest...> const& data) const
    {
        return FType::fold | data;
    }
};

constexpr inline auto fold = toGFunc<1>(Fold{});

class Foldr
{
public:
    template <typename Func, typename Init, template <typename...> class T, typename MData, typename... Rest, typename FType = FoldableType<T<MData, Rest...>>>
    constexpr auto operator()(Func func, Init init, T<MData, Rest...> const& list) const
    {
        return FType::foldr | func | init | list;
    }
};

constexpr inline auto foldr = toGFunc<3>(Foldr{});

class FoldMap
{
public:
    template <typename Func, template <typename...> class T, typename MData, typename... Rest, typename FType = FoldableType<T<MData, Rest...>>>
    constexpr auto operator()(Func&& func, T<MData, Rest...> const& data) const
    {
        return FType::foldMap | func | data;
    }
};

constexpr inline auto foldMap = toGFunc<2>(FoldMap{});

////////// Traversable ////////////

template <template<typename...> typename Type, typename... Args>
class Traversable;

template <typename T>
using TraversableType = typename TypeClassTrait<Traversable, std::decay_t<T>>::Type;

class Traverse
{
public:
    template <typename Func, template <typename...> class T, typename MData, typename... Rest, typename TType = TraversableType<T<MData, Rest...>>>
    constexpr auto operator()(Func&& f, T<MData, Rest...> const& data) const
    {
        return TType::traverse | f | data;
    }
};

constexpr inline auto traverse = toGFunc<2>(Traverse{});

class SequenceA
{
public:
    template <template <typename...> class T, typename MData, typename... Rest, typename TType = TraversableType<T<MData, Rest...>>>
    constexpr auto operator()(T<MData, Rest...> const& data) const
    {
        return TType::sequenceA | data;
    }
};

constexpr inline auto sequenceA = toGFunc<1>(SequenceA{});

/////////////// Monoid Traits ///////////////

template <template<typename...> typename Type, typename... Args>
class Monoid;

template <typename T>
struct MonoidTrait;

template <typename T>
using MonoidType = typename MonoidTrait<std::decay_t<T>>::Type;

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

constexpr inline auto mappend = toGFunc<2>(Mappend{});

class Mconcat
{
public:
    template <template <typename...> class C, typename MData, typename... Rest>
    constexpr auto operator()(C<MData, Rest...> const& data) const
    {
        using MType = MonoidType<MData>;
        return MType::mconcat | data;
    }
    template <typename Repr, typename Ret, typename FirstArg, typename... Rest>
    constexpr auto operator()(data::Function<Repr, Ret, FirstArg, Rest...> const& func) const
    {
        using MType = MonoidType<data::Function<Repr, Ret, FirstArg, Rest...>>;
        return MType::mconcat | func;
    }
};

constexpr inline auto mconcat = toGFunc<1>(Mconcat{});

/////////////// Monoid ///////////////

template <typename MType>
class MonoidConcat
{
public:
    constexpr static auto mconcat = toGFunc<1>([](auto v) { return data::foldl | MType::mappend | MType::mempty | v; });
};

template <template<typename...> typename Type, typename... Args>
class MonoidBase{};

template <template<typename...> typename Type, typename... Args>
class ContainerMonoidBase
{
public:
    const static Type<Args...> mempty;

    constexpr static auto mappend = data::toFunc<>([](Type<Args...> const& lhs, Type<Args...> const& rhs)
    {
        auto const r = data::ChainView{data::RefView{lhs}, data::RefView{rhs}};
        Type<Args...> result;
        for (auto e : r)
        {
            result.insert(result.end(), e);
        }
        return result;
    });
};

template <template<typename...> typename Type, typename... Args>
const Type<Args...> ContainerMonoidBase<Type, Args...>::mempty{};

template <typename... Args>
class MonoidBase<std::vector, Args...> : public ContainerMonoidBase<std::vector, Args...>{};

template <typename... Args>
class MonoidBase<std::list, Args...> : public ContainerMonoidBase<std::list, Args...>{};

template <typename... Args>
class MonoidBase<std::basic_string, Args...> : public ContainerMonoidBase<std::basic_string, Args...>{};

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
constexpr auto getAll = data::from;
constexpr auto getAny = data::from;
constexpr auto getFirst = data::from;
constexpr auto getLast = data::from;
constexpr auto getZipList = data::from;
constexpr auto appEndo = data::from;

using All = AllImpl<bool>;
using Any = AnyImpl<bool>;

template <typename Data>
class MonoidBase<Product, Data>
{
public:
    constexpr static auto mempty = Product<Data>{1};

    constexpr static auto mappend = data::toFunc<>([](Product<Data> const& lhs, Product<Data> const& rhs)
    {
        return Product<Data>{lhs.get() * rhs.get()};
    });
};

template <typename Data>
class MonoidBase<Sum, Data>
{
public:
    constexpr static auto mempty = Sum<Data>{0};

    constexpr static auto mappend = data::toFunc<>([](Sum<Data> const& lhs, Sum<Data> const& rhs)
    {
        return Sum<Data>{lhs.get() + rhs.get()};
    });
};

template <>
class MonoidBase<DummyTemplateClass, Any>
{
public:
    constexpr static auto mempty = Any{false};

    constexpr static auto mappend = data::toFunc<>([](Any lhs, Any rhs)
    {
        return Any{lhs.get() || rhs.get()};
    });
};

template <>
class MonoidBase<DummyTemplateClass, All>
{
public:
    constexpr static auto mempty = All{true};

    constexpr static auto mappend = data::toFunc<>([](All lhs, All rhs)
    {
        return All{lhs.get() && rhs.get()};
    });
};

template <typename Data>
class MonoidBase<First, Data>
{
public:
    constexpr static auto mempty = First<Data>{data::nothing};

    constexpr static auto mappend = data::toFunc<>([](First<Data> lhs, First<Data> rhs)
    {
        return (getFirst | lhs) == data::nothing ? rhs : lhs;
    });
};

template <typename Data>
class MonoidBase<Last, Data>
{
public:
    constexpr static auto mempty = Last<Data>{data::nothing};

    constexpr static auto mappend = data::toFunc<>([](Last<Data> lhs, Last<Data> rhs)
    {
        return (getLast | rhs) == data::nothing ? lhs : rhs;
    });
};

template <typename Data>
class MonoidBase<ZipList, Data>
{
    using DataMType = MonoidType<Data>;
public:
    const static ZipList<Data> mempty;

    constexpr static auto mappend = toGFunc<2>([](auto lhs, auto rhs)
    {
        return toZipList <o> data::to<std::list> || data::zipWith | DataMType::mappend | data::nonOwnedRange(lhs) | data::nonOwnedRange(rhs);
    });
};

template <typename Data>
const ZipList<Data> MonoidBase<ZipList, Data>::mempty = ZipList{MonoidType<Data>::mempty};

template <>
class MonoidBase<Endo>
{
public:
    constexpr static auto mempty = toEndo | id;

    constexpr static auto mappend = data::toGFunc<2>([](auto&& lhs, auto&& rhs)
    {
        return toEndo || lhs.get() <o> rhs.get();
    });
};

enum class Ordering
{
    kLT,
    kEQ,
    kGT
};

constexpr auto compare = toGFunc<2>([](auto lhs, auto rhs)
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

template <>
class MonoidBase<DummyTemplateClass, Ordering>
{
public:
    constexpr static auto mempty = Ordering::kEQ;

    constexpr static auto mappend = data::toFunc<>([](Ordering lhs, Ordering rhs)
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

    constexpr static auto mappend = data::toFunc<>([](_O_, _O_)
    {
        return _o_;
    });
};

template <template<typename...> typename Type, typename... Args>
class Monoid : public MonoidBase<Type, Args...>, public MonoidConcat<MonoidBase<Type, Args...>>
{};

template <typename Data>
class Monoid<data::Range, Data>
{
public:
    constexpr static auto mempty = data::ownedRange(data::EmptyView<Data>{});

    constexpr static auto mappend = toGFunc<2>([](auto lhs, auto rhs)
    {
        return data::ownedRange(data::ChainView{lhs, rhs});
    });

    constexpr static auto mconcat = toGFunc<1>([](auto const& nested)
    {
        if constexpr (data::isTupleLikeV<decltype(nested)>)
        {
            return std::apply([](auto&&... rngs)
            {
                return data::ChainView{rngs...};
            }
            , nested);
        }
        else
        {
            return data::ownedRange(data::JoinView{nested});
        }
    });
};

template <typename GFunc>
class MonoidBase<DummyTemplateClass, GenericFunctionTag, GFunc>
{
public:
    constexpr static auto mempty = toGFunc<1>([](auto x)
    {
        using RetType = std::invoke_result_t<GFunc, decltype(x)>;
        return MonoidType<RetType>::mempty;
    });

    constexpr static auto mappend = toGFunc<2>([](auto f, auto g)
    {
        return toGFunc<1>([f=std::move(f), g=std::move(g)](auto arg)
        {
            using RetType = std::invoke_result_t<GFunc, decltype(arg)>;
            using MType = MonoidType<RetType>;
            return (f | arg) <MType::mappend> (g | arg);
        });
    });
};

template <typename InArg, typename RetType>
class MonoidBase<data::Function, InArg, RetType>
{
public:
    constexpr static auto mempty = data::toFunc<>([](InArg)
    {
        return MonoidType<RetType>::mempty;
    });

    constexpr static auto mappend = toGFunc<2>([](auto f, auto g)
    {
        return data::toFunc<>([f=std::move(f), g=std::move(g)](InArg arg)
        {
            using MType = MonoidType<RetType>;
            return (f | arg) <MType::mappend> (g | arg);
        });
    });
};

namespace impl
{
template <size_t... I, typename Func, typename Tuple1, typename Tuple2>
constexpr static auto zipTupleWithImpl(Func func, Tuple1&& lhs, Tuple2&& rhs, std::index_sequence<I...>)
{
    static_assert(std::tuple_size_v<std::decay_t<Tuple1>> == sizeof...(I));
    static_assert(std::tuple_size_v<std::decay_t<Tuple2>> == sizeof...(I));
    return std::make_tuple((func | std::get<I>(std::forward<Tuple1>(lhs)) | std::get<I>(std::forward<Tuple2>(rhs)))...);
}
}

template <typename Func, typename Tuple1, typename Tuple2>
constexpr static auto zipTupleWith(Func func, Tuple1&& lhs, Tuple2&& rhs)
{
    return impl::zipTupleWithImpl(func, std::forward<Tuple1>(lhs), std::forward<Tuple2>(rhs), std::make_index_sequence<std::tuple_size_v<std::decay_t<Tuple1>>>{});
}

template <typename... Ts>
class MonoidBase<std::tuple, Ts...>
{
public:
    const static decltype(std::make_tuple(MonoidType<Ts>::mempty...)) mempty;

    constexpr static auto mappend = data::toFunc<>([](std::tuple<Ts...> lhs, std::tuple<Ts...> rhs)
    {
        constexpr auto mapp = toGFunc<2>(Mappend{});
        return zipTupleWith(mapp, lhs, rhs);
    });
};

template <typename... Ts>
const decltype(std::make_tuple(MonoidType<Ts>::mempty...)) MonoidBase<std::tuple, Ts...>::mempty = std::make_tuple(MonoidType<Ts>::mempty...);

template <typename Data>
class MonoidBase<data::Maybe, Data>
{
public:
    const static data::Maybe<Data> mempty;

    constexpr static auto mappend = data::toFunc<>([](data::Maybe<Data> const& lhs, data::Maybe<Data> const& rhs)
    {
        return std::visit(overload(
            [](data::Just<Data> const& l, data::Just<Data> const& r) -> data::Maybe<Data>
            {
                using MType = MonoidType<Data>;
                return data::Just{MType::mappend | l.data | r.data};
            },
            [&](data::Nothing, data::Just<Data> const&)
            {
                return rhs;
            },
            [&](auto, data::Nothing)
            {
                return lhs;
            }
        ),
        static_cast<data::MaybeBase<Data> const&>(lhs),
        static_cast<data::MaybeBase<Data> const&>(rhs));
    });
};

template <typename Data>
const data::Maybe<Data> MonoidBase<data::Maybe, Data>::mempty = data::nothing;

template <template <typename...> class C, typename Data>
struct MonoidTraitImpl
{
    using Type = Monoid<C, Data>;
};

template <typename Data, typename... Rest>
struct MonoidTrait<std::vector<Data, Rest...>> : MonoidTraitImpl<std::vector, Data> {};

template <typename Data, typename... Rest>
struct MonoidTrait<std::list<Data, Rest...>> : MonoidTraitImpl<std::list, Data> {};

template <typename Data, typename... Rest>
struct MonoidTrait<std::basic_string<Data, Rest...>> : MonoidTraitImpl<std::basic_string, Data> {};

template <typename Data, typename... Rest>
struct MonoidTrait<data::Range<Data, Rest...>> : MonoidTraitImpl<data::Range, Data> {};

template <typename Data>
struct MonoidTrait<data::Maybe<Data>> : MonoidTraitImpl<data::Maybe, Data> {};

template <typename Data>
struct MonoidTrait<ZipList<Data>> : MonoidTraitImpl<ZipList, Data> {};

template <typename Data>
struct MonoidTrait<Sum<Data>> : MonoidTraitImpl<Sum, Data> {};

template <typename Data>
struct MonoidTrait<Product<Data>> : MonoidTraitImpl<Product, Data> {};

template <typename Data>
struct MonoidTrait<First<Data>> : MonoidTraitImpl<First, Data> {};

template <typename Data>
struct MonoidTrait<Last<Data>> : MonoidTraitImpl<Last, Data> {};

template <typename... Args>
struct MonoidTrait<std::tuple<Args...>>
{
    using Type = Monoid<std::tuple, std::decay_t<Args>...>;
};

template <>
struct MonoidTrait<Any>
{
    using Type = Monoid<DummyTemplateClass, Any>;
};

template <>
struct MonoidTrait<All>
{
    using Type = Monoid<DummyTemplateClass, All>;
};

template <typename Func>
struct MonoidTrait<Endo<Func>>
{
    using Type = Monoid<Endo>;
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
struct MonoidTrait<data::GenericFunction<nbArgs, Repr>>
{
    using Type = Monoid<DummyTemplateClass, GenericFunctionTag, data::GenericFunction<nbArgs, Repr>>;
};

template <typename Repr, typename Ret, typename FirstArg, typename... Rest>
struct MonoidTrait<data::Function<Repr, Ret, FirstArg, Rest...>>
{
    using RetT = std::invoke_result_t<data::Function<Repr, Ret, FirstArg, Rest...>, FirstArg>;
    using Type = Monoid<data::Function, FirstArg, RetT>;
};

/////////// Foldable ///////////

template <template<typename...> typename Type, typename... Args>
class FoldableBase
{
public:
    constexpr static auto foldr = toGFunc<3>([](auto&& f, auto&& z, auto&& t)
    {
        return appEndo || hspp::foldMap | (toEndo <o> f) | t || z;
    });
    constexpr static auto foldMap = toGFunc<2>([](auto&& func, auto&& ta)
    {
        using Data = decltype(*ta.begin());
        using MData = std::invoke_result_t<decltype(func), Data>;
        return hspp::foldr | (mappend <o> func) | MonoidType<MData>::mempty | ta;
    });
    constexpr static auto fold = hspp::foldMap | id;
    constexpr static auto toRange = toGFunc<1>([](auto const& tm)
    {
        return nonOwnedRange(data::RefView{tm});
    });
    constexpr static auto null = hspp::foldr | [](auto, bool){ return false; } | true;
};

template <template<typename...> typename Type, typename... Args>
class Foldable : public FoldableBase<Type, Args...>
{
public:
    constexpr static auto foldr = data::listFoldr;
};

template <typename... Init>
class Foldable<std::tuple, Init...> : public FoldableBase<std::tuple, Init...>
{
public:
    constexpr static auto foldMap = toGFunc<2>([](auto&& func, auto&& ta)
    {
        static_assert(std::tuple_size_v<std::decay_t<decltype(ta)>> == sizeof...(Init)+1);
        auto const& last = std::get<sizeof...(Init)>(ta);
        return func | last;
    });
    constexpr static auto foldr = toGFunc<3>([](auto&& func, auto&& z, auto&& ta)
    {
        static_assert(std::tuple_size_v<std::decay_t<decltype(ta)>> == sizeof...(Init)+1);
        auto const& last = std::get<sizeof...(Init)>(ta);
        return func | last | z;
    });
};

template <>
class FoldableBase<data::Maybe>
{
public:
    constexpr static auto foldr = toGFunc<3>([](auto&& func, auto&& z, auto&& ta)
    {
        if (ta == data::nothing)
        {
            return z;
        }
        return func | ta.value() | z;
    });
    constexpr static auto foldMap = toGFunc<2>([](auto&& func, auto&& ta)
    {
        using Data = decltype(ta.value());
        using MData = std::invoke_result_t<decltype(func), Data>;
        return foldr | (mappend <o> func) | MonoidType<MData>::mempty | ta;
    });
};

/////////// Functor ///////////

template <template<typename...> typename Type, typename... Ts>
class ContainerFunctor
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

template <template<typename...> typename Type, typename... Ts>
class Functor;

template <typename... Ts>
class Functor<std::vector, Ts...> : public ContainerFunctor<std::vector, Ts...>
{};

template <typename... Ts>
class Functor<std::list, Ts...> : public ContainerFunctor<std::list, Ts...>
{};

template <typename... Ts>
class Functor<data::Parser, Ts...>
{};

template <>
class Functor<data::Range>
{
public:
    template <typename Func, typename Arg, typename Repr>
    constexpr static auto fmap(Func const& func, data::Range<Arg, Repr> const& in)
    {
        return data::ownedRange(data::MapView{in, func});
    }
};

template <typename... Init>
class Functor<std::tuple, Init...>
{
public:
    template <typename Func, typename Last>
    constexpr static auto fmap(Func const& func, std::tuple<Init..., Last> in)
    {
        constexpr auto sizeMinusOne = sizeof...(Init);
        auto const last = std::get<sizeMinusOne>(in);
        return std::tuple_cat(subtuple<0, sizeMinusOne>(std::move(in)), std::make_tuple(func(std::move(last))));
    }
};

template <>
class Functor<data::Maybe>
{
public:
    template <typename Func, typename Arg>
    constexpr static auto fmap(Func const& func, data::Maybe<Arg> const& in)
    {
        using R = std::invoke_result_t<Func, Arg>;
        return std::visit(overload(
            [](data::Nothing) -> data::Maybe<R>
            {
                return data::nothing;
            },
            [func](data::Just<Arg> const& j) -> data::Maybe<R>
            {
                return data::Just<R>(func(j.data));
            }
        ), static_cast<data::MaybeBase<Arg>const &>(in));
    }
};

template <typename FirstArg>
class Functor<data::Function, FirstArg>
{
public:
    template <typename Func, typename Repr, typename Ret, typename... Args>
    constexpr static auto fmap(Func&& func, data::Function<Repr, Ret, FirstArg, Args...> const& in)
    {
        return o | func | in;
    }
};

template <typename FirstArg>
class Functor<data::Reader, FirstArg>
{
public:
    template <typename Func, typename Repr, typename Ret, typename... Args>
    constexpr static auto fmap(Func&& func, data::Reader<FirstArg, Ret, Repr> const& in)
    {
        return data::toReader || func <o> (data::runReader | in);
    }
};

template <>
class Functor<DummyTemplateClass, GenericFunctionTag>
{
public:
    template <typename Func, size_t nbArgs, typename Repr>
    constexpr static auto fmap(Func&& func, data::GenericFunction<nbArgs, Repr> const& in)
    {
        return o | func | in;
    }
};

template <>
class Functor<data::IO>
{
public:
    template <typename Func1, typename Data, typename Func2>
    constexpr static auto fmap(Func1 const& func, data::IO<Data, Func2> const& in)
    {
        return data::io([=]{ return func(in.run()); });
    }
};

template <template<typename...> class Type, typename... Ts>
class ApplicativeBase;

template <template<typename...> class Type, typename... Ts>
class Applicative : public Functor<Type, Ts...>, public ApplicativeBase<Type, Ts...>
{};

template <template<typename...> class Type, typename... Ts>
class ContainerApplicativeBase
{
public:
    constexpr static auto pure = toGFunc<1> | [](auto in)
    {
        return Type<std::decay_t<decltype(in)>>{in};
    };
    template <typename Func, typename Arg>
    constexpr static auto ap(Type<Func> const& func, Type<Arg> const& in)
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

template <typename... Ts>
class ApplicativeBase<std::vector, Ts...> : public ContainerApplicativeBase<std::vector, Ts...>
{};

template <typename... Ts>
class ApplicativeBase<std::list, Ts...> : public ContainerApplicativeBase<std::list, Ts...>
{};

template <typename... Ts>
class ApplicativeBase<std::basic_string, Ts...> : public ContainerApplicativeBase<std::basic_string, Ts...>
{};

template <>
class Applicative<data::Range>
{
public:
    constexpr static auto pure = toGFunc<1> | [](auto&& in)
    {
        return data::ownedRange(data::SingleView{std::forward<decltype(in)>(in)});
    };
    template <typename Func, typename Arg, typename Repr1, typename Repr2>
    constexpr static auto ap(data::Range<Func, Repr1> const& func, data::Range<Arg, Repr2> const& in)
    {
        auto view = data::MapView{data::ProductView{func, in}, [](auto&& tuple) { return std::get<0>(tuple)(std::get<1>(tuple)); }};
        return data::ownedRange(view);
    }
};

template <typename... Init>
class Applicative<std::tuple, Init...> : public Functor<std::tuple, Init...>
{
public:
    constexpr static auto pure = toGFunc<1> | [](auto in)
    {
        return std::make_tuple(MonoidType<Init>::mempty..., std::move(in));
    };
    template <typename Func, typename Last>
    constexpr static auto ap(std::tuple<Init..., Func> const& funcs, std::tuple<Init..., Last> const& args)
    {
        constexpr auto sizeMinusOne = sizeof...(Init);
        auto const func = std::get<sizeMinusOne>(funcs);
        auto const last = std::get<sizeMinusOne>(args);
        auto const init = mappend | takeTuple<sizeMinusOne>(funcs) | takeTuple<sizeMinusOne>(args);
        return std::tuple_cat(init, std::make_tuple(func(last)));
    }
};

template <>
class Applicative<data::Maybe> : public Functor<data::Maybe>
{
public:
    constexpr static auto pure = data::just;
    template <typename Func, typename Arg>
    constexpr static auto ap(data::Maybe<Func> const& func, data::Maybe<Arg> const& in)
    {
        using R = std::invoke_result_t<Func, Arg>;
        return std::visit(overload(
            [](data::Just<Func> const& f, data::Just<Arg> const& a) -> data::Maybe<R>
            {
                return data::Just<R>{f.data(a.data)};
            },
            [](auto, auto) -> data::Maybe<R>
            {
                return data::nothing;
            }
        ),
        static_cast<data::MaybeBase<Func>const &>(func),
        static_cast<data::MaybeBase<Arg>const &>(in)
        );
    }
};

template <>
class Applicative<data::IO> : public Functor<data::IO>
{
public:
    constexpr static auto pure = toGFunc<1> | [](auto in)
    {
        return data::ioData(std::move(in));
    };
    template <typename Func, typename Arg, typename Func1, typename Func2>
    constexpr static auto ap(data::IO<Func, Func1> const& func, data::IO<Arg, Func2> const& in)
    {
        return data::io([=]{ return func.run()(in.run()); });
    }
};

template <typename FirstArg>
class Applicative<data::Function, FirstArg> : public Functor<data::Function, FirstArg>
{
public:
    constexpr static auto pure = toGFunc<1> | [](auto ret)
    {
        return data::toFunc<>([ret=std::move(ret)](FirstArg){ return ret; });
    };
    template <typename Func1, typename Func2>
    constexpr static auto ap(Func1 func, Func2 in)
    {
        return data::toFunc<>(
            [func=std::move(func), in=std::move(in)](FirstArg arg)
            {
                return func(arg)(in(arg));
            });
    }
};

template <typename FirstArg>
class Applicative<data::Reader, FirstArg> : public Functor<data::Reader, FirstArg>
{
public:
    constexpr static auto pure = toGFunc<1> | [](auto ret)
    {
        return data::toReader || data::toFunc<>([ret=std::move(ret)](FirstArg){ return ret; });
    };
    template <typename Reader1, typename Reader2>
    constexpr static auto ap(Reader1 func, Reader1 in)
    {
        return data::toReader || data::toFunc<>(
            [func=std::move(func), in=std::move(in)](FirstArg arg)
            {
                return data::runReader | func | arg || data::runReader | in | arg;
            });
    }
};

template <typename S>
class Applicative<data::State, S> : public Functor<data::State, S>
{
public:
    constexpr static auto pure = toGFunc<1> | [](auto ret)
    {
        return data::toState | data::toFunc<>([ret=std::move(ret)](S st){ return std::make_tuple(ret, st); });
    };
    // template <typename Func, typename State>
    // constexpr static auto ap(Func func, State in)
    // {
    //     return data::toState | data::toFunc<>(
    //         [func=std::move(func), in=std::move(in)](FirstArg arg)
    //         {
    //             return data::runReader | func | arg || data::runReader | in | arg;
    //         });
    // }
};

template <>
class Applicative<data::Parser> : public Functor<data::Parser>
{
public:
    constexpr static auto pure = toGFunc<1> | [](auto a)
    {
        return data::toParser || data::toFunc<> | [a=std::move(a)](std::string cs){ return std::vector{std::make_tuple(a, cs)}; };
    };
};

template <>
class Applicative<DummyTemplateClass, GenericFunctionTag> : public Functor<DummyTemplateClass, GenericFunctionTag>
{
public:
    constexpr static auto pure = toGFunc<1> | [](auto ret)
    {
        return toGFunc<1>([=](auto){ return ret; });
    };
    template <typename Func1, typename Func2>
    constexpr static auto ap(Func1 func, Func2 in)
    {
        return toGFunc<1>([f=std::move(func), g=std::move(in)](auto arg) {return f(arg)(g(arg)); });
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

constexpr auto pure = toGFunc<1>([](auto const& data)
{
    return pureImpl(data);
}
);

constexpr auto return_ = pure;

class DeferredGuard;

template <typename T>
class IsDeferred : public std::false_type
{};
template <typename T>
class IsDeferred<DeferredPure<T>> : public std::true_type
{};
template <>
class IsDeferred<DeferredGuard> : public std::true_type
{};
template <typename T>
constexpr static auto isDeferredV = IsDeferred<std::decay_t<T>>::value;

class Ap
{
public:
    template <typename Func, typename Data>
    constexpr auto operator()(DeferredPure<Func> const& func, Data const& data) const
    {
        using ApType = ApplicativeType<Data>;
        return ApType::ap(ApType::pure(func.mData), data);
    }
    template <typename Func, typename Data>
    constexpr auto operator()(Func const& func, DeferredPure<Data> const& in) const
    {
        using ApType = ApplicativeType<Func>;
        return ApType::ap(func, ApType::pure(in.mData));
    }
    template <typename Func, typename Data>
    constexpr auto operator()(Func const& func, Data const& in) const
    {
        using ApType1 = ApplicativeType<Func>;
        using ApType2 = ApplicativeType<Data>;
        static_assert(std::is_same_v<ApType1, ApType2>);
        return ApType1::ap(func, in);
    }
};

constexpr inline auto ap = toGFunc<2>(Ap{});

/////////// Monad ///////////

template <template<typename...> class Type, typename... Ts>
class MonadBase;

template <typename MonadB>
class MonadRShift
{
public:
    constexpr static auto rshift = toGFunc<2>
    ([](auto x, auto y){
        return MonadB::bind(x, [y](auto) { return y; });
    });
};

template <template<typename...> class Type, typename... Ts>
class Monad : public Applicative<Type, Ts...>, public MonadBase<Type, Ts...>, public MonadRShift<MonadBase<Type, Ts...>>
{
public:
    constexpr static auto return_ = Applicative<Type, Ts...>::pure;
};

template <template<typename...> class Type, typename... Ts>
class ContainerMonadBase
{
public:
    template <typename... Args, typename Func>
    constexpr static auto bind(Type<Args...> const& arg, Func const& func)
    {
        return mconcat || fmap | func | arg;
    }
};

template <typename... Ts>
class MonadBase<std::vector, Ts...> : public ContainerMonadBase<std::vector, Ts...>
{};

template <typename... Ts>
class MonadBase<std::list, Ts...> : public ContainerMonadBase<std::list, Ts...>
{};

template <typename... Ts>
class MonadBase<std::basic_string, Ts...> : public ContainerMonadBase<std::basic_string, Ts...>
{};

template <typename... Ts>
class MonadBase<data::Range, Ts...> : public ContainerMonadBase<data::Range, Ts...>
{};

template <>
class MonadBase<data::Maybe>
{
public:
    template <typename Arg, typename Func>
    constexpr static auto bind(data::Maybe<Arg> const& arg, Func const& func)
    {
        using R = std::invoke_result_t<Func, Arg>;
        return std::visit(overload(
            [](data::Nothing) -> R
            {
                return data::nothing;
            },
            [func](data::Just<Arg> const& j) -> R
            {
                return func(j.data);
            }
        ), static_cast<data::MaybeBase<Arg> const&>(arg));
    }
};

template <typename... Init>
class MonadBase<std::tuple, Init...>
{
public:
    template <typename Last, typename Func>
    constexpr static auto bind(std::tuple<Init..., Last> const& args, Func&& func)
    {
        constexpr auto sizeMinusOne = sizeof...(Init);
        auto const last = std::get<sizeMinusOne>(args);
        auto const lastResult = func | last;
        auto const init = mappend | takeTuple<sizeMinusOne>(args) | takeTuple<sizeMinusOne>(lastResult);
        return std::tuple_cat(init, std::make_tuple(std::get<sizeMinusOne>(lastResult)));
    }
};

template <>
class MonadBase<data::IO>
{
public:
    template <typename Arg, typename Func1, typename Func>
    constexpr static auto bind(data::IO<Arg, Func1> const& arg, Func const& func)
    {
        return data::io([=]{ return func(arg.run()).run(); });
    }
};

template <>
class MonadBase<DummyTemplateClass, GenericFunctionTag>
{
public:
    template <size_t nbArgs, typename Repr, typename Func>
    constexpr static auto bind(data::GenericFunction<nbArgs, Repr> const& m, Func k)
    {
        return (flip | std::move(k)) <ap> m;
    }
};

template <typename FirstArg>
class MonadBase<data::Function, FirstArg>
{
public:
    template <typename Repr, typename Ret, typename... Rest, typename Func>
    constexpr static auto bind(data::Function<Repr, Ret, FirstArg, Rest...> const& m, Func k)
    {
        return (flip | std::move(k)) <ap> m;
    }
};

template <typename FirstArg>
class MonadBase<data::Reader, FirstArg>
{
public:
    template <typename Ret, typename Repr, typename Func>
    constexpr static auto bind(data::Reader<FirstArg, Ret, Repr> const& m, Func k)
    {
        return (flip | std::move(k)) <ap> m;
    }
};

template <typename S>
class MonadBase<data::State, S>
{
public:
    template <typename A, typename Repr, typename Func>
    constexpr static auto bind(data::State<S, A, Repr> const& act, Func k)
    {
        return data::toState | toFunc<>([=](S st)
        {
            auto&& [x, st_] = data::runState | act | st;
            return data::runState | (k | x) | st_;
        });
    }
};

template <>
class MonadBase<data::Parser>
{
public:
    template <typename A, typename Repr, typename Func>
    constexpr static auto bind(data::Parser<A, Repr> const& p, Func f)
    {
        return data::toParser | toFunc<>([=](std::string cs)
        {
            auto&& tempResult = data::runParser | p | cs;
            auto const cont = toGFunc<1> | [f=std::move(f)](auto tu)
            {
                auto&& [a, cs] = tu;
                return return_ || data::runParser | f(a) | cs;
            };
            return mconcat || (tempResult >>= cont);
        });
    }
};

template <typename T>
using MonadType = typename TypeClassTrait<Monad, std::decay_t<T>>::Type;

template <typename T, typename Enable = void>
class IsMonad : public std::false_type
{};
template <typename T>
class IsMonad<T, std::void_t<MonadType<T>>> : public std::true_type
{};
template <typename T>
constexpr static auto isMonadV = IsMonad<std::decay_t<T>>::value;

static_assert(isMonadV<std::vector<int>>);

/////////// MonadPlus //////////

template <template<typename...> class Type, typename... Ts>
class MonadZero;

template <template<typename...> class Type, typename... Ts>
class MonadPlus;

template <typename T>
struct MonadZeroTrait;

template <template <typename...> class C, typename Data, typename... Rest>
struct MonadZeroTrait<C<Data, Rest...>>
{
    using Type = MonadZero<C, Data>;
};

template <typename T>
using MonadZeroType = typename MonadZeroTrait<T>::Type;

template <typename T>
struct MonadPlusTrait;

template <template <typename...> class C, typename Data, typename... Rest>
struct MonadPlusTrait<C<Data, Rest...>>
{
    using Type = MonadPlus<C, Data>;
};

template <typename T>
using MonadPlusType = typename MonadPlusTrait<T>::Type;

class Mplus
{
public:
    template <typename T1, typename T2>
    constexpr auto operator()(T1 t1, T2 t2) const
    {
        using MPT = MonadPlusType<T1>;
        static_assert(std::is_same_v<MPT, MonadPlusType<T2>>);
        return MPT::mplus | t1 | t2;
    }
};

constexpr auto mplus = toGFunc<2> | Mplus{};

template <template<typename...> class Type, typename... Ts>
class MonadZero : public Monoid<Type, Ts...>
{
public:
    const static decltype(Monoid<Type, Ts...>::mempty) mzero;
};

template <template<typename...> typename Type, typename... Args>
const decltype(Monoid<Type, Args...>::mempty) MonadZero<Type, Args...>::mzero = Monoid<Type, Args...>::mempty;

constexpr auto failIO = toFunc<> | [](std::string s)
{
    return data::io(
        [=]{
            throw std::runtime_error{s};
        }
    );
};

inline const auto failIOMZero = failIO | "mzero";

template <typename A>
class MonadZero<data::IO, A>
{
public:
    const static decltype(failIOMZero) mzero;
};

template <typename A>
const decltype(failIOMZero) MonadZero<data::IO, A>::mzero = failIOMZero;

template <typename A>
class MonadZero<data::Parser, A>
{
public:
    constexpr static auto mzero = data::toParser || toFunc<> | [](std::string)
    {
        return std::vector<std::tuple<A, std::string>>{};
    };
};

template <template<typename...> class Type, typename... Ts>
class MonadPlus : public MonadZero<Type, Ts...>
{
public:
    constexpr static auto mplus = Monoid<Type, Ts...>::mappend;
};

template <typename A>
class MonadPlus<data::IO, A> : public MonadZero<data::IO, A>
{
public:
    // constexpr static auto mplus;
};


template <typename A>
class MonadPlus<data::Parser, A>
{
public:
    constexpr static auto mplus = toGFunc<2> | [](auto p, auto q)
    {
        return data::toParser || toFunc<> | [=](std::string cs)
        {
            return (data::runParser | p | cs) <hspp::mplus> (data::runParser | q | cs);
        };
    };
};

/////////// Traversable ///////////

template <template<typename...> typename Type, typename... Args>
class TraversableBase
{
public:
    constexpr static auto traverse = hspp::sequenceA <o> (fmap | id);
    constexpr static auto sequenceA = hspp::traverse | id;
};

template <template<typename...> typename Type, typename... Args>
class Traversable : public TraversableBase<Type, Args...>
{
public:
    constexpr static auto traverse = toGFunc<2>([](auto&& f, auto&& ta)
    {
        auto const consF = toGFunc<2>([&](auto&& x, auto&& ys)
        {
            return data::cons <fmap> (f | x) <ap> ys;
        });
        using OutterResultDataType = decltype(f <fmap> ta);
        using ResultDataType = DataType<OutterResultDataType>;
        using DataT = DataType<ResultDataType>;
        return data::listFoldr | consF || ApplicativeType<ResultDataType>::pure(Type<DataT>{}) || ta;
    });
};

template <typename... Args>
class Traversable<data::Maybe, Args...> : public TraversableBase<data::Maybe, Args...>
{
public:
    constexpr static auto traverse = toGFunc<2>([](auto&& f, auto&& ta)
    {
        if (ta == data::nothing)
        {
            using ResultDataType = decltype(f | ta.value());
            using DataT = DataType<ResultDataType>;
            return ApplicativeType<ResultDataType>::pure(data::Maybe<DataT>{});
        }
        return data::just <fmap> (f | ta.value());
    });
};

template <typename... Args>
class Traversable<std::tuple, Args...> : public TraversableBase<std::tuple, Args...>
{
public:
    constexpr static auto traverse = toGFunc<2>([](auto&& f, auto&& in)
    {
        constexpr auto sizeMinusOne = std::tuple_size_v<std::decay_t<decltype(in)>> - 1;
        auto last = std::get<sizeMinusOne>(in);
        auto const func = toGFunc<1>([in=std::forward<decltype(in)>(in)](auto&& lastElem)
        {
            constexpr auto sizeMinusOne = std::tuple_size_v<std::decay_t<decltype(in)>> - 1;
            return std::tuple_cat(subtuple<0, sizeMinusOne>(std::move(in)), std::make_tuple(lastElem));
        });
        return func <fmap> (f | last);
    });
};

class DeferredGuard
{
public:
    bool cond;
};

constexpr auto guard = toFunc<> | [](bool flag)
{
    return DeferredGuard{flag};
};


template <typename ClassT>
constexpr auto guardImpl(bool b)
{
    if constexpr (data::isRangeV<ClassT>)
    {
        return data::ownedRange(data::FilterView{Monad<data::Range>::return_(_o_), [b](auto){ return b; }});
    }
    else
    {
        if constexpr (data::isIOV<ClassT>)
        {
            return data::io([=]{
                if (!b)
                {
                    failIOMZero.run();
                }
                return _o_;
            });
        }
        else
        {
            return b ? MonadType<ClassT>::return_( _o_) : MonadPlusType<ReplaceDataTypeWith<ClassT, _O_>>::mzero;
        }
    }
}

constexpr auto show = toGFunc<1>([](auto&& d)
{
    std::stringstream os;
    os << std::boolalpha << d;
    return os.str();
});

template <typename T>
constexpr auto read = data::toFunc<>([](std::string const& d)
{
    std::stringstream is{d};
    T t;
    is >> t;

    if (is.bad())
    {
        throw std::runtime_error{"Invalid read!"};
    }
    return t;
});

constexpr auto print = putStrLn <o> show;

template <typename ClassT, typename T, typename = std::enable_if_t<std::is_same_v<MonadType<ClassT>, MonadType<T>>, void>>
constexpr auto evalDeferredImpl(T&& t)
{
    return std::forward<T>(t);
}

template <typename ClassT, typename T>
constexpr auto evalDeferredImpl(DeferredPure<T> t)
{
    return MonadType<ClassT>::return_(t.mData);
}

template <typename ClassT>
constexpr auto evalDeferredImpl(DeferredGuard g)
{
    return guardImpl<ClassT>(g.cond);
}

template <typename ClassT>
constexpr auto evalDeferred = toGFunc<1>([](auto&& d)
{
    return evalDeferredImpl<ClassT>(d);
});

// >>= is right-assocative in C++, have to add some parens when chaining the calls.
template <typename Arg, typename Func, typename Ret = std::invoke_result_t<Func, Arg>>
constexpr auto operator>>=(DeferredPure<Arg> const& arg, Func const& func)
{
    using MType = MonadType<Ret>;
    return MType::bind(evalDeferred<Ret> | arg, func);
}

template <typename MonadData, typename Func>
constexpr auto operator>>=(MonadData const& data, Func const& func)
{
    using MType = MonadType<MonadData>;
    return MType::bind(data, evalDeferred<MonadData> <o> func);
}

template <typename Deferred, typename MonadData, typename = std::enable_if_t<isDeferredV<Deferred>, void>>
constexpr auto operator>>(Deferred const& arg, MonadData const& data)
{
    using MType = MonadType<MonadData>;
    return MType::rshift | (evalDeferred<MonadData> | arg) | data;
}

template <typename MonadData1, typename MonadData2,
    std::enable_if_t<std::is_same_v<MonadType<MonadData1>, MonadType<MonadData2>>, void*> = nullptr>
constexpr auto operator>>(MonadData1 const& lhs, MonadData2 const& rhs)
{
    using MType = MonadType<MonadData1>;
    return MType::rshift | lhs || rhs;
}

template <typename MonadData1, typename MonadData2, typename MType = MonadType<MonadData1>,
    typename = std::enable_if_t<isDeferredV<MonadData2>, bool>>
constexpr auto operator>>(MonadData1 const& lhs, MonadData2 const& rhs)
{
    return MType::rshift | lhs || evalDeferred<MonadData1> | rhs;
}

// for IO
template <typename MData>
constexpr inline auto replicateM_Impl (size_t times, MData const& mdata)
{
    static_assert(isMonadV<MData>);
    return data::io(
        [=]
        {
            for (size_t i = 0; i < times; ++i)
            {
                mdata.run();
            }
            return _o_;
        }
    );
}

constexpr inline auto replicateM_ = toGFunc<2>([](size_t times, auto mdata)
{
    return replicateM_Impl(times, mdata);
});

constexpr inline auto forever = toGFunc<1>([](auto io_)
{
    static_assert(isMonadV<decltype(io_)>);
    return data::io(
        [=]
        {
            while (true)
            {
                io_.run();
            }
            return _o_;
        }
    );
});


constexpr auto all = toGFunc<1>([](auto p)
{
    return getAll <o> (foldMap | (toAll <o> p));
});

constexpr auto any = toGFunc<1>([](auto p)
{
    return getAny <o> (foldMap | (toAny <o> p));
});

constexpr auto sum = getSum <o> (foldMap | toSum);

constexpr auto product = getProduct <o> (foldMap | toProduct);

constexpr inline auto elem = any <o> data::equalTo;

constexpr inline auto length = getSum <o> (foldMap || data::const_ | (toSum | 1));


} // namespace hspp

#endif // HSPP_TYPECLASS_H
/*
 *  Copyright (c) 2022 Bowen Fu
 *  Distributed Under The Apache-2.0 License
 */

#ifndef HSPP_DO_NOTATION_H
#define HSPP_DO_NOTATION_H

namespace hspp
{

namespace doN
{
template <typename T>
class Nullary;

template <typename F>
class LetExpr : public F
{
public:
    LetExpr(F f)
    : F{std::move(f)}
    {}
};

template <typename T>
class IsLetExpr : public std::false_type
{
};

template <typename T>
class IsLetExpr<LetExpr<T>> : public std::true_type
{
};

template <typename T>
constexpr auto isLetExprV = IsLetExpr<std::decay_t<T>>::value;

// make sure Id is not a const obj.
template <typename T>
class Id
{
    using OptT = std::optional<T>;
    std::shared_ptr<OptT> mT = std::make_shared<OptT>();
public:
    constexpr Id() = default;
    constexpr auto const& value() const
    {
        if (!mT)
        {
            throw std::runtime_error{"Invalid id!"};
        }
        if (!mT->has_value())
        {
            std::cerr << "mT : " << mT.get() << std::endl;
            throw std::runtime_error{"Id has no binding!"};
        }
        return mT->value();
    }
    constexpr void bind(T v) const
    {
        const_cast<OptT&>(*mT) = std::move(v);
    }

    constexpr auto operator=(T const& d)
    {
        bind(d);
        return LetExpr([]{});
    }

    // return let expr
    template <typename F>
    constexpr auto operator=(Nullary<F> const& f)
    {
        static_assert(std::is_same_v<T, std::invoke_result_t<Nullary<F>>>);
        return LetExpr([*this, f]{ bind(f()); });
    }

};

template <typename T>
class Nullary : public T
{
public:
    using T::operator();
};

template <typename T>
constexpr auto nullary(T const &t)
{
    return Nullary<T>{t};
}

template <typename T>
constexpr auto toTENullaryImpl(Nullary<T> const &t)
{
    return nullary(std::function<std::invoke_result_t<T>>{t});
}

constexpr auto toTENullary = toGFunc<1> | [](auto const& t)
{
    return toTENullaryImpl(t);
};

template <typename T>
class IsNullary : public std::false_type
{
};

template <typename T>
class IsNullary<Nullary<T>> : public std::true_type
{
};

template <typename T>
constexpr auto isNullaryV = IsNullary<std::decay_t<T>>::value;

template <typename ClassT, typename T , typename = std::enable_if_t<isNullaryV<T>, void>>
constexpr auto evalDeferredImpl(T&& t)
{
    static_assert(std::is_same_v<MonadType<ClassT>, MonadType<std::invoke_result_t<T>>>);
    return t();
}

template <typename T>
class IsNullaryOrId : public IsNullary<T>
{
};

template <typename T>
class IsNullaryOrId<Id<T>> : public std::true_type
{
};

template <typename T>
constexpr auto isNullaryOrIdV = IsNullaryOrId<std::decay_t<T>>::value;

template <typename M>
class DeMonad
{
    using T = DataType<M>;
public:
    constexpr DeMonad(M const& m, Id<T> id)
    : mM{m}
    , mId{std::move(id)}
    {
    }
    constexpr decltype(auto) m() const
    {
        return mM;
    }
    constexpr auto id() const -> Id<T>
    {
        return mId;
    }

private:
    // FIXME reference for lvalue, value for rvalue
    // std::reference_wrapper<M const> mM;
    M const mM;
    Id<T> mId;
};

template <typename... Ts>
class IsDeMonad : public std::false_type
{};
template <typename... Ts>
class IsDeMonad<DeMonad<Ts...>> : public std::true_type
{};
template <typename T>
constexpr static auto isDeMonadV = IsDeMonad<std::decay_t<T>>::value;


template <typename M, typename = MonadType<M>>
constexpr auto operator<= (Id<DataType<M>>& id, M const& m)
{
    return DeMonad{m, id};
}

template <typename D, typename N, typename = MonadType<std::invoke_result_t<Nullary<N>>>>
constexpr auto operator<= (Id<D>& id, Nullary<N> const& n)
{
    using MT = std::invoke_result_t<Nullary<N>>;
    return nullary([=] { return DeMonad<MT>{evaluate_(n), id}; });
}

template <typename T>
class EvalTraits
{
public:
    template <typename... Args>
    constexpr static decltype(auto) evalImpl(T const &v)
    {
        return v;
    }
};

template <typename T>
class EvalTraits<Nullary<T>>
{
public:
    constexpr static decltype(auto) evalImpl(Nullary<T> const &e) { return e(); }
};

// Only allowed in nullary
template <typename T>
class EvalTraits<Id<T>>
{
public:
    constexpr static decltype(auto) evalImpl(Id<T> const &id)
    {
        return id.value();
    }
};

template <typename T>
constexpr decltype(auto) evaluate_(T const &t)
{
    return EvalTraits<T>::evalImpl(t);
}

#define UN_OP_FOR_NULLARY(op)                                               \
    template <typename T, std::enable_if_t<isNullaryOrIdV<T>, bool> = true> \
    constexpr auto operator op(T&&t)                                  \
    {                                                                       \
        return nullary([=] { return op evaluate_(t); });                    \
    }

#define BIN_OP_FOR_NULLARY(op)                                                  \
    template <typename T, typename U,                                           \
              std::enable_if_t<isNullaryOrIdV<T> || isNullaryOrIdV<U>, bool> =  \
                  true>                                                         \
    constexpr auto operator op(T t, U u)                                    \
    {                                                                           \
        return nullary([t=std::move(t), u=std::move(u)] { return evaluate_(t) op evaluate_(u); });           \
    }

using hspp::operator>>;

BIN_OP_FOR_NULLARY(|)
BIN_OP_FOR_NULLARY(||)
BIN_OP_FOR_NULLARY(>>)
BIN_OP_FOR_NULLARY(*)
BIN_OP_FOR_NULLARY(+)
BIN_OP_FOR_NULLARY(==)
BIN_OP_FOR_NULLARY(%)
BIN_OP_FOR_NULLARY(<)
BIN_OP_FOR_NULLARY(>)
BIN_OP_FOR_NULLARY(&&)
BIN_OP_FOR_NULLARY(-)

template <typename T, typename BodyBaker>
constexpr auto funcWithParams(Id<T> const& param, BodyBaker const& bodyBaker)
{
    return [=](T const& t)
    {
        // bind before baking body.
        param.bind(t);
        auto result = evaluate_(bodyBaker());
        return result;
    };
}

template <typename MClass, typename Head>
constexpr auto doImpl(Head const& head)
{
    return evalDeferred<MClass> | head;
}

template <typename MClass, typename N, typename... Rest, typename = std::enable_if_t<isDeMonadV<std::invoke_result_t<Nullary<N>>>, void>>
constexpr auto doImplNullaryDeMonad(Nullary<N> const& dmN, Rest const&... rest)
{
    return doImpl<MClass>(dmN(), rest...);
}

template <typename MClass1, typename F, typename... Rest>
constexpr auto doImpl(LetExpr<F> const& le, Rest const&... rest)
{
    le();
    return evaluate_(doImpl<MClass1>(rest...));
}

template <typename MClass1, typename MClass2, typename... Rest>
constexpr auto doImpl(DeMonad<MClass2> const& dm, Rest const&... rest)
{
    static_assert(std::is_same_v<MonadType<MClass1>, MonadType<MClass2>>);
    auto const bodyBaker = [=] { return doImpl<MClass2>(rest...);};
    return dm.m() >>= funcWithParams(dm.id(), bodyBaker);
}

template <typename MClass, typename Head, typename... Rest, typename = std::enable_if_t<!isDeMonadV<Head> && !isLetExprV<Head>, void>>
constexpr auto doImpl(Head const& head, Rest const&... rest)
{
    if constexpr (isNullaryOrIdV<Head>)
    {
        if constexpr (isDeMonadV<std::invoke_result_t<Head>>)
        {
            return doImplNullaryDeMonad<MClass>(head, rest...);
        }
        else
        {
            return (evalDeferred<MClass> | head) >> doImpl<MClass>(rest...);
        }
    }
    else
    {
        return (evalDeferred<MClass> | head) >> doImpl<MClass>(rest...);
    }
}

// Get class type that is a monad.
template <typename T, typename Enable = void>
struct MonadClassImpl;

template <>
struct MonadClassImpl<std::tuple<>>
{
    using Type = void;
};

template <typename T1, typename T2, typename Enable1 = void, typename Enable2 = void>
struct SameMType : std::false_type{};

template <typename T1, typename T2>
struct SameMType<T1, T2, std::void_t<MonadType<T1>>, std::void_t<MonadType<T2>>>
{
    constexpr static auto value = std::is_same_v<MonadType<T1>, MonadType<T2>>;
};

static_assert(SameMType<hspp::data::Maybe<int>, hspp::data::Maybe<int>>::value);

template <typename Head, typename... Rest>
struct MonadClassImpl<std::tuple<Head, Rest...>, std::enable_if_t<isMonadV<Head>, void>>
{
    using Type = Head;
private:
    using RType = typename MonadClassImpl<std::tuple<Rest...>>::Type;
    static_assert(std::is_same_v<void, RType> || SameMType<Type, RType>::value);
};

template <typename Head, typename... Rest>
struct MonadClassImpl<std::tuple<DeMonad<Head>, Rest...>, std::enable_if_t<isMonadV<Head>, void>>
{
    using Type = Head;
private:
    using RType = typename MonadClassImpl<std::tuple<Rest...>>::Type;
    static_assert(std::is_same_v<void, RType> || SameMType<Type, RType>::value);
};

template <typename Head, typename... Rest>
struct MonadClassImpl<std::tuple<Head, Rest...>, std::enable_if_t<!isMonadV<Head> && !isDeMonadV<Head>, void>>
{
    using Type = typename MonadClassImpl<std::tuple<Rest...>>::Type;
};


// Get class type that is a monad.
template <typename... Ts>
struct MonadClass
{
    using Type = typename MonadClassImpl<std::tuple<Ts...>>::Type;
    static_assert(!std::is_same_v<Type, void>);
};

template <typename... Ts>
using MonadClassType = typename MonadClass<std::decay_t<Ts>...>::Type;

static_assert(isMonadV<data::Maybe<int>>);
static_assert(std::is_same_v<MonadClassType<data::Maybe<int>>, data::Maybe<int>>);

template <typename Head, typename... Rest>
constexpr auto do_(Head const& head, Rest const&... rest)
{
    using MClass = MonadClassType<Head, Rest...>;
    auto result = doImpl<MClass>(head, rest...);
    static_assert(!isNullaryOrIdV<decltype(result)>);
    return result;
}

template <typename... Args>
constexpr auto doInner(Args&&... args)
{
    return nullary([=] { return do_(evaluate_(args)...); });
}

template <typename Head, typename... Rest>
constexpr auto _(Head const& head, Rest const&... rest)
{
    return do_(rest..., return_ | head);
}

constexpr auto if_ = guard;

// used for doN, so that Id/Nullary can be used with ifThenElse.
constexpr auto ifThenElse = toGFunc<3> | [](auto pred, auto then_, auto else_)
{
    using MClass = MonadClassType<decltype(evaluate_(then_)), decltype(evaluate_(else_))>;
    return nullary([pred=std::move(pred), then_=std::move(then_), else_=std::move(else_)] { return evaluate_(pred) ? (evalDeferred<MClass> | evaluate_(then_)) : (evalDeferred<MClass> | evaluate_(else_)); });
};

} // namespace doN

} // namespace hspp

#endif // HSPP_DO_NOTATION_H
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
using data::Parser;
using data::toParser;
using data::runParser;

using data::TEParser;
using data::toTEParser;

constexpr auto alt = toGFunc<2> | [](auto p, auto q)
{
    return data::toParser <o> data::toFunc<> | [=](std::string cs)
    {
        auto const tmp = data::runParser | (p <mplus> q) | cs;
        if (tmp.empty())
        {
            return tmp;
        }
        std::remove_const_t<decltype(tmp)> tmp2;
        tmp2.push_back(tmp.front());
        return tmp2;
    };
};

constexpr auto item = toParser | toFunc<>([](std::string cs) -> std::vector<std::tuple<char, std::string>>
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
        toParser || toFunc<> | [flag = p | c, posParser = Monad<Parser>::return_ | c, negParser = MonadZero<Parser, char>::mzero]
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

inline const auto space = many || sat | isSpace;

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
/*
Reference:

Du Bois, A.R. (2011). An Implementation of Composable Memory Transactions in Haskell. In: Apel, S., Jackson, E. (eds) Software Composition. SC 2011. Lecture Notes in Computer Science, vol 6708. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-22045-6_3
*/

#include <vector>
#include <list>
#include <cmath>
#include <cctype>
#include <thread>
#include <atomic>
#include <shared_mutex>
#include <mutex>
#include <any>
#include <type_traits>
#include <map>
#include <typeindex>
#include <set>
#include <array>

using namespace hspp;
using namespace hspp::data;
using namespace std::literals;
using namespace hspp::doN;

namespace hspp
{

namespace concurrent
{

template <typename Func>
auto forkIOImpl(IO<_O_, Func> io_)
{
    return io(
        [io_]
        {
            std::thread t{[io_]{
                io_.run();
            }};
            t.detach();
            return t.get_id();
        }
    );
}

constexpr auto forkIO = toGFunc<1> | [](auto io_){
    return forkIOImpl(io_);
};

constexpr auto threadDelay = toFunc<> | [](size_t microseconds)
{
    return io(
        [microseconds]{
            std::this_thread::sleep_for(std::chrono::microseconds{microseconds});
            return _o_;
        }
    );
};

template <typename A>
struct MVar;

template <typename A>
class Atomic
{
public:
    Atomic() = default;
    Atomic(A value)
    : mValue{std::move(value)}
    {}
    bool compareExchangeStrong(A& expected, A const& desired)
    {
        std::unique_lock lc{mLock};
        if (expected != mValue)
        {
            expected = mValue;
            return false;
        }
        mValue = desired;
        return true;
    }
    void store(A const& a)
    {
        std::unique_lock lc{mLock};
        mValue = a;
    }
    A load() const
    {
        std::shared_lock lc{mLock};
        return mValue;
    }
    std::shared_mutex& lock() const
    {
        return mLock;
    }
    A mValue;
private:
    template <typename B>
    friend constexpr auto takeMVarImpl(MVar<B> const& a);
    mutable std::shared_mutex mLock;
};

template <typename A>
struct MVar
{
    using Data = A;
    using T = Atomic<std::optional<A>>;
    std::shared_ptr<T> data = std::make_shared<T>();
    MVar() = default;
    MVar(A a)
    : data{std::make_shared<T>(std::make_optional(std::move(a)))}
    {}
};

template <typename A>
constexpr auto newEmptyMVar = io([]{ return MVar<A>{}; });

template <typename A>
constexpr auto newMVarImpl(A a)
{
    return io([a=std::move(a)]{ return MVar<A>{std::move(a)}; });
}

constexpr auto newMVar = toGFunc<1> | [](auto a)
{
    return newMVarImpl(a);
};

template <typename A>
constexpr auto takeMVarImpl(MVar<A> const& a)
{
    return io([a]
    {
        while (true)
        {
            {
                std::unique_lock lock{a.data->lock()};
                if (a.data->mValue.has_value())
                {
                    auto result = std::move(a.data->mValue.value());
                    a.data->mValue.reset();
                    return result;
                }
            }
            std::this_thread::yield();
        }
    });
}

constexpr auto takeMVar = toGFunc<1> | [](auto a)
{
    return takeMVarImpl(a);
};

template <typename A>
constexpr auto putMVarImpl(MVar<A> a, A new_)
{
    return io([a, new_=std::optional<A>{std::move(new_)}]
    {
        auto old_ = std::optional<A>{};
        while (!a.data->compareExchangeStrong(old_, new_))
        {
            old_ = std::optional<A>{};
            std::this_thread::yield();
        }
        return _o_;
    });
}

constexpr auto putMVar = toGFunc<2> | [](auto a, auto new_)
{
    return putMVarImpl(std::move(a), std::move(new_));
};

template <typename A>
constexpr auto tryPutMVarImpl(MVar<A>& a, A new_)
{
    return io([a, new_]
    {
        std::unique_lock lock{a.data->lock()};
        if (!a.data->mValue.has_value())
        {
            a.data->mValue = new_;
            return true;
        }
        return false;
    });
}

constexpr auto tryPutMVar = toGFunc<2> | [](auto a, auto new_)
{
    return tryPutMVarImpl(a, new_);
};

// can be optimized to use shared_lock.
constexpr auto readMVar = toGFunc<1> | [](auto m){
    using T = std::decay_t<decltype((takeMVar | m).run())>;
    Id<T> a;
    return do_(
        a <= (takeMVar | m),
        putMVar | m | a,
        return_ | a
    );
};

template <typename A>
class Async : public MVar<A>
{};

constexpr auto toAsync = toGFunc<1> | [](auto a)
{
    return Async<typename decltype(a)::Data>{a};
};

constexpr auto async = toGFunc<1> | [](auto action){
    using A = std::decay_t<decltype(action.run())>;
    Id<MVar<A>> var;
    Id<A> r;
    return do_(
        var <= newEmptyMVar<A>,
        forkIO || do_( // why doInner does not work here?
            r <= action,
            putMVar | var | r),
        return_ | (toAsync | var)
    );
};

constexpr auto wait = toGFunc<1> | [](auto aVar)
{
    return readMVar | aVar;
};

// For STM

using Integer = int64_t;

template <typename A>
struct IORef
{
    using Data = A;
    using Repr = Atomic<A>;
    std::shared_ptr<Repr> data = std::make_shared<Repr>();
};

template <typename A>
auto initIORef(A a)
{
    return std::make_shared<typename IORef<A>::Repr>(std::move(a));
}

constexpr auto newIORef = toGFunc<1> | [](auto a)
{
    using A = decltype(a);
    return io([a=std::move(a)]
    {
        return IORef<A>{initIORef(std::move(a))};
    });
};

constexpr auto newLock = newIORef | Integer{1};

template <typename A>
auto atomCASImpl(IORef<A> ptr, A old, A new_)
{
    return io(
        [ptr, old, new_]
        {
            auto old_ = old;
            auto result = ptr.data->compareExchangeStrong(old_, new_);
            return result;
        }
    );
}

constexpr auto atomCAS = toGFunc<3> | [](auto const& ptr, auto const& old, auto const& new_)
{
    return atomCASImpl(ptr, old, new_);
};

using ID = Integer;

// Integer => locked : even transaction id or odd, free : odd write stamp
using Lock = IORef<Integer>;

template <typename A>
struct TVar
{
    Lock lock;
    ID id;
    IORef<Integer> writeStamp;
    IORef<A> content;
    IORef<std::vector<MVar<_O_>>> waitQueue;
};

template <typename A>
auto toTVarImpl(Lock lock, ID id, IORef<Integer> writeStamp, IORef<A> content, IORef<std::vector<MVar<_O_>>> waitQueue)
{
    return TVar<A>{std::move(lock), std::move(id), std::move(writeStamp), std::move(content), std::move(waitQueue)};
}

constexpr auto toTVar = toGFunc<5> | [](Lock lock, ID id, IORef<Integer> writeStamp, auto content, IORef<std::vector<MVar<_O_>>> waitQueue)
{
    return toTVarImpl(std::move(lock), std::move(id), std::move(writeStamp), std::move(content), std::move(waitQueue));
};

constexpr auto readIORef = toGFunc<1> | [](auto const& ioRef)
{
    return io([ioRef]() -> typename std::decay_t<decltype(ioRef)>::Data
    {
        return ioRef.data->load();
    });
};

inline IORef<Integer> idref = (newIORef | Integer{0}).run();

inline IO<Integer> newID = []
{
    Id<Integer> cur;
    Id<bool> changed;
    return toTEIO | do_(
        cur <= (readIORef | idref),
        changed <= (atomCAS | idref | cur | (cur+1)),
        ifThenElse | changed | (toTEIO | (Monad<IO>::return_ | (cur+1))) | newID
    );
}();

constexpr auto readLock = readIORef;

constexpr auto hassert = toFunc<> | [](bool result, std::string const& msg)
{
    return io([=]
    {
        if (!result)
        {
            throw std::runtime_error{msg};
        }
        return _o_;
    });
};

constexpr auto newTVarIO = toGFunc<1> | [](auto a)
{
    using A = decltype(a);
    Id<Lock> lock;
    Id<Integer> ws;
    Id<ID> id;
    Id<IORef<Integer>> writeStamp;
    Id<IORef<A>> content;
    Id<IORef<std::vector<MVar<_O_>>>> waitQueue;
    return do_(
		lock <= newLock,
		ws <= (readLock | lock),
		id <= newID,
		hassert | (odd | ws) | "newtvar: value in lock is not odd!",
		writeStamp <= (newIORef | ws),
		content <= (newIORef | a),
		waitQueue <= (newIORef | std::vector<MVar<_O_>>{}),
		return_ | (toTVar | lock | id | writeStamp | content | waitQueue)
    );
};

struct RSE
{
    ID id;
    Lock lock;
    IORef<Integer> writeStamp;
    IORef<std::vector<MVar<_O_>>> waitQueue;
};

inline constexpr bool operator<(RSE const& lhs, RSE const& rhs)
{
    auto result = (lhs.id <compare> rhs.id);
    return result == Ordering::kLT;
}

constexpr auto writeIORef = toGFunc<2> | [](auto const& ioRef, auto const& data)
{
    return io([=]() -> _O_
    {
        ioRef.data->store(data);
        return _o_;
    });
};


using AnyCommitters = std::map<const std::type_index, std::function<void(std::any)>>;
// inline is ok? Move to cpp otherwise. Or choose to use virtual dispatch instead of map.
inline AnyCommitters& anyCommitters()
{
    static AnyCommitters committers{};
    return committers;
}

template <typename A>
struct WSEData;

template <typename A>
class Commiter
{
public:
    auto operator()(std::any wseData) const
    {
        auto [iocontent, v] = std::any_cast<WSEData<A> const&>(wseData);
        (writeIORef | iocontent | v).run();
    }
};

template <typename A>
struct WSEData;

template <typename T>
class CommitterRegister
{
public:
    constexpr CommitterRegister()
    {
        anyCommitters().emplace(std::type_index(typeid(WSEData<T>)), Commiter<T>{});
    }
};

template <typename A>
struct WSEData
{
    WSEData(IORef<A> content_, A newValue_)
    : content{content_}
    , newValue{newValue_}
    {
        static const CommitterRegister<A> dummy;
    }
    IORef<A> content;
    A newValue;
};

struct WSE
{
    Lock lock;
    IORef<Integer> writeStamp;
    IORef<std::vector<MVar<_O_>>> waitQueue;
    std::any wseData;
};

constexpr auto toWSE = toGFunc<5> | [](Lock lock, IORef<Integer> writeStamp, IORef<std::vector<MVar<_O_>>> waitQueue, auto content, auto newValue)
{
    return WSE{lock, writeStamp, waitQueue, WSEData<decltype(newValue)>{content, newValue}};
};

using ReadSet = IORef<std::set<RSE>>;
using WriteSet = IORef<std::map<ID, WSE>>;

using TId = Integer;
using Stamp = Integer;

struct TState
{
    // A transaction id is always the standard thread id.
    TId transId;
    Integer readStamp;
    ReadSet readSet;
    WriteSet writeSet;
};

constexpr auto toTState = toGFunc<4> | [](TId transId, Integer readStamp, ReadSet readSet, WriteSet writeSet)
{
    return TState{transId, readStamp, readSet, writeSet};
};

constexpr auto getWriteSet = toFunc<> | [](TState ts)
{
    return ts.writeSet;
};

// inline ok? unique for each translation unit?
inline IORef<Integer> globalClock{initIORef<Integer>(1)};

constexpr auto incrementGlobalClockImpl = yCombinator | [](auto const& self) -> IO<Integer>
{
    Id<Integer> ov;
    Id<bool> changed;
    return toTEIO | do_(
        ov <= (readIORef | globalClock),
        changed <= (atomCAS | globalClock | ov | (ov+2)),
        ifThenElse || changed
                   || (toTEIO || hspp::Monad<IO>::return_ | (ov+2))
                   || nullary(self)
    );
};

const auto incrementGlobalClock = incrementGlobalClockImpl();

template <typename A>
class Valid : public std::pair<TState, A>
{
public:
    using T = A;
    using std::pair<TState, A>::pair;
};

class Retry : public TState
{};

class Invalid : public TState
{};

template <typename A>
using TResultBase = std::variant<Valid<A>, Retry, Invalid>;

template <typename A>
class TResult : public TResultBase<A>
{
public:
    using DataT = A;
    using std::variant<Valid<A>, Retry, Invalid>::variant;
};

constexpr auto toValid = toGFunc<2> | [](TState ts, auto a)
{
    return TResult<decltype(a)>{Valid<decltype(a)>{ts, a}};
};

template <typename A>
constexpr auto toRetry = toFunc<> | [](TState ts)
{
    return TResult<A>{Retry{ts}};
};

template <typename A, typename Func>
class STM
{
    static_assert(std::is_invocable_v<Func, TState>);
    using RetT = std::invoke_result_t<Func, TState>;
    static_assert(isIOV<RetT>);
    static_assert(std::is_same_v<DataType<RetT>, TResult<A>>);
public:
    constexpr STM(Func func)
    : mFunc{std::move(func)}
    {}
    auto run(TState tState) const -> RetT
    {
        return mFunc(tState);
    }
private:
    Func mFunc;
};

template <typename Func>
constexpr auto toSTMImpl(Func func)
{
    using RetT = std::invoke_result_t<Func, TState>;
    static_assert(isIOV<RetT>);
    using A = typename DataType<RetT>::DataT;
    return STM<A, Func>{func};
}

constexpr auto toSTM = toGFunc<1> | [](auto func)
{
    return toSTMImpl(func);
};

} // namespace concurrent

template <template <template<typename...> typename Type, typename... Ts> class TypeClassT, typename... Args>
struct TypeClassTrait<TypeClassT, concurrent::STM<Args...>>
{
    using Type = TypeClassT<concurrent::STM>;
};

template <typename A, typename Repr>
struct DataTrait<concurrent::STM<A, Repr>>
{
    using Type = A;
};

template <>
class Applicative<concurrent::STM>
{
public:
    constexpr static auto pure = toGFunc<1> | [](auto x)
    {
        return concurrent::toSTM | [=](concurrent::TState tState) { return ioData(concurrent::toValid | tState | x); };
    };
};


template <>
class MonadBase<concurrent::STM>
{
public:
    template <typename A, typename Repr, typename Func>
    constexpr static auto bind(concurrent::STM<A, Repr> const& t1, Func const& func)
    {
        return concurrent::toSTM || [=](concurrent::TState tState)
        {
            Id<concurrent::TResult<A>> tRes;
            auto const dispatchResult = toFunc<> | [=](concurrent::TResult<A> tResult)
            {
                return io([=]
                {
                    using T = typename std::decay_t<decltype(func(std::declval<A>()).run(std::declval<concurrent::TState>()).run())>::DataT;
                    using RetT = concurrent::TResult<T>;
                    return std::visit(overload(
                        [=](concurrent::Valid<A> const& v_) -> RetT
                        {
                            auto [nTState, v] = v_;
                            auto t2 = func(v);
                            return t2.run(nTState).run();
                        },
                        [=](concurrent::Retry const& retry_) -> RetT
                        {
                            return RetT{retry_};
                        },
                        [=](concurrent::Invalid const& invalid_) -> RetT
                        {
                            return RetT{invalid_};
                        }
                    ), static_cast<concurrent::TResultBase<A> const&>(tResult));
                });
            };
            return do_(
                tRes <= t1.run(tState),
                dispatchResult | tRes
            );
        };
    }
};

namespace concurrent
{
constexpr auto newTVar = toGFunc<1> | [](auto a)
{
    return toSTM | [a](TState tState)
    {
        return (newTVarIO | a) >>= (return_ <o> (toValid | tState));
    };
};

constexpr auto putWS = toFunc<> | [](WriteSet ws, ID id, WSE wse)
{
    return io([=]
    {
        std::unique_lock lc{ws.data->lock()};
        ws.data->mValue.emplace(id, wse);
        return _o_;
    });
};

template <typename... Ts>
class IsSTM : public std::false_type
{};
template <typename... Ts>
class IsSTM<STM<Ts...>> : public std::true_type
{};
template <typename T>
constexpr static auto isSTMV = IsSTM<std::decay_t<T>>::value;

constexpr auto putRS = toFunc<> | [](ReadSet rs, RSE entry)
{
    return io([=]
    {
        std::unique_lock lc{rs.data->lock()};
        rs.data->mValue.insert(entry);
        return _o_;
    });
};

constexpr auto lookUpWS = toFunc<> | [](WriteSet ws, ID id)
{
    return io([=]() -> std::optional<WSE>
    {
        std::shared_lock lc{ws.data->lock()};
        auto iter = ws.data->mValue.find(id);
        if (iter == ws.data->mValue.end())
        {
            return {};
        }
        return iter->second;
    });
};

template <typename A>
constexpr auto writeTVarImpl(TVar<A> const& tvar, A const& newValue)
{
    return toSTM | [=](auto tState)
    {
        auto [lock, id, wstamp, content, queue] = tvar;
        WSE wse = (toWSE | lock | wstamp | queue | content | newValue);
        return do_(
            putWS | (getWriteSet | tState) | id | wse,
            return_ | (toValid | tState | _o_)
        );
    };
}

constexpr auto writeTVar = toGFunc<2> | [](auto tvar, auto newValue)
{
    return writeTVarImpl(tvar, newValue);
};

constexpr auto isLocked = even;

constexpr auto isLockedIO = toFunc<> | [](Lock lock)
{
    auto io_ = (readIORef | lock) >>= (return_ <o> isLocked);
    return io_.run();
};

template <typename A>
constexpr auto readTVarImpl(TVar<A> const tvar)
{
    return toSTM | [=](TState tState)
    {
        Id<std::optional<WSE>> mwse;
        auto const handleMWse = toFunc<> | [=](TState tState, std::optional<WSE> mwse_)
        {
            return io([=]() -> TResult<A>
            {
                if (mwse_.has_value())
                {
                    auto const& wse = mwse_.value();
                    auto const& wseData = std::any_cast<WSEData<A> const&>(wse.wseData);
                    return toValid | tState | wseData.newValue;
                }
                else
                {
                    auto [lock, id, wstamp, content, queue] = tvar;

                    auto lockVal = (readLock | lock).run();
                    if (isLocked | lockVal)
                    {
                        return Invalid{tState};
                    }
                    auto const result = (readIORef | content).run();
                    auto lockVal2 = (readLock | lock).run();
                    if ((lockVal != lockVal2) || (lockVal > (tState.readStamp)))
                    {
                        return Invalid{tState};
                    }
                    auto io_ = putRS | tState.readSet | RSE{id, lock, wstamp, queue};
                    io_.run();
                    return toValid | tState | result;
                }
            });
        };
        return do_(
            mwse <= (lookUpWS | (getWriteSet | tState) | tvar.id),
            handleMWse | tState | mwse
        );
    };
}

constexpr auto readTVar = toGFunc<1> | [](auto tvar)
{
    return readTVarImpl(tvar);
};

constexpr auto myTId = io([]() -> TId
{
    TId result = 2 * static_cast<Integer>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    return result;
});

constexpr auto newTState = io([]
{
    auto const readStamp = readIORef(globalClock).run();
    return TState{myTId.run(), readStamp, {}, {}};
});

constexpr auto unlock = toGFunc<1> | [](auto tid)
{
    return mapM_ | [=](auto pair)
    {
        auto [iows, lock] = pair;
        Id<Integer> ws;
        Id<bool> unlocked;
        return do_(
            ws <= (readIORef | iows),
            unlocked <= (atomCAS | lock | tid | ws),
            hassert | unlocked | "COULD NOT UNLOCK LOCK",
            return_ | _o_
        );
    };
};

constexpr auto getLocks = toGFunc<2> | [](auto tid, auto wsList)
{
    return io([=]() -> std::pair<bool, std::vector<std::pair<IORef<Integer>, Lock>>>
    {
        std::vector<std::pair<IORef<Integer>, Lock>> locks;
        for (auto [_, wse] : wsList)
        {
            auto lock = wse.lock;
            auto iowstamp = wse.writeStamp;
            auto lockValue = (readLock | lock).run();
            if (isLocked | lockValue)
            {
                (hassert | (lockValue != tid) | "Locking WS: lock already held by me!!").run();
                return std::make_pair(false, locks);
            }
            else
            {
                auto r = (atomCAS | lock | lockValue | tid).run();
                if (r)
                {
                    locks.emplace_back(iowstamp, lock);
                    continue;
                }
                else
                {
                    return std::make_pair(false, locks);
                }
            }
        }
        return std::make_pair(true, locks);
    });
};

inline auto validateReadSet2(typename ReadSet::Data rseLst, Integer readStamp, Integer myid)
{
    return io([=]() -> bool
    {
        if (rseLst.empty())
        {
            return true;
        }
        for (RSE rse: rseLst)
        {
            auto lock = rse.lock;
            auto iowstamp = rse.writeStamp;
            auto lockValue = (readLock | lock).run();
            if ((isLocked | lockValue) && (lockValue != myid))
            {
                return false;
            }
            else
            {
                if (lockValue != myid)
                {
                    if (lockValue > readStamp)
                    {
                        return false;
                    }
                    else
                    {
                        continue;
                    }
                }
                else
                {
                    auto wstamp = (readIORef | iowstamp).run();
                    if (wstamp > readStamp)
                    {
                        return false;
                    }
                    else
                    {
                        continue;
                    }
                }
            }
        }
        return true;
    });
}

constexpr auto validateReadSet = toGFunc<3> | [](auto ioReadSet, Integer readStamp, TId myId)
{
	auto readS = (readIORef | ioReadSet).run();
	return validateReadSet2(readS, readStamp, myId);
};

// https://en.cppreference.com/w/cpp/utility/any/type
// any_committer

inline auto commitAny(std::any wseData)
{
    auto idx = std::type_index(wseData.type());
    auto iter = anyCommitters().find(idx);
    (hassert | iter != anyCommitters().end() | "Cannot find commiter of this type!").run();
    iter->second(wseData);
}

constexpr auto commitChangesToMemory = toFunc<> | [](Integer wstamp, typename WriteSet::Data wset)
{
    auto commit = [wstamp](auto wsePair)
    {
        return io([=]() -> _O_
        {
            auto [id, wse] = wsePair;
            auto iowstamp = wse.writeStamp;
            (writeIORef | iowstamp | wstamp).run();
            commitAny(wse.wseData);
            return _o_;
        });
    };

    return mapM_ | commit | wset;
};

constexpr auto unblockThreads = toFunc<> | [](std::pair<ID, WSE> const& pair)
{
    return io([=]() -> _O_
    {
        auto const& queue = pair.second.waitQueue;
        auto listMVars = (readIORef | queue).run();
        (mapM_ | [](auto mvar)
        {
            return tryPutMVar | mvar | _o_;
        } | listMVars).run();
        (writeIORef | queue | std::vector<MVar<_O_>>{}).run();
        return _o_;
    });
};

constexpr auto wakeUpBQ = mapM_ | unblockThreads;

constexpr auto validateAndAcquireLocks = toFunc<> | [](Integer readStamp, Integer myId, ReadSet::Data const& readSet)
{
    return io([=]() -> std::pair<bool, std::vector<std::pair<IORef<Integer>, Lock>>>
    {
        std::vector<std::pair<IORef<Integer>, Lock>> locks;
        for (RSE rse : readSet)
        {
            auto lock = rse.lock;
            auto wstamp = rse.writeStamp;
            auto lockValue = (readLock | lock).run();
            if (isLocked | lockValue)
            {
                (hassert | (lockValue != myId) | "validate and lock readset: already locked by me!!!").run();
                return std::make_pair(false, locks);
            }
            else
            {
                if (lockValue > readStamp)
                {
                    return std::make_pair(false, locks);
                }
                else
                {
                    auto r = (atomCAS | lock | lockValue | myId).run();
                    if (r)
                    {
                        locks.emplace_back(wstamp, lock);
                    }
                    else
                    {
                        return std::make_pair(false, locks);
                    }
                }
            }
        }
        return std::make_pair(true, locks);
    });
};

constexpr auto addToWaitQueues = toFunc<> | [](MVar<_O_> mvar)
{
    return mapM_ | [=](RSE rse)
    {
        auto lock = rse.lock;
        auto iomlist = rse.waitQueue;
        return io([=]() -> _O_
        {
            (hassert | (isLockedIO | lock) | "AddtoQueues: tvar not locked!!!").run();
            auto list = (readIORef | iomlist).run();
            list.push_back(mvar);
            return (writeIORef | iomlist | list).run();
        });
    };
};


template <typename A, typename Func>
auto atomicallyImpl(STM<A, Func> const& stmac) -> IO<A>
{
    Id<TState> ts;
    Id<TResult<A>> r;
    auto const dispatchResult = toFunc<> | [=](TResult<A> tResult)
    {
        return io([=]
        {
            return std::visit(overload(
                [=](Valid<A> const& v_) -> A
                {
                    auto [nts, a] = v_;
                    auto wslist = (readIORef | nts.writeSet).run();
                    auto [success, locks] = (getLocks | nts.transId | wslist).run();
                    if (success)
                    {
						auto wstamp = incrementGlobalClock.run();
						auto valid = (validateReadSet | nts.readSet | nts.readStamp | nts.transId).run();
						if (valid)
                        {
                            (commitChangesToMemory | wstamp | wslist).run();
                            (wakeUpBQ | wslist).run();
                            (unlock | nts.transId | locks).run();
                            return a;
                        }
                        else
                        {
                            (unlock | nts.transId | locks).run();
                            return atomicallyImpl(stmac).run();
                        }
                    }
                    else
                    {
						unlock | nts.transId | locks;
						return atomicallyImpl(stmac).run();
                    }
                    return A{};
                },
                [=](Retry const& nts)
                {
                    auto rs = (readIORef | nts.readSet).run();
                    auto lrs = rs;
                    auto [valid, locks] = (validateAndAcquireLocks | nts.readStamp | nts.transId | lrs).run();
                    if (valid)
                    {
                        auto waitMVar = newEmptyMVar<_O_>.run();
                        (addToWaitQueues | waitMVar | lrs).run();
                        (unlock | nts.transId | locks).run();
                        // Looks like this line will block. No one is responsible in waking waitqueues in
                        (takeMVar | waitMVar).run();
                        return atomicallyImpl(stmac).run();
                    }
                    else
                    {
                        unlock | nts.transId | locks;
                        return atomicallyImpl(stmac).run();
                    }
                },
                [=](Invalid const&)
                {
                    return atomicallyImpl(stmac).run();
                }
            ), static_cast<TResultBase<A> const&>(tResult));
        });
    };
    return toTEIO | do_(
        r <= stmac.run(newTState.run()),
        dispatchResult | r
    );
}

constexpr auto atomically = toGFunc<1> | [](auto const& stmac)
{
    return atomicallyImpl(stmac);
};

// todo create a deferred TResult that can be converted to any TResult;
template <typename A>
constexpr auto retry = toSTM | [](TState tState)
{
    return ioData(toRetry<A>(tState));
};

template <typename Data, typename Func>
constexpr auto toTESTMImpl(STM<Data, Func> const& p)
{
    return STM<Data, std::function<IO<TResult<Data>>(TState)>>{[p](TState tState)
    {
        return toTEIO | p.run(tState);
    }};
}

constexpr auto toTESTM = toGFunc<1> | [](auto p)
{
    return toTESTMImpl(p);
};

constexpr auto cloneIORef = toGFunc<1> | [](auto ioRef)
{
    using Data = typename std::decay_t<decltype(ioRef)>::Data;
    Id<Data> value;
    return do_(
        value <= (readIORef | ioRef),
        newIORef | value
    );
};

constexpr auto cloneTState = toFunc<> | [](TState tstate)
{
    Id<WriteSet> nws;
    Id<ReadSet> nrs;
    return do_(
        nws <= (cloneIORef | tstate.writeSet),
        nrs <= (cloneIORef | tstate.readSet),
        return_ | (toTState | tstate.transId | tstate.readStamp | nrs | nws)
    );
};

constexpr auto setUnion = toGFunc<2> | [](auto set1, auto set2)
{
    using Set = decltype(set1);
    static_assert(std::is_same_v<Set, decltype(set2)>);
    set1.merge(set2);
    return set1;
};

constexpr auto mergeTStates = toFunc<> | [](TState ts1, TState ts2)
{
    using Data = typename ReadSet::Data;
    Id<Data> rs1;
    Id<Data> rs2;
    Id<Data> nrs;
    return do_(
        rs1 <= (readIORef | ts1.readSet),
        rs2 <= (readIORef | ts2.readSet),
        nrs = (setUnion | rs1 | rs2),
        writeIORef | ts1.readSet | nrs,
        return_ | ts1
    );
};


template <typename A, typename Repr1, typename Repr2>
auto orElseImpl(STM<A, Repr1> const& s1, STM<A, Repr2> const& s2)
{
    return toSTM | [=](TState tstate)
    {
        using TResType = decltype(s1.run(tstate).run());
        auto dispatchTRes1 = toFunc<> | [=](TResType const& tRes1, TState const& tsCopy)
        {
            return io([=]() -> TResType
            {
                auto retry1 = std::get_if<Retry>(static_cast<TResultBase<A> const*>(&tRes1));
                if (retry1)
                {
                    auto nTState1 = static_cast<TState const&>(*retry1);
                    auto tRes2 = s2.run(tsCopy).run();
                    using TRes2Type = decltype(tRes2);
                    using DataT = typename TRes2Type::DataT;
                    return std::visit(overload(
                        [=](Retry const& retry2) -> TRes2Type
                        {
                            auto nTState2 = static_cast<TState const&>(retry2);
                            auto fTState = (mergeTStates | nTState2 | nTState1).run();
                            return toRetry<DataT> | fTState;
                        },
                        [=](Valid<DataT> const& valid2) -> TRes2Type
                        {
                            auto [nTState2, r] = valid2;
                            auto fTState = (mergeTStates | nTState2 | nTState1).run();
                            return (toValid | fTState | r);
                        },
                        [=](Invalid const&) -> TRes2Type
                        {
                            return tRes2;
                        }), static_cast<TResultBase<A> const&>(tRes2));
                }
                return tRes1;
            });
        };

        Id<TState> tsCopy;
        Id<TResType> tRes1;
        return do_(
            tsCopy <= (cloneTState | tstate),
            tRes1 <= s1.run(tstate),
            dispatchTRes1 | tRes1 | tsCopy
        );
    };
}

constexpr auto orElse = toGFunc<2> | [](auto s1, auto s2)
{
    return orElseImpl(s1, s2);
};

template <typename A>
class TMVar : public TVar<data::Maybe<A>>
{
};

constexpr auto toTMVar = toGFunc<1> | [](auto t)
{
    using A = DataType<DataType<decltype(t)>>;
    return TMVar<A>{t};
};

template <typename A>
constexpr auto newEmptyTMVar = ((newTVar | Maybe<A>{}) >>= (Monad<STM>::return_ <o> toTMVar));

template <typename A>
constexpr auto takeTMVarImpl(TMVar<A> const& t)
{
    auto dispatch = toFunc<> | [=](Maybe<A> m)
    {
        if (m.hasValue())
        {
            return toTESTM | do_(
                (writeTVar | t | data::Maybe<A>{}),
                return_ | m.value()
            );
        }
        return toTESTM | retry<A>;
    };

    Id<Maybe<A>> m;
    auto result = do_(
        m <= (readTVar | t),
        dispatch | m
    );
    using RetT = decltype(result);
    static_assert(isSTMV<RetT>);
    static_assert(std::is_same_v<DataType<RetT>, A>);
    return result;
}

constexpr auto takeTMVar = toGFunc<1> | [](auto t)
{
    return takeTMVarImpl(t);
};

template <typename A>
constexpr auto putTMVarImpl(TMVar<A> const& t, A const& a)
{
    auto dispatch = toFunc<> | [=](Maybe<A> m)
    {
        if (m.hasValue())
        {
            return toTESTM | retry<_O_>;
        }
        return toTESTM | do_(
            (writeTVar | t | (just | a)),
            return_ | _o_
        );
    };

    Id<Maybe<A>> m;
    auto result = do_(
        m <= (readTVar | t),
        dispatch | m
    );
    using RetT = decltype(result);
    static_assert(isSTMV<RetT>);
    static_assert(std::is_same_v<DataType<RetT>, _O_>);
    return result;
}

constexpr auto putTMVar = toGFunc<2> | [](auto t, auto a)
{
    return putTMVarImpl(t, a);
};

template <typename A>
struct Stream;

template <typename A>
using StreamPtr = std::shared_ptr<Stream<A>>;

template <typename A>
using Item = std::pair<A, StreamPtr<A>>;

template <typename A>
using StreamBase = MVar<Item<A>>;

template <typename A>
struct Stream : public StreamBase<A>
{
public:
    Stream(StreamBase<A> sb)
    : StreamBase<A>{std::move(sb)}
    {}
};

template <typename A>
constexpr auto toStreamPtrImpl(StreamBase<A> sb)
{
    return std::make_shared<Stream<A>>(Stream<A>{std::move(sb)});
};

constexpr auto toStreamPtr = toGFunc<1> | [](auto sb)
{
    return toStreamPtrImpl(std::move(sb));
};

template <typename T>
constexpr auto cast = toGFunc<1> | [](auto v)
{
    return static_cast<T>(v);
};

template <typename T>
constexpr auto makeShared = toGFunc<1> | [](auto v)
{
    return std::make_shared<T>(std::move(v));
};

constexpr auto toItem = toGFunc<2> | [](auto a, auto b)
{
    using A = decltype(a);
    using B = decltype(b);
    static_assert(std::is_same_v<StreamPtr<A>, B>);
    return Item<A>{a, std::move(b)};
};

template <typename A>
struct Chan : public std::array<MVar<StreamPtr<A>>, 2>
{};

template <typename A>
constexpr auto toChanImpl(MVar<StreamPtr<A>> a, MVar<StreamPtr<A>> b)
{
    return Chan<A>{a, b};
}

constexpr auto toChan = toGFunc<2> | [](auto a, auto b)
{
    return toChanImpl(std::move(a), std::move(b));
};

template <typename A>
constexpr auto newChanImpl()
{
    Id<StreamBase<A>> hole_;
    Id<StreamPtr<A>> hole;
    Id<MVar<StreamPtr<A>>> readVar, writeVar;
    return do_(
        hole_ <= newEmptyMVar<Item<A>>,
        hole = toStreamPtr | hole_,
        readVar <= (newMVar | hole),
        writeVar <= (newMVar | hole),
        return_ | (toChan | readVar | writeVar)
    );
};

template <typename A>
inline const auto newChan = newChanImpl<A>();

template <typename A>
constexpr auto writeChanImpl(Chan<A> chan, A val)
{
    auto writeVar = chan[1];
    Id<MVar<Item<A>>> newHole_;
    Id<StreamPtr<A>> newHole;
    Id<StreamPtr<A>> oldHole;
    return do_(
        newHole_ <= newEmptyMVar<Item<A>>,
        newHole = makeShared<Stream<A>> || cast<Stream<A>> | newHole_,
        oldHole <= (takeMVar | writeVar),
        putMVar | (cast<StreamBase<A>> | (deref | oldHole)) | (toItem | val | newHole),
        putMVar | writeVar | newHole
    );
};

constexpr auto writeChan = toGFunc<2> | [](auto chan, auto val)
{
    return writeChanImpl(chan, val);
};

template <typename A>
constexpr auto readChanImpl(Chan<A> chan)
{
    auto readVar = chan[0];
    Id<StreamPtr<A>> stream;
    Id<StreamBase<A>> stream_;
    Id<Item<A>> item;
    return do_(
        stream <= (takeMVar | readVar),
        stream_ = cast<StreamBase<A>> || deref | stream,
        item <= (readMVar || stream_),
        putMVar | readVar | (snd | item),
        return_ || fst | item
    );
};

constexpr auto readChan = toGFunc<1> | [](auto chan)
{
    return readChanImpl(chan);
};

template <typename A>
constexpr auto dupChanImpl(Chan<A> chan)
{
    auto writeVar = chan[1];
    Id<StreamPtr<A>> hole;
    Id<MVar<StreamPtr<A>>> newReadVar;
    return do_(
        hole <= (readMVar | writeVar),
        newReadVar <= (newMVar | hole),
        return_ | (toChan | newReadVar | writeVar)
    );
}

constexpr auto dupChan = toGFunc<1> | [](auto chan)
{
    return dupChanImpl(chan);
};
} // namespace concurrent

} // namespace hspp
