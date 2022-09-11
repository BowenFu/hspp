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

template <typename Repr, typename Ret, typename... Rest>
struct DataTrait<data::Function<Repr, Ret, Rest...>>
{
    using Type = Ret;
};

template <typename A, typename Repr>
struct DataTrait<data::Range<A, Repr>>
{
    using Type = A;
};

template <typename LeftT, typename RightT>
struct DataTrait<data::Either<LeftT, RightT>>
{
    using Type = RightT;
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

template <template <template<typename...> typename, typename...> class TypeClassT, typename... Args>
struct TypeClassTrait<TypeClassT, data::Maybe<Args...>>
{
    using Type = TypeClassT<data::Maybe>;
};

template <template <template<typename...> typename, typename...> class TypeClassT, typename LeftT, typename RightT>
struct TypeClassTrait<TypeClassT, data::Either<LeftT, RightT>>
{
    using Type = TypeClassT<data::Either, LeftT>;
};

template <template <template<typename...> typename, typename...> class TypeClassT, typename... Args>
struct TypeClassTrait<TypeClassT, data::IO<Args...>>
{
    using Type = TypeClassT<data::IO>;
};

template <template <template<typename...> typename Type, typename... Ts> class TypeClassT, typename Repr, typename Ret, typename Arg, typename... Rest>
struct TypeClassTrait<TypeClassT, data::Function<Repr, Ret, Arg, Rest...>>
{
    using Type = TypeClassT<data::Function, Arg>;
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
    constexpr static auto mempty = First<Data>{data::Nothing{}};

    constexpr static auto mappend = data::toFunc<>([](First<Data> lhs, First<Data> rhs)
    {
        return (getFirst | lhs) == data::nothing<Data> ? rhs : lhs;
    });
};

template <typename Data>
class MonoidBase<Last, Data>
{
public:
    constexpr static auto mempty = Last<Data>{data::Nothing{}};

    constexpr static auto mappend = data::toFunc<>([](Last<Data> lhs, Last<Data> rhs)
    {
        return (getLast | rhs) == data::nothing<Data> ? lhs : rhs;
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
        if (lhs.hasValue() && rhs.hasValue())
        {
            using MType = MonoidType<Data>;
            return data::just || MType::mappend | lhs.value() | rhs.value();
        }
        if (!lhs.hasValue())
        {
            return rhs;
        }
        return lhs;
    });
};

template <typename Data>
const data::Maybe<Data> MonoidBase<data::Maybe, Data>::mempty = data::nothing<Data>;

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
        if (!ta.hasValue())
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
        std::transform(in.begin(), in.end(), std::back_inserter(result), [&](auto e){ return std::invoke(func, e); });
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
        return std::tuple_cat(subtuple<0, sizeMinusOne>(std::move(in)), std::make_tuple(std::invoke(func, std::move(last))));
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
        if (in.hasValue())
        {
            return data::just | std::invoke(func, in.value());
        }
        return data::nothing<R>;
    }
};

template <typename LeftT>
class Functor<data::Either, LeftT>
{
public:
    template <typename Func, typename RightT>
    constexpr static auto fmap(Func const& func, data::Either<LeftT, RightT> const& in)
    {
        using NewRightT = std::invoke_result_t<Func, RightT>;
        using ResultT = data::Either<LeftT, NewRightT>;
        if (in.isRight())
        {
            return static_cast<ResultT>(std::invoke(func, in.right()));
        }
        return static_cast<ResultT>(in.left());
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
        return std::tuple_cat(init, std::make_tuple(std::invoke(func, last)));
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
        if (func.hasValue() && in.hasValue())
        {
            return data::just | std::invoke(func.value(), in.value());
        }
        return data::nothing<R>;
    }
};

template <typename LeftT>
class Applicative<data::Either, LeftT> : public Functor<data::Either, LeftT>
{
public:
    constexpr static auto pure = [](auto r)
    {
        return static_cast<data::Either<LeftT, decltype(r)>>(data::toRight | r);
    };
    template <typename Func, typename Arg>
    constexpr static auto ap(data::Either<LeftT, Func> const& func, data::Either<LeftT, Arg> const& in)
    {
        using NewRightT = std::invoke_result_t<Func, Arg>;
        if (func.isRight())
        {
            return func <fmap> in;
        }
        return static_cast<data::Either<LeftT, NewRightT>>(data::toLeft | func.left());
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
        return data::io([=]{ return std::invoke(func.run(), in.run()); });
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
        if (arg.hasValue())
        {
            return std::invoke(func, arg.value());
        }
        return static_cast<R>(data::Nothing{});
    }
};

template <typename LeftT>
class MonadBase<data::Either, LeftT>
{
public:
    template <typename RightT, typename Func>
    constexpr static auto bind(data::Either<LeftT, RightT> const& arg, Func const& func)
    {
        using R = std::invoke_result_t<Func, RightT>;
        if (arg.isRight())
        {
            return std::invoke(func, arg.right());
        }
        return static_cast<R>(data::toLeft | arg.left());
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
        return data::io([=]{ return std::invoke(func, arg.run()).run(); });
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
        if (!ta.hasValue())
        {
            using ResultDataType = decltype(f | ta.value());
            using DataT = DataType<ResultDataType>;
            return ApplicativeType<ResultDataType>::pure(data::nothing<DataT>);
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
    return MType::bind(data, o | evalDeferred<MonadData> | func);
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

// Promote a function to a monad.
// liftM   :: (Monad m) => (a1 -> r) -> m a1 -> m r
constexpr auto liftM = toGFunc<2> | [](auto f, auto m1)
{
    return m1 >>= [=](auto x1){ return return_ | f (x1); };
};

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

template <typename D>
constexpr auto operator<= (Id<D>& id, DeferredPure<D> const& n)
{
    return id = n.mData;
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
    return toFunc<> | [=](T const& t)
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
