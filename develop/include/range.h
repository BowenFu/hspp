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
