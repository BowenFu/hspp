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
        if (ta == data::Nothing{})
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
                return data::nothing<R>;
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
                return data::nothing<R>;
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
            [=](data::Nothing) -> R
            {
                return data::Nothing{};
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
        if (ta == data::Nothing{})
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
