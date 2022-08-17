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
