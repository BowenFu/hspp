
#ifndef HSPP_MONAD_TRANS_H
#define HSPP_MONAD_TRANS_H

namespace hspp
{

/////////// MonadTrans ////////////
// class MonadTrans t where
//     -- | Lift a computation from the argument monad to the constructed monad.
//     lift :: (Monad m) => m a -> t m a
template <template<template<typename...> class, typename...> class T, typename...>
class MonadTrans;

template <template <typename...> class M, typename A>
class MaybeT
{
public:
    M<data::Maybe<A>> runMaybeT;
};

constexpr auto toMaybeT = toGFunc<1> | [](auto ma)
{
    return MaybeT{ma};
};

template <>
class MonadTrans<MaybeT>
{
public:
    constexpr static auto lift = data::Compose{}(toMaybeT, (liftM | data::just));
};

template <template <template<typename...> class, typename...> class M>
class Lift
{
public:
    template <typename Func>
    constexpr auto operator()(Func const& func) const
    {
        using MTType = MonadTrans<M>;
        return MTType::lift(func);
    }
};

template <template <template<typename...> class, typename...> class M>
constexpr inline auto lift = toGFunc<1>(Lift<M>{});

} // namespace hspp

#endif // HSPP_MONAD_TRANS_H
