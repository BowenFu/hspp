#include "hspp.h"
#include "monadTrans.h"
#include <memory>
#include <variant>
#include <cassert>
#include "common.h"

#if !defined(FOR_CLANG)

using namespace hspp;
using namespace hspp::doN;
using namespace hspp::data;
using namespace std::literals;

// Based on haskell version at https://www.schoolofhaskell.com/school/to-infinity-and-beyond/pick-of-the-week/coroutines-for-streaming/part-1-pause-and-resume

// data PauseT m r
//   = RunT (m (PauseT m r))
//   | DoneT r

template <template <typename...> class M, typename R>
struct RunT;

template <typename R>
class DoneT
{
public:
    R r;
};

template <template <typename...> class M, typename R>
using PauseTBase = std::variant<RunT<M, R>, DoneT<R>>;

template <template <typename...> class M, typename R>
class PauseT : public PauseTBase<M, R>
{
public:
    using PauseTBase<M, R>::PauseTBase;
};

template <template <typename...> class M, typename R>
class PauseTPtr : public std::shared_ptr<PauseT<M, R>>
{
public:
    PauseTPtr(std::shared_ptr<PauseT<M, R>> ptr)
    : std::shared_ptr<PauseT<M, R>>{std::move(ptr)}
    {}
};

template <template <typename...> class M, typename R>
struct RunT
{
    M<PauseTPtr<M, R>> data;
};

template <template <typename...> class M, typename R, typename... Ts>
constexpr auto toDoneTPtrImpl(R r) -> PauseTPtr<M, R>
{
    return std::make_shared<PauseT<M, R>>(DoneT<R>{r});
}

template <template <typename...> class M>
constexpr auto toDoneTPtr = toGFunc<1> | [](auto d)
{
    return toDoneTPtrImpl<M>(d);
};

template <template <typename...> class M, typename R, typename... Ts>
constexpr auto toRunPtrImpl(M<PauseTPtr<M, R>, Ts...> d) -> PauseTPtr<M, R>
{
    return std::make_shared<PauseT<M, R>>(RunT<M, R>{d});
}

constexpr auto toRunTPtr = toGFunc<1> | [](auto d)
{
    return toRunPtrImpl(d);
};

template <typename R>
using PausePtrIO = PauseTPtr<IO, R>;

namespace hspp
{

class ApplicativePausePtr
{
public:
    constexpr static auto pure = toGFunc<1> | [](auto x)
    {
        return DoneT<decltype(x)>(x);
    };
};

template <>
class Applicative<PausePtrIO> : public ApplicativePausePtr
{};

template <template <typename...> class M>
class MonadBasePause
{
public:
    template <typename R, typename Func>
    constexpr static auto bind(PauseTPtr<M, R> const& t1, Func const& func) -> std::invoke_result_t<Func, R>
    {
        using RetT = std::invoke_result_t<Func, R>;
        return std::visit(
            overload(
                [&](DoneT<R> const& r) -> RetT
                {
                    return func(r.r);
                },
                [&](RunT<M, R> const& r) -> RetT
                {
                    return toRunTPtr(
                        liftM | [=](PauseTPtr<M, R> const& v) { return bind(v, func); } | r.data
                    );
                })
            , static_cast<PauseTBase<M, R> const&>(*t1));
    }
};

template <>
class MonadBase<PausePtrIO> : public MonadBasePause<IO>
{};

template <>
class MonadTrans<PauseTPtr>
{
public:
    // use IO for now.
    constexpr static auto lift = toRunTPtr <o> (liftM | toDoneTPtr<IO>);
};

template <template <template<typename...> typename Type, typename... Ts> class TypeClassT, typename R>
struct TypeClassTrait<TypeClassT, PausePtrIO<R>>
{
    using Type = TypeClassT<PausePtrIO>;
};


} // namespace hspp

// pause :: Monad m => PauseT m ()
// pause = DoneT ()
auto const pause = toDoneTPtr<IO>(_o_);

auto const example2 = do_(
  lift<PauseTPtr> || putStrLn | "Step 1",
  pause,
  lift<PauseTPtr> || putStrLn | "Step 2",
  pause,
  (lift<PauseTPtr> || putStrLn | "Step 3")
);

template <template <typename...> class M, typename R>
constexpr auto runNTImpl(int n, PauseTPtr<M, R> p) -> M<PauseTPtr<M, R>>;

constexpr auto runNT = toGFunc<2> | [](int n, auto p)
{
    return runNTImpl(n, p);
};

template <template <typename...> class M, typename R>
constexpr auto runNTImpl(int n, PauseTPtr<M, R> p) -> M<PauseTPtr<M, R>>
{
    return toTEIO | io([=]() -> PauseTPtr<M, R>
    {
        if (n == 0)
        {
            return p;
        }
        if (std::get_if<DoneT<R>>(p.get()))
        {
            return p;
        }
        assert (n >= 0);
        return (std::get<RunT<M, R>>(*p).data >>= ([n](auto p){ return runNTImpl(n-1, p); })).run();
    });
}

template <template <typename...> class M, typename R>
constexpr auto fullRunImpl(PauseTPtr<M, R> p) -> M<_O_>
{
    return toTEIO | io([=]() -> _O_
    {
        if (auto ptr = std::get_if<DoneT<R>>(p.get()))
        {
            return ptr->r;
        }
        return (std::get<RunT<M, R>>(*p).data >>= ([](auto p){ return fullRunImpl(p); })).run();
    });
}

constexpr auto fullRun = toGFunc<1> | [](auto p)
{
    return fullRunImpl(p);
};

int main()
{
    static_cast<void>(Applicative<PausePtrIO>::pure);
    static_cast<void>(MonadTrans<PauseTPtr>::lift);
    Id<PauseTPtr<IO, _O_>> rest;
    auto main_ = do_(
        rest <= (runNT | 2 | example2),
        putStrLn | "=== should print through step 2 ===",
        runNT | 1 | rest,
        // remember, IO Foo is just a recipe for Foo, not a Foo itself
        // so we can run that recipe again
        fullRun | rest,
        fullRun | example2
    );
    main_.run();

    return 0;
}

#else
int main()
{
    return 0;
}
#endif // !defined(FOR_CLANG)
