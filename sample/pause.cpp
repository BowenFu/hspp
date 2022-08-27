#include "hspp.h"
#include <memory>
#include <variant>
#include <cassert>

#if !defined(FOR_CLANG)
auto expectTrue(bool x)
{
    if (!x)
    {
        throw std::runtime_error{"False in expectedTrue!"};
    }
}

template <typename T>
auto expectEq(T const& l, T const& r)
{
    if (l != r)
    {
        std::stringstream ss;
        ss << l << " != " << r;
        throw std::runtime_error{ss.str()};
    }
}

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
using PauseTPtr = std::shared_ptr<PauseT<M, R>>;

template <template <typename...> class M, typename R>
struct RunT
{
    M<PauseTPtr<M, R>> data;
};

template <typename R>
using PauseIO = PauseT<IO, R>;

namespace hspp
{

template <>
class Applicative<PauseIO>
{
public:
    constexpr static auto pure = toGFunc<1> | [](auto x)
    {
        return DoneT<decltype(x)>(x);
    };
};

template <>
class MonadBase<PauseIO>
{
    template <typename R, typename Func>
    constexpr static auto bind(PauseIO<R> const& t1, Func const& func)
    {
        return std::visit(
            overload(
                [=](DoneT<R> r)
                {
                    return func(r.r);
                },
                [=](RunT<IO, R> r)
                {
                    return RunT<IO, R>{
                        liftM | [=](auto v) { return v >>= func; } | r.data
                    };
                }
            ), t1);
    };
};

// instance MonadTrans PauseT where
//   lift m = RunT $ liftM DoneT m

} // namespace hspp

// pause :: Monad m => PauseT m ()
// pause = DoneT ()

template <template <typename...> class M, typename R>
const auto done = std::make_shared<PauseT<M, R>>(DoneT<R>{});

template <template <typename...> class M, typename R, typename... Ts>
constexpr auto toRunImpl(M<PauseTPtr<M, R>, Ts...> d) -> PauseTPtr<M, R>
{
    return std::make_shared<PauseT<M, R>>(RunT<M, R>{d});
}

constexpr auto toRun = toGFunc<1> | [](auto d)
{
    return toRunImpl(d);
};

const auto pauseExample = toRun || toTEIO | do_(
  putStrLn | "Let's begin",
  putStrLn | "Step 1",
  return_ || (toRun || toTEIO | do_(
    putStrLn | "Step 2",
    return_ || (toRun || toTEIO | do_(
      putStrLn | "Step 3",
      putStrLn | "Yay, we're done!",
      return_ | done<IO, _O_>
    ))
  ))
);

template <template <typename...> class M, typename R>
constexpr auto runNImpl(int n, PauseTPtr<M, R> p) -> M<PauseTPtr<M, R>>;

constexpr auto runN = toGFunc<2> | [](int n, auto p)
{
    return runNImpl(n, p);
};

template <template <typename...> class M, typename R>
constexpr auto runNImpl(int n, PauseTPtr<M, R> p) -> M<PauseTPtr<M, R>>
{
    return toTEIO | io([=]() -> PauseTPtr<M, R>
    {
        if (n == 0)
        {
            return p;
        }
        if (p == done<M, R>)
        {
            return p;
        }
        assert (n >= 0);
        return (std::get<RunT<M, R>>(*p).data >>= ([n](auto p){ return runNImpl(n-1, p); })).run();
    });
}

template <template <typename...> class M, typename R>
constexpr auto fullRunImpl(PauseTPtr<M, R> p) -> M<_O_>
{
    return toTEIO | io([=]() -> _O_
    {
        if (p == done<M, R>)
        {
            return _o_;
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
    static_cast<void>(Applicative<PauseIO>::pure);
    Id<PauseTPtr<IO, _O_>> rest;
    auto main_ = do_(
        rest <= (runN | 2 | pauseExample),
        putStrLn | "=== should print through step 2 ===",
        runN | 1 | rest,
        // remember, IO Foo is just a recipe for Foo, not a Foo itself
        // so we can run that recipe again
        fullRun | rest,
        fullRun | pauseExample
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
