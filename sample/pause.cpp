#include "hspp.h"
#include <memory>
#include <cassert>

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

// data Pause m
//   = Run (m (Pause m))
//   | Done

template <template <typename...> class M>
struct Run;

class Done{};

template <template <typename...> class M>
using Pause = std::variant<Run<M>, Done>;

template <template <typename...> class M>
using PausePtr = std::shared_ptr<Pause<M>>;

template <template <typename...> class M>
struct Run
{
    M<PausePtr<M>> data;
};

template <template <typename...> class M>
const auto done = std::make_shared<Pause<M>>(Done{});

template <template <typename...> class M>
constexpr auto toRunImpl(M<PausePtr<M>> d) -> PausePtr<M>
{
    return std::make_shared<Pause<M>>(Run<M>{d});
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
      return_ | done<IO>
    ))
  ))
);

template <template <typename...> class M>
constexpr auto runNImpl(int n, PausePtr<M> p) -> M<PausePtr<M>>;

constexpr auto runN = toGFunc<2> | [](int n, auto p)
{
    return runNImpl(n, p);
};

template <template <typename...> class M>
constexpr auto runNImpl(int n, PausePtr<M> p) -> M<PausePtr<M>>
{
    return toTEIO | io([=]() -> PausePtr<M>
    {
        if (n == 0)
        {
            return p;
        }
        if (p == done<IO>)
        {
            return done<IO>;
        }
        assert (n >= 0);
        return (std::get<Run<M>>(*p).data >>= ([n](auto p){ return runNImpl(n-1, p); })).run();
    });
}

template <template <typename...> class M>
constexpr auto fullRunImpl(PausePtr<M> p) -> M<_O_>
{
    return toTEIO | io([=]() -> _O_
    {
        if (p == done<IO>)
        {
            return _o_;
        }
        return (std::get<Run<M>>(*p).data >>= ([](auto p){ return fullRunImpl(p); })).run();
    });
};

constexpr auto fullRun = toGFunc<1> | [](auto p)
{
    return fullRunImpl(p);
};

int main()
{
    Id<PausePtr<IO>> rest;
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
