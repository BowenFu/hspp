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
class Pause;

template <template <typename...> class M>
using PausePtr = std::shared_ptr<Pause<M>>;

template <template <typename...> class M>
struct Run
{
    M<PausePtr<M>> data;
};

class Done{};

template <template <typename...> class M>
using PauseBase = std::variant<Run<M>, Done>;

template <template <typename...> class M>
class Pause : public PauseBase<M>
{
public:
    using PauseBase<M>::PauseBase;
};

template <template <typename...> class M>
const auto done = std::make_shared<Pause<M>>(Done{});

template <template <typename...> class M>
constexpr auto toRunImpl(M<PausePtr<M>> d) -> PausePtr<M>
{
    return std::make_shared<Pause<M>>(Run<M>{d});
};

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

// runN :: Monad m => Int -> Pause m -> m (Pause m)
// runN 0 p = return p
// runN _ Done = return Done
// runN n (Run m)
//   | n < 0     = fail "Invalid argument to runN" -- ewww I just used fail.
//   | otherwise = m >>= runN (n - 1)

// fullRun :: Monad m => Pause m -> m ()
// fullRun Done = return ()
// fullRun (Run m) = m >>= fullRun

// -- show Check the result
// main = do
//   rest <- runN 2 pauseExample1
//   putStrLn "=== should print through step 2 ==="
//   Done <- runN 1 rest
//   -- remember, IO Foo is just a recipe for Foo, not a Foo itself
//   -- so we can run that recipe again
//   fullRun rest
//   fullRun pauseExample1

int main()
{
    return 0;
}
