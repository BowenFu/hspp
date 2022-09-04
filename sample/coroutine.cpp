#include "hspp.h"
#include "monadTrans.h"
#include <memory>
#include <variant>
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

// Developed based on Haskell version at: https://www.schoolofhaskell.com/user/school/to-infinity-and-beyond/pick-of-the-week/coroutines-for-streaming/part-3-stacking-interfaces

// newtype Producing o i m r
//   = Producing { resume :: m (ProducerState o i m r) }

// newtype Consuming r m i o
//   = Consuming { provide :: i -> Producing o i m r }

template <template <typename...> class M, typename O, typename I, typename R>
struct Producing;

template <template <typename...> class M, typename O, typename I, typename R, typename Func = std::function<Producing<M, O, I, R>(I)>>
struct Consuming
{
  Func provide;
  static_assert(std::is_invocable_v<Func, I>);
  static_assert(std::is_same_v<std::invoke_result_t<Func, I>, Producing<M, O, I, R>>);
};

template <template <typename...> class M, typename O, typename I, typename R>
constexpr auto toConsuming = toGFunc<1> | [](auto provide)
{
  return Consuming<M, O, I, R, decltype(provide)>{provide};
};

template <template <typename...> class M, typename O, typename I, typename R>
struct Produced
{
  O o;
  Consuming<M, O, I, R> consuming;
};

template <typename R>
struct Done
{
  R r;
};

template <template <typename...> class M, typename O, typename I, typename R>
using ProducerState = std::variant<Produced<M, O, I, R>, Done<R>>;

template <template <typename...> class M, typename O, typename I>
constexpr auto toDone = toGFunc<1> | [](auto r)
{
  return ProducerState{Done{r}};
};

template <template <typename...> class M, typename O, typename I>
constexpr auto toProduced = toGFunc<2> | [](auto o, auto consuming)
{
  return ProducerState{Produced{o, consuming}};
};

template <template <typename...> class M, typename O, typename I, typename R>
struct Producing
{
  M<ProducerState<M, O, I, R>> resume;
};

constexpr auto resume = toGFunc<1> | [](auto p)
{
  return p.resume;
};

constexpr auto toProducing = toGFunc<1> | [](auto r)
{
  return Producing{r};
};

template <typename O, typename I, typename R>
using ProducingIO = Producing<IO, O, I, R>;

// instance (Functor m) => Functor (Producing o i m) where
//    fmap f p = Producing $ fmap (fmap f) (resume p)

class FunctorProducing
{
public:
  constexpr auto static fmap = toGFunc<2> | [](auto f, auto p)
  {
    return toProducing || ::fmap | (::fmap | f) | (resume | p);
  };
};

template <>
class Functor<ProducingIO> : public FunctorProducing
{};

// instance (Functor m) => Functor (ProducerState o i m) where
//   fmap f (Done x) = Done (f x)
//   fmap f (Produced o k) = Produced o $ Consuming (fmap f . provide k)

template <typename O, typename I, typename R>
using ProducerStateIO = ProducerState<IO, O, I, R>;

template <template <typename...> class M>
class FunctorProducerState
{
public:
  template <typename Func, typename O, typename I, typename R>
  constexpr auto static fmap(Func f, ProducerState<M, O, I, R> const& ps)
  {
    return std::visit(overload(
      [&](Done<R> const& d){ return toDone<M, O, I> | (f | d.r); },
      [&](Produced<M, O, I, R> const& p){
        auto [o, k] = p;
        return toProduced<M, O, I> | o || toConsuming<M, O, I, R> | (::fmap | (f <o> (k.provide)));
      }
    ) , ps);
  };
};

template <>
class Functor<ProducerStateIO> : public FunctorProducerState<IO>
{};

// instance (Functor m, Monad m) => Applicative (Producing o i m) where
//    pure = return
//    (<*>) = ap

template <template <typename...> class M>
class ApplicativeProducing
{
public:
  template <typename O, typename I, typename R>
  constexpr auto static pure(R r)
  {
//    return x = Producing $ return (Done x)
    return toProducing || Monad<M>::return_ | (toDone<M, O, I> | r);
  };
};

template <>
class Applicative<ProducingIO> : public ApplicativeProducing<IO>
{};

// instance (Monad m) => Monad (Producing o i m) where
//    return x = Producing $ return (Done x)
//    p >>= f = Producing $ resume p >>= \s -> case s of
//      Done x -> resume (f x)
//      Produced o k ->
//       return $ Produced o $ Consuming ((>>= f) . provide k)

template <template <typename...> class M>
class MonadProducing
{
public:
  template <typename O, typename I, typename R>
  constexpr auto static return_(R r)
  {
    return ApplicativeProducing<M>::pure(r);
  };
  template <typename O, typename I, typename R, typename Func>
  constexpr auto static bind(Producing<M, O, I, R> p, Func f)
  {
    return toProducing || (resume | p) >>= [=](ProducerState<M, O, I, R> const& s)
    {
      return std::visit(overload(
        [&](Done<R> const& d){ return resume | (f | d.r); },
        [&](Produced<M, O, I, R> const& p){
          auto [o, k] = p;
          return toProduced<M, O, I> | o || toConsuming<M, O, I, R> | ([=](auto m){ return bind(m, f); } <o> (k.provide));
        }
      ) , s);
    };
  };
};

template <>
class MonadBase<ProducingIO> : public MonadProducing<IO>
{};


// instance MonadTrans (Producing o i) where
//    lift = Producing . liftM Done

// instance MFunctor (Producing o i) where
//   hoist f = go where
//     go p = Producing $ f $ liftM map' (resume p)
//     map' (Done r) = Done r
//     map' (Produced o k) = Produced o $ Consuming (go . provide k)

// yield :: Monad m => o -> Producing o i m i
// yield o = Producing $ return $ Produced o $ Consuming return

// infixl 0 $$

// ($$) :: Monad m => Producing a b m r -> Consuming r m a b -> m r
// producing $$ consuming = resume producing >>= \s -> case s of
//   Done r -> return r
//   Produced o k -> provide consuming o $$ k


// -- show
// example1 :: Producing String String IO ()
// example1 = do
//   name <- yield "What's your name? "
//   lift $ putStrLn $ "Hello, " ++ name
//   color <- yield "What's your favorite color? "
//   lift $ putStrLn $ "I like " ++ color ++ ", too."

// -- this comes in handy for defining Consumers
// foreverK :: Monad m => (a -> m a) -> a -> m r
// foreverK f = go where
//   go a = f a >>= go

// stdOutIn :: Consuming r IO String String
// stdOutIn = Consuming $ foreverK $ \str -> do
//   lift $ putStrLn str
//   lift getLine >>= yield

// stdInOut :: Producing String String IO r
// stdInOut = provide stdOutIn ""

// main = example1 $$ stdOutIn

int main()
{
    (void)FunctorProducing::fmap;
    return 0;
}