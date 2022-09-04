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

template <template <typename...> class M, typename O, typename I, typename R, typename Func>
struct Consuming;

template <template <typename...> class M, typename O, typename I, typename R, typename Func = std::function<Producing<M, O, I, R>(I)>>
using ConsumingPtr = std::shared_ptr<Consuming<M, O, I, R, Func>>;

template <template <typename...> class M, typename O, typename I, typename R>
struct Produced
{
  O o;
  ConsumingPtr<M, O, I, R> consuming;
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
  using R = decltype(r);
  return ProducerState<M, O, I, R>{Done<R>{r}};
};

template <template <typename...> class M, typename O, typename I, typename R>
constexpr auto toProducedImpl(O o, ConsumingPtr<M, O, I, R> consuming)
{
  return ProducerState<M, O, I, R>{Produced<M, O, I, R>{o, consuming}};
};

constexpr auto toProduced = toGFunc<2> | [](auto o, auto consuming)
{
  return toProducedImpl(o, consuming);
};

template <template <typename...> class M, typename O, typename I, typename R>
struct Producing
{
  using OT = O;
  using IT = I;
  using RT = R;
  template <typename Func>
  using ConsumingT = Consuming<M, O, I, R, Func>;

  M<ProducerState<M, O, I, R>> resume;
};

constexpr auto resume = toGFunc<1> | [](auto p)
{
  return p.resume;
};

constexpr auto provide = toGFunc<1> | [](auto p)
{
  return p.provide;
};

template <template <typename...> class M, typename O, typename I, typename R>
constexpr auto toProducingImpl(M<ProducerState<M, O, I, R>> r)
{
  return Producing<M, O, I, R>{std::move(r)};
};

constexpr auto toProducing = toGFunc<1> | [](auto r)
{
  return toProducingImpl(r);
};


template <template <typename...> class M, typename O, typename I, typename R, typename Func = std::function<Producing<M, O, I, R>(I)>>
struct Consuming
{
  Func provide;
private:
  static_assert(std::is_invocable_v<Func, I>);
  using RT = std::invoke_result_t<Func, I>;
  using P = Producing<M, O, I, R>;
  constexpr static bool same = std::is_same_v<P, RT>;
  static_assert(std::is_same_v<P, RT>);
  constexpr static bool constructible = std::is_constructible_v<P, RT>;
  static_assert(same || constructible);
};

template <typename Func>
class InferConsuming
{
  using FuncTrait = FunctionTrait<decltype(&Func::operator())>;
  using ProducingT = typename FuncTrait::Ret;
  using IT = typename ProducingT::I;
  using I_T = typename FuncTrait::template Arg<0>;
  static_assert(std::is_same_v<IT, I_T>);
public:
  using ConsumingT = typename ProducingT::template ConsumingT<Func>;
};

template <typename Func>
constexpr auto toConsumingPtrImpl(Func provide)
{
  using ConsumingT = typename InferConsuming<Func>::ConsumingT;
  return std::make_shared<ConsumingT>(provide);
};

constexpr auto toConsumingPtr = toGFunc<1> | [](auto provide)
{
  return toConsumingPtrImpl(provide);
};

template <typename O, typename I, typename R>
using ProducingIO = Producing<IO, O, I, R>;

// instance (Functor m) => Functor (Producing o i m) where
//    fmap f p = Producing $ fmap (fmap f) (resume p)

template <template <typename...> class M, typename O, typename I, typename R>
class FunctorProducing
{
public:
  constexpr auto static fmap = toGFunc<2> | [](auto f, auto p)
  {
    return toProducing || ::fmap | (::fmap | f) | (resume | p);
  };
};

template <typename O, typename I, typename R>
class Functor<ProducingIO, O, I, R> : public FunctorProducing<IO, O, I, R>
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
        return toProduced | o || toConsumingPtr | (::fmap | (f <o> (k.provide)));
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

template <template <typename...> class M, typename O, typename I, typename R>
class ApplicativeProducing
{
public:
  constexpr static auto pure = toGFunc<1> | [](auto r)
  {
    return toProducing || Monad<M>::return_ | (toDone<M, O, I> | r);
  };
};

template <typename O, typename I, typename R>
class Applicative<ProducingIO, O, I, R> : public ApplicativeProducing<IO, O, I, R>
{};

// instance (Monad m) => Monad (Producing o i m) where
//    return x = Producing $ return (Done x)
//    p >>= f = Producing $ resume p >>= \s -> case s of
//      Done x -> resume (f x)
//      Produced o k ->
//       return $ Produced o $ Consuming ((>>= f) . provide k)

template <template <typename...> class M, typename O, typename I, typename R>
class MonadProducing
{
public:
  template <typename Func>
  constexpr auto static bind(Producing<M, O, I, R> p, Func f)
  {
    return toProducing || (resume | p) >>= [=](ProducerState<M, O, I, R> const& s)
    {
      return std::visit(overload(
        [&](Done<R> const& d){ return resume | (f | d.r); },
        [&](Produced<M, O, I, R> const& p){
          auto [o, k] = p;
          return toProduced | o || toConsumingPtr | ([=](auto m){ return bind(m, f); } <o> (k.provide));
        }
      ) , s);
    };
  };
};

template <typename O, typename I, typename R>
class MonadBase<ProducingIO, O, I, R> : public MonadProducing<IO, O, I, R>
{};


// instance MonadTrans (Producing o i) where
//    lift = Producing . liftM Done

template <typename O, typename I, typename R>
class MonadTrans<Producing, O, I, R>
{
public:
    // use IO for now.
    constexpr static auto lift = toProducing <o> (liftM | toDone<IO, O, I, R>);
};

// instance MFunctor (Producing o i) where
//   hoist f = go where
//     go p = Producing $ f $ liftM map' (resume p)
//     map' (Done r) = Done r
//     map' (Produced o k) = Produced o $ Consuming (go . provide k)


// yield :: Monad m => o -> Producing o i m i
// yield o = Producing $ return $ Produced o $ Consuming return

template <template <typename...> class M, typename O, typename I, typename R>
constexpr auto yield = toGFunc<1> | [](auto o)
{
  // for IO
  return toProducing || (Monad<M>::return_ || toProduced | o | (toConsumingPtr | Monad<ProducingIO, O, I, R>::return_));
};

// infixl 0 $$

// ($$) :: Monad m => Producing a b m r -> Consuming r m a b -> m r
// producing $$ consuming = resume producing >>= \s -> case s of
//   Done r -> return r
//   Produced o k -> provide consuming o $$ k

template <template <typename...> class M, typename O, typename I, typename R>
constexpr auto SS = toGFunc<2> | [](auto producing, auto consuming)
{
  return toProducing || (Monad<M>::return_ || toProduced | o | (toConsumingPtr | Monad<ProducingIO, O, I>::return_));
  return (resume | producing) >>= [=](auto s)
  {
    return std::visit(overload(
      [&](Done<R> const& d){ return Monad<M>::return_ | d.r; },
      [&](Produced<M, O, I, R> const& p){
        auto [o, k] = p;
        return SS<M, O, I, R>(provide | consuming | o, k);
      }
    ) , s);
  };
};


// -- show
// example1 :: Producing String String IO ()
// example1 = do
//   name <- yield "What's your name? "
//   lift $ putStrLn $ "Hello, " ++ name
//   color <- yield "What's your favorite color? "
//   lift $ putStrLn $ "I like " ++ color ++ ", too."

// using O = std::string;
// using I = std::string;
// using R = hspp::data::IO<std::__1::variant<Produced<IO, std::__1::basic_string<char>, std::__1::basic_string<char>, std::__1::basic_string<char> >, Done<std::__1::basic_string<char> > >>;

// Id<std::string> name, color;
// const auto example1 = do_(
//   name <= (yield<IO, O, I, R> | "What's your name? "),
//   lift<IO> || putStrLn | ("Hello, " + name),
//   color <= (yield<IO, std::string, std::string, std::string> | "What's your favorite color? "),
//   lift<IO> || putStrLn | ("I like " + color + ", too.")
// );

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
    (void)FunctorProducing<IO, std::string, std::string, _O_>::fmap;
    return 0;
}