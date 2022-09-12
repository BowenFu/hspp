#include "hspp.h"
#include "monadTrans.h"
#include <cassert>
#include <memory>
#include <variant>

#if !defined(FOR_WIN)

using namespace hspp;
using namespace hspp::doN;
using namespace hspp::data;
using namespace std::literals;

// Developed based on Haskell version at:
// https://www.schoolofhaskell.com/user/school/to-infinity-and-beyond/pick-of-the-week/coroutines-for-streaming/part-3-stacking-interfaces

// newtype Producing o i m r
//   = Producing { resume :: m (ProducerState o i m r) }

// newtype Consuming r m i o
//   = Consuming { provide :: i -> Producing o i m r }

template <template <typename...> class M, typename O, typename I, typename R>
struct Producing;

template <template <typename...> class M, typename O, typename I, typename R>
struct Consuming;

template <template <typename...> class M, typename O, typename I, typename R>
using ConsumingPtr = std::shared_ptr<Consuming<M, O, I, R>>;

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
}

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
    using ConsumingT = Consuming<M, O, I, R>;

    M<ProducerState<M, O, I, R>> resume;
};

constexpr auto resume = toGFunc<1> | [](auto p)
{
    return p.resume;
};

constexpr auto provide = toGFunc<1> | [](auto p)
{
    assert(p);
    return p->provide;
};

template <template <typename...> class M, typename O, typename I, typename R, typename... Ts>
constexpr auto toProducingImpl(M<ProducerState<M, O, I, R>, Ts...> r)
{
    return Producing<M, O, I, R>{static_cast<M<ProducerState<M, O, I, R>>>(r)};
}

constexpr auto toProducing = toGFunc<1> | [](auto r)
{
    return toProducingImpl(r);
};


template <template <typename...> class M, typename O, typename I, typename R>
struct Consuming
{
    using Func = TEFunction<Producing<M, O, I, R>, I>;
    Func provide;
    template <typename F>
    Consuming(F f)
    : provide{toTEFunc<Producing<M, O, I, R>, I>(std::move(f))}
    {}
};

template <typename Func>
class InferConsuming
{
    using FuncTrait = FunctionTrait<decltype(&Func::operator())>;
    using ProducingT = typename FuncTrait::RetT;
    using IT = typename ProducingT::I;
    using I_T = typename FuncTrait::template ArgT<0>;
    static_assert(std::is_same_v<IT, I_T>);
public:
    using ConsumingT = typename ProducingT::ConsumingT;
};

template <typename Func>
constexpr auto toConsumingPtrImpl(Func provide)
{
    using ConsumingT = typename InferConsuming<Func>::ConsumingT;
    return std::make_shared<ConsumingT>(std::move(provide));
}

constexpr auto toConsumingPtr = toGFunc<1> | [](auto provide)
{
    return toConsumingPtrImpl(provide);
};

template <template <typename...> class M, typename O, typename I, typename R>
constexpr auto toConsumingPtr_ = toGFunc<1> | [](auto provide)
{
    using ConsumingT = Consuming<M, O, I, R>;
    return std::make_shared<ConsumingT>(std::move(provide));
};

template <typename O, typename I, typename R>
using ProducingIO = Producing<IO, O, I, R>;

// instance (Functor m) => Functor (Producing o i m) where
//    fmap f p = Producing $ fmap (fmap f) (resume p)

template <template <typename...> class M>
class FunctorProducing
{
public:
    constexpr auto static fmap = toGFunc<2> | [](auto f, auto p)
    {
        return toProducing || ::fmap | (::fmap | f) | (resume | p);
    };
};

template <>
class hspp::Functor<ProducingIO> : public FunctorProducing<IO>
{};

// instance (Functor m) => Functor (ProducerState o i m) where
//   fmap f (Done x) = Done (f x)
//   fmap f (Produced o k) = Produced o $ Consuming (fmap f . provide k)

template <typename O, typename I, typename R>
using ProducerStateIO = ProducerState<IO, O, I, R>;

template <template <typename...> class M, typename O, typename I>
class FunctorProducerState
{
public:
    template <typename Func, typename R>
    constexpr auto static fmap(Func f, ProducerState<M, O, I, R> const& ps)
    {
        using RetT = ProducerState<M, O, I, std::invoke_result_t<Func, R>>;
        return std::visit(overload(
            [&](Done<R> const& d) -> RetT
            {
                return toDone<M, O, I> | (f | d.r);
            },
            [&](Produced<M, O, I, R> const& p) -> RetT {
                auto [o, k] = p;
                return toProduced | o || toConsumingPtr | (::fmap | (f <o> (provide | k)));
            }
        ) , ps);
    }
};

template <typename O, typename I>
class hspp::Functor<ProducerStateIO, O, I> : public FunctorProducerState<IO, O, I>
{};

// instance (Functor m, Monad m) => Applicative (Producing o i m) where
//    pure = return
//    (<*>) = ap

template <template <typename...> class M, typename O, typename I>
class ApplicativeProducing
{
public:
    constexpr static auto pure = toGFunc<1> | [](auto r)
    {
        return toProducing || Monad<M>::return_ | (toDone<M, O, I> | r);
    };
};

template <typename O, typename I>
class hspp::Applicative<ProducingIO, O, I> : public ApplicativeProducing<IO, O, I>
{};

// instance (Monad m) => Monad (Producing o i m) where
//    return x = Producing $ return (Done x)
//    p >>= f = Producing $ resume p >>= \s -> case s of
//      Done x -> resume (f x)
//      Produced o k ->
//       return $ Produced o $ Consuming ((>>= f) . provide k)

template <template <typename...> class M, typename O, typename I>
class MonadProducing
{
public:
    template <typename Func, typename R>
    constexpr auto static bind(Producing<M, O, I, R> p, Func f)
        -> Producing<M, O, I, typename std::invoke_result_t<Func, R>::RT>
    {
        auto result = toProducing || ((resume | p) >>= [=](ProducerState<M, O, I, R> const &s)
        {
            using RetT = std::decay_t<decltype(resume | f(std::declval<R>()))>;
            return std::visit(overload(
                [&](Done<R> const &d) -> RetT
                {
                    return resume | f(d.r);
                },
                [&](Produced<M, O, I, R> const &p) -> RetT
                {
                  auto [o_, k] = p;
                            auto ret = Monad<IO>::return_ ||
                                (toProduced | o_ ||
                                    toConsumingPtr_<M, O, I, typename std::invoke_result_t<Func, R>::RT> |
                                        ([=](auto m) { return bind(m, f); } <o> (provide | k))
                                );
                  return static_cast<RetT>(ret);
                }
            ), s);
        });
        using RetT = Producing<M, O, I, typename std::invoke_result_t<Func, R>::RT>;
        return static_cast<RetT>(std::move(result));
    }
};

template <typename O, typename I>
class hspp::MonadBase<ProducingIO, O, I> : public MonadProducing<IO, O, I>
{};

// instance MonadTrans (Producing o i) where
//    lift = Producing . liftM Done

template <typename O, typename I> class hspp::MonadTrans<Producing, O, I>
{
public:
    // use IO for now.
    constexpr static auto lift = toProducing<o>(liftM | toDone<IO, O, I>);
};

template <template <template <typename...> typename Type, typename... Ts> class TypeClassT, typename O, typename I, typename R>
struct hspp::TypeClassTrait<TypeClassT, ProducingIO<O, I, R>>
{
    using Type = TypeClassT<ProducingIO, O, I>;
};

template <typename O, typename I, typename R>
struct hspp::DataTrait<ProducingIO<O, I, R>>
{
    using Type = R;
};

// instance MFunctor (Producing o i) where
//   hoist f = go where
//     go p = Producing $ f $ liftM map' (resume p)
//     map' (Done r) = Done r
//     map' (Produced o k) = Produced o $ Consuming (go . provide k)

// yield :: Monad m => o -> Producing o i m i
// yield o = Producing $ return $ Produced o $ Consuming return

template <template <typename...> class M, typename O, typename I, typename R>
constexpr auto yield = toFunc<> | [](O o) {
    // for IO
    return toProducing ||
        (Monad<M>::return_ || toProduced | o | (toConsumingPtr_<IO, O, I, R> | Monad<ProducingIO, O, I>::return_));
};

// infixl 0 $$

// ($$) :: Monad m => Producing a b m r -> Consuming r m a b -> m r
// producing $$ consuming = resume producing >>= \s -> case s of
//   Done r -> return r
//   Produced o k -> provide consuming o $$ k

// let the two coroutines hand over control to each other by turn.

template <template <typename...> class M, typename O, typename I, typename R>
constexpr auto SSImpl(Producing<M, O, I, R> const &producing, ConsumingPtr<M, O, I, R> const &consuming) -> M<R>
{
    return (resume | producing) >>= [=](ProducerState<M, O, I, R> const &s)
    {
        using RetT = M<R>;
        return std::visit(overload(
            [&](Done<R> const &d) -> RetT
            {
                return Monad<M>::return_ | d.r;
            },
            [&](Produced<M, O, I, R> const &p) -> RetT
            {
              auto [o, k] = p;
              return SSImpl<M, O, I, R>(provide | consuming | o, k);
            }
        ), s);
    };
}

constexpr auto SS = toGFunc<2> | [](auto const &producing, auto const &consuming)
{
    return SSImpl(producing, consuming);
};

// -- show
// example1 :: Producing String String IO ()
// example1 = do
//   name <- yield "What's your name? "
//   lift $ putStrLn $ "Hello, " ++ name
//   color <- yield "What's your favorite color? "
//   lift $ putStrLn $ "I like " ++ color ++ ", too."

using O = std::string;
using I = std::string;

// coroutine for handling strings
Id<std::string> name, color;
auto const example1 = do_(
    name <= (yield<IO, O, I, std::string> | "What's your name? "),
    lift<Producing, O, I> || putStrLn | ("Hello, " + name),
    color <= (yield<IO, O, I, std::string> | "What's your favorite color? "),
    lift<Producing, O, I> || putStrLn | ("I like " + color + ", too.")
);

// -- this comes in handy for defining Consumers
// foreverK :: Monad m => (a -> m a) -> a -> m r
// foreverK f = go where
//   go a = f a >>= go

constexpr auto foreverK = toGFunc<1> | [](auto f)
{
    using FuncTrait = FunctionTrait<decltype(&decltype(f)::operator())>;
    using ArgT = typename FuncTrait::template ArgT<0>;

    return yCombinator | [=](auto const& self, ArgT a) -> ProducingIO<O, I, _O_>
    {
        return f(a) >>= self;
    };
};

// stdOutIn :: Consuming r IO String String
// stdOutIn = Consuming $ foreverK $ \str -> do
//   lift $ putStrLn str
//   lift getLine >>= yield

auto const foreverKResult = foreverK | [](std::string str) -> Producing<IO, O, I, std::string>
{
    return do_(
        lift<Producing, O, I> || putStrLn | str,
        (lift<Producing, O, I> | getLine) >>= yield<IO, O, I, std::string>
    );
};

// coroutine for handling io
auto const stdOutIn = toConsumingPtr_<IO, O, I, _O_> || foreverKResult;

// stdInOut :: Producing String String IO r
// stdInOut = provide stdOutIn ""
// auto const stdInOut = provide | stdOutIn | "";

// main = example1 $$ stdOutIn

int main()
{
    // FIXME: no sure why, but the following line is needed to instantiate some templates, otherwise we will see linker
    // issue.
    (void)foreverKResult("");
    // let the two coroutines hand over control to each other by return.
    auto io_ = example1 <SS> stdOutIn;
    io_.run();
    return 0;
}

#else
int main()
{
    return 0;
}
#endif // !defined(FOR_WIN)
