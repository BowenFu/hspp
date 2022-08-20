
// Developed based on Haskell version at: https://www.schoolofhaskell.com/user/school/to-infinity-and-beyond/pick-of-the-week/coroutines-for-streaming/part-3-stacking-interfaces

// import Control.Applicative
// import Control.Monad
// import Control.Monad.Morph
// import Control.Monad.Trans.Class

// newtype Producing o i m r
//   = Producing { resume :: m (ProducerState o i m r) }

// newtype Consuming r m i o
//   = Consuming { provide :: i -> Producing o i m r }

template <typename O, typename I, template <typename> class M, typename R>
struct Producing;

template <typename O, typename I, template <typename> class M, typename R, typename Func = std::function<Producing<O, I, M, R>(I)>>
struct Consuming
{
  Func provide;
  static_assert(std::is_invocable_v<Func, I>);
  static_assert(std::is_same_v<std::invoke_result_t<Func, I>, Producing<O, I, M, R>>);
};

template <typename O, typename I, template <typename> class M, typename R>
struct Produced
{
  O o;
  Consuming<O, I, M, R> consuming;
};

template <typename R>
struct Done
{
  R r;
};

template <typename O, typename I, template <typename> class M, typename R>
using ProducerState = std::variant<Produced<O, I, M, R>, Done<R>>;

template <typename O, typename I, template <typename> class M, typename R>
struct Producing
{
  M<ProducerState<O, I, M, R>> resume;
};

// data ProducerState o i m r
//   = Produced o (Consuming r m i o)
//   | Done r

// instance (Functor m) => Functor (Producing o i m) where
//    fmap f p = Producing $ fmap (fmap f) (resume p)

// instance (Functor m) => Functor (ProducerState o i m) where
//   fmap f (Done x) = Done (f x)
//   fmap f (Produced o k) = Produced o $ Consuming (fmap f . provide k)

// instance (Functor m, Monad m) => Applicative (Producing o i m) where
//    pure = return
//    (<*>) = ap

// instance (Monad m) => Monad (Producing o i m) where
//    return x = Producing $ return (Done x)
//    p >>= f = Producing $ resume p >>= \s -> case s of
//      Done x -> resume (f x)
//      Produced o k ->
//       return $ Produced o $ Consuming ((>>= f) . provide k)

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