#include "hspp.h"
#include <vector>
#include <list>
#include <gtest/gtest.h>
#include <cmath>
#include <cctype>
#include <thread>
#include <atomic>
#include <shared_mutex>
#include <any>

using namespace hspp;
using namespace hspp::data;
using namespace hspp::parser;
using namespace std::literals;
using namespace hspp::doN;

namespace concurrent
{

template <typename Func>
auto forkIOImpl(IO<_O_, Func> io_)
{
    return io(
        [io_]
        {
            std::thread t{[io_]{
                io_.run();
            }};
            t.detach();
            return t.get_id();
        }
    );
}

constexpr auto forkIO = toGFunc<1> | [](auto io_){
    return forkIOImpl(io_);
};

constexpr auto threadDelay = toFunc<> | [](size_t microseconds)
{
    return io(
        [microseconds]{
            std::this_thread::sleep_for(std::chrono::microseconds{microseconds});
            return _o_;
        }
    );
};

TEST(forkIO, 1)
{
    auto io_ = do_(
        forkIO | (replicateM_ | 10000U | (putChar | 'A')),
        (replicateM_ | 10000U | (putChar | 'B'))
    );
    io_.run();
}

constexpr auto setReminder = toFunc<> | [](std::string const& s)
{
    Id<size_t> n;
    return do_(
        n = (hspp::read<size_t> | s), // let expression.
        putStr | "Ok, I'll remind you in ",
        print | n,
        putStrLn | " seconds",
        threadDelay | (1000000U * n),
        print | n,
        putStrLn | " seconds is up! BING!BEL"
    );
};

TEST(forkIO, 2)
{
    Id<std::string> s;
    auto io_ = forever || do_(
        s <= getLine,
        forkIO || setReminder | s
    );
    // io.run();
    (void)io_;

}

TEST(forkIO, 3)
{
    IO<_O_> loop0 = []{ return _o_; };
    auto loop = loop0;

    Id<std::string> s;
    loop = toTEIO | do_(
        s <= getLine,
        ifThenElse(s == "exit")
            || loop0
            || toTEIO | (doInner(forkIO || setReminder | s,
                nullary([&]{ return loop;}))) // capturing by ref is important, so that loop is not fixed to loop0.
    );

    // loop.run();
    (void)loop;
}

template <typename A>
struct MVar
{
    using Data = A;
    using T = std::pair<std::optional<A>, std::shared_mutex>;
    std::shared_ptr<T> data = std::make_shared<T>();
    MVar() = default;
    MVar(A a)
    : data{std::make_shared<T>(a, {})}
    {}
};

template <typename A>
constexpr auto newEmptyMVar = io([]{ return MVar<A>{}; });

template <typename A>
constexpr auto newMVarImpl(A a)
{
    return io([&]{ return MVar<A>{std::move(a)}; });
}

constexpr auto newMVar = toGFunc<1> | [](auto a)
{
    return newMVarImpl(a);
};

template <typename A>
constexpr auto takeMVarImpl(MVar<A> const& a)
{
    return io([a]
    {
        while (true)
        {
            {
                std::unique_lock lock{a.data->second};
                if (a.data->first.has_value())
                {
                    auto result = std::move(a.data->first.value());
                    a.data->first.reset();
                    return result;
                }
            }
            std::this_thread::yield();
        }
    });
}

constexpr auto takeMVar = toGFunc<1> | [](auto a)
{
    return takeMVarImpl(a);
};

template <typename A>
constexpr auto putMVarImpl(MVar<A>& a, A new_)
{
    return io([a, new_]
    {
        while (true)
        {
            {
                std::unique_lock lock{a.data->second};
                if (!a.data->first.has_value())
                {
                    a.data->first = new_;
                    return _o_;
                }
            }
            std::this_thread::yield();
        }
    });
}

constexpr auto putMVar = toGFunc<2> | [](auto a, auto new_)
{
    return putMVarImpl(a, new_);
};

// can be optimized to use shared_lock.
constexpr auto readMVar = toGFunc<1> | [](auto m){
    using T = std::decay_t<decltype((takeMVar | m).run())>;
    Id<T> a;
    return do_(
        a <= (takeMVar | m),
        putMVar | m | a,
        return_ | a
    );
};

TEST(MVar, 1)
{
    (void)newMVar;

    Id<MVar<char>> m;
    Id<char> r;
    auto const io_ = do_(
        m <= newEmptyMVar<char>,
        forkIO || putMVar | m | 'x',
        r <= (takeMVar | m),
        print | r
    );
    io_.run();
}

TEST(MVar, 2)
{
    Id<MVar<char>> m;
    Id<char> r;
    auto io_ = do_(
        m <= newEmptyMVar<char>,
        forkIO || doInner(
            putMVar | m | 'x',
            putMVar | m | 'y'
        ),
        r <= (takeMVar | m),
        print | r,
        r <= (takeMVar | m),
        print | r
    );
    io_.run();
}

TEST(MVar, 3)
{
    Id<MVar<char>> m;
    auto io_ = do_(
        m <= newEmptyMVar<char>,
        takeMVar | m
    );
    // stuck
    (void)io_;
    // io_.run();
}

class Message : public std::string{};
class Stop : public MVar<_O_>{};

using LogCommand = std::variant<Message, Stop>;
class Logger : public MVar<LogCommand>{};

constexpr auto toStop = toFunc<> | [](MVar<_O_> mo)
{
    return Stop{std::move(mo)};
};

constexpr auto toLogCommnad = toGFunc<1> | [](auto l)
{
    return LogCommand{std::move(l)};
};

constexpr auto toLogger = toFunc<> | [](MVar<LogCommand> mlc)
{
    return Logger{std::move(mlc)};
};

const auto logger = toFunc<> | [] (Logger m)
{
    auto const dispatchCmd = toGFunc<2> | [](LogCommand const& lc, auto const& loop)
    {
        return std::visit(overload(
            [&](Message const& msg){
                return toTEIO | do_(print | msg, loop());
            },
            [](Stop s){
                return toTEIO | do_(putStrLn | "logger: stop", putMVar | s | _o_);
            }
        ), lc);
    };


    auto loop = yCombinator | [=](auto const& self) -> IO<_O_>
    {
        Id<LogCommand> cmd;
        return toTEIO | do_(
            cmd <= (takeMVar | m),
            dispatchCmd | cmd | self
        );
    };
    return loop();
};

auto initLogger()
{
    Id<MVar<LogCommand>> m;
    Id<Logger> l;
    return do_(
        m <= newEmptyMVar<LogCommand>,
        l = (toLogger | m),
        forkIO | (logger | l),
        return_ | l
    );
}

constexpr auto logMessage = toFunc<> | [](Logger m, std::string s)
{
    return putMVar | m | LogCommand{Message{s}};
};

constexpr auto logStop = toFunc<> | [](Logger m)
{
    Id<MVar<_O_>> s;
    return do_(
        s <= newEmptyMVar<_O_>,
        putMVar | m || (toLogCommnad || toStop | s),
        takeMVar | s
    );
};

TEST(MVar, logger)
{
    Id<Logger> l;
    auto io_ = do_(
        l <= initLogger(),
        logMessage | l | "hello",
        logMessage | l | "bye",
        logStop | l
    );

    testing::internal::CaptureStdout();
    io_.run();
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_EQ(output, "hello\nbye\nlogger: stop\n");
}

template <typename A>
class Async : public MVar<A>
{};

constexpr auto toAsync = toGFunc<1> | [](auto a)
{
    return Async<typename decltype(a)::Data>{a};
};

constexpr auto async = toGFunc<1> | [](auto action){
    using A = std::decay_t<decltype(action.run())>;
    Id<MVar<A>> var;
    Id<A> r;
    return do_(
        var <= newEmptyMVar<A>,
        forkIO || do_( // why doInner does not work here?
            r <= action,
            putMVar | var | r),
        return_ | (toAsync | var)
    );
};

constexpr auto wait = toGFunc<1> | [](auto aVar)
{
    return readMVar | aVar;
};

TEST(Async, 1)
{
    Id<Async<std::string>> a1;
    Id<Async<std::string>> a2;
    Id<std::string> r1;
    Id<std::string> r2;
    auto io_ = do_(
        a1 <= (async | ioData("12345"s)),
        a2 <= (async | ioData("67890"s)),
        r1 <= (wait | a1),
        r2 <= (wait | a2),
        print | r1,
        print | r2
    );

    testing::internal::CaptureStdout();
    io_.run();
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_EQ(output, "12345\n67890\n");
}

template <typename A>
struct IORef
{
    using T = std::atomic<A>;
    std::shared_ptr<T> data = std::make_shared<T>();
};

template <typename A>
auto initIORef(A a)
{
    return std::make_shared<typename IORef<A>::T>(std::move(a));
}

template <typename A>
auto atomCASImpl(IORef<A> ptr, A old, A new_)
{
    return io(
        [ptr, old, new_]
        {
            auto old_ = old;
            return ptr.data->compare_exchange_strong(old_, new_);
        }
    );
}

constexpr auto atomCAS = toGFunc<3> | [](auto const& ptr, auto const& old, auto const& new_)
{
    return atomCASImpl(ptr, old, new_);
};

using Integer = int64_t;
using ID = Integer;

// Integer => locked : even transaction id or odd, free : odd write stamp
class Lock : public IORef<Integer>
{};

template <typename A>
struct TVar
{
    Lock lock;
    ID id;
    IORef<Integer> writeStamp;
    IORef<A> content;
    IORef<std::vector<MVar<_O_>>> waitQueue;
};

template <typename A>
struct RSE
{
    ID id;
    Lock lock;
    IORef<Integer> writeStamp;
    IORef<std::vector<MVar<_O_>>> waitQueue;
};

template <typename A>
struct WSE
{
    Lock lock;
    IORef<Integer> writeStamp;
    IORef<A> content;
    IORef<std::vector<MVar<_O_>>> waitQueue;
    A newValue;
};

constexpr auto toWSE = toGFunc<5> | [](Lock lock, IORef<Integer> writeStamp, auto content, IORef<std::vector<MVar<_O_>>> waitQueue, auto newValue)
{
    return WSE{lock, writeStamp, content, waitQueue, newValue};
};

// optimize me later
class ReadSet
{
    using T = std::map<ID, std::any>;
public:
    std::shared_ptr<T> data = std::make_shared<T>();
};

using WriteSet = ReadSet;

using TId = std::thread::id;
using Stamp = Integer;

struct TState
{
    // A transaction id is always the standard thread id.
    TId transId;
    Integer readStamp;
    ReadSet readSet;
    WriteSet writeSet;
};

constexpr auto getWriteSet = toFunc<> | [](TState ts)
{
    return ts.writeSet;
};

IORef<Integer> globalClock{initIORef<Integer>(1)};

constexpr auto readIORef = toGFunc<1> | [](auto const& ioRef)
{
    return io([&ioRef]{
        return ioRef.data->load();
    });
};

constexpr auto incrementGlobalClockImpl = yCombinator | [](auto const& self) -> IO<Integer>
{
    Id<Integer> ov;
    Id<bool> changed;
    return toTEIO | do_(
        ov <= (readIORef | globalClock),
        changed <= (atomCAS | globalClock | ov | (ov+2)),
        ifThenElse || changed
                   || (toTEIO || hspp::Monad<IO>::return_ | (ov+2))
                   || nullary(self)
    );
};

const auto increamentGlobalClock = incrementGlobalClockImpl();

TEST(atomCAS, integer)
{
    auto a  = IORef<int>{initIORef(1)};
    auto old = 1;
    auto new_ = 2;
    auto io_ = atomCAS | a | old | new_;
    EXPECT_EQ(a.data->load(), 1);
    auto result = io_.run();
    EXPECT_TRUE(result);
    EXPECT_EQ(a.data->load(), 2);
}

TEST(atomCAS, clock)
{
    auto io_ = increamentGlobalClock;
    EXPECT_EQ(globalClock.data->load(), 1);
    io_.run();
    EXPECT_EQ(globalClock.data->load(), 3);
}

template <typename A>
class Valid : public std::pair<TState, A>
{};

constexpr auto toValid = toGFunc<2> | [](TState ts, auto a)
{
    return Valid{ts, a};
};

class Retry : public TState
{};

class Invalid : public TState
{};

template <typename A>
class TResult : public std::variant<Valid<A>, Retry, Invalid>
{
public:
    using DataT = A;
};

template <typename A, typename Func>
class STM
{
    static_assert(std::is_invocable_v<Func, TState>);
    using RetT = std::invoke_result_t<Func, TState>;
    static_assert(isIOV<RetT>);
    static_assert(std::is_same_v<DataType<RetT>, TResult<A>>);
public:
    constexpr STM(Func func)
    : mFunc{std::move(func)}
    {}
    auto run(TState tState) const
    {
        return mFunc(tState);
    }
private:
    Func mFunc;
};

template <typename Func>
constexpr auto toSTMImpl(Func func)
{
    using RetT = std::invoke_result_t<Func, TState>;
    static_assert(isIOV<RetT>);
    using A = typename DataType<RetT>::DataT;
    return STM<A, Func>{func};
}

constexpr auto toSTM = toGFunc<1> | [](auto func)
{
    return toSTMImpl(func);
};

constexpr auto castToPtr = toGFunc<1> | [](auto obj)
{
    return ioData(std::any{obj});
};

constexpr auto putWS = toFunc<> | [](WriteSet ws, ID id, std::any ptr)
{
    return io([=]
    {
        ws.data->emplace(id, ptr);
        return _o_;
    });
};

constexpr auto putRS = putWS;

constexpr auto lookUpWS = toFunc<> | [](WriteSet ws, ID id)
{
    return io([=]() -> Maybe<std::any>
    {
        auto iter = ws.data->find(id);
        if (iter == ws.data->end())
        {
            return nothing;
        }
        return just | iter->second;
    });
};

template <typename A>
constexpr auto writeTVarImpl(TVar<A> const& tvar, A const& newValue)
{
    return toSTM | [=](auto tState)
    {
        auto [lock, id, wstamp, content, queue] = tvar;
        Id<std::any> ptr;
        return do_(
            ptr <= (castToPtr | (toWSE | lock | wstamp | content | queue | newValue)),
            putWS | (getWriteSet | tState) | id | ptr,
            return_ | (toValid | tState | _o_)
        );
    };
}

constexpr auto writeTVar = toGFunc<2> | [](auto tvar, auto newValue)
{
    return writeTVarImpl(tvar, newValue);
};

template <typename A>
constexpr auto readTVarImpl(TVar<A> const tvar)
{
    return toSTM | [=](TState tState)
    {
        Id<Maybe<std::any>> mptr;
        auto const handleMPtr = toFunc<> | [=](TState tState, Maybe<std::any> mptr_)
        {
            return io([=]() -> TResult<A>
            {
                if (mptr_.hasValue())
                {
                    auto const& value = mptr_.value();
                    auto const& wse = std::any_cast<WSE<A> const&>(value);
                    return toValid | tState | wse.newValue;
                }
                else
                {
                    auto [lock, id, wstamp, content, queue] = tvar;

                    auto const lockVal = lock.data.get();
                    auto const isLocked = (lockVal % 2 == 0);
                    if (isLocked)
                    {
                        return Invalid{tState};
                    }
                    auto const result = content.data.get();
                    auto const lockVal2 = lock.data.get();
                    if ((lockVal != lockVal2) || (lockVal > (tState.readStamp)))
                    {
                        return Invalid{tState};
                    }
                    auto io_ = putRS | tState.readSet | RSE{id, lock, wstamp, queue};
                    io_.run();
                    return toValid | tState | result;
                }
            });
        };
        return do_(
            mptr <= (lookUpWS | (getWriteSet | tState) | tvar.id),
            handleMPtr | tState | mptr
        );
    };
}

constexpr auto readTVar = toGFunc<1> | [](auto tvar)
{
    return readTVarImpl(tvar);
};

} // namespace concurrent

namespace hspp
{
    using namespace concurrent;
template <>
class Applicative<STM>
{
public:
    template <typename A, typename Repr>
    constexpr static auto pure(A x)
    {
        return toSTM | [=](auto tState) { return ioData<TResult<A>>(toValid | tState | x); };
    }
};


template <>
class MonadBase<STM>
{
public:
    template <typename A, typename Repr, typename Func>
    constexpr static auto bind(STM<A, Repr> const& t1, Func const& f)
    {
        return toSTM || [=](TState tState)
        {
            Id<TResult<A>> tRes;
            auto const dispatchResult = toFunc<> | [=](TResult<A> tResult)
            {
                return io([=]
                {
                    return std::visit(overload(
                        [=](Valid<A> const& v_) -> TResult<A>
                        {
                            auto [nTState, v] = v_;
                            auto t2 = func(v);
                            return t2(nTState).run();
                        },
                        [=](Retry const&)
                        {
                            return tResult;
                        },
                        [=](Invalid const&)
                        {
                            return tResult;
                        }
                    ), tResult);
                });
            };
            return do_(
                tRes <= t1(tState),
                dispatchResult | tRes
            );
        };
    }
};

} // namespace hspp

namespace concurrent
{

constexpr auto newTState = io([]
{
    return TState{std::this_thread::get_id(), {}, {}, {}};
});

template <typename A, typename Func>
auto atomicallyImpl(STM<A, Func> const& stmac)
{
    Id<TState> ts;
    Id<TResult<A>> r;
    auto const dispatchResult = toFunc<> | [=](TResult<A> tResult)
    {
        return io([=]
        {
            return std::visit(overload(
                [=](Valid<A> const& v_) -> A
                {
                    (void)v_;
                    // auto [nts, a] = v_;
                    // auto transid = nts.transId;
                    // auto writeSet = nts.writeSet;
                    return A{};
                },
                [=](Retry const&)
                {
                    return A{};
                },
                [=](Invalid const& nts)
                {
                    return atomicallyImpl(stmac).run();
                }
            ), tResult);
        });
    };
    return do_(
        ts <= newTState,
        r <= stmac.run(ts),
        dispatchResult | r
    );
}

} // namespace concurrent
