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
#include <type_traits>
#include <map>
#include <typeindex>
#include <variant>

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

template <typename A>
constexpr auto tryPutMVarImpl(MVar<A>& a, A new_)
{
    return io([a, new_]
    {
        std::unique_lock lock{a.data->second, std::defer_lock};
        if (lock.try_lock())
        {
            if (!a.data->first.has_value())
            {
                a.data->first = new_;
                return true;
            }
        }
        return false;
    });
}

constexpr auto tryPutMVar = toGFunc<2> | [](auto a, auto new_)
{
    return tryPutMVarImpl(a, new_);
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

using Integer = int64_t;

template <typename A>
struct IORefTrait
{
    constexpr static bool kSUPPORT_CAS = false;
};

template <>
struct IORefTrait<Integer>
{
    constexpr static bool kSUPPORT_CAS = true;
};

template <typename A>
struct IORef
{
    using Data = A;
    using Repr = std::conditional_t<IORefTrait<A>::kSUPPORT_CAS, std::atomic<A>, A>;
    std::shared_ptr<Repr> data = std::make_shared<Repr>();
};

template <typename A>
auto initIORef(A a)
{
    return std::make_shared<typename IORef<A>::Repr>(std::move(a));
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

// constexpr auto newTVar = toGFunc<1> | [](auto a)
// {
//     using A = decltype(a);
//     return io([&]
//     {
//         auto const readStamp = readIORef(globalClock).run();
//         return TVar<A>{myTId.run(), initIORef(readStamp), {}, {}};
//     });
// };

struct RSE
{
    ID id;
    Lock lock;
    IORef<Integer> writeStamp;
    IORef<std::vector<MVar<_O_>>> waitQueue;
};

bool operator<(RSE const& lhs, RSE const& rhs)
{
    auto result = (lhs.id <compare> rhs.id);
    return result == Ordering::kLT;
}

constexpr auto writeIORef = toGFunc<2> | [](auto const& ioRef, auto const& data)
{
    return io([=]() -> _O_
    {
        *ioRef.data = data;
        return _o_;
    });
};

static std::map<const std::type_index, std::function<void(std::any)>> anyCommiters{};

template <typename A>
struct WSEData;

template <typename A>
class Commiter
{
public:
    auto operator()(std::any wseData) const
    {
        auto [iocontent, v] = std::any_cast<WSEData<A> const&>(wseData);
        writeIORef | iocontent | v;
    }
};

template <typename T>
class CommitterRegister
{
public:
    constexpr CommitterRegister()
    {
        anyCommiters.emplace(std::type_index(typeid(T)), Commiter<T>{});
    }
};

template <typename A>
struct WSEData
{
    constexpr static CommitterRegister<A> dummy{};
    IORef<A> content;
    A newValue;
};

struct WSE
{
    Lock lock;
    IORef<Integer> writeStamp;
    IORef<std::vector<MVar<_O_>>> waitQueue;
    std::any wseData;
};

constexpr auto toWSE = toGFunc<5> | [](Lock lock, IORef<Integer> writeStamp, IORef<std::vector<MVar<_O_>>> waitQueue, auto content, auto newValue)
{
    return WSE{lock, writeStamp, waitQueue, WSEData<decltype(newValue)>{content, newValue}};
};

using ReadSet = IORef<std::set<RSE>>;
using WriteSet = IORef<std::map<ID, WSE>>;

using TId = Integer;
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
    return io([ioRef]() -> typename std::decay_t<decltype(ioRef)>::Data
    {
        return *ioRef.data;
    });
};

constexpr auto readLock = readIORef;

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

const auto incrementGlobalClock = incrementGlobalClockImpl();

TEST(atomCAS, integer)
{
    auto a  = IORef<Integer>{initIORef<Integer>(1)};
    Integer old = 1;
    Integer new_ = 2;
    auto io_ = atomCAS | a | old | new_;
    EXPECT_EQ(a.data->load(), 1);
    auto result = io_.run();
    EXPECT_TRUE(result);
    EXPECT_EQ(a.data->load(), 2);
}

TEST(atomCAS, clock)
{
    auto io_ = incrementGlobalClock;
    EXPECT_EQ(globalClock.data->load(), 1);
    io_.run();
    EXPECT_EQ(globalClock.data->load(), 3);
}

template <typename A>
class Valid : public std::pair<TState, A>
{
public:
    using T = A;
    using std::pair<TState, A>::pair;
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
    using std::variant<Valid<A>, Retry, Invalid>::variant;
};

constexpr auto toValid = toGFunc<2> | [](TState ts, auto a)
{
    return TResult<decltype(a)>{Valid<decltype(a)>{ts, a}};
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
    auto run(TState tState) const -> RetT
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

constexpr auto putWS = toFunc<> | [](WriteSet ws, ID id, WSE wse)
{
    return io([=]
    {
        ws.data->emplace(id, wse);
        return _o_;
    });
};

constexpr auto putRS = toFunc<> | [](ReadSet rs, RSE entry)
{
    return io([=]
    {
        rs.data->insert(entry);
        return _o_;
    });
};

constexpr auto lookUpWS = toFunc<> | [](WriteSet ws, ID id)
{
    return io([=]() -> std::optional<WSE>
    {
        auto iter = ws.data->find(id);
        if (iter == ws.data->end())
        {
            return {};
        }
        return iter->second;
    });
};

template <typename A>
constexpr auto writeTVarImpl(TVar<A> const& tvar, A const& newValue)
{
    return toSTM | [=](auto tState)
    {
        auto [lock, id, wstamp, content, queue] = tvar;
        WSE wse = (toWSE | lock | wstamp | queue | content | newValue);
        return do_(
            putWS | (getWriteSet | tState) | id | wse,
            return_ | (toValid | tState | _o_)
        );
    };
}

constexpr auto writeTVar = toGFunc<2> | [](auto tvar, auto newValue)
{
    return writeTVarImpl(tvar, newValue);
};

constexpr auto isLocked = even;

constexpr auto isLockedIO = toFunc<> | [](Lock lock)
{
    auto const v = (readIORef | lock).run();
    return (isLocked | v);
};

template <typename A>
constexpr auto readTVarImpl(TVar<A> const tvar)
{
    return toSTM | [=](TState tState)
    {
        Id<std::optional<WSE>> mwse;
        auto const handleMWse = toFunc<> | [=](TState tState, std::optional<WSE> mwse_)
        {
            return io([=]() -> TResult<A>
            {
                if (mwse_.has_value())
                {
                    auto const& wse = mwse_.value();
                    auto const& wseData = std::any_cast<WSEData<A> const&>(wse.wseData);
                    return toValid | tState | wseData.newValue;
                }
                else
                {
                    auto [lock, id, wstamp, content, queue] = tvar;

                    auto lockVal = (readLock | lock).run();
                    if (isLocked | lockVal)
                    {
                        return Invalid{tState};
                    }
                    auto const result = (readIORef | content).run();
                    auto lockVal2 = (readLock | lock).run();
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
            mwse <= (lookUpWS | (getWriteSet | tState) | tvar.id),
            handleMWse | tState | mwse
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
    constexpr static auto pure = toGFunc<1> | [](auto x)
    {
        using A = decltype(x);
        return toSTM | [=](auto tState) { return ioData<TResult<A>>(toValid | tState | x); };
    };
};


template <>
class MonadBase<STM>
{
public:
    template <typename A, typename Repr, typename Func>
    constexpr static auto bind(STM<A, Repr> const& t1, Func const& func)
    {
        return toSTM || [=](TState tState)
        {
            Id<TResult<A>> tRes;
            auto const dispatchResult = toFunc<> | [=](TResult<A> tResult)
            {
                return io([=]
                {
                    using T = typename std::decay_t<decltype(func(std::declval<A>()).run(std::declval<TState>()).run())>::DataT;
                    using RetT = TResult<T>;
                    return std::visit(overload(
                        [=](Valid<A> const& v_) -> RetT
                        {
                            auto [nTState, v] = v_;
                            auto t2 = func(v);
                            return t2.run(nTState).run();
                        },
                        [=](Retry const& retry_) -> RetT
                        {
                            return toValid | tState | T{};
                        },
                        [=](Invalid const& invalid_) -> RetT
                        {
                            return toValid | tState | T{};
                        }
                    ), tResult);
                });
            };
            return do_(
                tRes <= t1.run(tState),
                dispatchResult | tRes
            );
        };
    }
};

} // namespace hspp

namespace concurrent
{

constexpr auto myTId = io([]() -> TId
{
    return 2 * std::hash<std::thread::id>{}(std::this_thread::get_id());
});

constexpr auto newTState = io([]
{
    auto const readStamp = readIORef(globalClock).run();
    return TState{myTId.run(), readStamp, {}, {}};
});

constexpr auto hassert = toFunc<> | [](bool result, std::string const& msg)
{
    return io([=]
    {
        if (!result)
        {
            throw std::runtime_error{msg};
        }
        return _o_;
    });
};

constexpr auto unlock = toGFunc<1> | [](auto tid)
{
    return mapM_ | [=](auto pair)
    {
        auto [iows, lock] = pair;
        Id<Integer> ws;
        Id<bool> unlocked;
        return do_(
            ws <= (readIORef | iows),
            unlocked <= (atomCAS | lock | tid | ws),
            hassert | unlocked | "COULD NOT UNLOCK LOCK",
            return_ | _o_
        );
    };
};

constexpr auto getLocks = toGFunc<2> | [](auto tid, auto wsList)
{
    return io([=]() -> std::pair<bool, std::vector<std::pair<IORef<Integer>, Lock>>>
    {
        bool success = true;
        std::vector<std::pair<IORef<Integer>, Lock>> locks;
        for (auto [_, wse] : wsList)
        {
            auto lock = wse.lock;
            auto iowstamp = wse.writeStamp;
            auto lockValue = (readLock | lock).run();
            if (isLocked | lockValue)
            {
                hassert | (lockValue != tid) | "Locking WS: lock already held by me!!";
                return std::make_pair(false, locks);
            }
            else
            {
                auto r = (atomCAS | lock | lockValue | tid).run();
                if (r)
                {
                    locks.emplace_back(iowstamp, lock);
                    continue;
                }
                else
                {
                    return std::make_pair(false, locks);
                }
            }
        }
        return std::make_pair(true, locks);
    });
};

auto validateReadSet2(typename ReadSet::Data rseLst, Integer readStamp, Integer myid)
{
    return io([=]() -> bool
    {
        if (rseLst.empty())
        {
            return true;
        }
        for (RSE rse: rseLst)
        {
            auto lock = rse.lock;
            auto iowstamp = rse.writeStamp;
            auto lockValue = (readLock | lock).run();
            if ((isLocked | lockValue) && (lockValue != myid))
            {
                return false;
            }
            else
            {
                if (lockValue != myid)
                {
                    if (lockValue > readStamp)
                    {
                        return false;
                    }
                    else
                    {
                        continue;
                    }
                }
                else
                {
                    auto wstamp = (readIORef | iowstamp).run();
                    if (wstamp > readStamp)
                    {
                        return false;
                    }
                    else
                    {
                        continue;
                    }
                }
            }
        }
        return true;
    });
}

constexpr auto validateReadSet = toGFunc<3> | [](auto ioReadSet, Integer readStamp, TId myId)
{
	auto readS = (readIORef | ioReadSet).run();
	return validateReadSet2(readS, readStamp, myId);
};

// https://en.cppreference.com/w/cpp/utility/any/type
// any_committer

auto commitAny(std::any wseData)
{
    auto idx = std::type_index(wseData.type());
    auto iter = anyCommiters.find(idx);
    hassert | iter != anyCommiters.end() | "Cannot find commiter of this type!";
    iter->second(wseData);
}

constexpr auto commitChangesToMemory = toFunc<> | [](Integer wstamp, typename WriteSet::Data wset)
{
    auto commit = [wstamp](auto wsePair)
    {
        return io([=]() -> _O_
        {
            auto [id, wse] = wsePair;
            auto iowstamp = wse.writeStamp;
            (writeIORef | iowstamp | wstamp).run();
            commitAny(wse.wseData);
            return _o_;
        });
    };

    return mapM_ | commit | wset;
};

constexpr auto unblockThreads = toFunc<> | [](std::pair<ID, WSE> const& pair)
{
    return io([=]() -> _O_
    {
        auto const& queue = pair.second.waitQueue;
        auto listMVars = (readIORef | queue).run();
        mapM_ | [](auto mvar) { return tryPutMVar | mvar | _o_; } | listMVars;
        writeIORef | queue | std::vector<MVar<_O_>>{};
        return _o_;
    });
};

constexpr auto wakeUpBQ = mapM_ | unblockThreads;

constexpr auto validateAndAcquireLocks = toFunc<> | [](Integer readStamp, Integer myId, ReadSet::Data const& readSet)
{
    return io([=]() -> std::pair<bool, std::vector<std::pair<IORef<Integer>, Lock>>>
    {
        std::vector<std::pair<IORef<Integer>, Lock>> locks;
        for (RSE rse : readSet)
        {
            auto lock = rse.lock;
            auto wstamp = rse.writeStamp;
            auto lockValue = (readLock | lock).run();
            if (isLocked | lockValue)
            {
                hassert | (lockValue != myId) | "validate and lock readset: already locked by me!!!";
                return std::make_pair(false, locks);
            }
            else
            {
                if (lockValue > readStamp)
                {
                    return std::make_pair(false, locks);
                }
                else
                {
                    auto r = (atomCAS | lock | lockValue | myId).run();
                    if (r)
                    {
                        locks.emplace_back(wstamp, lock);
                    }
                    else
                    {
                        return std::make_pair(false, locks);
                    }
                }
            }
        }
        return std::make_pair(true, locks);
    });
};

constexpr auto addToWaitQueues = toFunc<> | [](MVar<_O_> mvar)
{
    return mapM_ | [=](RSE rse)
    {
        auto lock = rse.lock;
        auto iomlist = rse.waitQueue;
        return io([=]() -> _O_
        {
            (hassert | (isLockedIO | lock) | "AddtoQueues: tvar not locked!!!").run();
            auto list = (readIORef | iomlist).run();
            list.push_back(mvar);
            return (writeIORef | iomlist | list).run();
        });
    };
};


template <typename A, typename Func>
auto atomicallyImpl(STM<A, Func> const& stmac) -> IO<A>
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
                    auto [nts, a] = v_;
                    auto ti = myTId.run();
                    assert(ti == (nts.transId));
                    (void)ti;
                    (void)a;
                    auto wslist = (readIORef | nts.writeSet).run();
                    auto [success, locks] = (getLocks | nts.transId | wslist).run();
                    if (success)
                    {
						auto wstamp = incrementGlobalClock.run();
						auto valid = (validateReadSet | nts.readSet | nts.readStamp | nts.transId).run();
						if (valid)
                        {
                            (commitChangesToMemory | wstamp | wslist).run();
                            (wakeUpBQ | wslist).run();
                            (unlock | nts.transId | locks).run();
                            return a;
                        }
                        else
                        {
                            (unlock | nts.transId | locks).run();
                            return atomicallyImpl(stmac).run();
                        }
                    }
                    else
                    {
						unlock | nts.transId | locks;
						return atomicallyImpl(stmac).run();
                    }
                    return A{};
                },
                [=](Retry const& nts)
                {
                    auto rs = (readIORef | nts.readSet).run();
                    auto lrs = rs;
                    auto [valid, locks] = (validateAndAcquireLocks | nts.readStamp | nts.transId | lrs).run();
                    if (valid)
                    {
                        auto waitMVar = newEmptyMVar<_O_>.run();
                        (addToWaitQueues | waitMVar | lrs).run();
                        (unlock | nts.transId | locks).run();
                        (takeMVar | waitMVar).run();
                        return atomicallyImpl(stmac).run();
                    }
                    else
                    {
                        unlock | nts.transId | locks;
                        return atomicallyImpl(stmac).run();
                    }
                },
                [=](Invalid const& nts)
                {
                    return atomicallyImpl(stmac).run();
                }
            ), tResult);
        });
    };
    return toTEIO | do_(
        r <= stmac.run(newTState.run()),
        dispatchResult | r
    );
}

constexpr auto atomically = toGFunc<1> | [](auto const& stmac)
{
    return atomicallyImpl(stmac);
};

} // namespace concurrent

template <template <template<typename...> typename Type, typename... Ts> class TypeClassT, typename... Args>
struct hspp::TypeClassTrait<TypeClassT, STM<Args...>>
{
    using Type = TypeClassT<STM>;
};

template <typename A, typename Repr>
struct hspp::DataTrait<STM<A, Repr>>
{
    using Type = A;
};

TEST(atomically, 1)
{
    constexpr auto transferSTM = toFunc<> | [](TVar<Integer> ma, TVar<Integer> mb, Integer d)
    {
        Id<Integer> wa, wb;
        return do_(
            wa <= (readTVar | ma),
            wb <= (readTVar | mb),
            writeTVar | ma | (wa - d),
            writeTVar | mb | (wb + d)
        );
    };
    constexpr auto transfer = toFunc<> | [=](TVar<Integer> ma, TVar<Integer> mb, Integer d)
    {
        return atomically || transferSTM | ma | mb | d;
    };
   // Id<TVar<Integer>> ia, ib;
    // auto io_ = do_(
    //     ia <= newTV
    // );

}