/*
Reference:

Du Bois, A.R. (2011). An Implementation of Composable Memory Transactions in Haskell. In: Apel, S., Jackson, E. (eds) Software Composition. SC 2011. Lecture Notes in Computer Science, vol 6708. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-22045-6_3
*/

#include <vector>
#include <list>
#include <cmath>
#include <cctype>
#include <thread>
#include <atomic>
#include <shared_mutex>
#include <mutex>
#include <any>
#include <type_traits>
#include <map>
#include <typeindex>
#include <set>

using namespace hspp;
using namespace hspp::data;
using namespace hspp::parser;
using namespace std::literals;
using namespace hspp::doN;

namespace hspp
{

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

template <typename A>
struct MVar;

template <typename A>
class Atomic
{
public:
    Atomic() = default;
    Atomic(A value)
    : mValue{std::move(value)}
    {}
    bool compareExchangeStrong(A& expected, A desired)
    {
        std::unique_lock lc{mLock};
        if (expected != mValue)
        {
            expected = mValue;
            return false;
        }
        mValue = desired;
        return true;
    }
    void store(A const& a)
    {
        std::unique_lock lc{mLock};
        mValue = a;
    }
    A load() const
    {
        std::shared_lock lc{mLock};
        return mValue;
    }
    std::shared_mutex& lock() const
    {
        return mLock;
    }
    A mValue;
private:
    template <typename B>
    friend constexpr auto takeMVarImpl(MVar<B> const& a);
    mutable std::shared_mutex mLock;
};

template <typename A>
struct MVar
{
    using Data = A;
    using T = Atomic<std::optional<A>>;
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
                std::unique_lock lock{a.data->lock()};
                if (a.data->mValue.has_value())
                {
                    auto result = std::move(a.data->mValue.value());
                    a.data->mValue.reset();
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
    return io([a, new_=std::optional<A>{new_}]
    {
        auto old_ = std::optional<A>{};
        while (!a.data->compareExchangeStrong(old_, new_))
        {
            old_ = std::optional<A>{};
            std::this_thread::yield();
        }
        return _o_;
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
        std::unique_lock lock{a.data->lock()};
        if (!a.data->mValue.has_value())
        {
            a.data->mValue = new_;
            return true;
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

// For STM

using Integer = int64_t;

template <typename A>
struct IORef
{
    using Data = A;
    using Repr = Atomic<A>;
    std::shared_ptr<Repr> data = std::make_shared<Repr>();
};

template <typename A>
auto initIORef(A a)
{
    return std::make_shared<typename IORef<A>::Repr>(std::move(a));
}

constexpr auto newIORef = toGFunc<1> | [](auto a)
{
    using A = decltype(a);
    return io([a=std::move(a)]
    {
        return IORef<A>{initIORef(std::move(a))};
    });
};

constexpr auto newLock = newIORef | Integer{1};

template <typename A>
auto atomCASImpl(IORef<A> ptr, A old, A new_)
{
    return io(
        [ptr, old, new_]
        {
            auto old_ = old;
            auto result = ptr.data->compareExchangeStrong(old_, new_);
            return result;
        }
    );
}

constexpr auto atomCAS = toGFunc<3> | [](auto const& ptr, auto const& old, auto const& new_)
{
    return atomCASImpl(ptr, old, new_);
};

using ID = Integer;

// Integer => locked : even transaction id or odd, free : odd write stamp
using Lock = IORef<Integer>;

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
auto toTVarImpl(Lock lock, ID id, IORef<Integer> writeStamp, IORef<A> content, IORef<std::vector<MVar<_O_>>> waitQueue)
{
    return TVar<A>{std::move(lock), std::move(id), std::move(writeStamp), std::move(content), std::move(waitQueue)};
}

constexpr auto toTVar = toGFunc<5> | [](Lock lock, ID id, IORef<Integer> writeStamp, auto content, IORef<std::vector<MVar<_O_>>> waitQueue)
{
    return toTVarImpl(std::move(lock), std::move(id), std::move(writeStamp), std::move(content), std::move(waitQueue));
};

constexpr auto readIORef = toGFunc<1> | [](auto const& ioRef)
{
    return io([ioRef]() -> typename std::decay_t<decltype(ioRef)>::Data
    {
        return ioRef.data->load();
    });
};

IORef<Integer> idref = (newIORef | Integer{0}).run();

IO<Integer> newID = []
{
    Id<Integer> cur;
    Id<bool> changed;
    return toTEIO | do_(
        cur <= (readIORef | idref),
        changed <= (atomCAS | idref | cur | (cur+1)),
        ifThenElse | changed | (toTEIO | (Monad<IO>::return_ | (cur+1))) | newID
    );
}();

constexpr auto readLock = readIORef;

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

constexpr auto newTVarIO = toGFunc<1> | [](auto a)
{
    using A = decltype(a);
    Id<Lock> lock;
    Id<Integer> ws;
    Id<ID> id;
    Id<IORef<Integer>> writeStamp;
    Id<IORef<A>> content;
    Id<IORef<std::vector<MVar<_O_>>>> waitQueue;
    return do_(
		lock <= newLock,
		ws <= (readLock | lock),
		id <= newID,
		hassert | (odd | ws) | "newtvar: value in lock is not odd!",
		writeStamp <= (newIORef | ws),
		content <= (newIORef | a),
		waitQueue <= (newIORef | std::vector<MVar<_O_>>{}),
		return_ | (toTVar | lock | id | writeStamp | content | waitQueue)
    );
};

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
        ioRef.data->store(data);
        return _o_;
    });
};


using AnyCommitters = std::map<const std::type_index, std::function<void(std::any)>>;
// TODO: move definition to cpp
AnyCommitters& anyCommitters()
{
    static AnyCommitters committers{};
    return committers;
}

template <typename A>
struct WSEData;

template <typename A>
class Commiter
{
public:
    auto operator()(std::any wseData) const
    {
        auto [iocontent, v] = std::any_cast<WSEData<A> const&>(wseData);
        (writeIORef | iocontent | v).run();
    }
};

template <typename A>
struct WSEData;

template <typename T>
class CommitterRegister
{
public:
    constexpr CommitterRegister()
    {
        anyCommitters().emplace(std::type_index(typeid(WSEData<T>)), Commiter<T>{});
    }
};

template <typename A>
struct WSEData
{
    WSEData(IORef<A> content_, A newValue_)
    : content{content_}
    , newValue{newValue_}
    {
        static const CommitterRegister<A> dummy;
    }
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

constexpr auto toTState = toGFunc<4> | [](TId transId, Integer readStamp, ReadSet readSet, WriteSet writeSet)
{
    return TState{transId, readStamp, readSet, writeSet};
};

constexpr auto getWriteSet = toFunc<> | [](TState ts)
{
    return ts.writeSet;
};

IORef<Integer> globalClock{initIORef<Integer>(1)};

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
using TResultBase = std::variant<Valid<A>, Retry, Invalid>;

template <typename A>
class TResult : public TResultBase<A>
{
public:
    using DataT = A;
    using std::variant<Valid<A>, Retry, Invalid>::variant;
};

constexpr auto toValid = toGFunc<2> | [](TState ts, auto a)
{
    return TResult<decltype(a)>{Valid<decltype(a)>{ts, a}};
};

template <typename A>
constexpr auto toRetry = toFunc<> | [](TState ts)
{
    return TResult<A>{Retry{ts}};
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

constexpr auto newTVar = toGFunc<1> | [](auto a)
{
    return toSTM | [a](TState tState)
    {
        using A = decltype(a);
        Id<TVar<A>> tvar;
        return do_(
			tvar <= (newTVarIO | a),
			return_ | (toValid | tState | tvar)
        );
    };
};


constexpr auto putWS = toFunc<> | [](WriteSet ws, ID id, WSE wse)
{
    return io([=]
    {
        std::unique_lock lc{ws.data->lock()};
        ws.data->mValue.emplace(id, wse);
        return _o_;
    });
};

template <typename... Ts>
class IsSTM : public std::false_type
{};
template <typename... Ts>
class IsSTM<STM<Ts...>> : public std::true_type
{};
template <typename T>
constexpr static auto isSTMV = IsSTM<std::decay_t<T>>::value;

constexpr auto putRS = toFunc<> | [](ReadSet rs, RSE entry)
{
    return io([=]
    {
        std::unique_lock lc{rs.data->lock()};
        rs.data->mValue.insert(entry);
        return _o_;
    });
};

constexpr auto lookUpWS = toFunc<> | [](WriteSet ws, ID id)
{
    return io([=]() -> std::optional<WSE>
    {
        std::shared_lock lc{ws.data->lock()};
        auto iter = ws.data->mValue.find(id);
        if (iter == ws.data->mValue.end())
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
    Id<Integer> v;
    return do_(
        v <= (readIORef | lock),
        return_ | (isLocked | v)
    ).run();
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


constexpr auto myTId = io([]() -> TId
{
    TId result = 2 * std::hash<std::thread::id>{}(std::this_thread::get_id());
    return result;
});

constexpr auto newTState = io([]
{
    auto const readStamp = readIORef(globalClock).run();
    return TState{myTId.run(), readStamp, {}, {}};
});

constexpr auto unlock = toGFunc<1> | [](auto tid)
{
    return mapM_ | [=](auto pair)
    {
        auto [iows, lock] = pair;
        Id<Integer> ws;
        Id<bool> unlocked;
        return do_(
            ws <= (readIORef | iows),
            print | tid,
            print | ws,
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
                (hassert | (lockValue != tid) | "Locking WS: lock already held by me!!").run();
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
    auto iter = anyCommitters().find(idx);
    (hassert | iter != anyCommitters().end() | "Cannot find commiter of this type!").run();
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
        (mapM_ | [](auto mvar)
        {
            return tryPutMVar | mvar | _o_;
        } | listMVars).run();
        (writeIORef | queue | std::vector<MVar<_O_>>{}).run();
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
                (hassert | (lockValue != myId) | "validate and lock readset: already locked by me!!!").run();
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
                        // Looks like this line will block. No one is responsible in waking waitqueues in
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
            ), static_cast<TResultBase<A> const&>(tResult));
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

// todo create a deferred TResult that can be converted to any TResult;
template <typename A>
constexpr auto retry = toSTM | [](TState tState)
{
    return ioData(toRetry<A>(tState));
};

template <typename Data, typename Func>
constexpr auto toTESTMImpl(STM<Data, Func> const& p)
{
    return STM<Data, std::function<IO<TResult<Data>>(TState)>>{[p](TState tState)
    {
        return toTEIO | p.run(tState);
    }};
}

constexpr auto toTESTM = toGFunc<1> | [](auto p)
{
    return toTESTMImpl(p);
};

constexpr auto cloneIORef = toGFunc<1> | [](auto ioRef)
{
    using Data = typename std::decay_t<decltype(ioRef)>::Data;
    Id<Data> value;
    return do_(
        value <= (readIORef | ioRef),
        newIORef | value
    );
};

constexpr auto cloneTState = toFunc<> | [](TState tstate)
{
    Id<WriteSet> nws;
    Id<ReadSet> nrs;
    return do_(
        nws <= (cloneIORef | tstate.writeSet),
        nrs <= (cloneIORef | tstate.readSet),
        return_ | (toTState | tstate.transId | tstate.readStamp | nrs | nws)
    );
};

constexpr auto setUnion = toGFunc<2> | [](auto set1, auto set2)
{
    using Set = decltype(set1);
    static_assert(std::is_same_v<Set, decltype(set2)>);
    set1.merge(set2);
    return set1;
};

constexpr auto mergeTStates = toFunc<> | [](TState ts1, TState ts2)
{
    using Data = typename ReadSet::Data;
    Id<Data> rs1;
    Id<Data> rs2;
    Id<Data> nrs;
    return do_(
        rs1 <= (readIORef | ts1.readSet),
        rs2 <= (readIORef | ts2.readSet),
        nrs = (setUnion | rs1 | rs2),
        writeIORef | ts1.readSet | nrs,
        return_ | ts1
    );
};


template <typename A, typename Repr1, typename Repr2>
auto orElseImpl(STM<A, Repr1> const& s1, STM<A, Repr2> const& s2)
{
    return toSTM | [=](TState tstate)
    {
        using TResType = decltype(s1.run(tstate).run());
        auto dispatchTRes1 = toFunc<> | [=](TResType const& tRes1, TState const& tsCopy)
        {
            return io([=]() -> TResType
            {
                auto retry1 = std::get_if<Retry>(static_cast<TResultBase<A> const*>(&tRes1));
                if (retry1)
                {
                    auto nTState1 = static_cast<TState const&>(*retry1);
                    auto tRes2 = s2.run(tsCopy).run();
                    using TRes2Type = decltype(tRes2);
                    using DataT = typename TRes2Type::DataT;
                    return std::visit(overload(
                        [=](Retry const& retry2) -> TRes2Type
                        {
                            auto nTState2 = static_cast<TState const&>(retry2);
                            auto fTState = (mergeTStates | nTState2 | nTState1).run();
                            return toRetry<DataT> | fTState;
                        },
                        [=](Valid<DataT> const& valid2) -> TRes2Type
                        {
                            auto [nTState2, r] = valid2;
                            auto fTState = (mergeTStates | nTState2 | nTState1).run();
                            return (toValid | fTState | r);
                        },
                        [=](Invalid const& invalid2) -> TRes2Type
                        {
                            return tRes2;
                        }), static_cast<TResultBase<A> const&>(tRes2));
                }
                return tRes1;
            });
        };

        Id<TState> tsCopy;
        Id<TResType> tRes1;
        return do_(
            tsCopy <= (cloneTState | tstate),
            tRes1 <= s1.run(tstate),
            dispatchTRes1 | tRes1 | tsCopy
        );
    };
}

constexpr auto orElse = toGFunc<2> | [](auto s1, auto s2)
{
    return orElseImpl(s1, s2);
};

template <typename A>
class TMVar : public TVar<data::Maybe<A>>
{
};

constexpr auto toTMVar = toGFunc<1> | [](auto t)
{
    using A = DataType<DataType<decltype(t)>>;
    return TMVar<A>{t};
};

template <typename A>
inline auto newEmptyTMVar = []
{
    Id<TVar<Maybe<A>>> t;
    auto result = do_(
        t <= (newTVar | Maybe<A>{}),
        return_ | (toTMVar | t)
    );
    using RetT = decltype(result);
    static_assert(isSTMV<RetT>);
    static_assert(std::is_same_v<DataType<RetT>, TMVar<A>>);
    return result;
}();

template <typename A>
constexpr auto takeTMVarImpl(TMVar<A> const& t)
{
    auto dispatch = toFunc<> | [=](Maybe<A> m)
    {
        if (m.hasValue())
        {
            return toTESTM | do_(
                (writeTVar | t | data::nothing),
                return_ | m.value()
            );
        }
        return toTESTM | retry<A>;
    };

    Id<Maybe<A>> m;
    auto result = do_(
        m <= (readTVar | t),
        dispatch | m
    );
    using RetT = decltype(result);
    static_assert(isSTMV<RetT>);
    static_assert(std::is_same_v<DataType<RetT>, A>);
    return result;
}

constexpr auto takeTMVar = toGFunc<1> | [](auto t)
{
    return takeTMVarImpl(t);
};

} // namespace concurrent

template <template <template<typename...> typename Type, typename... Ts> class TypeClassT, typename... Args>
struct TypeClassTrait<TypeClassT, concurrent::STM<Args...>>
{
    using Type = TypeClassT<concurrent::STM>;
};

template <typename A, typename Repr>
struct DataTrait<concurrent::STM<A, Repr>>
{
    using Type = A;
};

template <>
class Applicative<concurrent::STM>
{
public:
    constexpr static auto pure = toGFunc<1> | [](auto x)
    {
        using A = decltype(x);
        return concurrent::toSTM | [=](auto tState) { return ioData<concurrent::TResult<A>>(concurrent::toValid | tState | x); };
    };
};


template <>
class MonadBase<concurrent::STM>
{
public:
    template <typename A, typename Repr, typename Func>
    constexpr static auto bind(concurrent::STM<A, Repr> const& t1, Func const& func)
    {
        return concurrent::toSTM || [=](concurrent::TState tState)
        {
            Id<concurrent::TResult<A>> tRes;
            auto const dispatchResult = toFunc<> | [=](concurrent::TResult<A> tResult)
            {
                return io([=]
                {
                    using T = typename std::decay_t<decltype(func(std::declval<A>()).run(std::declval<concurrent::TState>()).run())>::DataT;
                    using RetT = concurrent::TResult<T>;
                    return std::visit(overload(
                        [=](concurrent::Valid<A> const& v_) -> RetT
                        {
                            auto [nTState, v] = v_;
                            auto t2 = func(v);
                            return t2.run(nTState).run();
                        },
                        [=](concurrent::Retry const& retry_) -> RetT
                        {
                            return RetT{retry_};
                        },
                        [=](concurrent::Invalid const& invalid_) -> RetT
                        {
                            return RetT{invalid_};
                        }
                    ), static_cast<concurrent::TResultBase<A> const&>(tResult));
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
