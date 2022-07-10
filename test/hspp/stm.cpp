#include "hspp.h"
#include <vector>
#include <list>
#include <gtest/gtest.h>
#include <cmath>
#include <cctype>
#include <thread>
#include <atomic>
#include <shared_mutex>

using namespace hspp;
using namespace hspp::data;
using namespace hspp::parser;
using namespace std::literals;
using namespace hspp::doN;

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
    using T = std::pair<std::optional<A>, std::shared_mutex>;
    std::shared_ptr<T> data = std::make_shared<T>();
    MVar() = default;
    MVar(A a)
    : data{std::make_shared<T>(a, {})}
    {}
};

static_assert(std::atomic<std::optional<int64_t>>::is_always_lock_free);

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
            if (a.data->first.has_value())
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
            if (!a.data->first.has_value())
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

#if 0
class Message : public std::string{};
class Stop : public MVar<_O_>{};

using LogCommand = std::variant<Message, Stop>;
class Logger : public MVar<LogCommand>{};


auto logger(Logger& m)
{
    IO<_O_> loop0 = []{ return _o_; };
    auto loop = loop0;

    auto const dispatchCmd = toFunc<> | [&loop](LogCommand const& lc)
    {
        return std::visit(overload(
            [&](Message const& msg){
                return toTEIO | do_(print | msg, loop);
            },
            [](Stop s){
                return toTEIO | do_(putStrLn | "logger: stop", putMVar | s | _o_);
            }
        ), lc);
    };

    Id<LogCommand> cmd;
    loop = toTEIO | do_(
        cmd <= (takeMVar | m),
        dispatchCmd | cmd
    );
}

constexpr auto initLoggerImpl()
{
    Id<LogCommand> m;
    Id<Logger> l;
    return do_(
        m <= newEmptyMVar<LogCommand>,
        l = (Logger | m),
        forkIO | (logger | l),
        return_ | l
    );
}
#endif // 0

template <typename A>
struct IORef
{
    std::atomic<A> a;
};

template <typename A>
auto atomCAS(IORef<A>& ptr, A& old, A new_)
{
    return io(
        [&ptr, &old, new_]
        {
            return ptr.a.compare_exchange_strong(old, new_);
        }
    );
}

#if 0
using Integer = int64_t;
using ID = Integer;

template <typename A>
struct TVar
{
    Lock lock;
    ID id;
    IORef<Integer> writeStamp;
    IORef<A> content;
    IORef<std::vector<MVar<_O_>>> waitQueue;
};

// optimize me later
using ReadSet = std::map<ID, std::any>;
using WriteSet = std::map<ID, std::any>;

using TId = Integer;
using Stamp = Integer;

struct TState
{
    TId transId;
    Integer readStamp;
    ReadSet readSet;
    WriteSet writeSet;
};

#endif // 0

TEST(atomCAS, integer)
{
    auto a  = IORef<int>{1};
    auto old = 1;
    auto new_ = 2;
    auto io_ = atomCAS(a, old, new_);
    EXPECT_EQ(a.a.load(), 1);
    auto result = io_.run();
    EXPECT_TRUE(result);
    EXPECT_EQ(a.a.load(), 2);
}