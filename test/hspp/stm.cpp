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

} // namespace concurrent