#include "hspp.h"
#include <vector>
#include <list>
#include <gtest/gtest.h>
#include <cmath>
#include <cctype>
#include <thread>

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

TEST(forkIO, 2)
{
    auto setReminder = toFunc<> | [](std::string const& s)
    {
        Id<size_t> n;
        return do_(
            putStr | "Ok, I'll remind you in ",
            print | s,
            putStrLn | " seconds",
            threadDelay | (1000000U * (hspp::read<size_t> | s)),
            print | s,
            putStrLn | " seconds is up! BING!BEL"
        );
        // return do_(
        //     n = (hspp::read<size_t> | s),
        //     putStr | "Ok, I'll remind you in ",
        //     print | n,
        //     putStrLn | " seconds",
        //     threadDelay | (1000000U * n),
        //     print | n,
        //     putStrLn | " seconds is up! BING!BEL"
        // );
    };

    Id<std::string> s;
    // std::string s = "1";
    auto io = forever || do_(
        s <= getLine,
        forkIO || setReminder | s
    );
    // io.run();
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