#include <gtest/gtest.h>
#include "hspp.h"
#include "concurrent.h"

using namespace hspp::data;
using namespace std::literals;
using namespace hspp::doN;
using namespace hspp::concurrent;

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
    auto io_ = newEmptyMVar<char> >>= takeMVar;
    // Expected to be stuck.
    (void)io_;
}

class Message : public std::string{};
class Stop : public MVar<_O_>{};

bool operator==(Stop const& lhs, Stop const& rhs)
{
    return lhs.data->load() == rhs.data->load();
}

bool operator!=(Stop const& lhs, Stop const& rhs)
{
    return !(lhs == rhs);
}

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

auto const logger = toFunc<> | [] (Logger m)
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

TEST(Async, 1)
{
    using hspp::concurrent::wait;
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

TEST(newID, 1)
{
    EXPECT_EQ(newID.run(), 1);
    EXPECT_EQ(newID.run(), 2);
}

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

// Some tests borrowed from https://www.schoolofhaskell.com/school/advanced-haskell/beautiful-concurrency/3-software-transactional-memory

using Account = TVar<Integer>;

constexpr auto limitedWithdrawSTM = toFunc<> | [](Account acc, Integer amount)
{
    Id<Integer> bal;
    auto result = do_(
        bal <= (readTVar | acc),
        ifThenElse | (amount >0 && amount > bal)
            | (toTESTM | retry<_O_>)
            | (toTESTM |(writeTVar | acc | (bal - amount)))
    );
    using RetT = decltype(result);
    static_assert(isSTMV<RetT>);
    static_assert(std::is_same_v<DataType<RetT>, _O_>);
    return result;
};

constexpr auto withdrawSTM = toFunc<> | [](Account acc, Integer amount)
{
    Id<Integer> bal;
    auto result = do_(
        bal <= (readTVar | acc),
        writeTVar | acc | (bal - amount)
    );
    using RetT = decltype(result);
    static_assert(isSTMV<RetT>);
    static_assert(std::is_same_v<DataType<RetT>, _O_>);
    return result;
};

constexpr auto depositSTM = toFunc<> | [](Account acc, Integer amount)
{
    return withdrawSTM | acc | (- amount);
};

constexpr auto transfer = toFunc<> | [](Account from, Account to, Integer amount)
{
    auto result = atomically | do_(
        depositSTM | to | amount,
        limitedWithdrawSTM | from | amount
    );
    using RetT = decltype(result);
    static_assert(isIOV<RetT>);
    static_assert(std::is_same_v<DataType<RetT>, _O_>);
    return result;
};

constexpr auto showAccount = toFunc<> | [](Account acc)
{
    auto result = atomically | (readTVar | acc);
    using RetT = decltype(result);
    static_assert(isIOV<RetT>);
    static_assert(std::is_same_v<DataType<RetT>, Integer>);
    return result;
};

TEST(atomically, 1)
{
    Id<Account> from, to;
    Id<Integer> v1, v2;
    auto io_ = do_(
        from <= (atomically | (newTVar | Integer{200})),
        to   <= (newTVarIO | Integer{100}),
        transfer | from | to | 50,
        v1 <= (showAccount | from),
        v2 <= (showAccount | to),
        hassert | (v1 == 150) | "v1 should be 150",
        hassert | (v2 == 150) | "v2 should be 150"
    );

    io_.run();
}


constexpr auto delayDeposit = toFunc<> | [](Account acc, Integer amount)
{
    Id<Integer> bal;
    return do_(
        putStr | "Getting ready to deposit money...hunting through pockets...\n",
        threadDelay | 3000,
        putStr | "OK! Depositing now!\n",
        atomically | do_(
            bal <= (readTVar | acc),
            writeTVar | acc | (bal + amount)
        )
    );
};

TEST(atomically, 2)
{
    Id<Account> acc;
    Id<Integer> bal;
    auto io_ = do_(
        acc <= (newTVarIO | Integer{100}),
        forkIO | (delayDeposit | acc | 1),
        putStr | "Trying to withdraw money...\n",
        atomically | (limitedWithdrawSTM | acc | 101),
        putStr | "Successful withdrawal!\n",
        bal <= (showAccount | acc),
        hassert | (bal == 0) | "bal should be 0"
    );

    io_.run();
}

// (limitedWithdraw2 acc1 acc2 amt) withdraws amt from acc1,
// if acc1 has enough money, otherwise from acc2.
// If neither has enough, it retries.
constexpr auto limitedWithdraw2 = toFunc<> | [](Account acc1, Account acc2, Integer amt)
{
    return orElse | (limitedWithdrawSTM | acc1 | amt) | (limitedWithdrawSTM | acc2 | amt);
};

constexpr auto showAcc = toFunc<> | [](std::string name, Account acc)
{
    Id<Integer> bal;
    return do_(
        bal <= atomically (readTVar | acc),
        putStr | (name + ": $"),
        putStrLn | (show | bal)
    );
};

TEST(atomically, 3)
{
    Id<Account> acc1, acc2;
    Id<Integer> v1, v2;
    auto io_ = do_(
        acc1 <= (atomically | (newTVar | Integer{100})),
        acc2 <= (atomically | (newTVar | Integer{100})),
        showAcc | "Left pocket" | acc1,
        showAcc | "Right pocket" | acc2,
        forkIO | (delayDeposit | acc2 | 1),
        print | "Withdrawing $101 from either pocket...",
        atomically | (limitedWithdraw2 | acc1 | acc2 | Integer{101}),
        print | "Successful withdrawal!",
        showAcc | "Left pocket" | acc1,
        showAcc | "Right pocket" | acc2,
        v1 <= (showAccount | acc1),
        v2 <= (showAccount | acc2),
        hassert | (v1 == 100) | "v1 should be 100",
        hassert | (v2 == 0) | "v2 should be 0"
    );

    io_.run();
}

TEST(TMVar, 1)
{
    Id<TMVar<int>> ta, tb;
    Id<int> a, b;
    auto io_ = atomically | do_(
        ta <= newEmptyTMVar<int>,
        tb <= newEmptyTMVar<int>,
        putTMVar | ta | 10,
        putTMVar | tb | 20,
        a <= (takeTMVar | ta),
        b <= (takeTMVar | tb),
        return_ | (makeTuple<2> | a | b)
    );
    auto result = io_.run();
    EXPECT_EQ(result, std::make_tuple(10, 20));
}

TEST(Chan, 0)
{
    auto hole_ = newEmptyMVar<Item<int>>.run();
    auto hole = toStreamPtr | hole_;
    auto readVar = (newMVar | hole).run();
}

TEST(Chan, 1)
{
    Id<Chan<int>> a, b;
    Id<int> x, y;
    auto io_ = do_(
        a <= newChan<int>,
        writeChan | a | 5
    );
    auto result = io_.run();
    EXPECT_EQ(result, _o_);
}

TEST(Chan, 2)
{
    Id<Chan<int>> a, b;
    auto io_ = do_(
        a <= newChan<int>,
        b <= (dupChan | a),
        forkIO | (replicateM_ | 10U || writeChan | a | 5),
        forkIO | (replicateM_ | 10U || readChan | a),
        replicateM_ | 10U || readChan | b
    );
    auto result = io_.run();
    EXPECT_EQ(result, _o_);
}
