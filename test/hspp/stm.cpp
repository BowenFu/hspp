#include "hspp.h"
#include <vector>
#include <list>
#include <gtest/gtest.h>
#include <cmath>
#include <cctype>

using namespace hspp;
using namespace hspp::data;
using namespace hspp::parser;
using namespace std::literals;

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

using WriteSet = std::map<ID, 

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