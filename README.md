# hspp

![hspp](./hspp.svg)

## hspp: bring Haskell Style Programming to C++.

![Standard](https://img.shields.io/badge/c%2B%2B-17/20-blue.svg)

![Platform](https://img.shields.io/badge/platform-linux-blue)
![Platform](https://img.shields.io/badge/platform-osx-blue)
![Platform](https://img.shields.io/badge/platform-win-blue)

[![CMake](https://github.com/BowenFu/hspp/actions/workflows/cmake.yml/badge.svg)](https://github.com/BowenFu/hspp/actions/workflows/cmake.yml)
[![CMake](https://github.com/BowenFu/hspp/actions/workflows/sanitizers.yml/badge.svg)](https://github.com/BowenFu/hspp/actions/workflows/sanitizers.yml)
![GitHub license](https://img.shields.io/github/license/BowenFu/hspp.svg)
[![codecov](https://codecov.io/gh/BowenFu/hspp/branch/main/graph/badge.svg)](https://codecov.io/gh/BowenFu/hspp)


## Mom, can we have monadic do notation / monad comprehension in C++?

Here you are!

[badge.godbolt]: https://img.shields.io/badge/try-godbolt-blue

### Sample 1 for monadic do notation for a vector monad

Filter even numbers.

[godbolt1]: https://godbolt.org/z/MM7MW9MPY

[![Try it on godbolt][badge.godbolt]][godbolt1]

```c++
    using namespace hspp;
    using namespace hspp::doN;
    Id<int> i;
    auto const result = do_(
        i <= std::vector{1, 2, 3, 4},
        guard | (i % 2 == 0),
        return_ | i
    );
```

### Sample 2 for monad comprehension for a range monad

Obtain an infinite range of Pythagorean triples.

Haskell version

```haskell
triangles = [(i, j, k) | k <- [1..], i <- [1..k], j <- [i..k] , i^2 + j^2 == k^2]
```

[godbolt2]: https://godbolt.org/z/nnKozMYzo

[![Try it on godbolt][badge.godbolt]][godbolt2]

```c++
    using namespace hspp::doN;
    using namespace hspp::data;
    Id<int> i, j, k;
    auto const rng = _(
        makeTuple<3> | i | j | k,
        k <= (enumFrom | 1),
        i <= (iota | 1 | k),
        j <= (iota | i | k),
        if_ || (i*i + j*j == k*k)
    );
```

Equivalent version using RangeV3 would be [link](http://ericniebler.com/2014/04/27/range-comprehensions/)

```C++
using namespace ranges;

// Lazy ranges for generating integer sequences
auto const intsFrom = view::iota;
auto const ints = [=](int i, int j)
    {
        return view::take(intsFrom(i), j-i+1);
    };

// Define an infinite range of all the Pythagorean
// triples:
auto triples =
    view::for_each(intsFrom(1), [](int z)
    {
        return view::for_each(ints(1, z), [=](int x)
        {
            return view::for_each(ints(x, z), [=](int y)
            {
                return yield_if(x*x + y*y == z*z,
                    std::make_tuple(x, y, z));
            });
        });
    });
```

### Sample 3 for Maybe (similar to std::optional) Monad used in do notation

We have two functions, plus1, and showStr. With do notation we construct a new function that will accept an integer as argument and return a tuple of results of the two functions.

The sample is originated from Learn You a Haskell for Great Good!

"Pierre has decided to take a break from his job at the fish farm and try tightrope walking. He's not that bad at it, but he does have one problem: birds keep landing on his balancing pole!
Let's say that he keeps his balance if the number of birds on the left side of the pole and on the right side of the pole is within three."

Note that Pierre may also suddenly slip and fall when there is a banana.

Original Haskell version

```haskell
routine :: Maybe Pole
routine = do
    start <- return (0,0)
    first <- landLeft 2 start
    Nothing
    second <- landRight 2 first
    landLeft 1 second
```

C++ version using hspp

[godbolt3]: https://godbolt.org/z/9T5sa64nE

[![Try it on godbolt][badge.godbolt]][godbolt3]

```c++
Id<Pole> start, first, second;
auto const routine = do_(
    start <= return_ | Pole{0,0}),
    first <= (landLeft | 2 | start),
    nothing<Pole>,
    second <= (landRight | 2 | first),
    landLeft | 1 | second
);
```


### Sample 4 for Function Monad used in do notation

We have two functions, plus1, and showStr. With do notation we construct a new function that will accept an integer as argument and return a tuple of results of the two functions.

[godbolt4]: https://godbolt.org/z/d58Ezbxjz

[![Try it on godbolt][badge.godbolt]][godbolt4]

```c++
    auto plus1 = toFunc<> | [](int x){ return 1+x; };
    auto showStr = toFunc<> | [](int x){ return show | x; };

    Id<int> x;
    Id<std::string> y;
    auto go = do_(
        x <= plus1,
        y <= showStr,
        return_ || makeTuple<2> | x | y
    );
    auto result = go | 3;
    std::cout << std::get<0>(result) << std::endl;
    std::cout << std::get<1>(result) << std::endl;
```

### Sample 5 for parser combinator

Original haskell version [Monadic Parsing in Haskell](https://www.cambridge.org/core/journals/journal-of-functional-programming/article/monadic-parsing-in-haskell/E557DFCCE00E0D4B6ED02F3FB0466093)

```haskell
expr, term, factor, digit :: Parser Int

digit  = do {x <- token (sat isDigit); return (ord x - ord '0')}

factor = digit +++ do {symb "("; n <- expr; symbol ")"; return n}

term   = factor `chainl1` mulop

expr   = term   `chainl1` addop
```

C++ version
[parse_expr](https://github.com/BowenFu/hspp/blob/main/sample/parse_expr.cpp)

[godbolt5]: https://godbolt.org/z/r7WTYjGYa

[![Try it on godbolt][badge.godbolt]][godbolt5]

```c++
Id<char> x;
auto const digit = do_(
    x <= (token || sat | isDigit),
    return_ | (x - '0')
);

extern TEParser<int> const expr;

Id<int> n;
auto const factor =
    digit <alt>
        do_(
            symb | "("s,
            n <= expr,
            symb | ")"s,
            return_ | n
        );

auto const term = factor <chainl1> mulOp;

TEParser<int> const expr = toTEParser || (term <chainl1> addOp);
```

### Sample 6 for STM / concurrent

[concurrent.cpp](https://github.com/BowenFu/hspp/blob/main/test/hspp/concurrent.cpp)

[godbolt6]: https://godbolt.org/z/zj9Mc1h4h

[![Try it on godbolt][badge.godbolt]][godbolt6]


Transfer from one account to another one atomically.
```c++
Id<Account> from, to;
Id<Integer> v1, v2;
auto io_ = do_(
    from <= (newTVarIO | Integer{200}),
    to   <= (newTVarIO | Integer{100}),
    transfer | from | to | 50,
    v1 <= (showAccount | from),
    v2 <= (showAccount | to),
    hassert | (v1 == 150) | "v1 should be 150",
    hassert | (v2 == 150) | "v2 should be 150"
);
io_.run();
```

Withdraw from an account but waiting for sufficient money.
```c++
Id<Account> acc;
auto io_ = do_(
    acc <= (newTVarIO | Integer{100}),
    forkIO | (delayDeposit | acc | 1),
    putStr | "Trying to withdraw money...\n",
    atomically | (limitedWithdrawSTM | acc | 101),
    putStr | "Successful withdrawal!\n"
);

io_.run();
```

And we can also compose two STMs with `orElse`
```c++
// (limitedWithdraw2 acc1 acc2 amt) withdraws amt from acc1,
// if acc1 has enough money, otherwise from acc2.
// If neither has enough, it retries.
constexpr auto limitedWithdraw2 = toFunc<> | [](Account acc1, Account acc2, Integer amt)
{
    return orElse | (limitedWithdrawSTM | acc1 | amt) | (limitedWithdrawSTM | acc2 | amt);
};

Id<Account> acc1, acc2;
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
    showAcc | "Right pocket" | acc2
);

io_.run();
```

## Haskell vs Hspp (Incomplete list)

| Haskell       | Hspp |
| -------       | ---- |
| function      | Function / GenericFunction |
| f x y         | f \| x \| y |
| f $ g x       | f \|\| g \| x|
| f . g $ x     | f \<o\> g \|\| x|
| a \`f\` b     | a \<f\> b |
|[f x \| x <- xs, p x]| \_(f \| x, x <= xs, if\_ \|\| p \| x) |
| list (lazy)   | range |
| list (strict) | std::vector/list/forward_list|
| do {patA <- action1; action2} | do_(patA <= action1, action2) |
| f <$> v       | f \<fmap\> v |
| f <*> v       | f \<ap\> v |
| pure a        | pure \| a |
| m1 >> m2      | m1 >> m2 |
| m1 >>= f      | m1 >>= f |
| return a      | return_ \| a |


## Why bother?

The library is

1. for fun,
2. to explore the interesting features of Haskell,
3. to explore the boundary of C++,
4. to facilitate the translation of some interesting Haskell codes to C++.

This library is still in active development and not production ready.

Discussions / issues / PRs are welcome.

## Related

Haskell pattern matching is not covered in this repo. You may be interested in [match(it)](https://github.com/BowenFu/matchit.cpp) if you want to see how pattern matching works in C++.
