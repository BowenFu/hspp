# hspp

## hspp: Haskell Style Programming brought to C++.

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

### Sample 1 for monadic do notation

[godbolt1]: https://godbolt.org/z/7fTvTd3hT
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

### Sample 2 for monad comprehension


[godbolt2]: https://godbolt.org/z/M8Ynjvr3x

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

### Sample 3 for parser combinator

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

extern TEParser<int> const expr = toTEParser || (term <chainl1> addOp);
```

### Sample 4 for STM / concurrent

[concurrent.cpp](https://github.com/BowenFu/hspp/blob/main/test/hspp/concurrent.cpp)

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

Haskell pattern matching is not covered in this repo. You may be interested in [matchit(it)](https://github.com/BowenFu/matchit.cpp) if you want to see how pattern matching works in C++.
