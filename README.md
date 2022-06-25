# hspp

hspp: Haskell Style Programming brought to Cpp.

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

Sample 1

```c++
    using namespace hspp::doN;
    Id<int> i;
    Id<int> j;
    Id<int> k;
    auto result = _(
        makeTuple<3> | i | j | k,
        i <= (iota | 1 | 20),
        j <= (iota | i | 20),
        k <= (iota | j | 20),
        if_ || (i*i + j*j == k*k))
    );
```

Sample 2

```c++
    using namespace hspp::doN;
    Id<int> i;
    auto const result = do_(
        i <= std::vector{1, 2, 3, 4},
        guard | (i % 2 == 0),
        return_ | i
    );
```

Sample 3 (Parser Combinator)

Original haskell version (Monadic Parsing in Haskell)

```haskell
expr, term, factor, digit :: Parser Int

digit  = do {x <- token (sat isDigit); return (ord x - ord '0')}

factor = digit +++ do {symb "("; n <- expr; symbol ")"; return n}

term   = factor `chainl1` mulop

expr   = term   `chainl1` addop
```

C++ version

```c++
Id<char> x;
auto const digit = do_(
    x <= (token || sat | isDigit),
    return_ | (x - '0')
);

extern TEParser<int> const expr;

Id<int> n;
auto const factor =
    digit <triPlus>
        do_(
            symb | "("s,
            n <= expr,
            symb | ")"s,
            return_ | n
        );

auto const term = factor <chainl1> mulOp;

extern TEParser<int> const expr = toTEParser || (term <chainl1> addOp);
```

## Why bother?

The library is 

1. for fun,
2. to explore the interesting features of Haskell,
3. to explore the boundary of C++,
4. to facilitate the translation of some interesting Haskell codes to C++.

This library is still in active development and not production ready.

Discussions / issues / PRs are welcome.
