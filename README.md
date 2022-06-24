# hspp

hspp: Haskell Style Programming brought to Cpp.

Mom, can we have monadic do notation / monad comprehension in C++?

Here you are!

Sample 1

```c++
    using namespace hspp::doN;
    Id<int> i;
    Id<int> j;
    Id<int> k;
    auto result = _(
        makeTuple<3> | i | j | k,
        i <= iota(1, 20),
        j <= iota(1, 20),
        k <= iota(1, 20),
        if_ || (i < j) && (i*i + j*j == k*k)
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

![Standard](https://img.shields.io/badge/c%2B%2B-17/20-blue.svg)

![Platform](https://img.shields.io/badge/platform-linux-blue)
![Platform](https://img.shields.io/badge/platform-osx-blue)
![Platform](https://img.shields.io/badge/platform-win-blue)

[![CMake](https://github.com/BowenFu/hspp/actions/workflows/cmake.yml/badge.svg)](https://github.com/BowenFu/hspp/actions/workflows/cmake.yml)
[![CMake](https://github.com/BowenFu/hspp/actions/workflows/sanitizers.yml/badge.svg)](https://github.com/BowenFu/hspp/actions/workflows/sanitizers.yml)
![GitHub license](https://img.shields.io/github/license/BowenFu/hspp.svg)
[![codecov](https://codecov.io/gh/BowenFu/hspp/branch/main/graph/badge.svg)](https://codecov.io/gh/BowenFu/hspp)

