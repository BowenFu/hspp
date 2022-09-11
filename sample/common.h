auto expectTrue(bool x)
{
    if (!x)
    {
        throw std::runtime_error{"False in expectedTrue!"};
    }
}

template <typename T, typename Enable = void>
struct IsPrintable : std::false_type
{};

template <typename T>
struct IsPrintable<T, std::void_t<decltype(std::cout << std::declval<T>())>> : std::true_type
{};

template <typename T>
constexpr bool isPrintableV = IsPrintable<std::decay_t<T>>::value;

template <typename T, typename U>
auto equal(T l, U r) -> std::enable_if_t<std::is_floating_point_v<T> || std::is_floating_point_v<U>, bool>
{
    return (std::abs(l - r) < 0.005);
}

template <typename T, typename U>
auto equal(T l, U r) -> std::enable_if_t<!std::is_floating_point_v<T> && !std::is_floating_point_v<U>, bool>
{
    return l == r;
}

template <typename T, typename U>
auto expectEq(T const& l, U const& r) -> std::enable_if_t<(isPrintableV<T> && isPrintableV<U>)>
{
    if (!equal(l, r))
    {
        std::stringstream ss;
        ss << l << " != " << r;
        throw std::runtime_error{ss.str()};
    }
}

template <typename T, typename U>
auto expectEq(T const& l, U const& r) -> std::enable_if_t<!(isPrintableV<T> && isPrintableV<U>)>
{
    if (!equal(l, r))
    {
        throw std::runtime_error{"Not equal. Types not printable"};
    }
}
