#include "../count_sort/count_sort.hpp"
#include <catch2/catch.hpp>

TEST_CASE("sort_array", "[sort]")
{
    SECTION("Test 1")
    {
        std::vector<int> v = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        const auto v_orig = v;
        count_sort(v.data(), v.size(), 1);

        REQUIRE(std::is_sorted(begin(v), end(v)));
        REQUIRE_THAT(v, Catch::Matchers::UnorderedEquals(v_orig));
    }
    SECTION("Test 2")
    {
        std::vector<int> v = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
        const auto v_orig = v;
        count_sort(v.data(), v.size(), 2);

        REQUIRE(std::is_sorted(begin(v), end(v)));
        REQUIRE_THAT(v, Catch::Matchers::UnorderedEquals(v_orig));
    }
    SECTION("Test 3")
    {
        std::vector<int> v = {5, 4, 7, 2, 4, 8, 9, 5, 3, 3};
        const auto v_orig = v;
        count_sort(v.data(), v.size(), 3);

        REQUIRE(std::is_sorted(begin(v), end(v)));
        REQUIRE_THAT(v, Catch::Matchers::UnorderedEquals(v_orig));
    }
    SECTION("Test 4")
    {
        std::vector<int> v = {4, 3, 3, 3, 3, 3, 3, 3, 3, 2};
        const auto v_orig = v;
        count_sort(v.data(), v.size(), 4);

        REQUIRE(std::is_sorted(begin(v), end(v)));
        REQUIRE_THAT(v, Catch::Matchers::UnorderedEquals(v_orig));
    }
    SECTION("Test 5")
    {
        std::vector<int> v = {1, 3, 5, 4, 2, 6, 8, 7, 9, 8, 5, 3, 4, 3, 2, 4, 6, 7, 9, 8, 9, 0, 0};
        const auto v_orig = v;
        count_sort(v.data(), v.size(), 5);

        REQUIRE(std::is_sorted(begin(v), end(v)));
        REQUIRE_THAT(v, Catch::Matchers::UnorderedEquals(v_orig));
    }
}

TEST_CASE("sort_iterator", "[sort]")
{
    SECTION("Test 1")
    {
        std::vector<int> v = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        const auto v_orig = v;
        count_sort(begin(v), end(v), 1);

        REQUIRE(std::is_sorted(begin(v), end(v)));
        REQUIRE_THAT(v, Catch::Matchers::UnorderedEquals(v_orig));
    }
    SECTION("Test 2")
    {
        std::vector<int> v = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
        const auto v_orig = v;
        count_sort(begin(v), end(v), 2);

        REQUIRE(std::is_sorted(begin(v), end(v)));
        REQUIRE_THAT(v, Catch::Matchers::UnorderedEquals(v_orig));
    }
    SECTION("Test 3")
    {
        std::vector<int> v = {5, 4, 7, 2, 4, 8, 9, 5, 3, 3};
        const auto v_orig = v;
        count_sort(begin(v), end(v), 3);

        REQUIRE(std::is_sorted(begin(v), end(v)));
        REQUIRE_THAT(v, Catch::Matchers::UnorderedEquals(v_orig));
    }
    SECTION("Test 4")
    {
        std::vector<int> v = {4, 3, 3, 3, 3, 3, 3, 3, 3, 2};
        const auto v_orig = v;
        count_sort(begin(v), end(v), 4);

        REQUIRE(std::is_sorted(begin(v), end(v)));
        REQUIRE_THAT(v, Catch::Matchers::UnorderedEquals(v_orig));
    }
    SECTION("Test 5")
    {
        std::vector<int> v = {1, 3, 5, 4, 2, 6, 8, 7, 9, 8, 5, 3, 4, 3, 2, 4, 6, 7, 9, 8, 9, 0, 0};
        const auto v_orig = v;
        count_sort(begin(v), end(v), 5);

        REQUIRE(std::is_sorted(begin(v), end(v)));
        REQUIRE_THAT(v, Catch::Matchers::UnorderedEquals(v_orig));
    }
}
