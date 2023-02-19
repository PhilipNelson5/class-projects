#include <Vector/Vec2.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using Catch::Approx;

TEST_CASE("Calculates vector magnitude", "[magnitude]")
{
    SECTION("Test 1")
    {
        const Math::Vec2<double> v { 2, 5 };
        const double expected = 5.3851648071;
        const double result = Math::magnitude(v);

        CHECK( result == Approx(expected) );
    }

    SECTION("Test 2")
    {
        const Math::Vec2<double> v { 4, 2 };
        const double expected = 4.472135955;
        const double result = Math::magnitude(v);

        CHECK( result == Approx(expected) );
    }
}

TEST_CASE("Normalizes vectors", "[normalize]")
{
    SECTION("Test 1")
    {
        const Math::Vec2<double> v { 2, 5 };
        const Math::Vec2<double> expected = { 0.371391, 0.928477 };
        const auto result = Math::normalize(v);

        CHECK( result.x == Approx(expected.x));
        CHECK( result.y == Approx(expected.y));
    }

    SECTION("Test 2")
    {
        const Math::Vec2<double> v { 4, 2 };
        const Math::Vec2<double> expected = { 0.894427, 0.447214 };
        const auto result = Math::normalize(v);

        CHECK( result.x == Approx(expected.x));
        CHECK( result.y == Approx(expected.y));
    }
}

TEST_CASE("Rotate vector", "[rotate]")
{
    SECTION("Test 1")
    {
        const Math::Vec2<double> v { 3, 7 };
        const double deg = 31;
        const Math::Vec2<double> expected = { -1.03376, 7.54529 };
        const auto result = Math::rotate(v, deg);

        CHECK( result.x == Approx(expected.x));
        CHECK( result.y == Approx(expected.y));
    }

    SECTION("Test 2")
    {
        const Math::Vec2<double> v { -6, 10 };
        const double deg = 11.1;
        const Math::Vec2<double> expected = { -7.812976, 8.65779 };
        const auto result = Math::rotate(v, deg);

        CHECK( result.x == Approx(expected.x));
        CHECK( result.y == Approx(expected.y));
    }
}
