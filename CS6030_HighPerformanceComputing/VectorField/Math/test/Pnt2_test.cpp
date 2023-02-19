#include <Point/Pnt2.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using Catch::Approx;

TEST_CASE("Calculates distance between points", "[distance]")
{
    SECTION("Test 1")
    {
        const Math::Pnt2<double> a { 2, 5 };
        const Math::Pnt2<double> b { 4, 2 };
        const double expected = 3.60555;
        const double result = Math::distance(a, b);

        CHECK( result == Approx(expected) );
    }

    SECTION("Test 2")
    {
        const Math::Pnt2<double> a { 3, 7 };
        const Math::Pnt2<double> b { -10, 8 };
        const double expected = 13.0384;
        const double result = Math::distance(a, b);

        CHECK( result == Approx(expected) );
    }
}

TEST_CASE("Add vector to point", "[point+vec]")
{
    SECTION("Test 1")
    {
        const Math::Pnt2<double> p { 1, 1 };
        const Math::Vec2<double> v { 2, 5 };
        const Math::Pnt2<double> expected = { 3, 6 };
        const auto result = p + v;

        CHECK( result.x == Approx(expected.x));
        CHECK( result.y == Approx(expected.y));
    }

    SECTION("Test 2")
    {
        const Math::Pnt2<double> p { 10, -3 };
        const Math::Vec2<double> v { -2, 4 };
        const Math::Pnt2<double> expected = { 8, 1 };
        const auto result = p + v;

        CHECK( result.x == Approx(expected.x));
        CHECK( result.y == Approx(expected.y));
    }
}
