#include <VectorField/VectorField.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using Catch::Approx;

TEST_CASE("at accesses values from the Fector Field grid", "[at]")
{
    SECTION("Test 1")
    {
        VectorField<double> vf(2, 2);
        Math::Vec2<double> a = {0,0};
        Math::Vec2<double> b = {1,0};
        Math::Vec2<double> c = {0,1};
        Math::Vec2<double> d = {1,1};
        vf.at({0,0}) = a;
        vf.at({1,0}) = b;
        vf.at({0,1}) = c;
        vf.at({1,1}) = d;
        CHECK(vf.at({0,0}).x == a.x);
        CHECK(vf.at({0,0}).y == a.y);
        CHECK(vf.at({1,0}).x == b.x);
        CHECK(vf.at({1,0}).y == b.y);
        CHECK(vf.at({0,1}).x == c.x);
        CHECK(vf.at({0,1}).y == c.y);
        CHECK(vf.at({1,1}).x == d.x);
        CHECK(vf.at({1,1}).y == d.y);
    }
}

TEST_CASE("Vector Field will interpolate between grid points using bilinear interpolation", "[bilinear interpolation]")
{
    SECTION("Test 1")
    {
        VectorField<double> vf(2, 2);
        Math::Vec2<double> a = {0,0};
        Math::Vec2<double> b = {1,0};
        Math::Vec2<double> c = {0,1};
        Math::Vec2<double> d = {1,1};
        vf.at({0,0}) = a;
        vf.at({1,0}) = b;
        vf.at({0,1}) = c;
        vf.at({1,1}) = d;
        CHECK(vf({0.0,0.0}).value().x == Approx(1.0));
        CHECK(vf({1.0,1.0}).value().y == Approx(1.0));
        CHECK(vf({0.0,0.0}).value().x == Approx(0.0));
        CHECK(vf({0.0,0.0}).value().y == Approx(0.0));
        CHECK(vf({0.1,0.2}).value().x == Approx(0.1));
        CHECK(vf({0.1,0.2}).value().y == Approx(0.2));
    }

}
