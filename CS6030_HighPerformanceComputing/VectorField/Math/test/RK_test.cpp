#include <Point/Pnt2.hpp>
#include <Vector/Vec2.hpp>
#include <Solvers/RungeKutta.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using Catch::Approx;

TEST_CASE("Calculates the next point in function using Runge Kutta", "[runge kutta]")
{
    SECTION("Test 1")
    {
        // const double dt = .00001;
        // Math::Pnt2<double> p{ .33, .33 };
        // auto f = [](const Math::Pnt2<double> p) -> Math::Vec2<double> {
        //     return { std::sin(p.x), std::sin(p.y) };
        // };
        // const Vec2<double> expected = f({ p.x + dt, p.y + dt });
        
        // const Pnt2<double> result = Math::runge_kutta(f, p, dt);
        
        // CHECK( result.x == Approx(expected.x));
        // CHECK( result.y == Approx(expected.y));
    }
}
