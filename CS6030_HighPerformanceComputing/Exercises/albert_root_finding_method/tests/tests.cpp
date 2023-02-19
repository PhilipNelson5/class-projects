#include "aberth/aberth.hpp"
#include "function/function.hpp"
#include <catch2/catch.hpp>
#include <complex>

TEST_CASE("Function")
{
    SECTION("image")
    {
        Function::Function f({{1, -3.0}, {2, -2.0}, {3, 1.0}});
        {
            const auto expected = std::complex<double>(0.0);
            const auto result = Function::image(f, 3);

            CHECK(result.real() == Approx(expected.real()));
            CHECK(result.imag() == Approx(expected.imag()));
        }
        {
            const auto expected = std::complex<double>(20.0);
            const auto result = Function::image(f, 4);

            CHECK(result.real() == Approx(expected.real()));
            CHECK(result.imag() == Approx(expected.imag()));
        }
        {
            const auto expected = std::complex<double>(770.0);
            const auto result = Function::image(f, 10);

            CHECK(result.real() == Approx(expected.real()));
            CHECK(result.imag() == Approx(expected.imag()));
        }
        {
            const auto expected = std::complex<double>(-28.0, 16.0);
            const auto result = Function::image(f, {3, 2});

            CHECK(result.real() == Approx(expected.real()));
            CHECK(result.imag() == Approx(expected.imag()));
        }
        {
            const auto expected = std::complex<double>(31.682, -112.613);
            const auto result = Function::image(f, {5.4, -1.9});

            CHECK(result.real() == Approx(expected.real()));
            CHECK(result.imag() == Approx(expected.imag()));
        }
    }

    SECTION("image")
    {
        Function::Function f({{1, -3.0}, {2, -2.0}, {3, 1.0}});
        {
            const auto expected = std::complex<double>(0.0);
            const auto result = Function::derivative(f, -0.53518);

            CHECK(result.real() == Approx(expected.real()));
            CHECK(result.imag() == Approx(expected.imag()));
        }
        {
            const auto expected = std::complex<double>(0.0);
            const auto result = Function::derivative(f, 1.8685);

            CHECK(result.real() == Approx(expected.real()));
            CHECK(result.imag() == Approx(expected.imag()));
        }
    }
}
