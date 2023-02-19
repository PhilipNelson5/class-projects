#include <catch2/catch_all.hpp>
#include <catch2/catch_approx.hpp>
#include <Histogram/Histogram.hpp>

TEST_CASE("Calculate Bin Maxes", "[calculate_bin_maxes]") {
    SECTION("Test 1") {
        const double min_meas = 0, max_meas = 10;
        const int bin_count = 10;
        const std::vector<double> expected { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
        const std::vector<double> bin_maxes = calculate_bin_maxes(min_meas, max_meas, bin_count);

        REQUIRE_THAT(bin_maxes, Catch::Matchers::Equals(expected));
    }
    SECTION("Test 2") {
        const double min_meas = 0, max_meas = 5;
        const int bin_count = 10;
        const std::vector<double> expected { 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0 };
        const std::vector<double> bin_maxes = calculate_bin_maxes(min_meas, max_meas, bin_count);

        REQUIRE_THAT(bin_maxes, Catch::Matchers::Equals(expected));
    }
    SECTION("Test 3") {
        const double min_meas = 10, max_meas = 12;
        const int bin_count = 4;
        const std::vector<double> expected { 10.5, 11.0, 11.5, 12.0};
        const std::vector<double> bin_maxes = calculate_bin_maxes(min_meas, max_meas, bin_count);

        REQUIRE_THAT(bin_maxes, Catch::Matchers::Equals(expected));
    }
}

TEST_CASE("Get Bin", "[get_bin]") {
    const std::vector<double> bin_maxes { 1.1, 2.2, 3.3, 4.4, 5.5 };

    SECTION("Test 1") {
        const double elem = 0;
        const int expected = 0;
        const int bin = get_bin(elem, bin_maxes);

        REQUIRE(bin == expected);
    }   
    SECTION("Test 2") {
        const double elem = 1.1;
        const int expected = 1;
        const int bin = get_bin(elem, bin_maxes);

        REQUIRE(bin == expected);
    }   
    SECTION("Test 3") {
        const double elem = 4;
        const int expected = 3;
        const int bin = get_bin(elem, bin_maxes);

        REQUIRE(bin == expected);
    }   
    SECTION("Test 4") {
        const double elem = 4;
        const int expected = 3;
        const int bin = get_bin(elem, bin_maxes);

        REQUIRE(bin == expected);
    }   
    SECTION("Test 5") {
        const double elem = 5.49;
        const int expected = 4;
        const int bin = get_bin(elem, bin_maxes);

        REQUIRE(bin == expected);
    }   
}
