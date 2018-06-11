/*
* COPYRIGHT NOTICE, DISCLAIMER, and LICENSE:
*
*
* For the purposes of this copyright and license, "Contributing Authors"
* is defined as the following set of individuals:
*
*    Carlos Augusto Dietrich (cadietrich@gmail.com)
*
* This library is supplied "AS IS".  The Contributing Authors disclaim
* all warranties, expressed or implied, including, without limitation,
* the warranties of merchantability and of fitness for any purpose.
* The Contributing Authors assume no liability for direct, indirect,
* incidental, special, exemplary, or consequential damages, which may
* result from the use of the this library, even if advised of the
* possibility of such damage.
*
* Permission is hereby granted to use, copy, modify, and distribute this
* source code, or portions hereof, for any purpose, without fee, subject
* to the following restrictions:
*
* 1. The origin of this source code must not be misrepresented.
*
* 2. Altered versions must be plainly marked as such and must not be
*    misrepresented as being the original source.
*
* 3. This Copyright notice may not be removed or altered from any source
*    or altered source distribution.
*
* The Contributing Authors specifically permit, without fee, and
* encourage the use of this source code as a component in commercial
* products. If you use this source code in a product, acknowledgment
* is not required but would be appreciated.
*
*
* "Software is a process, it's never finished, it's always evolving.
* That's its nature. We know our software sucks. But it's shipping!
* Next time we'll do better, but even then it will be shitty.
* The only software that's perfect is one you're dreaming about.
* Real software crashes, loses data, is hard to learn and hard to use.
* But it's a process. We'll make it less shitty. Just watch!"
*/

#if !defined(MY_ANALYTICS_INCLUDED)
#define MY_ANALYTICS_INCLUDED

#include <vector>

#include "MomentsOfDistribution.h"

namespace my {
    namespace analytics {
        enum  MOMENTS_OF_DISTRIBUTION_TYPE
        {
            MEAN = 0,
            MEAN_DEVIATION = 1,
            STANDARD_DEVIATION = 2,
            VARIANCE = 3,
            SKEWNESS = 4,
            KURTOSIS = 5,
            _0TH_QUARTILE = 6,
            _1ST_QUARTILE = 7,
            _2ND_QUARTILE = 8,
            _3RD_QUARTILE = 9,
            _4TH_QUARTILE = 10,
            MEDIAN = 11
        };

        /**
        */
        inline std::string GetMomentsOfDistributionTypeString(int momentsOfDistributionType)
        {
            switch (momentsOfDistributionType) {
            case MEAN:
                return "Mean";
                break;
            case MEAN_DEVIATION:
                return "Mean Deviation";
                break;
            case STANDARD_DEVIATION:
                return "Standard Deviation";
                break;
            case VARIANCE:
                return "Variance";
                break;
            case SKEWNESS:
                return "Skewness";
                break;
            case KURTOSIS:
                return "Kurtosis";
                break;
            case _0TH_QUARTILE:
                return "0th Quartile";
                break;
            case _1ST_QUARTILE:
                return "1st Quartile";
                break;
            case _2ND_QUARTILE:
                return "2nd Quartile";
                break;
            case _3RD_QUARTILE:
                return "3rd Quartile";
                break;
            case _4TH_QUARTILE:
                return "4th Quartile";
                break;
            case MEDIAN:
                return "Median";
                break;
            };

            return std::string();
        }

        /**
        */
        template <typename T>
        bool Mean(const std::vector<T>& xArray, T& mean);

        /**
        */
        template <typename T>
        bool Moment(std::vector<T> xArray, my::analytics::CMomentsOfDistribution& momentsOfDistribution);

        // (http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient) "... measure of the linear correlation (dependence) between two variables X and Y, giving a value between +1 and ?1 inclusive, where 1 is total positive correlation, 0 is no correlation, and ?1 is total negative correlation."
        template <typename T>
        bool SamplePearsonCorrelationCoefficient(const std::vector<T>& xArray, const std::vector<T>& yArray, T &rho);

        // Kendall's Tau (Numerical Recipes (http://www2.units.it/ipl/students_area/imm2/files/Numerical_Recipes.pdf), p. 642). Given data arrays data1[1..n] and data2[1..n], this program returns Kendall's t as tau, its number of standard deviations from zero as z, and its two - sided significance level as prob. Small values of prob indicate a significant correlation (tau positive) or anticorrelation (tau negative).
        template <typename T>
        bool SamplePearsonCorrelationCoefficient(const std::vector<T>& xArray, const std::vector<T>& yArray, T &tau, T &standardDeviationsFromZero, T & twoSidedSignificanceLevel);
    };
}; // my

#endif // #if !defined(MY_ANALYTICS_INCLUDED)

