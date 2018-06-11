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

#include "Logger.h"
#include "MyMath.h"

#include "MyAnalytics.h"

namespace my {
    namespace analytics {

        /**
        */
        template <>
        bool Mean(const std::vector<double>& xArray, double& mean)
        {
            INT32 n = (INT32)xArray.size();

            if (!n)
            {
                // BUG: (04-Jul-2015) Disabled for performance reasons.
                //LOG_MESSAGE("There must be at least one measurement.");

                return false;
            }

            double s = 0,
                sp = 0;

            for (std::vector<double>::const_iterator dataIterator = xArray.begin(); dataIterator != xArray.end(); ++dataIterator)
            {
                HEALTH_CHECK(!MyMath::IsValid(*dataIterator), false);

                s += (*dataIterator);

                // BUG: (09-Jul-2015) A naive way to prevent overflows.
                if (s > 1.0e8)
                {
                    sp += s / (double)n;

                    s = 0;
                }
            }

            HEALTH_CHECK(!MyMath::IsValid(s), false);

            mean = s / (double)n + sp;

            return true;
        }

        /**
        */
        template <>
        bool Moment(std::vector<double> xArray, my::analytics::CMomentsOfDistribution& momentsOfDistribution)
        {
            INT32 n = (INT32)xArray.size();

            if (!n)
            {
                // BUG: (04-Jul-2015) Disabled for performance reasons.
                //LOG_MESSAGE("There must be at least one measurement.");

                return false;
            }

            double mean = 0;

            if (!Mean(xArray, mean))
            {
                LOG_ERROR();

                return false;
            }

            double s = 0,
                p = 0,
                meanDeviation = 0,
                variance = 0,
                skewness = 0,
                kurtosis = 0;

            for (std::vector<double>::const_iterator dataIterator = xArray.begin(); dataIterator != xArray.end(); ++dataIterator)
            {
                s = (*dataIterator) - mean;

                meanDeviation = meanDeviation + fabs(s);

                p = s * s;

                variance += p;

                p = p * s;

                skewness += p;

                p = p * s;

                kurtosis += p;
            }

            HEALTH_CHECK(!MyMath::IsValid(meanDeviation), false);

            meanDeviation = meanDeviation / (double)n;

            variance = variance / (double)(n - 1);

            if (!MyMath::IsValid(variance))
                return false;

            double standardDeviation = sqrt(variance);

            if (!MyMath::IsZero(variance))
            {
                skewness = skewness / ((double)n * pow(standardDeviation, 3.0));
                kurtosis = kurtosis / ((double)n * variance * variance) - 3.0;
            }
            // BUG: (23-Feb-2015) Disabled for performance reasons.
            //else
            //    LOG_MESSAGE("No skewness or kurtosis when zero variance.");

            HEALTH_CHECK(!MyMath::IsValid(skewness), false);
            HEALTH_CHECK(!MyMath::IsValid(kurtosis), false);

            momentsOfDistribution.SetNumberOfScoresInSample(n);
            momentsOfDistribution.SetMean(mean);
            momentsOfDistribution.SetMeanDeviation(meanDeviation);
            momentsOfDistribution.SetStandardDeviation(standardDeviation);
            momentsOfDistribution.SetVariance(variance);
            momentsOfDistribution.SetSkewness(skewness);
            momentsOfDistribution.SetKurtosis(kurtosis);

            std::sort(xArray.begin(), xArray.end());

            momentsOfDistribution.Set0thQuartile(xArray[0]);
            momentsOfDistribution.Set1stQuartile(xArray[(INT32)(0.25 * (double)n)]);
            momentsOfDistribution.Set2ndQuartile(xArray[(INT32)(0.5 * (double)n)]);
            momentsOfDistribution.Set3rdQuartile(xArray[(INT32)(0.75 * (double)n)]);
            momentsOfDistribution.Set4thQuartile(xArray[n - 1]);

            return true;
        }

        // (http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient) "... measure of the linear correlation (dependence) between two variables X and Y, giving a value between +1 and -1 inclusive, where 1 is total positive correlation, 0 is no correlation, and -1 is total negative correlation."
        template <>
        bool SamplePearsonCorrelationCoefficient(const std::vector<double>& xArray, const std::vector<double>& yArray, double &rho)
        {
            HEALTH_CHECK(xArray.empty(), false);
            HEALTH_CHECK(xArray.size() != yArray.size(), false);

            UINT32 n = (UINT32)xArray.size();

            double _1stMean = 0,
                _2ndMean = 0;

            if (!my::analytics::Mean(xArray, _1stMean))
            {
                LOG_ERROR();

                return false;
            }

            if (!my::analytics::Mean(yArray, _2ndMean))
            {
                LOG_ERROR();

                return false;
            }

            double numerator = 0,
                _1stDenominator = 0,
                _2ndDenominator = 0;

            for (UINT32 i = 0; i < n; ++i)
            {
                double tx = (xArray[i] - _1stMean),
                    ty = (yArray[i] - _2ndMean);

                numerator += tx * ty;

                _1stDenominator += tx * tx;
                _2ndDenominator += ty * ty;
            }

            HEALTH_CHECK(!MyMath::IsValid(numerator), false);
            HEALTH_CHECK(!MyMath::IsValid(_1stDenominator), false);
            HEALTH_CHECK(!MyMath::IsValid(_2ndDenominator), false);

            rho = numerator / (sqrt(_1stDenominator) * sqrt(_2ndDenominator));

            return true;
        }

        // Returns the complementary error function erfc(x) with fractional error everywhere less than 1.2 x 10-7.
        double erfcc(double x)
        {
            double t = 0,
                z = 0,
                ans = 0;

            z = fabs(x);

            t = 1.0 / (1.0 + 0.5 * z);

            ans = t * exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
                t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
                t * (-0.82215223 + t * 0.17087277)))))))));

            return x >= 0.0 ? ans : 2.0 - ans;
        }

        // Kendall's Tau (Numerical Recipes (http://www2.units.it/ipl/students_area/imm2/files/Numerical_Recipes.pdf), p. 642). Given data arrays data1[1..n] and data2[1..n], this program returns Kendall's t as tau, its number of standard deviations from zero as standardDeviationsFromZero, and its two - sided significance level as twoSidedSignificanceLevel. Small values of twoSidedSignificanceLevel indicate a significant correlation (tau positive) or anticorrelation (tau negative).
        template <>
        bool SamplePearsonCorrelationCoefficient(const std::vector<double>& xArray, const std::vector<double>& yArray, double &tau, double &standardDeviationsFromZero, double& twoSidedSignificanceLevel)
        {
            HEALTH_CHECK(xArray.empty(), false);
            HEALTH_CHECK(xArray.size() != yArray.size(), false);

            //double erfcc(T x);
            unsigned long n = (unsigned long)xArray.size(),
                n2 = 0,
                n1 = 0,
                k = 0,
                j = 0;
            long is = 0;
            double svar = 0,
                aa = 0,
                a2 = 0,
                a1 = 0;

            // Loop over first member of pair,
            for (j = 1; j < n; j++)
            {
                // and second member.
                for (k = (j + 1); k <= n; k++)
                {
                    a1 = xArray[j] - xArray[k];
                    a2 = yArray[j] - yArray[k];

                    aa = a1 * a2;

                    // Neither array has a tie.
                    if (aa)
                    {
                        ++n1;
                        ++n2;

                        aa > 0.0 ? ++is : --is;
                    }
                    // One or both arrays have ties.
                    else
                    {
                        // An "extra x" event.
                        if (a1)
                            ++n1;
                        // An "extra y" event.
                        if (a2)
                            ++n2;
                    }
                }
            }

            tau = is / (sqrt((double)n1) * sqrt((double)n2));

            svar = (4.0 * n + 10.0) / (9.0 * n * (n - 1.0));

            standardDeviationsFromZero = tau / sqrt(svar);

            // Significance.
            twoSidedSignificanceLevel = erfcc(fabs(standardDeviationsFromZero) / 1.4142136);

            return true;
        }
    };
}; // my
