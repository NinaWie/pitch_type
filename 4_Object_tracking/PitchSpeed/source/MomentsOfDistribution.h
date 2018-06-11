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

#include "Common.h"

#if !defined(MOMENTS_OF_DISTRIBUTION_INCLUDED)
#define MOMENTS_OF_DISTRIBUTION_INCLUDED

namespace my {
    namespace analytics {
        class CMomentsOfDistribution
        {
        public:
            CMomentsOfDistribution()
            {
                Create();
            }

            CMomentsOfDistribution(const CMomentsOfDistribution& momentsOfDistribution)
            {
                Copy(momentsOfDistribution);
            }

            void operator=(const CMomentsOfDistribution& momentsOfDistribution)
            {
                Copy(momentsOfDistribution);
            }

            my::int32 GetNumberOfScoresInSample() const
            {
                return m_numberOfScoresInSample;
            }

            void SetNumberOfScoresInSample(my::int32 numberOfScoresInSample)
            {
                m_numberOfScoresInSample = numberOfScoresInSample;
            }

            double GetMean() const
            {
                return m_mean;
            }

            void SetMean(double mean)
            {
                m_mean = mean;
            }

            double GetMeanDeviation() const
            {
                return m_meanDeviation;
            }

            void SetMeanDeviation(double meanDeviation)
            {
                m_meanDeviation = meanDeviation;
            }

            // SAMPLE STANDARD DEVIATION
            double GetStandardDeviation() const
            {
                return m_standardDeviation;
            }

            // SAMPLE STANDARD DEVIATION
            void SetStandardDeviation(double standardDeviation)
            {
                m_standardDeviation = standardDeviation;
            }

            double GetVariance() const
            {
                return m_variance;
            }
            
            void SetVariance(double variance)
            {
                m_variance = variance;
            }

            double GetSkewness() const
            {
                return m_skewness;
            }

            void SetSkewness(double skewness)
            {
                m_skewness = skewness;
            }

            double GetKurtosis() const
            {
                return m_kurtosis;
            }

            void SetKurtosis(double kurtosis)
            {
                m_kurtosis = kurtosis;
            }

            double Get0thQuartile() const
            {
                return m_0thQuartile;
            }

            void Set0thQuartile(double _0thQuartile)
            {
                m_0thQuartile = _0thQuartile;
            }

            double Get1stQuartile() const
            {
                return m_1stQuartile;
            }

            void Set1stQuartile(double _1stQuartile)
            {
                m_1stQuartile = _1stQuartile;
            }

            double Get2ndQuartile() const
            {
                return m_2ndQuartile;
            }

            void Set2ndQuartile(double _2ndQuartile)
            {
                m_2ndQuartile = _2ndQuartile;
            }

            double Get3rdQuartile() const
            {
                return m_3rdQuartile;
            }

            void Set3rdQuartile(double _3rdQuartile)
            {
                m_3rdQuartile = _3rdQuartile;
            }

            double Get4thQuartile() const
            {
                return m_4thQuartile;
            }

            void Set4thQuartile(double _4thQuartile)
            {
                m_4thQuartile = _4thQuartile;
            }

            void Clear()
            {
                Create();
            }

            bool IsValid() const
            {
                return (m_numberOfScoresInSample != 0) ||
                    // TRICKY: (03-Dec-2017) BACKWARD COMPATIBILITY
                    !my::IsNull(m_mean);
            }

        private:
            void Create()
            {
                m_numberOfScoresInSample = 0;
                m_mean = my::Null<double>();
                m_meanDeviation = my::Null<double>();
                m_standardDeviation = my::Null<double>();
                m_variance = my::Null<double>();
                m_skewness = my::Null<double>();
                m_kurtosis = my::Null<double>();
                m_0thQuartile = my::Null<double>();
                m_1stQuartile = my::Null<double>();
                m_2ndQuartile = my::Null<double>();
                m_3rdQuartile = my::Null<double>();
                m_4thQuartile = my::Null<double>();
            }

            void Copy(const CMomentsOfDistribution& momentsOfDistribution)
            {
                m_numberOfScoresInSample = momentsOfDistribution.m_numberOfScoresInSample;
                m_mean = momentsOfDistribution.m_mean;
                m_meanDeviation = momentsOfDistribution.m_meanDeviation;
                m_standardDeviation = momentsOfDistribution.m_standardDeviation;
                m_variance = momentsOfDistribution.m_variance;
                m_skewness = momentsOfDistribution.m_skewness;
                m_kurtosis = momentsOfDistribution.m_kurtosis;
                m_0thQuartile = momentsOfDistribution.m_0thQuartile;
                m_1stQuartile = momentsOfDistribution.m_1stQuartile;
                m_2ndQuartile = momentsOfDistribution.m_2ndQuartile;
                m_3rdQuartile = momentsOfDistribution.m_3rdQuartile;
                m_4thQuartile = momentsOfDistribution.m_4thQuartile;
            }

        protected:
            my::int32 m_numberOfScoresInSample;
            double m_mean;
            double m_meanDeviation;
            // SAMPLE STANDARD DEVIATION
            double m_standardDeviation;
            double m_variance;
            double m_skewness;
            double m_kurtosis;
            double m_0thQuartile;
            double m_1stQuartile;
            double m_2ndQuartile;
            double m_3rdQuartile;
            double m_4thQuartile;
        };
    };
}; // my

#endif // #if !defined(MOMENTS_OF_DISTRIBUTION_INCLUDED)

