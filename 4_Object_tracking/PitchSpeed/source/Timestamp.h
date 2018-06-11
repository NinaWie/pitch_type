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

#if !defined(TIMESTAMP_INCLUDED)
#define TIMESTAMP_INCLUDED

#include <string>

#include <boost/date_time.hpp>

#include "Common.h"

namespace my {
    class CTimestamp
    {
        CTimestamp()
        {
            Create();
        }

    public:
        CTimestamp(INT32 year, INT32 month, INT32 day, INT32 hours = 0, INT32 minutes = 0, INT32 seconds = 0, INT32 milliseconds = 0, INT32 microseconds = 0);
        
        CTimestamp(boost::posix_time::ptime posixTime);
        
        CTimestamp(INT64 secondsFromEpoch);
        
        CTimestamp(std::string timeString);

        CTimestamp(const CTimestamp& timestamp);

        void operator=(const CTimestamp& timestamp);

        bool operator<(const CTimestamp& timestamp) const;
        bool operator<=(const CTimestamp& timestamp) const;
        bool operator>(const CTimestamp& timestamp) const;
        bool operator>=(const CTimestamp& timestamp) const;

        bool operator==(const CTimestamp& timestamp) const;
        bool operator!=(const CTimestamp& timestamp) const;

        bool Set(INT32 year, INT32 month, INT32 day, INT32 hours = 0, INT32 minutes = 0, INT32 seconds = 0, INT32 milliseconds = 0, INT32 microseconds = 0);
        void Set(boost::posix_time::ptime posixTime);
        void Set(INT64 secondsFromEpoch);
        bool Set(std::string timeString);

        INT32 GetYear() const;
        INT32 GetMonth() const;
        std::string GetMonthAsString() const;
        INT32 GetDay() const;

        INT32 GetHours() const;
        INT32 GetMinutes() const;
        INT32 GetSeconds() const;
        INT64 GetMilliseconds() const;
        INT64 GetMicroseconds() const;

        void AddHours(INT32 hours);

        void AddMinutes(INT32 minutes);

        void AddSeconds(INT32 seconds);
        void AddSeconds(double seconds);

        void AddMilliseconds(INT32 milliseconds);

        void AddMicroseconds(my::int64 microseconds);

        boost::posix_time::ptime ToPosixTime() const;
        
        INT64 ToSecondsFromEpoch() const;
        INT64 ToMillisecondsFromEpoch() const;
        INT64 ToMicrosecondsFromEpoch() const;
        
        std::string ToString() const;

        static CTimestamp Now();

        INT64 DifferenceInSeconds(CTimestamp timestamp);
        INT64 DifferenceInMicroseconds(CTimestamp timestamp);

    private:
        void Create();
        void Copy(const CTimestamp& timestamp);

    protected:
        boost::posix_time::ptime m_time;
    };

    /**
    */
    inline INT64 DifferenceInMicroseconds(CTimestamp startTimestamp, CTimestamp endTimestamp)
    {
        return startTimestamp.DifferenceInMicroseconds(endTimestamp);
    }

    /**
    */
    inline INT64 DifferenceInMicroseconds(INT64 startSecondsFromEpoch, INT64 endSecondsFromEpoch)
    {
        return my::CTimestamp(startSecondsFromEpoch).DifferenceInMicroseconds(my::CTimestamp(endSecondsFromEpoch));
    }

    /**
    */
    template <>
    inline CTimestamp Null()
    {
        return CTimestamp(1970, 1, 1);
    }
}; // my

#endif //#if !defined(TIMESTAMP_INCLUDED)

