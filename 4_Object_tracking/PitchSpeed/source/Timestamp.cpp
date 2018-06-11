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

#include <time.h>
#include <ctime>

#include <boost/regex.hpp>

#include "Logger.h"

#include "Timestamp.h"

/**
*/
my::CTimestamp::CTimestamp(INT32 year, INT32 month, INT32 day, INT32 hours, INT32 minutes, INT32 seconds, INT32 milliseconds, INT32 microseconds)
    : m_time(boost::posix_time::ptime())
{
    if (!Set(year, month, day, hours, minutes, seconds, milliseconds, microseconds))
        LOG_ERROR();
}

/**
*/
my::CTimestamp::CTimestamp(boost::posix_time::ptime posixTime)
{
    Set(posixTime);
}

// CASTING!
my::CTimestamp::CTimestamp(INT64 secondsFromEpoch)
{
    Set(secondsFromEpoch);
}

/**
*/
my::CTimestamp::CTimestamp(std::string timeString)
{
    if (!Set(timeString))
        LOG_ERROR();
}

/**
*/
my::CTimestamp::CTimestamp(const CTimestamp& timestamp)
{
    Copy(timestamp);
}

/**
*/
void my::CTimestamp::operator=(const CTimestamp& timestamp)
{
    Copy(timestamp);
}

/**
*/
bool my::CTimestamp::operator<(const CTimestamp& timestamp) const
{
    return m_time < timestamp.m_time;
}

/**
*/
bool my::CTimestamp::operator<=(const CTimestamp& timestamp) const
{
    return m_time <= timestamp.m_time;
}

/**
*/
bool my::CTimestamp::operator>(const CTimestamp& timestamp) const
{
    return m_time > timestamp.m_time;
}

/**
*/
bool my::CTimestamp::operator>=(const CTimestamp& timestamp) const
{
    return m_time >= timestamp.m_time;
}

/**
*/
bool my::CTimestamp::operator==(const CTimestamp& timestamp) const
{
    return m_time == timestamp.m_time;
}

/**
*/
bool my::CTimestamp::operator!=(const CTimestamp& timestamp) const
{
    return m_time != timestamp.m_time;
}

/**
*/
bool my::CTimestamp::Set(INT32 year, INT32 month, INT32 day, INT32 hours, INT32 minutes, INT32 seconds, INT32 milliseconds, INT32 microseconds)
{
    try
    {
        m_time = boost::posix_time::ptime(boost::gregorian::date(year, month, day)) + boost::posix_time::hours(hours) + boost::posix_time::minutes(minutes) + boost::posix_time::seconds(seconds) + boost::posix_time::millisec(milliseconds) + boost::posix_time::microseconds(microseconds);
    }
    catch (...)
    {
        m_time = boost::posix_time::ptime();

        LOG_ERROR();

        return false;
    }

    return true;
}

/**
*/
void my::CTimestamp::Set(boost::posix_time::ptime posixTime)
{
    m_time = posixTime;
}

// CASTING!
void my::CTimestamp::Set(INT64 secondsFromEpoch)
{
    INT32 microseconds = 0,
        milliseconds = 0;

    // TRICKY: (01-Jul-2015) Microseconds (Wed, 16 Nov 5138 09:46:39 GMT)
    if (secondsFromEpoch > 99999999999999)
    {
        microseconds = secondsFromEpoch % 1000;

        secondsFromEpoch /= 1000;
    }
    // TRICKY: (01-Jul-2015) Microseconds (Wed, 16 Nov 5138 09:46:39 GMT)
    if (secondsFromEpoch > 99999999999)
    {
        milliseconds = secondsFromEpoch % 1000;

        secondsFromEpoch /= 1000;
    }

    m_time = boost::posix_time::from_time_t(secondsFromEpoch);

    AddMicroseconds(microseconds);
    AddMilliseconds(milliseconds);
}

/**
*/
bool my::CTimestamp::Set(std::string timeString)
{
    // (BEGIN OF) TESTING: (08-Sep-2017) TESTING A NEW APPROACH BASED ON boost::posix_time::time_from_string
    m_time = boost::posix_time::ptime();

    if (timeString.empty())
    {
        LOG_ERROR();

        return false;
    }

    boost::replace_all(timeString, "T", " ");
    boost::replace_all(timeString, "Z", " ");
    
    boost::trim(timeString);

    // YYYY-MM-DD HH:MM:SS
    // YYYY-MM-DD HH:MM:SS.NNN
    // YYYY-MM-DD HH:MM:SS.NNNNNN
    // YYYY-MM-DD HH:MM:SS.NNNNNNN
    try
    {
        boost::regex re("\\d{4}\\-\\d{2}\\-\\d{2}\\s\\d{2}\\:\\d{2}\\:(\\d{2}|\\d{2}\\.\\d{3,7})");

        if (boost::regex_match(timeString, re))
            m_time = boost::posix_time::time_from_string(timeString);
    }
    catch (...)
    {
        LOG_ERROR();
    }

    // BUG: (09-09-2017) YYYY-MM-DD
    try
    {
        boost::regex re("\\d{4}\\-\\d{2}\\-\\d{2}");

        if (boost::regex_match(timeString, re))
            m_time = boost::posix_time::time_from_string(timeString + " 00:00:00");
    }
    catch (...)
    {
        LOG_ERROR();
    }

    //// DEBUG ONLY! (08-Sep-2017)
    //std::cerr << "INPUT: " << timeString << ", OUTPUT: " << boost::posix_time::to_iso_extended_string(m_time) << std::endl;

    return true;
    // (END OF) TESTING: (08-Sep-2017) TESTING A NEW APPROACH BASED ON boost::posix_time::time_from_string

    //// (BEGIN OF) TESTING: (08-Sep-2017) TESTING AN APPROACH BASED ON std::locale
    //const std::locale formatArray[] = {
    //    // TRICKY: (26-Jun-2015) Fractional seconds.
    //    std::locale(std::locale::classic(), new boost::posix_time::time_input_facet("%Y-%m-%d %H:%M:%S")),
    //    std::locale(std::locale::classic(), new boost::posix_time::time_input_facet("%Y-%m-%d %H:%M:%s")),
    //    std::locale(std::locale::classic(), new boost::posix_time::time_input_facet("%Y/%m/%d %H:%M:%S")),
    //    std::locale(std::locale::classic(), new boost::posix_time::time_input_facet("%d.%m.%Y %H:%M:%S")),
    //    std::locale(std::locale::classic(), new boost::posix_time::time_input_facet("%Y-%m-%d")) };
    //size_t formatCount = sizeof(formatArray) / sizeof(formatArray[0]);

    //m_time = boost::posix_time::ptime();

    //for (size_t formatIndex = 0; formatIndex < formatCount; ++formatIndex)
    //{
    //    std::istringstream is(timeString);

    //    try
    //    {
    //        is.imbue(formatArray[formatIndex]);

    //        is >> m_time;
    //    }
    //    catch (...)
    //    {
    //        LOG_ERROR();

    //        return false;
    //    }

    //    if (m_time != boost::posix_time::ptime())
    //    {
    //        return true;
    //    }
    //}

    //return false;
    //// (END OF) TESTING: (08-Sep-2017) TESTING AN APPROACH BASED ON std::locale
}

/**
*/
INT32 my::CTimestamp::GetYear() const
{
    return m_time.date().year();
}

/**
*/
INT32 my::CTimestamp::GetMonth() const
{
    return (INT32)m_time.date().month().as_number();
}

/**
*/
std::string my::CTimestamp::GetMonthAsString() const
{
    return m_time.date().month().as_long_string();
}

/**
*/
INT32 my::CTimestamp::GetDay() const
{
    return m_time.date().day();
}

/**
*/
INT32 my::CTimestamp::GetHours() const
{
    return m_time.time_of_day().hours();
}

/**
*/
INT32 my::CTimestamp::GetMinutes() const
{
    return m_time.time_of_day().minutes();
}

/**
*/
INT32 my::CTimestamp::GetSeconds() const
{
    return m_time.time_of_day().seconds();
}

/**
*/
INT64 my::CTimestamp::GetMilliseconds() const
{
    // TRICKY: (28-Jun-2015) boost::posix_time arithmetic only! 
    return (INT64)m_time.time_of_day().total_milliseconds() - 1000 * m_time.time_of_day().total_seconds();
}

/**
*/
INT64 my::CTimestamp::GetMicroseconds() const
{
    // TRICKY: (28-Jun-2015) boost::posix_time arithmetic only! 
    return (INT64)m_time.time_of_day().total_microseconds() - 1000 * m_time.time_of_day().total_milliseconds();
}

/**
*/
void my::CTimestamp::AddHours(INT32 hours)
{
    m_time += boost::posix_time::hours(hours);
}

/**
*/
void my::CTimestamp::AddMinutes(INT32 minutes)
{
    m_time += boost::posix_time::minutes(minutes);
}

/**
*/
void my::CTimestamp::AddSeconds(INT32 seconds)
{
    m_time += boost::posix_time::seconds(seconds);
}

/**
*/
void my::CTimestamp::AddSeconds(double seconds)
{
    m_time += boost::posix_time::microseconds((INT64)(1000000.0 * seconds));
}

/**
*/
void my::CTimestamp::AddMilliseconds(INT32 milliseconds)
{
    m_time += boost::posix_time::milliseconds(milliseconds);
}

/**
*/
void my::CTimestamp::AddMicroseconds(my::int64 microseconds)
{
    m_time += boost::posix_time::microseconds(microseconds);
}

/**
*/
boost::posix_time::ptime my::CTimestamp::ToPosixTime() const
{
    return m_time;
}

/**
*/
INT64 my::CTimestamp::ToSecondsFromEpoch() const
{
    boost::posix_time::ptime epoch(boost::gregorian::date(1970, 1, 1));

    INT64 secondsFromEpoch = (INT64)(m_time - epoch).ticks() / boost::posix_time::time_duration::ticks_per_second();

#if defined(_DEBUG)
    // BUG: (29-Jun-2015) Milliseconds are only allowed after Sun, 09 Sep 2001 01:46:40 GMT.
    if (secondsFromEpoch > 1000000000)
    {
        if (GetMilliseconds() != 0)
            LOG_MESSAGE("Milliseconds are being ignored on time computation.");

        if (GetMicroseconds() != 0)
            LOG_MESSAGE("Microseconds are being ignored on time computation.");
    }
#endif //#if defined(_DEBUG)

    return secondsFromEpoch;
}

/**
*/
INT64 my::CTimestamp::ToMillisecondsFromEpoch() const
{
    boost::posix_time::ptime epoch(boost::gregorian::date(1970, 1, 1));

    INT64 secondsFromEpoch = (INT64)(m_time - epoch).ticks() / boost::posix_time::time_duration::ticks_per_second();

    // BUG: (29-Jun-2015) Milliseconds are only allowed after Sun, 09 Sep 2001 01:46:40 GMT.
    if (secondsFromEpoch > 1000000000)
    {
        // milliseconds
        secondsFromEpoch = 1000 * secondsFromEpoch + GetMilliseconds();

#if defined(_DEBUG)
        if (GetMicroseconds() != 0)
            LOG_MESSAGE("Microseconds are being ignored on time computation.");
#endif //#if defined(_DEBUG)
    }
#if defined(_DEBUG)
    else if ((GetMilliseconds() != 0) ||
        (GetMicroseconds() != 0))
    {
        LOG_MESSAGE("Milliseconds are only allowed after Sun, 09 Sep 2001 01:46:40 GMT.");
    }
#endif //#if defined(_DEBUG)

    return secondsFromEpoch;
}

/**
*/
INT64 my::CTimestamp::ToMicrosecondsFromEpoch() const
{
    boost::posix_time::ptime epoch(boost::gregorian::date(1970, 1, 1));

    INT64 secondsFromEpoch = (INT64)(m_time - epoch).ticks() / boost::posix_time::time_duration::ticks_per_second();

    // BUG: (29-Jun-2015) Milliseconds are only allowed after Sun, 09 Sep 2001 01:46:40 GMT.
    if (secondsFromEpoch > 1000000000)
    {
        // milliseconds
        secondsFromEpoch = 1000 * secondsFromEpoch + GetMilliseconds();

        // microseconds
        secondsFromEpoch = 1000 * secondsFromEpoch + GetMicroseconds();
    }
#if defined(_DEBUG)
    else if ((GetMilliseconds() != 0) ||
        (GetMicroseconds() != 0))
    {
        LOG_MESSAGE("Milliseconds are only allowed after Sun, 09 Sep 2001 01:46:40 GMT.");
    }
#endif //#if defined(_DEBUG)

    return secondsFromEpoch;
}

/**
*/
std::string my::CTimestamp::ToString() const
{
    return boost::posix_time::to_iso_extended_string(m_time);
}

/**
*/
my::CTimestamp my::CTimestamp::Now()
{
    return CTimestamp(boost::posix_time::microsec_clock::local_time());
}

/**
*/
INT64 my::CTimestamp::DifferenceInSeconds(CTimestamp timestamp)
{
    return (timestamp.m_time - m_time).total_seconds();
}

/**
*/
INT64 my::CTimestamp::DifferenceInMicroseconds(CTimestamp timestamp)
{
    return (timestamp.m_time - m_time).total_microseconds();
}

/**
*/
void my::CTimestamp::Create()
{
    m_time = boost::posix_time::ptime();
}

/**
*/
void my::CTimestamp::Copy(const CTimestamp& timestamp)
{
    m_time = timestamp.m_time;
}

// SUPPORT FOR THE NEXT (IN-HOUSE) VERSION OF TIMESTAMP:

// http://stlib.sourceforge.net/docs/Timestamp_8cc-source.html

// http://stackoverflow.com/questions/7960318/convert-seconds-since-1970-into-date-and-vice-versa
//#define DAYSPERWEEK (7)
//#define DAYSPERNORMYEAR (365U)
//#define DAYSPERLEAPYEAR (366U)
//
//#define SECSPERDAY (86400UL) /* == ( 24 * 60 * 60) */
//#define SECSPERHOUR (3600UL) /* == ( 60 * 60) */
//#define SECSPERMIN (60UL) /* == ( 60) */
//
//#define LEAPYEAR(year)          (!((year) % 4) && (((year) % 100) || !((year) % 400)))
//
//const int _ytab[2][12] = {
//    { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 },
//    { 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 }
//};
//
///****************************************************
//* Class:Function    : getSecsSomceEpoch
//* Input     : uint16_t epoch date (ie, 1970)
//* Input     : uint8 ptr to returned month
//* Input     : uint8 ptr to returned day
//* Input     : uint8 ptr to returned years since Epoch
//* Input     : uint8 ptr to returned hour
//* Input     : uint8 ptr to returned minute
//* Input     : uint8 ptr to returned seconds
//* Output        : uint32_t Seconds between Epoch year and timestamp
//* Behavior      :
//*
//* Converts MM/DD/YY HH:MM:SS to actual seconds since epoch.
//* Epoch year is assumed at Jan 1, 00:00:01am.
//****************************************************/
//uint32_t getSecsSinceEpoch(uint16_t epoch, uint8_t month, uint8_t day, uint8_t years, uint8_t hour, uint8_t minute, uint8_t second)
//{
//    unsigned long secs = 0;
//    int countleap = 0;
//    int i;
//    int dayspermonth;
//
//    secs = years * (SECSPERDAY * 365);
//    for (i = 0; i < (years - 1); i++)
//    {
//        if (LEAPYEAR((epoch + i)))
//            countleap++;
//    }
//    secs += (countleap * SECSPERDAY);
//
//    secs += second;
//    secs += (hour * SECSPERHOUR);
//    secs += (minute * SECSPERMIN);
//    secs += ((day - 1) * SECSPERDAY);
//
//    if (month > 1)
//    {
//        dayspermonth = 0;
//
//        if (LEAPYEAR((epoch + years))) // Only counts when we're on leap day or past it
//        {
//            if (month > 2)
//            {
//                dayspermonth = 1;
//            }
//            else if (month == 2 && day >= 29) {
//                dayspermonth = 1;
//            }
//        }
//
//        for (i = 0; i < month - 1; i++)
//        {
//            secs += (_ytab[dayspermonth][i] * SECSPERDAY);
//        }
//    }
//
//    return secs;
//}
