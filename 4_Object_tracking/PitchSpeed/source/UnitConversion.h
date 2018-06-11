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

#if !defined(UNIT_CONVERSION_INCLUDED)
#define UNIT_CONVERSION_INCLUDED

namespace UnitConversion
{
    // TIME

    //Millisecond
    //Second
    //Minute
    //Hour

    /**
    */
    template <typename T>
    T Picosecond2Microsecond(T psec);

    /**
	*/
	template <typename T>
	T Microsecond2Millisecond(T musec);

	/**
	*/
	template <typename T>
	T Microsecond2Second(T musec);

	/**
	*/
	template <typename T>
	T Millisecond2Second(T msec);

    /**
    */
    template <typename T>
    T Second2Microsecond(T sec);

    /**
    */
    template <typename T>
    T Second2Millisecond(T sec);

    // LENGTH

    //Centimeters
	//Feet
	//Inch
	//Miles
	//Kilometers
	//Meter

    /**
    */
    template <typename T>
    T Centimeters2Inch(T centimeters);

    /**
	*/
	template <typename T>
	T Feet2Inch(T feet);

    /**
    */
    template <typename T>
    T Feet2Centimeters(T feet);

	/**
	*/
	template <typename T>
	T Feet2Meter(T feet);

    /**
    */
    template <typename T>
    T Feet2Millimeters(T feet);

    /**
	*/
	template <typename T>
	T Inch2Feet(T inch);

    /**
    */
    template <typename T>
    T Meter2Feet(T meter);

    /**
    */
    template <typename T>
    T Millimeters2Feet(T millimiter)
    {
        return millimiter * 0.0032808398950131;
    }

    /**
    */
    template <typename T>
    T Yard2Feet(T yard);

    /**
    */
    template <typename T>
    T Inch2Millimeters(T inch)
    {
        return inch * 25.4;
    }

    // VELOCITY

    //FeetPerSecond,
    //FeetPerMilllisecond,
    //MilesPerHour,
    //KilometerPerHour,
    //MeterPerSecond

    /**
    */
    template <typename T>
    T FeetPerMillisecondToFeetPerSecond(T feetMsec);

    /**
    */
    template <typename T>
    T FeetPerSecond2MeterPerSecond(T feetPerSecond);

    /**
    */
    template <typename T>
    T FeetPerSecondToMilesPerHour(T feetSec);

    /**
    */
    template <typename T>
    T KilometerPerHour2FeetPerMilllisecond(T kilometerPerHour);

    /**
    */
    template <typename T>
    T KilometerPerHour2FeetPerSecond(T kilometerPerHour);

	/**
	*/
	template <typename T>
	T MeterPerSecond2FeetPerSecond(T meterPerSecond);

	/**
	*/
	template <typename T>
	T MeterPerSecond2MilesPerHour(T meterPerSecond);
	
	/**
	"In metric, one millimiter of water occupies one cubic
	centimeter, weights one gram, and requires one calorie of
	enery to heat up by one degree centigrade - which is 1 per-
	cent of the difference beween its freezing point and its
	boiling point. An amount of hydrogen weighing the same
	amount has exactly one mole of atoms in it.
	Whereas in the American system, the answer to "How 
	much energy does it take to boil a room-temperature gal-
	lon of water?" is "Go fuck yourself," because you can't 
	directly relate any of those quantities."
    */
    template <typename T>
    T MilesPerHour2FeetPerMilllisecond(T milesPerHour);

    /**
    */
    template <typename T>
    T MilesPerHour2FeetPerSecond(T milesPerHour);    

    // ACCELERATION

    template <typename T>
    T FeetPerSquareSecondToMetersPerSquareSecond(T feetPerSquareSecond);

    // TEMPERATURE

    /**
    */
    template <typename T>
    T Celcius2Fahrenheit(T celcius)
    {
        return celcius * 1.8 + 32.0;
    }

    /**
    */
    template <typename T>
    T Fahrenheit2Celcius(T fahrenheit)
    {
        return (fahrenheit - 32.0) / 1.8;
    }
};

#endif //#if !defined(UNIT_CONVERSION_INCLUDED)

