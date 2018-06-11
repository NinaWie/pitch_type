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

#if !defined(MY_MATH_INCLUDED)
#define MY_MATH_INCLUDED

#include <math.h>
#include <float.h>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <climits>

#include "Logger.h"
#include "Common.h"

#define COS_10 0.98480775301220805936674302458952
#define COS_11_25 0.98078528040323044912618223613424
#define COS_20 0.93969262078590838405410927732473
#define COS_22_5 0.92387953251128675612818318939679
#define COS_30 0.86602540378443864676372317075294
#define COS_45 0.70710678118654752440084436210485
#define COS_60 0.5
#define COS_67_5 0.3826834323650897717284599840304

#define SIN_10 0.17364817766693034885171662676931
#define SIN_11_25 0.19509032201612826784828486847702
#define SIN_20 0.34202014332566873304409961468226
#define SIN_22_5 0.3826834323650897717284599840304
#define SIN_30 0.5
#define SIN_45 0.70710678118654752440084436210485
#define SIN_60 0.86602540378443864676372317075294
#define SIN_67_5 0.92387953251128675612818318939679

/* sqrt(2) */
#define MY_SQRT_2 1.4142135623730950488016887242097

namespace MyMath
{
    
	/**
	 */
	template <typename T>
	T SquaredMagnitude(T x, T y)
	{
		return x*x + y*y;
	}

	/**
	 */
	template <typename T>
	T SquaredMagnitude(T x, T y, T z)
	{
		return x*x + y*y + z*z;
	}

	/**
	 */
	template <typename T>
	T SquaredMagnitude2(T v[2])
	{
		return v[0]*v[0] + v[1]*v[1];
	}

	/**
	 */
	template <typename T>
	T SquaredMagnitude3(T v[3])
	{
		return v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
	}

	/**
	 */
	template <typename T>
	T SquareRoot(T value);

	template <>
	inline float SquareRoot(float value)
    {
        return sqrtf(value);
    }

	template <>
	inline double SquareRoot(double value)
    {
        return sqrt(value);
    }

	/**
    Portuguese, because fuck you.
    Elementos de uma subtração:
      10 (minuendo)
    -  2 (subtraendo)
     ___
       8 (diferença ou resto)
	*/
	template <typename T>
	void Subtract2(T *minuendo, const T *subtraendo)
	{
		HEALTH_CHECK(!minuendo, /*false*/);
        HEALTH_CHECK(!subtraendo, /*false*/);

        *minuendo++ -= *subtraendo++;
        *minuendo   -= *subtraendo;
	}

	/**
    Portuguese, because fuck you.
    Elementos de uma subtração:
      10 (minuendo)
    -  2 (subtraendo)
     ___
       8 (diferença ou resto)
	*/
	template <typename T>
	void Subtract2(const T *minuendo, const T *subtraendo, T* resto)
	{
        HEALTH_CHECK(!minuendo, /*false*/);
        HEALTH_CHECK(!subtraendo, /*false*/);
        HEALTH_CHECK(!resto, /*false*/);

        *resto++ = *minuendo++ - *subtraendo++;
        *resto   = *minuendo   - *subtraendo;
	}

	/**
	Portuguese, because fuck you.
	Elementos de uma subtração:
	10 (minuendo)
	-  2 (subtraendo)
	___
	8 (diferença ou resto)
	*/
	template <typename T1, typename T2, typename T3>
	void Subtract2(const T1 *minuendo, const T2 *subtraendo, T3* resto)
	{
		HEALTH_CHECK(!minuendo, /*false*/);
		HEALTH_CHECK(!subtraendo, /*false*/);
		HEALTH_CHECK(!resto, /*false*/);

		*resto++ = (T3)(*minuendo++ - *subtraendo++);
		*resto = (T3)(*minuendo - *subtraendo);
	}

	/**
    Portuguese, because fuck you.
    Elementos de uma subtração:
      10 (minuendo)
    -  2 (subtraendo)
     ___
       8 (diferença ou resto)
	*/
	template <typename T>
	void Subtract3(T *minuendo, const T *subtraendo)
	{
        HEALTH_CHECK(!minuendo, /*false*/);

        *minuendo++ -= *subtraendo++;
        *minuendo++ -= *subtraendo++;
        *minuendo   -= *subtraendo;
	}

	/**
    Portuguese, because fuck you.
    Elementos de uma subtração:
      10 (minuendo)
    -  2 (subtraendo)
     ___
       8 (diferença ou resto)
	*/
	template <typename T>
	void Subtract3(const T *minuendo, const T *subtraendo, T* resto)
	{
        HEALTH_CHECK(!resto, /*false*/);

        *resto++ = *minuendo++ - *subtraendo++;
        *resto++ = *minuendo++ - *subtraendo++;
        *resto   = *minuendo   - *subtraendo;
	}

	/**
    Portuguese, because fuck you.
    Elementos de uma subtração:
      10 (minuendo)
    -  2 (subtraendo)
     ___
       8 (diferença ou resto)
	*/
	template <typename T>
	void Subtract3(T minuendo0, T minuendo1, T minuendo2, T subtraendo0, T subtraendo1, T subtraendo2, T* resto)
	{
        HEALTH_CHECK(!resto, /*false*/);

        *resto++ = minuendo0 - subtraendo0;
        *resto++ = minuendo1 - subtraendo1;
        *resto   = minuendo2 - subtraendo2;
	}

	///////////////////////////////////////////////////////////////////////////
	// T                                                                     //
	///////////////////////////////////////////////////////////////////////////

    /**
	 */
	template <typename T>
	T Tangent(T angle);

	template <>
	inline double Tangent(double angle)
	{
        return tan(angle);
	}

	template <>
	inline float Tangent(float angle)
	{
        return tanf(angle);
	}

    // BUG: (??-???-????) It's here because some compilers demand definitions before the first call.
    template <typename T>
    T Maximum();

    template <>
    inline INT8 Maximum()
    {
        return MAX_INT8;
    }

    template <>
    inline INT16 Maximum()
    {
        return MAX_INT16;
    }

    template <>
    inline INT32 Maximum()
    {
        return MAX_INT32;
    }

    template <>
    inline INT64 Maximum()
    {
        return MAX_INT64;
    }

    template <>
    inline float Maximum()
    {
        return FLT_MAX;
    }

    template <>
    inline double Maximum()
    {
        return DBL_MAX;
    }

    // BUG: (??-???-????) It's here because some compilers demand definitions before the first call.
    template <typename T>
    T Minimum();

    template <>
    inline INT8 Minimum()
    {
        return MIN_INT8;
    }

    template <>
    inline INT16 Minimum()
    {
        return MIN_INT16;
    }

    template <>
    inline INT32 Minimum()
    {
        return MIN_INT32;
    }

    template <>
    inline INT64 Minimum()
    {
        return MIN_INT64;
    }

    template <>
    inline float Minimum()
    {
        return -FLT_MAX;
    }

    template <>
    inline double Minimum()
    {
        return -DBL_MAX;
    }

	// BUG: (??-???-????) It's here because some compilers demand definitions before the first call.
	template <typename T>
	void CrossProduct3(const T u[3], const T v[3], T result[3])
	{
		result[0] = u[1] * v[2] - u[2] * v[1];
		result[1] = u[2] * v[0] - u[0] * v[2];
		result[2] = u[0] * v[1] - u[1] * v[0];
	}

	// BUG: (??-???-????) It's here because some compilers demand definitions before the first call.
	template <typename T>
	T DotProduct3(const T* u, const T* v)
	{
		return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
	}

	// BUG: (??-???-????) It's here because some compilers demand definitions before the first call.
	template <typename T> 
	T Epsilon();

    template <>
	inline float Epsilon()
	{
		return FLT_EPSILON;
	}

    template <>
	inline double Epsilon()
	{
		return DBL_EPSILON;
	}

	// BUG: (??-???-????) It's here because some compilers demand definitions before the first call.
	template <typename T>
	T EuclideanDistance2(const T *a, const T *b);

    template <>
	inline double EuclideanDistance2(const double *a, const double *b)
	{
		double dx,
			dy;

		dx = b[0] - a[0];
		dy = b[1] - a[1];

		return sqrt(dx*dx + dy*dy);
	}

	template <>
	inline float EuclideanDistance2(const float *a, const float *b)
	{
		float dx,
			dy;

		dx = b[0] - a[0];
		dy = b[1] - a[1];

		return sqrtf(dx*dx + dy*dy);
	}

	// BUG: (??-???-????) It's here because some compilers demand definitions before the first call.
	template <typename T>
	T EuclideanDistance3(const T a[3], const T b[3]);

	template <>
	inline double EuclideanDistance3(const double a[3], const double b[3])
	{
		double dx,
			dy,
			dz;

		dx = b[0] - a[0];
		dy = b[1] - a[1];
		dz = b[2] - a[2];

		return sqrt(dx*dx + dy*dy + dz*dz);
	}

	template <>
	inline float EuclideanDistance3(const float a[3], const float b[3])
	{
		float dx,
			dy,
			dz;

		dx = b[0] - a[0];
		dy = b[1] - a[1];
		dz = b[2] - a[2];

		return sqrtf(dx*dx + dy*dy + dz*dz);
	}

	// BUG: (??-???-????) It's here because some compilers demand definitions before the first call.
	template <typename T> 
	T Pi();

	template <> 
	inline float Pi()
	{
		return 3.1415927f;
	}

	template <> 
	inline double Pi()
	{
		return 3.1415926535897931;
	}

	// BUG: (??-???-????) It's here because some compilers demand definitions before the first call.
	template <typename T>
	T SquaredEuclideanDistance3(const T *a, const T *b)
	{
		HEALTH_CHECK(!a, 0);
		HEALTH_CHECK(!b, 0);

		T dx = b[0] - a[0],
			dy = b[1] - a[1],
			dz = b[2] - a[2];

		return dx*dx + dy*dy + dz*dz;
	}

    // BUG: (??-???-????) It's here because some compilers demand definitions before the first call.
    // The scalar triple product of three vectors A, B, and C is denoted [A,B,C] and defined by
    // [A,B,C] = A.(BxC)
    //         = B.(CxA)
    //         = C.(AxB)
    //         = det(ABC)
    //           |A_1 A_2 A_3|
    //         = |B_1 B_2 B_3|
    //           |C_1 C_2 C_3|
    // where A.B denotes a dot product, AxB denotes a cross product, det(A)=|A| denotes a determinant, and A_i, B_i, and C_i are components of the vectors A, B, and C, respectively.
	template <typename T>
	T TripleProduct3(const T A[3], const T B[3], const T C[3])
	{
		T BC[3] = { 0 };

		CrossProduct3<T>(B, C, BC);

		return DotProduct3<T>(A, BC);
	}

    // BUG: (??-???-????) It's here because some compilers demand definitions before the first call.
    template <typename T>
    T IsZero(T v);

    template <>
    inline double IsZero(double v)
    {
        return (fabs(v) < DBL_EPSILON);
    }

    template <>
    inline float IsZero(float v)
    {
        return (fabsf(v) < FLT_EPSILON);
    }

    // BUG: (??-???-????) It's here because some compilers demand definitions before the first call.
    template <typename T>
    bool IsValid(T value);

    template <>
    inline bool IsValid(float value)
    {
#if defined(_WIN32)
        return !my::IsNull(value) &&
            // -INF, +INF, NaN
            (_finite(value) != 0) &&
            // NaN
            (_isnan(value) == 0);
#else // #if defined(_WIN32)
        return !my::IsNull(value) &&
            !std::isinf(value) &&
            !std::isnan(value);
#endif // #if defined(_WIN32)
    }

    template <>
    inline bool IsValid(double value)
    {
#if defined(_WIN32)
        return !my::IsNull(value) &&
            // -INF, +INF, NaN
            (_finite(value) != 0) &&
            // NaN
            (_isnan(value) == 0);
#else // #if defined(_WIN32)
        return !my::IsNull(value) &&
            !std::isinf(value) &&
            !std::isnan(value);
#endif // #if defined(_WIN32)
    }

    ///////////////////////////////////////////////////////////////////////////
	// A                                                                     //
	///////////////////////////////////////////////////////////////////////////

	/**
	*/
	template <typename T>
	void Add2(T* sum, const T *addend)
	{
        *sum++ = *addend++;
        *sum   = *addend;
	}

	/**
	*/
	template <typename T>
	void Add2(const T *addend0, const T *addend1, T* sum)
	{
        *sum++ = *addend0++ + *addend1++;
        *sum   = *addend0   + *addend1;
	}

    /**
    */
    template <typename T>
    void Add3(T *sum, const T *addend)
    {
        *sum++ += *addend++;
        *sum++ += *addend++;
        *sum   += *addend;
    }

	/**
	*/
	template <typename T>
	void Add3(const T *addend0, const T *addend1, T* sum)
	{
        *sum++ = *addend0++ + *addend1++;
        *sum++ = *addend0++ + *addend1++;
        *sum   = *addend0   + *addend1;
	}

	/**
    Given in radians.
	*/
    template <typename T>
    T AngleBetweenVectors3(const T *u, const T *v);

    template <>
    inline double AngleBetweenVectors3(const double *u, const double *v)
    {
	    double uv_dot = 0,
		    u_norm = 0,
		    v_norm = 0,
		    angle = 0,
		    angle_cos = 0;

	    HEALTH_CHECK(!u, 0);
	    HEALTH_CHECK(!v, 0);

	    uv_dot = MyMath::DotProduct3(u, v);

	    u_norm = sqrt(MyMath::DotProduct3(u, u));
	    v_norm = sqrt(MyMath::DotProduct3(v, v));

	    angle_cos = uv_dot / u_norm / v_norm;

	    if (angle_cos <= -1.0)
		    angle = MyMath::Pi<double>();
	    else if (1.0 <= angle_cos)
		    angle = 0.0;
	    else
		    angle = acos(angle_cos);

	    return angle;
    }

	/**
	*/
	template <typename T>
	bool ApplyConvolutionKernel(const T *image, int width, int height, const T *kernel, int kernelSize, T *outputImage)
	{
        int i,
            j;
        int kc,
            ki,
            kj;
        int pi,
            pj;
        T *outputValue;

        HEALTH_CHECK(width < 1, false);
        HEALTH_CHECK(height < 1, false);
        HEALTH_CHECK(kernelSize < 2, false);
        HEALTH_CHECK(!image, false);
        HEALTH_CHECK(!outputImage, false);

        kc = kernelSize>>1;

        for (j=0; j<height; ++j)
        {
            for (i=0; i<width; ++i)
            {
                outputValue = outputImage + j*width + i;

                (*outputValue) = 0;

                for (kj=0; kj<kernelSize; ++kj)
                {
                    pj = std::max(std::min(j - kc + kj, height - 1), 0);

                    for (ki=0; ki<kernelSize; ++ki)
                    {
                        pi = std::max(std::min(i - kc + ki, width - 1), 0);

                        (*outputValue) += (*(kernel + kj*kernelSize + ki))*(*(image + pj*width + pi));
                    }
                }
            }
        }

        return true;
	}

	/**
	*/
	template <typename T1, typename T>
	void Assign(T1 x, T *to)
	{
        // Near useless.
        HEALTH_CHECK(!to, /*false*/);

        *to   = (T)x;
	}

	/**
	*/
	template <typename T1, typename T2, typename T>
	void Assign(T1 x, T2 y, T *to)
	{
        // Near useless.
        HEALTH_CHECK(!to, /*false*/);

        *to++ = (T)x;
        *to   = (T)y;
	}

	/**
	*/
	template <typename T1, typename T2, typename T3, typename T>
	void Assign(T1 x, T2 y, T3 z, T *to)
	{
        // Near useless.
        HEALTH_CHECK(!to, /*false*/);

        *to++ = (T)x;
        *to++ = (T)y;
        *to   = (T)z;
	}

	/**
	*/
	template <typename T1, typename T2, typename T3, typename T4, typename T>
	void Assign(T1 x, T2 y, T3 z, T4 w, T *to)
	{
        // Near useless.
        HEALTH_CHECK(!to, /*false*/);

        *to++ = (T)x;
        *to++ = (T)y;
        *to++ = (T)z;
        *to   = (T)w;
	}

	/**
	*/
	template <typename T>
	void Assign2(const T *from, T *to)
	{
		// Near useless.
		HEALTH_CHECK(!to, /*false*/);

		*to++ = *from++;
		*to = *from;
	}

	/**
	*/
	template <typename T1, typename T2>
	void Assign2(const T1 *from, T2 *to)
	{
		// Near useless.
		HEALTH_CHECK(!to, /*false*/);

		*to++ = (T2)*from++;
		*to = (T2)*from;
	}

	/**
	*/
    template <typename T1, typename T2>
    void Assign3(const T1 *from, T2 *to)
    {
        // Near useless.
        HEALTH_CHECK(!from, /*false*/);
        // Near useless.
        HEALTH_CHECK(!to, /*false*/);

        to[0] = from[0];
        to[1] = from[1];
        to[2] = from[2];
    }

	/**
	*/
    template <typename T1, typename T2>
    void Assign3(T1 a, T1 b, T1 c, T2 to)
	{
        to[0] = a;
        to[1] = b;
        to[2] = c;
    }

	/**
	*/
	template <typename T>
	void Assign4(T value, T *to)
	{
		// Near useless.
		HEALTH_CHECK(!to, /*false*/);

		*to++ = value;
		*to++ = value;
		*to++ = value;
		*to = value;
	}

	/**
	*/
	template <typename T1, typename T2>
	void Assign4(const T1 *from, T2 *to)
	{
        // Near useless.
        HEALTH_CHECK(!to, /*false*/);

        *to++ = (T2)*from++;
        *to++ = (T2)*from++;
        *to++ = (T2)*from++;
        *to   = (T2)*from;
	}

    /**
    */
    template <typename T1, typename T2>
    void Assign9(T1 from, T2 to)
    {
        to[0] = from[0];
        to[1] = from[1];
        to[2] = from[2];
        to[3] = from[3];
        to[4] = from[4];
        to[5] = from[5];
        to[6] = from[6];
        to[7] = from[7];
        to[8] = from[8];
    }

    /**
	*/
	template <typename T1, typename T2>
	void Assign16(const T1 *from, T2 *to)
	{
        // Near useless.
        HEALTH_CHECK(!to, /*false*/);

        *to++ = (T2)*from++;
        *to++ = (T2)*from++;
        *to++ = (T2)*from++;
        *to++ = (T2)*from++;
        *to++ = (T2)*from++;
        *to++ = (T2)*from++;
        *to++ = (T2)*from++;
        *to++ = (T2)*from++;
        *to++ = (T2)*from++;
        *to++ = (T2)*from++;
        *to++ = (T2)*from++;
        *to++ = (T2)*from++;
        *to++ = (T2)*from++;
        *to++ = (T2)*from++;
        *to++ = (T2)*from++;
        *to   = (T2)*from;
	}

	///////////////////////////////////////////////////////////////////////////
	// B                                                                     //
	///////////////////////////////////////////////////////////////////////////

	/**
	All triangle points shall be valid 2d arrays.
	http://www.gamedev.net/topic/621445-barycentric-coordinates-c-code-check/
	 */
	template <typename T>
	bool BarycentricCoordinates2(const T A[2], const T B[2], const T C[2], const T p[2], T lambda[3])
	{
		T den;

		den = (T)1/((B[1] - C[1])*(A[0] - C[0]) + (C[0] - B[0])*(A[1] - C[1]));

        lambda[0] = ((B[1] - C[1])*(p[0] - C[0]) + (C[0] - B[0])*(p[1] - C[1]))*den;
        lambda[1] = ((C[1] - A[1])*(p[0] - C[0]) + (A[0] - C[0])*(p[1] - C[1]))*den;
        lambda[2] = 1 - lambda[0] - lambda[1];

        return true;
	}

	/**
	 */
    template <typename T>
    void Bezier3(T t, const T* C1, const T* C2, const T* C3, const T* C4, T* p);

    template <>
    inline void Bezier3(float t, const float* C1, const float* C2, const float* C3, const float* C4, float* p)
    {
        float t1,
            t2,
            t3,
            t4;

        t1 = t*t*t;
        t2 = 3.0f*t*t*(1.0f - t);
        t3 = 3.0f*t*(1.0f - t)*(1.0f - t);
        t4 = (1.0f - t)*(1.0f - t)*(1.0f - t);

        p[0] = C1[0]*t1 + C2[0]*t2 + C3[0]*t3 + C4[0]*t4;
        p[1] = C1[1]*t1 + C2[1]*t2 + C3[1]*t3 + C4[1]*t4;
        p[2] = C1[2]*t1 + C2[2]*t2 + C3[2]*t3 + C4[2]*t4;
    }

    template <>
    inline void Bezier3(double t, const double* C1, const double* C2, const double* C3, const double* C4, double* p)
    {
        double t1,
            t2,
            t3,
            t4;

        t1 = t*t*t;
        t2 = 3.0*t*t*(1.0 - t);
        t3 = 3.0*t*(1.0 - t)*(1.0 - t);
        t4 = (1.0 - t)*(1.0 - t)*(1.0 - t);

        p[0] = C1[0]*t1 + C2[0]*t2 + C3[0]*t3 + C4[0]*t4;
        p[1] = C1[1]*t1 + C2[1]*t2 + C3[1]*t3 + C4[1]*t4;
        p[2] = C1[2]*t1 + C2[2]*t2 + C3[2]*t3 + C4[2]*t4;
    }

	// Hypertexture (Ken Perlin, Eric M. Hoffert)
	template <typename T>
	T Bias(T bias, T t);

    // Hypertexture (Ken Perlin, Eric M. Hoffert)
    template <>
	inline double Bias(double bias, double t)
	{
		return pow(t, log(bias)/log(0.5));
	}

    // Hypertexture (Ken Perlin, Eric M. Hoffert)
    template <>
	inline float Bias(float bias, float t)
	{
		return powf(t, logf(bias)/logf(0.5f));
	}

    ///////////////////////////////////////////////////////////////////////////
	// C                                                                     //
	///////////////////////////////////////////////////////////////////////////

    // BUG: (08-Dec-2015) THE PARAMETER IS NOT INPUT / OUTPUT
    template < typename T >
    inline T Clamp(T value, T minimum, T maximum)
    {
        return std::min(std::max(value, minimum), maximum);
    }

    template < typename T >
    inline void Clamp(T& x, T& y, T valueMin, T valueMax)
    {
        x = std::min(std::max(x, valueMin), valueMax);
        y = std::min(std::max(y, valueMin), valueMax);
    }

    template < typename T >
    inline void Clamp(T& x, T& y, T& z, T valueMin, T valueMax)
    {
        x = std::min(std::max(x, valueMin), valueMax);
        y = std::min(std::max(y, valueMin), valueMax);
        z = std::min(std::max(z, valueMin), valueMax);
    }

    template < typename T >
    inline void Clamp(T& x, T& y, T& z, T& w, T valueMin, T valueMax)
    {
        x = std::min(std::max(x, valueMin), valueMax);
        y = std::min(std::max(y, valueMin), valueMax);
        z = std::min(std::max(z, valueMin), valueMax);
        w = std::min(std::max(w, valueMin), valueMax);
    }

    template <typename T>
	inline void Clamp2(T *v, T minVal, T maxVal)
	{
		HEALTH_CHECK(!v, /*false*/);

		v[0] = std::min(std::max(v[0], minVal), maxVal);
		v[1] = std::min(std::max(v[1], minVal), maxVal);
	}

	/**
	*/
	template <typename T>
	void ClosestPointOnSegment2(const T *p, const T *s0, const T *s1, T *cp)
	{
		HEALTH_CHECK(!p, /*false*/);
		HEALTH_CHECK(!s0, /*false*/);
		HEALTH_CHECK(!s1, /*false*/);
		HEALTH_CHECK(!cp, /*false*/);

		T m = MyMath::EuclideanDistance2(s1, s0),
			U = (((p[0] - s0[0]) * (s1[0] - s0[0])) + ((p[1] - s0[1]) * (s1[1] - s0[1]))) / (m * m);

		MyMath::Clamp(U, (T)0, (T)1);

		cp[0] = s0[0] + U * (s1[0] - s0[0]);
		cp[1] = s0[1] + U * (s1[1] - s0[1]);
	}

	/**
	*/
	template <typename T>
	void ClosestPointOnSegment3(const T *p, const T *s0, const T *s1, T *cp)
	{
		HEALTH_CHECK(!p, /*false*/);
		HEALTH_CHECK(!s0, /*false*/);
		HEALTH_CHECK(!s1, /*false*/);
		HEALTH_CHECK(!cp, /*false*/);

		T m = MyMath::EuclideanDistance3(s1, s0),
			U = (((p[0] - s0[0]) * (s1[0] - s0[0])) + ((p[1] - s0[1]) * (s1[1] - s0[1])) + ((p[2] - s0[2]) * (s1[2] - s0[2]))) / (m * m);

		MyMath::Clamp(U, (T)0, (T)1);

		cp[0] = s0[0] + U * (s1[0] - s0[0]);
		cp[1] = s0[1] + U * (s1[1] - s0[1]);
		cp[2] = s0[2] + U * (s1[2] - s0[2]);
	}

	template <typename T>
    inline T ClosestPointOnTriangle3(const T *vertex, const T *A, const T *B, const T *C, T *closestPoint)
    {
        T point[3];
        T l1,
            l2,
            l3;
        T minimumDistance,
            distance;

        minimumDistance = std::numeric_limits<T>::max();

        // DEBUG ONLY!
        int i,
            j,
            numberOfSamples = 20;

        for (i=0; i<=numberOfSamples; ++i)
        {
            for (j=0; j<=(numberOfSamples - i); ++j)
            {
                l1 = (T)i/numberOfSamples;
                l2 = (T)j/numberOfSamples;
                l3 = std::max((T)1 - l1 - l2, (T)0);
         
                point[0] = l1*A[0] + l2*B[0] + l3*C[0];
                point[1] = l1*A[1] + l2*B[1] + l3*C[1];
                point[2] = l1*A[2] + l2*B[2] + l3*C[2];

                distance = MyMath::SquaredEuclideanDistance3(vertex, point);

                if (minimumDistance > distance)
                {
                    minimumDistance = distance;

                    MyMath::Assign3(point, closestPoint);
                }
            }
        }

        HEALTH_CHECK(minimumDistance == std::numeric_limits<T>::max(), (T)-1);

        return minimumDistance;
    }

    /**
	 */
	template <typename T>
	T Cosine(T angle);

	template <>
	inline double Cosine(double angle)
	{
        return cos(angle);
	}

	template <>
	inline float Cosine(float angle)
	{
        return cosf(angle);
	}

    /**
	 */
	template <typename T>
	T CosineInterpolate(T y0, T y1, T mu);

	template <>
	inline double CosineInterpolate(double y0, double y1, double mu)
	{
		double mu2;
		
		mu2 = (1.0 - cos(mu*MyMath::Pi<double>()))/2.0;
		
		return (y0*(1.0 - mu2) + y1*mu2);
	}

	template <>
	inline float CosineInterpolate(float y0, float y1, float mu)
	{
		float mu2;
		
		mu2 = (1.0f - cos(mu*Pi<float>()))/2.0f;
		
		return (y0*(1.0f - mu2) + y1*mu2);
	}

	///////////////////////////////////////////////////////////////////////////
	// D                                                                     //
	///////////////////////////////////////////////////////////////////////////

	/**
	*/
	template <typename T> 
	T DegreesToRadians(T degrees);

	template <> 
	inline float DegreesToRadians(float degrees)
	{
		return degrees*Pi<float>()/180.0f;
	}

	template <> 
	inline double DegreesToRadians(double degrees)
	{
		return degrees*Pi<double>()/180.0;
	}
    
    /**
     */
    template <typename ValueType>
    bool Derivative(int n, const ValueType *x, const ValueType *y, ValueType *dy)
    {
        HEALTH_CHECK(n <= 0, false);
        HEALTH_CHECK(!x, false);
        HEALTH_CHECK(!y, false);
        HEALTH_CHECK(!dy, false);
        
        for (int i = 0; i < (n - 1); ++i)
        {
            ValueType num = y[i + 1] - y[i],
                den = x[i + 1] - x[i],
                // TRICKY: (09-Jul-2015) 0 / 0 WILL NOT RESULT ON NaN, BUT 0 INSTEAD
                d = 0;

            if (IsZero(den) &&
                !IsZero(num))
            {
                LOG_ERROR();

                return false;
            }
            else
                d = num / den;

            HEALTH_CHECK(!MyMath::IsValid(d), false);
            
            dy[i] = d;
        }
        
        dy[n - 1] = 0;
        
        return true;
    }

	/**
	 3x3 matrix determinant.
	 */
	template <typename T> 
	T Determinant(T m00, T m01, T m02, T m10, T m11, T m12, T m20, T m21, T m22)
	{
		T minorDeterminant[3],
			determinant;

		minorDeterminant[0] = m11*m12 - m21*m22;
		minorDeterminant[1] = m01*m02 - m21*m22;
		minorDeterminant[2] = m01*m02 - m11*m12;

		determinant = m00*minorDeterminant[0] - m10*minorDeterminant[1] + m20*minorDeterminant[2];

		return determinant;
	}

	/**
	 4x4 matrix determinant.
	 */
	template <typename T>
	float Determinant(T m00, T m01, T m02, T m03, T m10, T m11, T m12, T m13, T m20, T m21, T m22, T m23, T m30, T m31, T m32, T m33)
	{
		T minorDeterminant[4],
			determinant;

		minorDeterminant[0] = m11*(m22*m33 - m23*m32) - m12*(m21*m33 - m23*m31) + m13*(m21*m32 - m22*m31);
		minorDeterminant[1] = m10*(m22*m33 - m23*m32) - m12*(m20*m33 - m23*m30) + m13*(m20*m32 - m22*m30);
		minorDeterminant[2] = m10*(m21*m33 - m23*m31) - m11*(m20*m33 - m23*m30) + m13*(m20*m31 - m21*m30);
		minorDeterminant[3] = m10*(m21*m32 - m22*m31) - m11*(m20*m32 - m22*m30) + m12*(m20*m31 - m21*m30);

		determinant = m00*minorDeterminant[0] - m01*minorDeterminant[1] + m02*minorDeterminant[2] - m03*minorDeterminant[3];

		return determinant;
	}

    /**
	 */
	template <typename T>
	T DotProduct(T x0, T y0, T x1, T y1)
	{
		return x0 * x1 + y0 * y1;
	}

    /**
    */
    template <typename T>
    T DotProduct(T x0, T y0, T z0, T x1, T y1, T z1)
    {
        return x0 * x1 + y0 * y1 + z0 * z1;
    }

    /**
    */
    template <typename T>
    T DotProduct(T x0, T y0, T z0, T w0, T x1, T y1, T z1, T w1)
    {
        return x0 * x1 + y0 * y1 + z0 * z1 + w0 * w1;
    }

    /**
	 */
	template <typename T>
	T DotProduct2(const T u[2], const T v[2])
	{
		return u[0]*v[0] + u[1]*v[1];
	}

	/**
	 */
	template <typename T>
	T DotProduct4(const T u[4], const T v[4])
	{
		return u[0]*v[0] + u[1]*v[1] + u[2]*v[2] + u[3]*v[3];
	}

	///////////////////////////////////////////////////////////////////////////
	// E                                                                     //
	///////////////////////////////////////////////////////////////////////////

	///**
	// */
	//template <typename T> 
	//T Epsilon();

	//template <> 
	//inline float Epsilon()
	//{
	//	return FLT_EPSILON;
	//}

	//template <> 
	//inline double Epsilon()
	//{
	//	return DBL_EPSILON;
	//}

	/**
	 */
	template <typename T>
	T EuclideanDistance(T x0, T y0, T x1, T y1);

	template <>
	inline double EuclideanDistance(double x0, double y0, double x1, double y1)
	{
        double dx,
            dy;

        dx = x0 - x1;
        dy = y0 - y1;

		return sqrt(dx*dx + dy*dy);
	}

	template <>
	inline float EuclideanDistance(float x0, float y0, float x1, float y1)
	{
        float dx,
            dy;

        dx = x0 - x1;
        dy = y0 - y1;

		return sqrtf(dx*dx + dy*dy);
	}

	/**
	 */
	template <typename T>
	T EuclideanDistance(T x0, T y0, T z0, T x1, T y1, T z1);

	template <>
	inline double EuclideanDistance(double x0, double y0, double z0, double x1, double y1, double z1)
	{
        double dx,
            dy,
            dz;

        dx = x0 - x1;
        dy = y0 - y1;
        dz = z0 - z1;

		return sqrt(dx*dx + dy*dy + dz*dz);
	}

	template <>
	inline float EuclideanDistance(float x0, float y0, float z0, float x1, float y1, float z1)
	{
        float dx,
            dy,
            dz;

        dx = x0 - x1;
        dy = y0 - y1;
        dz = z0 - z1;

		return sqrtf(dx*dx + dy*dy + dz*dz);
	}

	///////////////////////////////////////////////////////////////////////////
	// G                                                                     //
	///////////////////////////////////////////////////////////////////////////

	/**
	 Unashamedly copied from 
	 http://cobweb.ecn.purdue.edu/~ebertd/texture/perlin/perlin.c
	 */
	template <typename T>
	T Gain(T gain, T t);

	template <>
	inline double Gain(double gain, double t)
	{
		double p;
		
		if (t < Epsilon<double>())
			return 0;
		else if (t > (1.0 - Epsilon<double>()))
			return 1;

		p = log(1.0 - gain)/log(0.5);

		if (t < 0.5)
			return pow(2.0*t, p)/2.0;
		else
			return 1.0 - pow(2.0*(1.0 - t), p)/2.0;
	}

	template <>
	inline float Gain(float gain, float t)
	{
		float p;
		
		if (t < Epsilon<float>())
			return 0;
		else if (t > (1.0f - Epsilon<float>()))
			return 1;

		p = logf(1.0f - gain)/log(0.5f);

		if (t < 0.5f)
			return powf(2.0f*t, p)/2.0f;
		else
			return 1.0f - powf(2.0f*(1.0f - t), p)/2.0f;
	}

    /**
     */
    template <typename T>
    bool GaussianKernel(T standardDeviation, int kernelSize, std::vector<T>& kernel)
    {
        int i,
            j,
            c;
        int di,
            dj;
        T _2s2,
            inv_pi_2s2,
            GxyAccum,
            Gxy;

        HEALTH_CHECK(standardDeviation <= 0, false);
        HEALTH_CHECK(kernelSize < 3, false);

        c = kernelSize>>1;

        HEALTH_CHECK(c == 0, false);

        _2s2 = (T)2*standardDeviation*standardDeviation;
        inv_pi_2s2 = (T)1/(MyMath::Pi<T>()*_2s2);

        GxyAccum = 0;

        kernel.clear();

        for (j=0; j<kernelSize; ++j)
        {
            dj = j - c;

            for (i=0; i<kernelSize; ++i)
            {
                di = i - c;

                Gxy = inv_pi_2s2*exp(-(T)(di*di + dj*dj)/_2s2);

                GxyAccum += Gxy;

                kernel.push_back(Gxy);
            }
        }

        HEALTH_CHECK(GxyAccum == 0, false);

        for (typename std::vector<T>::iterator kernelIterator=kernel.begin(); kernelIterator!=kernel.end(); ++kernelIterator)
            (*kernelIterator) /= GxyAccum;

        return true;
    }

	///////////////////////////////////////////////////////////////////////////
	// H                                                                     //
	///////////////////////////////////////////////////////////////////////////

	/**
	*/
	template <typename T>
	std::vector<UINT32> Histogram(const std::vector<T>& measurementArray, INT32 size)
	{
		double minimumValue = MyMath::Maximum<double>(),
			maximumValue = MyMath::Minimum<double>();

		for (typename std::vector<T>::const_iterator measurementIterator = measurementArray.begin(); measurementIterator != measurementArray.end(); ++measurementIterator)
		{
			minimumValue = std::min(minimumValue, (double)(*measurementIterator));
			maximumValue = std::max(maximumValue, (double)(*measurementIterator));
		}

		std::vector<UINT32> histogram(size);

		for (typename std::vector<T>::const_iterator measurementIterator = measurementArray.begin(); measurementIterator != measurementArray.end(); ++measurementIterator)
		{
			double t = ((double)(*measurementIterator) - minimumValue) / (maximumValue - minimumValue);

			INT32 bin = (INT32)(t*(double)(size - 1));

			++histogram[MyMath::Clamp(bin, 0, size - 1)];
		}

		return histogram;
	}

	///////////////////////////////////////////////////////////////////////////
	// I                                                                     //
	///////////////////////////////////////////////////////////////////////////

	/**
	*/
	template <typename T>
	void Inflate3(const T *A, const T *B, const T *C, T s, T *A1, T *B1, T *C1)
	{
        T center[3];

        center[0] = (A[0] + B[0] + C[0])/(T)3;
        center[1] = (A[1] + B[1] + C[1])/(T)3;
        center[2] = (A[2] + B[2] + C[2])/(T)3;

        A1[0] = center[0] + s*(A[0] - center[0]);
        A1[1] = center[1] + s*(A[1] - center[1]);
        A1[2] = center[2] + s*(A[2] - center[2]);

        B1[0] = center[0] + s*(B[0] - center[0]);
        B1[1] = center[1] + s*(B[1] - center[1]);
        B1[2] = center[2] + s*(B[2] - center[2]);

        C1[0] = center[0] + s*(C[0] - center[0]);
        C1[1] = center[1] + s*(C[1] - center[1]);
        C1[2] = center[2] + s*(C[2] - center[2]);
    }

	/**
	 */
	template <typename T>
	bool IsClose(T a, T b, T tolerance);

	/**
	 */
	template <>
    inline bool IsClose(double a, double b, double tolerance)
	{
		return (fabs(a - b) < tolerance);
	}

	/**
	 */
	template <>
    inline bool IsClose(float a, float b, float tolerance)
	{
		return (fabsf(a - b) < tolerance);
	}

	/**
	*/
	template <typename T>
    bool IsClose3(const T *a, const T *b, T tolerance);

	/**
	*/
	template <>
    inline bool IsClose3(const double *a, const double *b, double tolerance)
	{
		return (fabs(a[0] - b[0]) < tolerance) &&
			(fabs(a[1] - b[1]) < tolerance) &&
			(fabs(a[2] - b[2]) < tolerance);
	}

	/**
	*/
	template <>
    inline bool IsClose3(const float *a, const float *b, float tolerance)
	{
		return (fabsf(a[0] - b[0]) < tolerance) &&
			(fabsf(a[1] - b[1]) < tolerance) &&
			(fabsf(a[2] - b[2]) < tolerance);
	}

	///////////////////////////////////////////////////////////////////////////
	// L                                                                     //
	///////////////////////////////////////////////////////////////////////////

	/**
	 Least common multiple.
	 */
	inline int LCM(int a, int b)
	{
		int n;

		for (n=1; ; n++)
		{
			if (n%a == 0 && n%b == 0)
				return n;
		}
	}

	/**
	f*(x) = a + b*x
	error = $\sum_i^n (y_i - f*(x_i))^2$
	*/
	template <typename T>
	void LeastSquaresFitting(const T *x, const T *y, int n, T *a, T *b, T *error)
	{
		T sxy = 0,
			sx = 0,
			sy = 0,
			sx2 = 0;

		for (int i = 0; i != n; i++)
		{
			sxy += x[i] * y[i];

			sx += x[i];

			sy += y[i];

			sx2 += x[i] * x[i];
		}

		T xb = sx / n,
			yb = sy / n,
			e = 0;

		*b = (sxy - (sx*sy) / n) / (sx2 - (sx*sx / n));
		*a = yb - (*b)*xb;

		*error = 0;

		for (int i = 0; i != n; i++)
		{
			e = y[i] - ((*a) + (*b)*x[i]);

			*error += e*e;
		}
	}

	/**
	 */
	template <typename T>
	T LinearInterpolation(T a, T b, T mu)
	{
		//return (a*(T(1) - mu) + b*mu);
        return a + mu*(b - a);
	}

	/**
	 */
	template <typename T>
	void LinearInterpolation3(const T *a, const T *b, T mu, T *output)
	{
        output[0] = (a[0]*(T(1) - mu) + b[0]*mu);
        output[1] = (a[1]*(T(1) - mu) + b[1]*mu);
        output[2] = (a[2]*(T(1) - mu) + b[2]*mu);
	}

	/**
	 Unashamedly copied from 
	 http://www.math.niu.edu/~rusin/known-math/99/tetr_interp

	 (SIC)
	 ... a simple linear interpolation in barycentric coordinates 
	 (c1,c2,...,c_n,c_{n+1}) within the simplex defined by N+1 points and N+1 
	 values in R^n. In one dimension it would correspond to interpolate
	
	 x1 -> f1
	 x2 -> f2
	
	 f(c1*x1+c2*x2) = c1*f1 + c2*f2    
	
	 (c1+c2=1), all c_i>=0. In two dimensions
	
	 P1=(x1,y1) -> f1
	 P2=(x2,y2) -> f2
	 P3=(x3,y3) -> f3
	
	 f(c1*P1+c2*P2+c3*P3) = c1*f1+c2*f2+c3*f3
	
	 c1+c2+c3=1, all c_i>=0. In three dimensions
	
	 P1=(x1,y1,z1) -> f1
	 P2=(x2,y2,z2) -> f2
	 P3=(x3,y3,z3) -> f3
	 P4=(x4,y4,z4) -> f4
	
	 f(c1*P1+c2*P2+c3*P3+c4*P4) := c1*f1+c2*f2+c3*f3+c3*f4 
	
	 c1+c2+c3+c4=1, all c_i>=0. For any set of (a) 2 points which are not 
	 identical, (b) 3 points which are not on the same straigth line and (c) 4 
	 points which are not in the same plane. There is one and only one affine 
	 mapping from R, R^2, and R^3 into a linear space E such that f(P_i)=f_i, 
	 and that's probably the mapping you want because this is the "natural" 
	 mapping. Any set of 4 nonnegative real numbers (c1,c2,c3,c4) with sum 1 - in
	 the so-called barycentric coordinate system relative to the tetrahedron 
	 P1,P2,P3,P4 - will yield some (x,y,z)=c1*P1+c2*P2+c3*P3+c4*P4 inside the 
	 tetrahedron and some f by f=c1*f1+c2*f2+c3*f3+c4*f4 where f fulfills the 
	 four initial conditions f(P_i)=f_i if c_i=1 and the rest of the c_j=0 and 
	 where f is the unique affine mapping defined by these conditions. If one 
	 or more of the c_i are negative then the point is outside the tetrahedron 
	 but the mapping is still a continuation of the same mapping.
	
	 You may rewrite the above formula in a asymetric way by choosing any P_i 
	 as the "anchor point", for instance P1 
	
	 f((1-c2-c3-c4)P1 + c2P2 + c3P3 + c4)
	 = f(P1 + c2(P2-P1) + c3(P3-P1) + c3(P4-P1))
	 = f1 + c2*(f2-f1) + c3*(f3-f1) + c4*(f4-f1)
	
	 where P is in the tetrahedron if and only if c2,c3,c4 are >=0 and c2+c3+c4<=1, 
	 i.e. c1=1-c2-c3-c4 >=0.
	
	 No problem so far. What is your problem? If you start from the c_i your 
	 problem is solved. Do you want to get the corresponding c_i for an arbitrary 
	 (x,y,z)?

	 This may be written in an elegant way. The c2,c3,c4 are the solutions of the 
	 linear equation system

	 x2-x1 x3-x1 x4-x1   c2   x-x1
	 y2-y1 y3-y1 y4-y1 * c3 = y-y1
	 z2-z1 z3-z1 z4-z1   c4   z-z1
	
	 for c2,c3,c4 where x,y,z are the input parameters. If you substitute 
	 Q2:=P2-P1, Q3:=P3-P1, Q4=P4-P1 then 
	
	 x-x1 
	 c2 = < y-y1 , [Q3.Q4]/det(Q2.Q3.Q4) >
	 z-y1 
	
	 x-x1 
	 c3 = < y-y1 , [Q4.Q2]/det(Q2,Q3,Q4) >
	 z-y1 
	
	 x-x1 
	 c4 = < y-y1 , [Q2.Q3]/det(Q2,Q3,Q4) >
	 z-y1 

	 Where <*.*> denotes the dot product and [*,*] the cross product and 
	 det(*,*,*) the determinant which may be evaluated as 
	
	 det(Q2,Q3,Q4) = <Q2,[Q3,Q4]> = <Q3,[Q4,Q2]> = <Q4,[Q2,Q3]>
	
	 f(x,y,z) = f1 + c2*(f2-f1) + c3*(f3-f1) + c4*(f4-f1)
	 = (1-c2-c3-c4)*f1 + c2*f2 + c3*f3 *c4*f4
	 */
	template <typename T, int size>
	void LinearInterpolationInsideTetrahedron(const T p1[3], const T f1[size], const T p2[3], const T f2[size], const T p3[3], const T f3[size], const T p4[3], const T f4[size], const T p[3], T f[size])
	{
		int i;
		T Q2[3],
			Q3[3],
			Q4[3],
			Qp[3],
			determinantQ2Q3Q4,
			crossQ3Q4[3],
			crossQ4Q2[3],
			crossQ2Q3[3],
			c2, 
			c3,
			c4;

		for (i=0; i<3; i++)
		{
			Q2[i] = p2[i] - p1[i];
			Q3[i] = p3[i] - p1[i];
			Q4[i] = p4[i] - p1[i];

			Qp[i] = p[i] - p1[i];
		}

		determinantQ2Q3Q4 = TripleProduct3<T>(Q2, Q3, Q4);

		CrossProduct3<T>(Q3, Q4, crossQ3Q4);

		crossQ3Q4[0] /= determinantQ2Q3Q4;
		crossQ3Q4[1] /= determinantQ2Q3Q4;
		crossQ3Q4[2] /= determinantQ2Q3Q4;

		c2 = DotProduct3<T>(Qp, crossQ3Q4);

		CrossProduct3<T>(Q4, Q2, crossQ4Q2);

		crossQ4Q2[0] /= determinantQ2Q3Q4;
		crossQ4Q2[1] /= determinantQ2Q3Q4;
		crossQ4Q2[2] /= determinantQ2Q3Q4;

		c3 = DotProduct3<T>(Qp, crossQ4Q2);

		CrossProduct3<T>(Q2, Q3, crossQ2Q3);

		crossQ2Q3[0] /= determinantQ2Q3Q4;
		crossQ2Q3[1] /= determinantQ2Q3Q4;
		crossQ2Q3[2] /= determinantQ2Q3Q4;

		c4 = DotProduct<T>(Qp[0], Qp[1], Qp[2], crossQ2Q3[0], crossQ2Q3[1], crossQ2Q3[2]);

		for (i=0; i<size; ++i)
			f[i] = (T(1) - c2 - c3 - c4)*f1[i] + c2*f2[i] + c3*f3[i] + c4*f4[i];
	}

    /*
    Linear Regression
    y(x) = a + b x, for n samples
    The following assumes the standard deviations are unknown for x and y
    Return a, b and r the regression coefficient
    */
    inline bool LinearRegression(double *x, double *y, int n, double *a, double *b, double *r)
    {
        int i = 0;
        double sumx = 0, 
            sumy = 0, 
            sumx2 = 0, 
            sumy2 = 0, 
            sumxy = 0;
        double sxx = 0, 
            syy = 0, 
            sxy = 0;

        *a = 0;
        *b = 0;
        *r = 0;

        if (n < 2)
            return false;

        /* Conpute some things we need */
        for (i = 0; i < n; i++)
        {
            sumx += x[i];
            sumy += y[i];
            sumx2 += (x[i] * x[i]);
            sumy2 += (y[i] * y[i]);
            sumxy += (x[i] * y[i]);
        }

        sxx = sumx2 - sumx * sumx / n;
        syy = sumy2 - sumy * sumy / n;
        sxy = sumxy - sumx * sumy / n;

        /* Infinite slope (b), non existant intercept (a) */
        if (fabs(sxx) == 0)
            return false;

        /* Calculate the slope (b) and intercept (a) */
        *b = sxy / sxx;
        *a = sumy / n - (*b) * sumx / n;

        // The coefficient of correlation between the uncertainty in a and the uncertainty in b, which is a number between -1 and 1.
        if (fabs(syy) == 0)
            *r = 1;
        else
            *r = sxy / sqrt(sxx * syy);

        return true;
    }

	///////////////////////////////////////////////////////////////////////////
	// M                                                                     //
	///////////////////////////////////////////////////////////////////////////

    // Re-maps a number from one range to another. Numbers outside of the range are not clamped to the minimum and maximum parameters values, because out-of-range values are often intentional and useful.
    // value: the incoming value to be converted
    // start1: lower bound of the value's current range
    // stop1: upper bound of the value's current range
    // start2: lower bound of the value's target range
    // stop2: upper bound of the value's target range
    template < typename InputType, typename OutputType >
    OutputType Map(InputType value, InputType start1, InputType stop1, OutputType start2, OutputType stop2);

    template < >
    inline double Map(double value, double start1, double stop1, double start2, double stop2)
    {
        double t = (value - start1) / (stop1 - start1);

        if (stop2 > start2)
            t = fabs(t);

        return start2 + t * (stop2 - start2);
    }

    template < >
    inline double Map(INT64 value, INT64 start1, INT64 stop1, double start2, double stop2)
    {
        double t = (double)(value - start1) / (stop1 - start1);

        if (stop2 > start2)
            t = fabs(t);

        return start2 + t * (stop2 - start2);
    }

    template < >
    inline INT64 Map(double value, double start1, double stop1, INT64 start2, INT64 stop2)
    {
        double t = (value - start1) / (stop1 - start1);

        if (stop2 > start2)
            t = fabs(t);

        return (INT64)(t * (stop2 - start2) + start2);
    }

    template < >
    inline double Map(INT32 value, INT32 start1, INT32 stop1, double start2, double stop2)
    {
        double t = (double)(value - start1) / (stop1 - start1);

        if (stop2 > start2)
            t = fabs(t);

        return start2 + t * (stop2 - start2);
    }

    template < >
    inline INT32 Map(double value, double start1, double stop1, INT32 start2, INT32 stop2)
    {
        double t = (value - start1) / (stop1 - start1);

        if (stop2 > start2)
            t = fabs(t);

        return (INT32)(t * (stop2 - start2) + start2);
    }

    // Re-maps a number from one range to another. Numbers outside of the range are not clamped to the minimum and maximum parameters values, because out-of-range values are often intentional and useful.
    // value: the incoming value to be converted
    // start1: lower bound of the value's current range
    // stop1: upper bound of the value's current range
    // start2: lower bound of the value's target range
    // stop2: upper bound of the value's target range
    // TRICKY: (??-???-????) USING DOUBLE AS INTERNAL PRECISION (MIND THE CASTINGS)
    template < typename InputType, typename OutputType >
    void Map(InputType *valueArray, INT64 size, InputType start1, InputType stop1, OutputType start2, OutputType stop2);

    template < >
    inline void Map(double *valueArray, INT64 size, double start1, double stop1, double start2, double stop2)
    {
        HEALTH_CHECK(!valueArray, /*false*/);

        double *value = valueArray;

        while (size-- > 0)
        {
            double t = ((*value) - start1) / (stop1 - start1);

            if (stop2 > start2)
                t = fabs(t);

            (*value++) = start2 + t * (stop2 - start2);
        }
    }

    /**
	 */
	template <typename T>
	T Magnitude(T x, T y);

	template <>
	inline double Magnitude(double x, double y)
	{
		return sqrt(x*x + y*y);
	}

	template <>
	inline float Magnitude(float x, float y)
	{
		return sqrtf(x*x + y*y);
	}

	/**
	 */
	template <typename T>
	T Magnitude(T x, T y, T z);

	template <>
	inline double Magnitude(double x, double y, double z)
	{
		return sqrt(x*x + y*y + z*z);
	}

	template <>
	inline float Magnitude(float x, float y, float z)
	{
		return sqrtf(x*x + y*y + z*z);
	}

	/**
	 */
	template <typename T>
	T Magnitude2(const T* v);

	template <>
	inline double Magnitude2(const double *v)
	{
        HEALTH_CHECK(!v, 0);

		return sqrt(v[0]*v[0] + v[1]*v[1]);
	}

	template <>
	inline float Magnitude2(const float *v)
	{
        HEALTH_CHECK(!v, 0);

		return sqrtf(v[0]*v[0] + v[1]*v[1]);
	}

	/**
	 */
	template <typename T>
	T Magnitude3(const T *v);

	template <>
	inline double Magnitude3(const double *v)
	{
		HEALTH_CHECK(!v, 0);

		return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
	}

	template <>
	inline float Magnitude3(const float *v)
	{
		HEALTH_CHECK(!v, 0);

		return sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	}

    template < typename ValueType >
    bool MedianFilter(const ValueType* in, ValueType* out, int w, int N)
    {
        HEALTH_CHECK(!in, false);
        HEALTH_CHECK(!out, false);
        HEALTH_CHECK(in == out, false);
        HEALTH_CHECK(w < 3, false);

        std::vector<ValueType> window;

        int h = w / 2,
            l = -h;

        // even?
        if (w % 2)
            ++h;

        for (int i = 0; i < N; ++i)
        {
            int il = std::max(0, l),
                ih = std::min(N, h);

            window.assign(in + il, in + ih);

            std::sort(window.begin(), window.end());

            int cw = (int)window.size(),
                cwh = cw / 2;

            // even?
            if (cw % 2)
                out[i] = window[cwh];
            else
                out[i] = (ValueType)(0.5 * (window[cwh] + window[cwh - 1]));

            ++l;
            ++h;
        }

        return true;
    }

    /**
    */
#define MEDIAN_FILTER_SWAP(a, b) if (window[a] > window[b]) std::swap(window[a], window[b]);

    template < typename ValueType >
    bool MedianFilter3(const ValueType* in, ValueType* out, int N)
    {
        HEALTH_CHECK(!in, false);
        HEALTH_CHECK(!out, false);
        HEALTH_CHECK(in == out, false);
        HEALTH_CHECK(N < 3, false);

        double window[3] = { 0 };

        // 0

        out[0] = in[0];

        // [1, N - 1)

        for (int i = 1; i < N - 1; ++i)
        {
            for (int j = 0; j < 3; ++j)
                window[j] = in[i - 1 + j];

            // Network sort (3)
            MEDIAN_FILTER_SWAP(1, 2);
            MEDIAN_FILTER_SWAP(0, 2);
            MEDIAN_FILTER_SWAP(0, 1);

            out[i] = window[1];
        }

        // N - 1

        out[N - 1] = in[N - 1];

        return true;
    }

    template < typename ValueType >
    bool MedianFilter5(const ValueType* in, ValueType* out, int N)
    {
        HEALTH_CHECK(!in, false);
        HEALTH_CHECK(!out, false);
        HEALTH_CHECK(in == out, false);
        HEALTH_CHECK(N < 5, false);

        double window[5] = { 0 };

        // 0

        // (BEGIN OF) TESTING: (17-Jan-2018) CHANGING THE BEHAVIOR OF THE FILTER WHEN THERE ARE NO SAMPLES ENOUGH (N) FOR THE WINDOW. INSTEAD OF MAKING THE WINDOW ASYMMETRICAL, TAKING ALL THE AVAILABLE SAMPLES, WE'LL TAKE ONLY SAMPLES WHILE THE WINDOW IS SYMMETRICAL.
        //for (int j = 0; j < 3; ++j)
        //    window[j] = in[j];

        //// Network sort (3)
        //MEDIAN_FILTER_SWAP(1, 2);
        //MEDIAN_FILTER_SWAP(0, 2);
        //MEDIAN_FILTER_SWAP(0, 1);

        //out[0] = window[1];
        out[0] = in[0];
        // (END OF) TESTING: (17-Jan-2018) CHANGING THE BEHAVIOR OF THE FILTER WHEN THERE ARE NO SAMPLES ENOUGH (N) FOR THE WINDOW. INSTEAD OF MAKING THE WINDOW ASYMMETRICAL, TAKING ALL THE AVAILABLE SAMPLES, WE'LL TAKE ONLY SAMPLES WHILE THE WINDOW IS SYMMETRICAL.

        // 1

        // (BEGIN OF) TESTING: (17-Jan-2018) CHANGING THE BEHAVIOR OF THE FILTER WHEN THERE ARE NO SAMPLES ENOUGH (N) FOR THE WINDOW. INSTEAD OF MAKING THE WINDOW ASYMMETRICAL, TAKING ALL THE AVAILABLE SAMPLES, WE'LL TAKE ONLY SAMPLES WHILE THE WINDOW IS SYMMETRICAL.
        //for (int j = 0; j < 4; ++j)
        //    window[j] = in[j];

        //// Network sort (4)
        //MEDIAN_FILTER_SWAP(0, 1);
        //MEDIAN_FILTER_SWAP(2, 3);
        //MEDIAN_FILTER_SWAP(0, 2);
        //MEDIAN_FILTER_SWAP(1, 3);
        //// NO EFFECT ON MEDIAN(5)
        ////MEDIAN_FILTER_SWAP(1, 2);

        //out[1] = (window[1] + window[2]) / 2.0;
        for (int j = 0; j < 3; ++j)
            window[j] = in[j];

        // Network sort (3)
        MEDIAN_FILTER_SWAP(1, 2);
        MEDIAN_FILTER_SWAP(0, 2);
        MEDIAN_FILTER_SWAP(0, 1);

        out[1] = window[1];
        // (END OF) TESTING: (17-Jan-2018) CHANGING THE BEHAVIOR OF THE FILTER WHEN THERE ARE NO SAMPLES ENOUGH (N) FOR THE WINDOW. INSTEAD OF MAKING THE WINDOW ASYMMETRICAL, TAKING ALL THE AVAILABLE SAMPLES, WE'LL TAKE ONLY SAMPLES WHILE THE WINDOW IS SYMMETRICAL.

        // [2, N - 2)

        for (int i = 2; i < N - 2; ++i)
        {
            for (int j = 0; j < 5; ++j)
                window[j] = in[i - 2 + j];

            // Network sort (5)
            MEDIAN_FILTER_SWAP(0, 1);
            MEDIAN_FILTER_SWAP(3, 4);
            MEDIAN_FILTER_SWAP(2, 4);
            MEDIAN_FILTER_SWAP(2, 3);
            MEDIAN_FILTER_SWAP(0, 3);
            MEDIAN_FILTER_SWAP(0, 2);
            MEDIAN_FILTER_SWAP(1, 4);
            MEDIAN_FILTER_SWAP(1, 3);
            MEDIAN_FILTER_SWAP(1, 2);

            out[i] = window[2];
        }

        // N - 2

        // (BEGIN OF) TESTING: (17-Jan-2018) CHANGING THE BEHAVIOR OF THE FILTER WHEN THERE ARE NO SAMPLES ENOUGH (N) FOR THE WINDOW. INSTEAD OF MAKING THE WINDOW ASYMMETRICAL, TAKING ALL THE AVAILABLE SAMPLES, WE'LL TAKE ONLY SAMPLES WHILE THE WINDOW IS SYMMETRICAL.
        //for (int j = 0; j < 4; ++j)
        //    window[j] = in[N - 1 - j];

        //// Network sort (4)
        //MEDIAN_FILTER_SWAP(0, 1);
        //MEDIAN_FILTER_SWAP(2, 3);
        //MEDIAN_FILTER_SWAP(0, 2);
        //MEDIAN_FILTER_SWAP(1, 3);
        //// NO EFFECT ON MEDIAN(5)
        ////MEDIAN_FILTER_SWAP(1, 2);

        //out[N - 2] = (window[1] + window[2]) / 2.0;
        for (int j = 0; j < 3; ++j)
            window[j] = in[N - 1 - j];

        // Network sort (3)
        MEDIAN_FILTER_SWAP(1, 2);
        MEDIAN_FILTER_SWAP(0, 2);
        MEDIAN_FILTER_SWAP(0, 1);

        out[N - 2] = window[1];
        // (END OF) TESTING: (17-Jan-2018) CHANGING THE BEHAVIOR OF THE FILTER WHEN THERE ARE NO SAMPLES ENOUGH (N) FOR THE WINDOW. INSTEAD OF MAKING THE WINDOW ASYMMETRICAL, TAKING ALL THE AVAILABLE SAMPLES, WE'LL TAKE ONLY SAMPLES WHILE THE WINDOW IS SYMMETRICAL.

        // N - 1

        // (BEGIN OF) TESTING: (17-Jan-2018) CHANGING THE BEHAVIOR OF THE FILTER WHEN THERE ARE NO SAMPLES ENOUGH (N) FOR THE WINDOW. INSTEAD OF MAKING THE WINDOW ASYMMETRICAL, TAKING ALL THE AVAILABLE SAMPLES, WE'LL TAKE ONLY SAMPLES WHILE THE WINDOW IS SYMMETRICAL.
        //for (int j = 0; j < 3; ++j)
        //    window[j] = in[N - 1 - j];

        //// Network sort (3)
        //MEDIAN_FILTER_SWAP(1, 2);
        //MEDIAN_FILTER_SWAP(0, 2);
        //MEDIAN_FILTER_SWAP(0, 1);

        //out[N - 1] = window[1];
        out[N - 1] = in[N - 1];
        // (END OF) TESTING: (17-Jan-2018) CHANGING THE BEHAVIOR OF THE FILTER WHEN THERE ARE NO SAMPLES ENOUGH (N) FOR THE WINDOW. INSTEAD OF MAKING THE WINDOW ASYMMETRICAL, TAKING ALL THE AVAILABLE SAMPLES, WE'LL TAKE ONLY SAMPLES WHILE THE WINDOW IS SYMMETRICAL.

        return true;
    }

    template < typename ValueType >
    bool MedianFilter5(const ValueType* in, ValueType* out, int *constraint, int N)
    {
        HEALTH_CHECK(!in, false);
        HEALTH_CHECK(!out, false);
        HEALTH_CHECK(!constraint, false);
        HEALTH_CHECK(in == out, false);
        HEALTH_CHECK(N < 5, false);

        double window[5] = { 0 };

        // 0

        for (int j = 0; j < 3; ++j)
            window[j] = in[j];

        // Network sort (3)
        MEDIAN_FILTER_SWAP(1, 2);
        MEDIAN_FILTER_SWAP(0, 2);
        MEDIAN_FILTER_SWAP(0, 1);

        if (!constraint[0])
            out[0] = window[1];

        // 1

        for (int j = 0; j < 4; ++j)
            window[j] = in[j];

        // Network sort (4)
        MEDIAN_FILTER_SWAP(0, 1);
        MEDIAN_FILTER_SWAP(2, 3);
        MEDIAN_FILTER_SWAP(0, 2);
        MEDIAN_FILTER_SWAP(1, 3);
        // NO EFFECT ON MEDIAN(5)
        //MEDIAN_FILTER_SWAP(1, 2);

        if (!constraint[1])
            out[1] = (window[1] + window[2]) / 2.0;

        // [2, N - 2)

        for (int i = 2; i < N - 2; ++i)
        {
            for (int j = 0; j < 5; ++j)
                window[j] = in[i - 2 + j];

            // Network sort (5)
            MEDIAN_FILTER_SWAP(0, 1);
            MEDIAN_FILTER_SWAP(3, 4);
            MEDIAN_FILTER_SWAP(2, 4);
            MEDIAN_FILTER_SWAP(2, 3);
            MEDIAN_FILTER_SWAP(0, 3);
            MEDIAN_FILTER_SWAP(0, 2);
            MEDIAN_FILTER_SWAP(1, 4);
            MEDIAN_FILTER_SWAP(1, 3);
            MEDIAN_FILTER_SWAP(1, 2);

            if (!constraint[i])
                out[i] = window[2];
        }

        // N - 2

        for (int j = 0; j < 4; ++j)
            window[j] = in[N - 1 - j];

        // Network sort (4)
        MEDIAN_FILTER_SWAP(0, 1);
        MEDIAN_FILTER_SWAP(2, 3);
        MEDIAN_FILTER_SWAP(0, 2);
        MEDIAN_FILTER_SWAP(1, 3);
        // NO EFFECT ON MEDIAN(5)
        //MEDIAN_FILTER_SWAP(1, 2);

        if (!constraint[N - 2])
            out[N - 2] = (window[1] + window[2]) / 2.0;

        // N - 1

        for (int j = 0; j < 3; ++j)
            window[j] = in[N - 1 - j];

        // Network sort (3)
        MEDIAN_FILTER_SWAP(1, 2);
        MEDIAN_FILTER_SWAP(0, 2);
        MEDIAN_FILTER_SWAP(0, 1);

        if (!constraint[N - 1])
            out[N - 1] = window[1];

        return true;
    }

    template < typename ValueType >
    bool MedianFilter7(const ValueType* in, ValueType* out, int N)
    {
        HEALTH_CHECK(!in, false);
        HEALTH_CHECK(!out, false);
        HEALTH_CHECK(in == out, false);
        HEALTH_CHECK(N < 7, false);

        double window[7] = { 0 };

        // 0

        for (int j = 0; j < 4; ++j)
            window[j] = in[j];

        MEDIAN_FILTER_SWAP(0, 1);
        MEDIAN_FILTER_SWAP(2, 3);
        MEDIAN_FILTER_SWAP(0, 2);
        MEDIAN_FILTER_SWAP(1, 3);
        MEDIAN_FILTER_SWAP(1, 2);

        out[0] = (window[1] + window[2]) / 2.0;

        // 1

        for (int j = 0; j < 5; ++j)
            window[j] = in[j];

        MEDIAN_FILTER_SWAP(0, 1);
        MEDIAN_FILTER_SWAP(3, 4);
        MEDIAN_FILTER_SWAP(2, 4);
        MEDIAN_FILTER_SWAP(2, 3);
        MEDIAN_FILTER_SWAP(0, 3);
        MEDIAN_FILTER_SWAP(0, 2);
        MEDIAN_FILTER_SWAP(1, 4);
        MEDIAN_FILTER_SWAP(1, 3);
        MEDIAN_FILTER_SWAP(1, 2);

        out[1] = window[2];

        // 2

        for (int j = 0; j < 6; ++j)
            window[j] = in[j];

        MEDIAN_FILTER_SWAP(1, 2);
        MEDIAN_FILTER_SWAP(0, 2);
        MEDIAN_FILTER_SWAP(0, 1);
        MEDIAN_FILTER_SWAP(4, 5);
        MEDIAN_FILTER_SWAP(3, 5);
        MEDIAN_FILTER_SWAP(3, 4);
        MEDIAN_FILTER_SWAP(0, 3);
        MEDIAN_FILTER_SWAP(1, 4);
        MEDIAN_FILTER_SWAP(2, 5);
        MEDIAN_FILTER_SWAP(2, 4);
        MEDIAN_FILTER_SWAP(1, 3);
        MEDIAN_FILTER_SWAP(2, 3);

        out[2] = (window[2] + window[3]) / 2.0;

        // [3, N - 3)

        for (int i = 3; i < (N - 3); ++i)
        {
            for (int j = 0; j < 7; ++j)
                window[j] = in[i - 3 + j];

            MEDIAN_FILTER_SWAP(1, 2);
            MEDIAN_FILTER_SWAP(0, 2);
            MEDIAN_FILTER_SWAP(0, 1);
            MEDIAN_FILTER_SWAP(3, 4);
            MEDIAN_FILTER_SWAP(5, 6);
            MEDIAN_FILTER_SWAP(3, 5);
            MEDIAN_FILTER_SWAP(4, 6);
            MEDIAN_FILTER_SWAP(4, 5);
            MEDIAN_FILTER_SWAP(0, 4);
            MEDIAN_FILTER_SWAP(0, 3);
            MEDIAN_FILTER_SWAP(1, 5);
            MEDIAN_FILTER_SWAP(2, 6);
            MEDIAN_FILTER_SWAP(2, 5);
            MEDIAN_FILTER_SWAP(1, 3);
            MEDIAN_FILTER_SWAP(2, 4);
            MEDIAN_FILTER_SWAP(2, 3);

            out[i] = window[3];
        }

        // N - 3
        for (int j = 0; j < 6; ++j)
            window[j] = in[N - 1 - j];

        MEDIAN_FILTER_SWAP(1, 2);
        MEDIAN_FILTER_SWAP(0, 2);
        MEDIAN_FILTER_SWAP(0, 1);
        MEDIAN_FILTER_SWAP(4, 5);
        MEDIAN_FILTER_SWAP(3, 5);
        MEDIAN_FILTER_SWAP(3, 4);
        MEDIAN_FILTER_SWAP(0, 3);
        MEDIAN_FILTER_SWAP(1, 4);
        MEDIAN_FILTER_SWAP(2, 5);
        MEDIAN_FILTER_SWAP(2, 4);
        MEDIAN_FILTER_SWAP(1, 3);
        MEDIAN_FILTER_SWAP(2, 3);

        out[N - 3] = (window[2] + window[3]) / 2.0;

        // N - 2

        for (int j = 0; j < 5; ++j)
            window[j] = in[N - 1 - j];

        MEDIAN_FILTER_SWAP(0, 1);
        MEDIAN_FILTER_SWAP(3, 4);
        MEDIAN_FILTER_SWAP(2, 4);
        MEDIAN_FILTER_SWAP(2, 3);
        MEDIAN_FILTER_SWAP(0, 3);
        MEDIAN_FILTER_SWAP(0, 2);
        MEDIAN_FILTER_SWAP(1, 4);
        MEDIAN_FILTER_SWAP(1, 3);
        MEDIAN_FILTER_SWAP(1, 2);

        out[N - 2] = window[2];

        // N - 1

        for (int j = 0; j < 4; ++j)
            window[j] = in[N - 1 - j];

        MEDIAN_FILTER_SWAP(0, 1);
        MEDIAN_FILTER_SWAP(2, 3);
        MEDIAN_FILTER_SWAP(0, 2);
        MEDIAN_FILTER_SWAP(1, 3);
        MEDIAN_FILTER_SWAP(1, 2);

        out[N - 1] = (window[1] + window[2]) / 2.0;

        return true;
    }

    /**
	 Converted on 30.0335 degrees south, 51.2207 degrees west coordinates.
	 */
	template <typename T>
	inline T MetersToLatitude(T meters)
	{
		return (0.00089932160009573083/100.0)*meters;
	}

	/**
	 Converted on 30.0335 degrees south, 51.2207 degrees west coordinates.
	 */
	template <typename T>
	inline T MetersToLongitude(T meters)
	{
		return (0.0010000000001300626/100.0)*meters;
	}

	/**
	*/
	template <typename T>
	void Multiply2(T *v, T s)
	{
        *v++ *= s;
        *v   *= s;
	}

	/**
	*/
	template <typename T>
	void Multiply3(T *v, T s)
	{
        *v++ *= s;
        *v++ *= s;
        *v   *= s;
	}

	/**
	*/
	template <typename T>
	T Multiply3(const T *a, const T *b)
	{
        T r;

        r = (*a++)*(*b++);
        r += (*a++)*(*b++);
        r += (*a)*(*b);

        return r;
    }

	/**
	*/
	template <typename T>
	void Multiply4x4(const T *A, const T *B, T* R)
    {
        *(R     ) = *A * *(B +  0) + *(A + 4) * *(B +  1) + *(A +  8) * *(B +  2) + *(A + 12) * *(B +  3);
        *(R +  4) = *A * *(B +  4) + *(A + 4) * *(B +  5) + *(A +  8) * *(B +  6) + *(A + 12) * *(B +  7);
        *(R +  8) = *A * *(B +  8) + *(A + 4) * *(B +  9) + *(A +  8) * *(B + 10) + *(A + 12) * *(B + 11);
        *(R + 12) = *A * *(B + 12) + *(A + 4) * *(B + 13) + *(A +  8) * *(B + 14) + *(A + 12) * *(B + 15);
    
        ++A;    
    
        *(R +  1) = *A * *(B +  0) + *(A + 4) * *(B +  1) + *(A + 8) * *(B +  2) + *(A + 12) * *(B +  3);
        *(R +  5) = *A * *(B +  4) + *(A + 4) * *(B +  5) + *(A + 8) * *(B +  6) + *(A + 12) * *(B +  7);
        *(R +  9) = *A * *(B +  8) + *(A + 4) * *(B +  9) + *(A + 8) * *(B + 10) + *(A + 12) * *(B + 11);
        *(R + 13) = *A * *(B + 12) + *(A + 4) * *(B + 13) + *(A + 8) * *(B + 14) + *(A + 12) * *(B + 15);
    
        ++A;

        *(R +  2) = *A * *(B +  0) + *(A + 4) * *(B +  1) + *(A + 8) * *(B +  2) + *(A + 12) * *(B +  3);
        *(R +  6) = *A * *(B +  4) + *(A + 4) * *(B +  5) + *(A + 8) * *(B +  6) + *(A + 12) * *(B +  7);
        *(R + 10) = *A * *(B +  8) + *(A + 4) * *(B +  9) + *(A + 8) * *(B + 10) + *(A + 12) * *(B + 11);
        *(R + 14) = *A * *(B + 12) + *(A + 4) * *(B + 13) + *(A + 8) * *(B + 14) + *(A + 12) * *(B + 15);
    
        ++A;

        *(R +  3) = *A * *(B +  0) + *(A + 4) * *(B +  1) + *(A + 8) * *(B +  2) + *(A + 12) * *(B +  3);
        *(R +  7) = *A * *(B +  4) + *(A + 4) * *(B +  5) + *(A + 8) * *(B +  6) + *(A + 12) * *(B +  7);
        *(R + 11) = *A * *(B +  8) + *(A + 4) * *(B +  9) + *(A + 8) * *(B + 10) + *(A + 12) * *(B + 11);
        *(R + 15) = *A * *(B + 12) + *(A + 4) * *(B + 13) + *(A + 8) * *(B + 14) + *(A + 12) * *(B + 15);
    }

	///////////////////////////////////////////////////////////////////////////
	// N                                                                     //
	///////////////////////////////////////////////////////////////////////////

	/**
	 */
	template <typename T>
	T Normalize(T *x, T *y)
    {
#if defined(_DEBUG)
        if (!IsValid(*x))
            return 0;

        if (!IsValid(*y))
            return 0;
#endif //#if defined(_DEBUG)

        T norma = SquareRoot((*x) * (*x) + (*y) * (*y));

        if (norma > Epsilon<T>())
        {
            *x /= norma;
            *y /= norma;
        }

        if (!IsValid(*x))
            (*x) = 0;

        if (!IsValid(*y))
            (*y) = 0;

        return norma;
    }

    /**
    */
    template <typename T>
    T Normalize(T *x, T *y, T *z)
    {
#if defined(_DEBUG)
        if (!IsValid(*x))
            return 0;

        if (!IsValid(*y))
            return 0;

        if (!IsValid(*z))
            return 0;
#endif //#if defined(_DEBUG)

        T norma = SquareRoot((*x) * (*x) + (*y) * (*y) + (*z) * (*z));

        if (norma > Epsilon<T>())
        {
            *x /= norma;
            *y /= norma;
            *z /= norma;
        }

        if (!IsValid(*x))
            (*x) = 0;

        if (!IsValid(*y))
            (*y) = 0;

        if (!IsValid(*z))
            (*z) = 0;

        return norma;
    }

    /**
	 */
	template <typename T>
	T Normalize(T *x, T *y, T *z, T *w)
	{
#if defined(_DEBUG)
        if (!IsValid(*x))
            return 0;

        if (!IsValid(*y))
            return 0;

        if (!IsValid(*z))
            return 0;

        if (!IsValid(*w))
            return 0;
#endif //#if defined(_DEBUG)

        T norma = SquareRoot((*x) * (*x) + (*y) * (*y) + (*z) * (*z) + (*w) * (*w));

		if (norma > Epsilon<T>())
		{
			*x /= norma;
			*y /= norma;
            *z /= norma;
            *w /= norma;
        }

        if (!IsValid(*x))
            (*x) = 0;

        if (!IsValid(*y))
            (*y) = 0;

        if (!IsValid(*z))
            (*z) = 0;

        if (!IsValid(*w))
            (*w) = 0;

        return norma;
	}

	/**
	 */
	template <typename T>
	T Normalize2(T *v);

	template <>
	inline double Normalize2(double *v)
	{
		double norma;

		norma = sqrt(v[0]*v[0] + v[1]*v[1]);

		if (norma > Epsilon<double>())
		{
			v[0] /= norma;
			v[1] /= norma;
		}

        if (!IsValid(v[0]))
            v[0] = 0;

        if (!IsValid(v[1]))
            v[1] = 0;

        return norma;
	}

	template <>
	inline float Normalize2(float v[2])
	{
		float norma;

		norma = sqrtf(v[0]*v[0] + v[1]*v[1]);

		if (norma > Epsilon<float>())
		{
			v[0] /= norma;
			v[1] /= norma;
		}

        if (!IsValid(v[0]))
            v[0] = 0;

        if (!IsValid(v[1]))
            v[1] = 0;

        return norma;
	}

	/**
	 */
	template <typename T>
	T Normalize3(T v[3]);

	template <>
	inline double Normalize3(double v[3])
	{
		double norma;

		norma = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);

		if (norma > Epsilon<double>())
		{
			v[0] /= norma;
			v[1] /= norma;
			v[2] /= norma;
		}

        if (!IsValid(v[0]))
            v[0] = 0;

        if (!IsValid(v[1]))
            v[1] = 0;

        if (!IsValid(v[2]))
            v[2] = 0;

        return norma;
	}

	template <>
	inline float Normalize3(float v[3])
	{
		float norma;

		norma = sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);

		if (norma > Epsilon<float>())
		{
			v[0] /= norma;
			v[1] /= norma;
			v[2] /= norma;
		}

        if (!IsValid(v[0]))
            v[0] = 0;

        if (!IsValid(v[1]))
            v[1] = 0;

        if (!IsValid(v[2]))
            v[2] = 0;

        return norma;
	}

	/**
	*/
	template <typename T>
	int NumberOfDigits(T number);

	template <>
	inline int NumberOfDigits(INT64 number)
	{
		int count = 0;

		while (number != 0)
		{
			number /= 10;

			++count;
		}

		return count;
	}

	///////////////////////////////////////////////////////////////////////////
	// P                                                                     //
	///////////////////////////////////////////////////////////////////////////

	/**
	 */
	template <typename T> 
	void PlaneCoeff3(const T A[3], const T B[3], const T C[3], T planeCoeff[4])
    {
        T AB[3],
            AC[3];

        AB[0] = B[0] - A[0];
        AB[1] = B[1] - A[1];
        AB[2] = B[2] - A[2];

        Normalize3(AB);

        AC[0] = C[0] - A[0];
        AC[1] = C[1] - A[1];
        AC[2] = C[2] - A[2];

        Normalize3(AC);

        CrossProduct3(AB, AC, planeCoeff);
        Normalize3(planeCoeff);

        planeCoeff[3] = -Multiply3(A, planeCoeff);
    }

    /**
    */
    template <typename T>
    void PlaneCoeff3(const T p[3], const T normal[3], T planeCoeff[4])
    {
        Assign3(normal, planeCoeff);
        Normalize3(planeCoeff);

        planeCoeff[3] = -Multiply3(p, planeCoeff);
    }

    /**
     * http://www.codeforge.com/read/33081/POLINT.C__html
     * Given arrays xa[1..n] and ya[1..n], and given a value x, this routine 
     * returns a value y, and an error estimate dy. If P(x) is the polynomial 
     * of degree N - 1 such that P(xa_i) = ya_i, i = 1...n, then the returned 
     * value y = P(x).
	 */
	template <typename T> 
    bool polint(const T xa[], const T ya[], int n, T x, T *y, T *dy)
    {
	    int i,
            m,
            ns = 1;
	    T den,
            dif,
            dift,
            ho,
            hp,
            w;
	    T c[32],
            d[32];

        HEALTH_CHECK(n >= 32, false);

	    dif = fabs(x - xa[1]);
	    
	    for (i=1; i<=n; i++) 
        {
		    if ((dift = fabs(x - xa[i])) < dif) 
            {
			    ns = i;
			    dif = dift;
		    }

		    c[i] = ya[i];
		    d[i] = ya[i];
	    }

	    *y = ya[ns--];

	    for (m=1; m<n; m++) 
        {
		    for (i=1; i<=(n - m); i++) 
            {
			    ho = xa[i] - x;

                //HEALTH_CHECK((i + m) > n, false);

			    hp = xa[i + m] - x;

                //HEALTH_CHECK((i + 1) > n, false);

			    w = c[i + 1] - d[i];

			    if ((den = ho - hp) == 0) 
                {
                    LOG_ERROR();

                    return false;
                }

			    den = w/den;

			    d[i] = hp*den;
			    c[i] = ho*den;
		    }

            //HEALTH_CHECK(ns < 0, false);
            //HEALTH_CHECK((ns + 1) > n, false);

		    *y += (*dy = (2*ns < (n - m) ? c[ns + 1] : d[ns--]));
	    }

        return true;
    }

	/**
    * Returns an increasing number that is not the correct angle, but 
    * that may be used for sorting angles.
    */
	template <typename T> 
    inline T PseudoAngle3(const T* e0, const T* e1, const T* v)
    {
        T cos = DotProduct3(v, e0);
        T sin = DotProduct3(v, e1);

        if (sin < -Epsilon<T>())
            return ((T)3 + cos)*(T)2*Pi<T>();
        else
            return ((T)1 - cos)*(T)2*Pi<T>();
    }

    ///////////////////////////////////////////////////////////////////////////
	// R                                                                     //
	///////////////////////////////////////////////////////////////////////////

	/**
	 */
	template <typename T> 
	T RadiansToDegrees(T radians)
	{
		return radians*T(180)/Pi<T>();
	}

    // http://www.mif.vu.lt/atpazinimas/dip/FIP/fip-Smoothin.html
    template < typename ValueType >
    bool RecursiveGaussianFilter(const ValueType *in, ValueType *out, INT64 N, ValueType stddev);

    template < >
    inline bool RecursiveGaussianFilter(const double *in, double *out, INT64 N, double stddev)
    {
        HEALTH_CHECK(!in, false);
        HEALTH_CHECK(!out, false);
        HEALTH_CHECK(stddev < 0.5, false);

        // BUG (27-Aug-2015) 
        if (N < 3)
            return false;

        double q = 3.97156 - 4.14554 * sqrt(1.0 - 0.26891 * stddev);

        if (stddev >= 2.5)
            q = 0.98711 * stddev - 0.96330;

        double b0 = 1.57825 + q * (2.44413 + q * (1.4281 + 0.422205 * q)),
            b1 = q * (2.44413 + q * (2.85619 + 1.26661 * q)),
            b2 = q * q * ((-1.4281) + (-1.26661) * q),
            b3 = 0.422205 * q * q * q;

        HEALTH_CHECK(!MyMath::IsValid(b0), false);
        HEALTH_CHECK(MyMath::IsZero(b0), false);

        double B = 1.0 - (b1 + b2 + b3) / b0;

        HEALTH_CHECK(!MyMath::IsValid(B), false);

        std::vector<double> w;

        w.reserve(N);

        double v;

        // FORWARD DIFFERENCE EQUATION

        // 0
        v = B * in[0] + (b1 * in[0] + b2 * in[0] + b3 * in[0]) / b0;
        HEALTH_CHECK(!MyMath::IsValid(v), false);
        w.push_back(v);

        // 1
        v = B * in[1] + (b1 * w[0] + b2 * w[0] + b3 * w[0]) / b0;
        HEALTH_CHECK(!MyMath::IsValid(v), false);
        w.push_back(v);

        // 2
        v = B * in[2] + (b1 * w[1] + b2 * w[0] + b3 * w[0]) / b0;
        HEALTH_CHECK(!MyMath::IsValid(v), false);
        w.push_back(v);

        for (INT64 i = 3; i < N; ++i)
        {
            v = B * in[i] + (b1 * w[i - 1] + b2 * w[i - 2] + b3 * w[i - 3]) / b0;
            HEALTH_CHECK(!MyMath::IsValid(v), false);
            w.push_back(v);
        }

        // BACKWARD DIFFERENCE EQUATION

        // N - 1
        v = B * w[N - 1] + (b1 * w[N - 1] + b2 * w[N - 1] + b3 * w[N - 1]) / b0;
        HEALTH_CHECK(!MyMath::IsValid(v), false);
        out[N - 1] = v;

        // N - 2
        v = B * w[N - 2] + (b1 * out[N - 1] + b2 * out[N - 1] + b3 * out[N - 1]) / b0;
        HEALTH_CHECK(!MyMath::IsValid(v), false);
        out[N - 2] = v;

        // N - 3
        v = B * w[N - 3] + (b1 * out[N - 2] + b2 * out[N - 1] + b3 * out[N - 1]) / b0;
        HEALTH_CHECK(!MyMath::IsValid(v), false);
        out[N - 3] = v;

        for (INT64 i = N - 4; i >= 0; --i)
        {
            v = B * w[i] + (b1 * out[i + 1] + b2 * out[i + 2] + b3 * out[i + 3]) / b0;
            HEALTH_CHECK(!MyMath::IsValid(v), false);
            out[i] = v;
        }

        return true;
    }

    template < typename ValueType >
    std::vector<ValueType> RecursiveGaussianFilter(const std::vector<ValueType>& in, ValueType stddev)
    {
        INT64 N = in.size();

        HEALTH_CHECK(N == 0, in);

        std::vector<ValueType> out(in);

        if (!RecursiveGaussianFilter(&in[0], &out[0], N, stddev))
        {
            LOG_ERROR();

            return in;
        }

        return out;
    }

    // http://www.mif.vu.lt/atpazinimas/dip/FIP/fip-Smoothin.html
    template < typename ValueType >
    bool RecursiveGaussianDerivativeFilter(const ValueType *in, ValueType *out, INT64 N, ValueType stddev);

    template < >
    inline bool RecursiveGaussianDerivativeFilter(const double *in, double *out, INT64 N, double stddev)
    {
        HEALTH_CHECK(stddev < 0.5, false);
        HEALTH_CHECK(!in, false);
        HEALTH_CHECK(!out, false);

        // (BEGIN OF) BUG: (27-Aug-2015) 
        if (N < 4)
            return false;
        // (END OF) BUG: (27-Aug-2015) 

        double q = 3.97156 - 4.14554 * sqrt(1.0 - 0.26891 * stddev);

        if (stddev >= 2.5)
            q = 0.98711 * stddev - 0.96330;

        double b0 = 1.57825 + q * (2.44413 + q * (1.4281 + 0.422205 * q)),
            b1 = q * (2.44413 + q * (2.85619 + 1.26661 * q)),
            b2 = q * q * ((-1.4281) + (-1.26661) * q),
            b3 = 0.422205 * q * q * q;

        HEALTH_CHECK(!MyMath::IsValid(b0), false);
        HEALTH_CHECK(MyMath::IsZero(b0), false);

        double B = 1.0 - (b1 + b2 + b3) / b0;

        HEALTH_CHECK(!MyMath::IsValid(B), false);

        std::vector<double> w;

        w.reserve(N);

        double v;

        // FORWARD DIFFERENCE EQUATION

        // 0
        // TESTING: (02-Feb-2016) IGNORING INVALID ELEMENTS
        v = 0.0;
        HEALTH_CHECK(!MyMath::IsValid(v), false);
        w.push_back(v);

        // 1
        // TESTING: (02-Feb-2016) IGNORING INVALID ELEMENTS
        v = (B / 2.0) * (in[2] - in[0])/* + (b1 * w[0] + b2 * w[-1] + b3 * w[-2]) / b0*/;
        HEALTH_CHECK(!MyMath::IsValid(v), false);
        w.push_back(v);

        // 2
        // TESTING: (02-Feb-2016) IGNORING INVALID ELEMENTS
        v = (B / 2.0) * (in[3] - in[1]) + (b1 * w[1] /*+ b2 * w[0] + b3 * w[-1]*/) / b0;
        HEALTH_CHECK(!MyMath::IsValid(v), false);
        w.push_back(v);

        // 3
        v = (B / 2.0) * (in[4] - in[2]) + (b1 * w[2] + b2 * w[1]/* + b3 * w[0]*/) / b0;
        HEALTH_CHECK(!MyMath::IsValid(v), false);
        w.push_back(v);

        for (INT64 i = 4; i < (N - 1); ++i)
        {
            v = (B / 2.0) * (in[i + 1] - in[i - 1]) + (b1 * w[i - 1] + b2 * w[i - 2] + b3 * w[i - 3]) / b0;
            HEALTH_CHECK(!MyMath::IsValid(v), false);
            w.push_back(v);
        }

        // N - 1
        v = /*(B / 2.0) * (inRow[(N - 1) + 1] - inRow[(N - 1) - 1]) + */(b1 * w[(N - 1) - 1] + b2 * w[(N - 1) - 2] + b3 * w[(N - 1) - 3]) / b0;
        HEALTH_CHECK(!MyMath::IsValid(v), false);
        w.push_back(v);

        // BACKWARD DIFFERENCE EQUATION

        // N - 1
        v = B * w[N - 1] + (b1 * w[N - 1] + b2 * w[N - 1] + b3 * w[N - 1]) / b0;
        HEALTH_CHECK(!MyMath::IsValid(v), false);
        out[N - 1] = v;

        // N - 2
        v = B * w[N - 2] + (b1 * out[N - 1] + b2 * out[N - 1] + b3 * out[N - 1]) / b0;
        HEALTH_CHECK(!MyMath::IsValid(v), false);
        out[N - 2] = v;

        // N - 3
        v = B * w[N - 3] + (b1 * out[N - 2] + b2 * out[N - 1] + b3 * out[N - 1]) / b0;
        HEALTH_CHECK(!MyMath::IsValid(v), false);
        out[N - 3] = v;

        for (INT64 i = N - 4; i > 0; --i)
        {
            v = B * w[i] + (b1 * out[i + 1] + b2 * out[i + 2] + b3 * out[i + 3]) / b0;
            HEALTH_CHECK(!MyMath::IsValid(v), false);
            out[i] = v;
        }

        // BUG: (02-Feb-2016) W[0] IS NOT VALID!
        v = /*B * w[0] + */(b1 * out[/*0 + */1] + b2 * out[/*0 + */2] + b3 * out[/*0 + */3]) / b0;
        HEALTH_CHECK(!MyMath::IsValid(v), false);
        out[0] = v;

        return true;
    }

    template < typename ValueType >
    std::vector<ValueType> RecursiveGaussianDerivativeFilter(const std::vector<ValueType>& in, ValueType stddev)
    {
        INT64 N = in.size();

        HEALTH_CHECK(N == 0, in);

        std::vector<ValueType> out(in);

        if (!RecursiveGaussianDerivativeFilter(&in[0], &out[0], N, stddev))
        {
            LOG_ERROR();

            return in;
        }

        return out;
    }

    ///////////////////////////////////////////////////////////////////////////
	// S                                                                     //
	///////////////////////////////////////////////////////////////////////////

    // For a plane spanned by (normal, point) and a segment defined between (s0, s1):
    // 0 = disjoint (no intersection).
    // 1 = unique intersection.
    // 2 = the  segment lies in the plane.
    template <typename T>
    int SegmentPlaneIntersection3(const T *s0, const T *s1, const T *normal, const T *point, T *intersection)
    {
        HEALTH_CHECK(!s0, 0);
        HEALTH_CHECK(!s1, 0);
        HEALTH_CHECK(!normal, 0);
        HEALTH_CHECK(!point, 0);
        HEALTH_CHECK(!intersection, 0);

        T u[3] = { 0 },
            w[3] = { 0 };

        MyMath::Subtract3(s1, s0, u);
        MyMath::Subtract3(s0, point, w);

        T D = DotProduct3(normal, u),
            N = -DotProduct3(normal, w);

        // SEGMENT PARALLEL TO PLANE
        if (IsZero(D))
        {
            // SEGMENT LIES ON THE PLANE
            if (IsZero(N))
                return 2;
            // NO INTERSECTION
            else
                return 0;
        }

        T t = N / D;

        if ((t < 0) || 
            (t > 1))
        {
            return 0;
        }

        intersection[0] = s0[0] + t * u[0];
        intersection[1] = s0[1] + t * u[1];
        intersection[2] = s0[2] + t * u[2];

        return 1;
    }

    /*
    Determine the intersection point of two line segments. Return false 
    if the lines don't intersect.
    http://paulbourke.net/geometry/pointlineplane/pdb.c
    */
	template <typename T>
    bool SegmentSegmentIntersection(T x1, T y1, T x2, T y2, T x3, T y3, T x4, T y4, T *x, T *y)
    {
        T mua,
            mub;
        T denom,
            numera,
            numerb;

        denom  = (y4-y3) * (x2-x1) - (x4-x3) * (y2-y1);
        numera = (x4-x3) * (y1-y3) - (y4-y3) * (x1-x3);
        numerb = (x2-x1) * (y1-y3) - (y2-y1) * (x1-x3);

        /* Are the line coincident? */
        if (fabs(numera) < Epsilon<T>() && 
            fabs(numerb) < Epsilon<T>() && 
            fabs(denom) < Epsilon<T>()) 
        {
            *x = (x1 + x2) / 2;
            *y = (y1 + y2) / 2;

            return true;
        }

        /* Are the line parallel */
        if (fabs(denom) < Epsilon<T>()) 
        {
            *x = 0;
            *y = 0;

            return false;
        }

        /* Is the intersection along the the segments */
        mua = numera / denom;
        mub = numerb / denom;

        if (mua < 0 || 
            mua > 1 || 
            mub < 0 || 
            mub > 1) 
        {
            *x = 0;
            *y = 0;

            return false;
        }

        *x = x1 + mua * (x2 - x1);
        *y = y1 + mua * (y2 - y1);
        
        return true;
    }

    /**
    http://paulbourke.net/geometry/pointlineplane/

    Calculate the line segment PaPb that is the shortest route between 
    two lines P1P2 and P3P4. Calculate also the values of mua and mub 
    where
    
    Pa = P1 + mua (P2 - P1)
    Pb = P3 + mub (P4 - P3)
    
    Return FALSE if no solution exists.
    */
	template <typename T>
    bool SegmentSegmentIntersection3(const T *p1, const T *p2, const T *p3, const T *p4, T *pa, T *pb, T *mua, T *mub);

	template <>
    inline bool SegmentSegmentIntersection3(const double *p1, const double *p2, const double *p3, const double *p4, double *pa, double *pb, double *mua, double *mub)
    {
        double p13[3] = {0},
            p43[3] = {0},
            p21[3] = {0};
        double d1343 = 0,
            d4321 = 0,
            d1321 = 0,
            d4343 = 0,
            d2121 = 0;
        double numer = 0,
            denom = 0;

        p13[0] = p1[0] - p3[0];
        p13[1] = p1[1] - p3[1];
        p13[2] = p1[2] - p3[2];
        p43[0] = p4[0] - p3[0];
        p43[1] = p4[1] - p3[1];
        p43[2] = p4[2] - p3[2];

        if ((fabs(p43[0]) < Epsilon<double>()) && 
            (fabs(p43[1]) < Epsilon<double>()) && 
            (fabs(p43[2]) < Epsilon<double>()))
        {
            LOG_ERROR();

            return false;
        }

        p21[0] = p2[0] - p1[0];
        p21[1] = p2[1] - p1[1];
        p21[2] = p2[2] - p1[2];

        if ((fabs(p21[0]) < Epsilon<double>()) && 
            (fabs(p21[1]) < Epsilon<double>()) && 
            (fabs(p21[2]) < Epsilon<double>()))
        {
            LOG_ERROR();

            return false;
        }

        d1343 = p13[0]*p43[0] + p13[1]*p43[1] + p13[2]*p43[2];
        d4321 = p43[0]*p21[0] + p43[1]*p21[1] + p43[2]*p21[2];
        d1321 = p13[0]*p21[0] + p13[1]*p21[1] + p13[2]*p21[2];
        d4343 = p43[0]*p43[0] + p43[1]*p43[1] + p43[2]*p43[2];
        d2121 = p21[0]*p21[0] + p21[1]*p21[1] + p21[2]*p21[2];

        denom = d2121 * d4343 - d4321 * d4321;
        
        if (fabs(denom) < Epsilon<double>())
        {
            LOG_ERROR();

            return false;
        }

        numer = d1343*d4321 - d1321*d4343;

        *mua = numer/denom;
        *mub = (d1343 + d4321*(*mua))/d4343;

        pa[0] = p1[0] + (*mua)*p21[0];
        pa[1] = p1[1] + (*mua)*p21[1];
        pa[2] = p1[2] + (*mua)*p21[2];
        pb[0] = p3[0] + (*mub)*p43[0];
        pb[1] = p3[1] + (*mub)*p43[1];
        pb[2] = p3[2] + (*mub)*p43[2];

        return true;
    }

    // http://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
    template <typename T>
    inline int Sign(T x, std::false_type is_signed) {
        return T(0) < x;
    }

    template <typename T>
    inline int Sign(T x, std::true_type /*is_signed*/) {
        return (T(0) < x) - (x < T(0));
    }

    template <typename T>
    inline int Sign(T x) {
        return Sign(x, std::is_signed<T>());
    }

	/**
	 */
	template <typename T>
	T Sine(T angle);

	template <>
	inline double Sine(double angle)
	{
        return sin(angle);
	}

	template <>
	inline float Sine(float angle)
	{
        return sinf(angle);
	}

    /**
    */
    template <typename T>
    T SquaredDistance(T ax, T ay, T bx, T by)
    {
        T dx = bx - ax,
            dy = by - ay;

        return dx*dx + dy*dy;
    }

    /**
    */
    template <typename T>
    T SquaredDistance2(const T *a, const T *b)
    {
        HEALTH_CHECK(!a, 0);
        HEALTH_CHECK(!b, 0);

        T dx = b[0] - a[0],
            dy = b[1] - a[1];

        return dx*dx + dy*dy;
    }

    /**
	*/
	template <typename T>
	T SquaredEuclideanDistance2(const T *a, const T *b)
	{
		HEALTH_CHECK(!a, 0);
		HEALTH_CHECK(!b, 0);

		T dx = b[0] - a[0],
			dy = b[1] - a[1];

		return dx*dx + dy*dy;
	}

	/**
	 */
	template <typename T>
	T SquaredMagnitude(T x, T y);

	template <>
	inline double SquaredMagnitude(double x, double y)
	{
		return x*x + y*y;
	}

	template <>
	inline float SquaredMagnitude(float x, float y)
	{
		return x*x + y*y;
	}

	/**
	 */
	template <typename T>
	T SquaredMagnitude(T x, T y, T z);

	template <>
	inline double SquaredMagnitude(double x, double y, double z)
	{
		return x*x + y*y + z*z;
	}

	template <>
	inline float SquaredMagnitude(float x, float y, float z)
	{
		return x*x + y*y + z*z;
	}

	/**
	 */
	template <typename T>
	T SquaredMagnitude2(T v[2]);

	template <>
	inline double SquaredMagnitude2(double v[2])
	{
		return v[0]*v[0] + v[1]*v[1];
	}

	template <>
	inline float SquaredMagnitude2(float v[2])
	{
		return v[0]*v[0] + v[1]*v[1];
	}

	template <typename T>
	T SquaredPointToSegmentDistance(T pointX, T pointY, T segmentStartX, T segmentStartY, T segmentEndX, T segmentEndY)
	{
        T difference[2] = {0},
			direction[2] = {0},
			t = 0;
	  
		difference[0] = pointX - segmentStartX;
		difference[1] = pointY - segmentStartY;

		direction[0] = segmentEndX - segmentStartX;
		direction[1] = segmentEndY - segmentStartY;

		Normalize<T>(direction, direction + 1);
	  
		t = direction[0]*difference[0] + direction[1]*difference[1];

		difference[0] = segmentStartX + t*direction[0];
		difference[1] = segmentStartY + t*direction[1];

		direction[0] = pointX - difference[0];
		direction[1] = pointY - difference[1];

		return direction[0]*direction[0] + direction[1]*direction[1];
	}

    /**
    */
	template <typename T>
	T SquaredPointToSegmentDistance2(const T *point, const T *segmentStart, const T *segmentEnd)
	{
        T difference[2] = {0},
			direction[2] = {0},
			t = 0;

        HEALTH_CHECK(!point, 0);
        HEALTH_CHECK(!segmentStart, 0);
        HEALTH_CHECK(!segmentEnd, 0);
	  
		difference[0] = point[0] - segmentStart[0];
		difference[1] = point[1] - segmentStart[1];

		direction[0] = segmentEnd[0] - segmentStart[0];
		direction[1] = segmentEnd[1] - segmentStart[1];

		Normalize2<T>(direction);
	  
		t = direction[0]*difference[0] + direction[1]*difference[1];

		difference[0] = segmentStart[0] + t*direction[0];
		difference[1] = segmentStart[1] + t*direction[1];

		direction[0] = point[0] - difference[0];
		direction[1] = point[1] - difference[1];

		return direction[0]*direction[0] + direction[1]*direction[1];
	}

    /**
    A better (?) way to compute areas of bad triangles, from 
    "Miscalculating Area and Angles of a Needle-like Triangle".
    */
    template <typename T>
    T TriangleArea(const T *A, const T *B, const T *C)
    {
	    T a = 0,
		    b = 0,
		    c = 0;
	    T area = 0;
		
	    a = SquareRoot((C[0] - B[0])*(C[0] - B[0]) + (C[1] - B[1])*(C[1] - B[1]) + (C[2] - B[2])*(C[2] - B[2]));
	    b = SquareRoot((A[0] - C[0])*(A[0] - C[0]) + (A[1] - C[1])*(A[1] - C[1]) + (A[2] - C[2])*(A[2] - C[2]));
	    c = SquareRoot((B[0] - A[0])*(B[0] - A[0]) + (B[1] - A[1])*(B[1] - A[1]) + (B[2] - A[2])*(B[2] - A[2]));

        if (a < b) 
		    std::swap(a, b);

        if (a < c) 
		    std::swap(a, c);

        if (b < c) 
		    std::swap(b, c);

	    area = (T)0.25*SquareRoot((a + (b + c))*(c - (a - b))*(c + (a - b))*(a + (b - c)));

	    return area;
    }

    /**
    */
    template <typename T>
    bool TriangleNormal(const T *A, const T *B, const T *C, T *normal, T tolerance);

    template <>
    inline bool TriangleNormal(const double *A, const double *B, const double *C, double *normal, double tolerance)
    {
	    HEALTH_CHECK(!A, 0);
	    HEALTH_CHECK(!B, 0);
	    HEALTH_CHECK(!C, 0);
	    HEALTH_CHECK(!normal, 0);

	    double AB[3] = { 0 },
		    AC[3] = { 0 };

	    AB[0] = B[0] - A[0];
	    AB[1] = B[1] - A[1];
	    AB[2] = B[2] - A[2];

	    if (MyMath::SquaredMagnitude3(AB) < tolerance)
		    return false;

	    MyMath::Normalize3(AB);

	    AC[0] = C[0] - A[0];
	    AC[1] = C[1] - A[1];
	    AC[2] = C[2] - A[2];

	    if (MyMath::SquaredMagnitude3(AC) < tolerance)
		    return 0;

	    MyMath::Normalize3(AC);

	    MyMath::CrossProduct3(AB, AC, normal);

        MyMath::Normalize3(normal);

	    return true;
    }

    template <typename T>
    bool TriangleSegmentIntersection3(const T A[3], const T B[3], const T C[3], const T s0[3], const T s1[3], T tolerance, T lambda[3])
    {
        T segmentCenter[3],
            // Segment axis.
            segmentDirection[3],
            segmentHalfLength,
            offsetToOrigin[3];
        T AB[3],
            AC[3],
            triangleNormal[3];
        T DdN,
            sign;
        T DdQxE2,
            DdE1xQ;
        T offsetToOriginCrossAC[3],
            ABCrossoffsetToOrigin[3];
        T QdN,
            extDdN;
        T inv;

        // Precondition (|s1 - s0| > 0).
        if ((T)0.5*MyMath::Magnitude(s0[0] - s1[0], s0[1] - s1[1], s0[2] - s1[2]) < tolerance)
            return false;

        // Precondition (|A - B| > 0).
        if (MyMath::Magnitude(A[0] - B[0], A[1] - B[1], A[2] - B[2]) < tolerance)
            return false;
        // Precondition (|B - C| > 0).
        if (MyMath::Magnitude(B[0] - C[0], B[1] - C[1], B[2] - C[2]) < tolerance)
            return false;
        // Precondition (|C - A| > 0).
        if (MyMath::Magnitude(C[0] - A[0], C[1] - A[1], C[2] - A[2]) < tolerance)
            return false;

        segmentCenter[0] = (T)0.5*(s0[0] + s1[0]);
        segmentCenter[1] = (T)0.5*(s0[1] + s1[1]);
        segmentCenter[2] = (T)0.5*(s0[2] + s1[2]);

        MyMath::Subtract3(s1, s0, segmentDirection);
    
        segmentHalfLength = (T)0.5*MyMath::Magnitude3(segmentDirection);

        MyMath::Subtract3(segmentCenter, A, offsetToOrigin);
        MyMath::Subtract3(B, A, AB);
        MyMath::Subtract3(C, A, AC);

        MyMath::CrossProduct3(AB, AC, triangleNormal);

        HEALTH_CHECK(MyMath::Magnitude3(triangleNormal) < tolerance, false);

        // Solve Q + t*D = b1*E1 + b2*E2 (Q = offsetToOrigin, D = segment direction,
        // E1 = AB, E2 = AC, N = Cross(E1,E2)) by
        //   |Dot(D,N)|*b1 = sign(Dot(D,N))*Dot(D,Cross(Q,E2))
        //   |Dot(D,N)|*b2 = sign(Dot(D,N))*Dot(D,Cross(E1,Q))
        //   |Dot(D,N)|*t = -sign(Dot(D,N))*Dot(Q,N)
        DdN = MyMath::DotProduct3(segmentDirection, triangleNormal);

        if (DdN > MyMath::Epsilon<T>())
        {
            sign = (T)1;
        }
        else if (DdN < -MyMath::Epsilon<T>())
        {
            sign = (T)-1;
            DdN = -DdN;
        }
        else
        {
            // Segment and triangle are parallel, call it a "no intersection"
            // even if the segment does intersect.
            return false;
        }

        MyMath::CrossProduct3(offsetToOrigin, AC, offsetToOriginCrossAC);

        DdQxE2 = sign*MyMath::DotProduct3(segmentDirection, offsetToOriginCrossAC);

        if (DdQxE2 >= 0)
        {
            MyMath::CrossProduct3(AB, offsetToOrigin, ABCrossoffsetToOrigin);

            DdE1xQ = sign*MyMath::DotProduct3(segmentDirection, ABCrossoffsetToOrigin);

            if (DdE1xQ >= (T)0)
            {
                if (DdQxE2 + DdE1xQ <= DdN)
                {
                    // Line intersects triangle, check if segment does.
                    QdN = -sign*MyMath::DotProduct3(offsetToOrigin, triangleNormal);
                
                    extDdN = segmentHalfLength*DdN;

                    if ((-extDdN <= QdN) && 
                        (QdN <= extDdN))
                    {
                        if (DdN < tolerance)
                            return false;

                        // Segment intersects triangle.
                        inv = ((T)1)/DdN;

                        lambda[1] = DdQxE2*inv;
                        lambda[2] = DdE1xQ*inv;
                        lambda[0] = (T)1 - lambda[1] - lambda[2];
                 
                        return true;
                    }
                    // else: |t| > extent, no intersection
                }
                // else: b1+b2 > 1, no intersection
            }
            // else: b2 < 0, no intersection
        }
        // else: b1 < 0, no intersection

        return false;
    }

    /**
    Given in radians.
    */
    template <typename T>
    T DihedralAngle(const T *_1stA, const T *_1stB, const T *_1stC, const T *_2ndA, const T *_2ndB, const T *_2ndC, T tolerance);

    template <>
    inline double DihedralAngle(const double *_1stA, const double *_1stB, const double *_1stC, const double *_2ndA, const double *_2ndB, const double *_2ndC, double tolerance)
    {
	    HEALTH_CHECK(!_1stA, 0);
	    HEALTH_CHECK(!_1stB, 0);
	    HEALTH_CHECK(!_1stC, 0);
	    HEALTH_CHECK(!_2ndA, 0);
	    HEALTH_CHECK(!_2ndB, 0);
	    HEALTH_CHECK(!_2ndC, 0);

	    double _1stNormal[3] = { 0 },
		    _2ndNormal[3] = { 0 };

	    if (!TriangleNormal(_1stA, _1stB, _1stC, _1stNormal, tolerance))
		    return 0;

	    if (!TriangleNormal(_2ndA, _2ndB, _2ndC, _2ndNormal, tolerance))
		    return 0;

	    return AngleBetweenVectors3(_1stNormal, _2ndNormal);
    }

    // TESTING: (11-Mar-2015)
    template <typename T>
    const T *X()
    {
        static T x[3] = { 1, 0, 0 };

        return x;
    }

    // TESTING: (11-Mar-2015)
    template <typename T>
    const T *Y()
    {
        static T y[3] = { 0, 1, 0 };

        return y;
    }

    // TESTING: (11-Mar-2015)
    template <typename T>
    const T *Z()
    {
        static T z[3] = { 0, 0, 1 };

        return z;
    }
}

#endif //#if !defined(MY_MATH_INCLUDED)

