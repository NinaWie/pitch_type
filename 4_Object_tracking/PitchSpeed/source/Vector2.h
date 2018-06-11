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

#if !defined(VECTOR_2_INCLUDED)
#define VECTOR_2_INCLUDED

#include "MyMath.h"

namespace my {
    template <typename T>
    class CVector2
    {
    public:
        typedef T ValueType;

        CVector2()
            : m_x(my::Null<T>()),
            m_y(my::Null<T>()),
            m_invalidValue(my::Null<T>())
        {
        }

        CVector2(const CVector2& vector2)
            : m_x(vector2.m_x),
            m_y(vector2.m_y),
            m_invalidValue(my::Null<T>())
        {
        }

        CVector2(T value)
            : m_x(value),
            m_y(value),
            m_invalidValue(my::Null<T>())
        {
        }

        CVector2(T x, T y)
            : m_x(x),
            m_y(y),
            m_invalidValue(my::Null<T>())
        {
        }

        void operator=(const CVector2& vector2)
        {
            m_x = vector2.m_x;
            m_y = vector2.m_y;
            m_invalidValue = my::Null<T>();
        }

        template <typename T2>
        void operator=(const CVector2<T2>& vector2)
        {
#if defined(_DEBUG)
            if (!vector2.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            m_x = (T)vector2.m_x();
            m_y = (T)vector2.m_y();
            m_invalidValue = my::Null<T>();
        }

        // UNARY NEGATION OPERATOR
        CVector2 operator-() const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return CVector2(-m_x, -m_y);
        }

        CVector2 operator+(const CVector2& vector2) const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

#if defined(_DEBUG)
            if (!vector2.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return CVector2(m_x + vector2.m_x, m_y + vector2.m_y);
        }

        CVector2& operator+=(const CVector2& vector2)
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

#if defined(_DEBUG)
            if (!vector2.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            m_x += vector2.m_x;
            m_y += vector2.m_y;

            return (*this);
        }

        CVector2 operator-(const CVector2& vector2) const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

#if defined(_DEBUG)
            if (!vector2.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return CVector2(m_x - vector2.m_x, m_y - vector2.m_y);
        }

        CVector2& operator-=(const CVector2& vector2)
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

#if defined(_DEBUG)
            if (!vector2.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            m_x -= vector2.m_x;
            m_y -= vector2.m_y;

            return (*this);
        }

        CVector2 operator*(const T scale) const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

#if defined(_DEBUG)
            if (!MyMath::IsValid(scale))
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return CVector2(m_x * scale, m_y * scale);
        }

        CVector2 operator*(const CVector2& vector2) const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

#if defined(_DEBUG)
            if (!vector2.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return CVector2(m_x * vector2.m_x, m_y * vector2.m_y);
        }

        CVector2& operator*=(ValueType scale)
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

#if defined(_DEBUG)
            if (!MyMath::IsValid(scale))
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            m_x *= scale;
            m_y *= scale;

            return (*this);
        }

        CVector2& operator*=(const CVector2& vector2)
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

#if defined(_DEBUG)
            if (!vector2.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            m_x *= vector2.m_x;
            m_y *= vector2.m_y;

            return (*this);
        }

        CVector2 operator/(const T scale) const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

#if defined(_DEBUG)
            if (!MyMath::IsValid(scale))
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            if (MyMath::IsZero(scale))
                LOG_ERROR();

            return CVector2(m_x / scale, m_y / scale);
        }

        CVector2& operator/=(const T scale)
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

#if defined(_DEBUG)
            if (!MyMath::IsValid(scale))
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            if (MyMath::IsZero(scale))
                LOG_ERROR();

            m_x /= scale;
            m_y /= scale;

            return (*this);
        }

        bool operator==(const CVector2& vector2) const
        {
            return (m_x == vector2.m_x) &&
                (m_y == vector2.m_y);
        }

        bool operator!=(const CVector2& vector2) const
        {
            return (m_x != vector2.m_x) ||
                (m_y != vector2.m_y);
        }

        T operator[](int index) const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            if ((index < 0) ||
                (index > 1))
            {
                LOG_ERROR();

                return my::Null<T>();
            }

            return (&m_x)[index];
        }

        T& operator[](int index)
        {
            if ((index < 0) ||
                (index > 1))
            {
                LOG_ERROR();

                return m_invalidValue;
            }

            return (&m_x)[index];
        }

        T x() const
        {
#if defined(_DEBUG)
            if (!MyMath::IsValid(m_x))
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return m_x;
        }

        T& x()
        {
            return m_x;
        }

        T y() const
        {
#if defined(_DEBUG)
            if (!MyMath::IsValid(m_y))
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return m_y;
        }

        T& y()
        {
            return m_y;
        }

        CVector2<T> yx() const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return CVector2<T>(m_y, m_x);
        }

        void Set(T x, T y)
        {
            m_x = x;
            m_y = y;
        }

        void Set(const CVector2& vector2)
        {
            m_x = vector2.m_x;
            m_y = vector2.m_y;
        }

        void Clamp(T minimum, T maximum)
        {
#if defined(_DEBUG)
            if (!vector2.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

#if defined(_DEBUG)
            if (!MyMath::IsValid(minimum) ||
                !MyMath::IsValid(maximum))
            {
                LOG_ERROR();
            }
#endif //#if defined(_DEBUG)

            MyMath::Clamp(m_x, m_y, minimum, maximum);
        }

        T Length() const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return MyMath::Magnitude(m_x, m_y);
        }

        T Distance(const CVector2& vector2) const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

#if defined(_DEBUG)
            if (!vector2.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return MyMath::EuclideanDistance(m_x, m_y, vector2.m_x, vector2.m_y);
        }

        T Distance(T x, T y) const
        {
#if defined(_DEBUG)
            if (!MyMath::IsValid(x) ||
                !MyMath::IsValid(y))
            {
                LOG_ERROR();
            }
#endif //#if defined(_DEBUG)

#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return MyMath::EuclideanDistance(m_x, m_y, x, y);
        }

        T DistanceToSegment(const CVector2& s0, const CVector2& s1) const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

#if defined(_DEBUG)
            if (!s0.IsValid())
                LOG_ERROR();

            if (!s1.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            T pPointer[2] = { m_x, m_y },
                s0Pointer[2] = { s0.m_x, s0.m_y },
                s1Pointer[2] = { s1.m_x, s1.m_y };
            
            return MyMath::distance::PointToSegment2(pPointer, s0Pointer, s1Pointer);
        }

        CVector2<T>& Normalize(T epsilon = MyMath::Epsilon<T>())
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            T magnitude = MyMath::Magnitude(m_x, m_y);

            if (magnitude > epsilon)
            {
                m_x /= magnitude;
                m_y /= magnitude;
            }
            else
            {
                m_x = 0;
                m_y = 0;
            }

            return (*this);
        }

        T Dot(const CVector2& vector2) const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

#if defined(_DEBUG)
            if (!vector2.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return MyMath::DotProduct(m_x, m_y, vector2.m_x, vector2.m_y);
        }

        bool Equals(const CVector2& vector2, T epsilon = MyMath::Epsilon<T>()) const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

#if defined(_DEBUG)
            if (!vector2.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return MyMath::IsClose(m_x, vector2.m_x, epsilon) &&
                MyMath::IsClose(m_y, vector2.m_y, epsilon);
        }

        bool IsValid() const
        {
            return true;
        }

    protected:
        T m_x;
        T m_y;

        T m_invalidValue;
    };

    template <>
    inline bool CVector2<double>::IsValid() const
    {
        return MyMath::IsValid(m_x) &&
            MyMath::IsValid(m_y);
    }

    template <>
    inline bool CVector2<float>::IsValid() const
    {
        return MyMath::IsValid(m_x) &&
            MyMath::IsValid(m_y);
    }
}; // my

#endif // #if !defined(VECTOR_2_INCLUDED)

