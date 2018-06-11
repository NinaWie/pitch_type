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

#if !defined(VECTOR_3_INCLUDED)
#define VECTOR_3_INCLUDED

#include "MyMath.h"
#include "Vector2.h"

namespace my {
    template <typename T>
    class CVector3
    {
    public:
        typedef T ValueType;

        CVector3() 
        : m_x(my::Null<T>()),
        m_y(my::Null<T>()),
        m_z(my::Null<T>()),
        m_invalidValue(my::Null<T>())
        {
        }

        CVector3(const CVector3& vector3)
            : m_x(vector3.m_x),
            m_y(vector3.m_y),
            m_z(vector3.m_z),
            m_invalidValue(my::Null<T>())
        {
        }

        CVector3(T value)
            : m_x(value),
            m_y(value),
            m_z(value),
            m_invalidValue(my::Null<T>())
        {
        }

        CVector3(T x, T y, T z)
            : m_x(x),
            m_y(y),
            m_z(z),
            m_invalidValue(my::Null<T>())
        {
        }

        template <typename T2>
        CVector3(const CVector2<T2>& vector2, T z)
            : m_x(vector2.x()),
            m_y(vector2.y()),
            m_z(z),
            m_invalidValue(my::Null<T>())
        {
        }

        void operator=(const CVector3& vector3)
        {
            m_x = vector3.m_x;
            m_y = vector3.m_y;
            m_z = vector3.m_z;
            m_invalidValue = my::Null<T>();
        }

        template <typename T2>
        void operator=(const CVector3<T2>& vector3)
        {
#if defined(_DEBUG)
            if (!vector3.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            m_x = (T)vector3.m_x();
            m_y = (T)vector3.m_y();
            m_z = (T)vector3.m_z();
            m_invalidValue = my::Null<T>();
        }

        // UNARY NEGATION OPERATOR
        CVector3 operator-() const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return CVector3(-m_x, -m_y, -m_z);
        }

        CVector3 operator+(const CVector3& vector3) const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

#if defined(_DEBUG)
            if (!vector3.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)
            
            return CVector3(m_x + vector3.m_x, m_y + vector3.m_y, m_z + vector3.m_z);
        }

        CVector3& operator+=(const CVector3& vector3)
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)
            
#if defined(_DEBUG)
            if (!vector3.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            m_x += vector3.m_x;
            m_y += vector3.m_y;
            m_z += vector3.m_z;

            return (*this);
        }

        CVector3 operator-(const CVector3& vector3) const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)
            
#if defined(_DEBUG)
            if (!vector3.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return CVector3(m_x - vector3.m_x, m_y - vector3.m_y, m_z - vector3.m_z);
        }

        CVector3& operator-=(const CVector3& vector3)
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)
            
#if defined(_DEBUG)
            if (!vector3.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            m_x -= vector3.m_x;
            m_y -= vector3.m_y;
            m_z -= vector3.m_z;

            return (*this);
        }

        CVector3 operator*(const T scale) const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)
            
#if defined(_DEBUG)
            if (!MyMath::IsValid(scale))
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return CVector3(m_x * scale, m_y * scale, m_z * scale);
        }

        CVector3 operator*(const CVector3& vector3) const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)
            
#if defined(_DEBUG)
            if (!vector3.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return CVector3(m_x * vector3.m_x, m_y * vector3.m_y, m_z * vector3.m_z);
        }

        CVector3& operator*=(ValueType scale)
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
            m_z *= scale;

            return (*this);
        }

        CVector3& operator*=(const CVector3& vector3)
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)
            
#if defined(_DEBUG)
            if (!vector3.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            m_x *= vector3.m_x;
            m_y *= vector3.m_y;
            m_z *= vector3.m_z;

            return (*this);
        }

        CVector3 operator/(const T scale) const
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

            return CVector3(m_x / scale, m_y / scale, m_z / scale);
        }

        CVector3& operator/=(const T scale)
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
            m_z /= scale;

            return (*this);
        }

        bool operator==(const CVector3& vector3) const
        {
            return (m_x == vector3.m_x) &&
                (m_y == vector3.m_y) && 
                (m_z == vector3.m_z);
        }

        bool operator!=(const CVector3& vector3) const
        {
            return (m_x != vector3.m_x) ||
                (m_y != vector3.m_y) ||
                (m_z != vector3.m_z);
        }

        T operator[](int index) const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            if ((index < 0) ||
                (index > 2))
            {
                LOG_ERROR();

                return my::Null<T>();
            }

            return (&m_x)[index];
        }

        T& operator[](int index)
        {
            if ((index < 0) ||
                (index > 2))
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

        T z() const
        {
#if defined(_DEBUG)
            if (!MyMath::IsValid(m_z))
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return m_z;
        }

        T& z()
        {
            return m_z;
        }

        CVector2<T> xy() const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return CVector2<T>(m_x, m_y);
        }

        CVector2<T> xz() const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return CVector2<T>(m_x, m_z);
        }

        CVector2<T> yx() const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return CVector2<T>(m_y, m_x);
        }

        CVector2<T> yz() const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return CVector2<T>(m_y, m_z);
        }

        CVector3<T> xy0() const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return CVector3<T>(m_x, m_y, 0.0);
        }

        CVector3<T> x0z() const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return CVector3<T>(m_x, 0.0, m_z);
        }

        void Set(T x, T y, T z)
        {
            m_x = x;
            m_y = y;
            m_z = z;
        }

        void Set(const CVector3& vector3)
        {
            m_x = vector3.m_x;
            m_y = vector3.m_y;
            m_z = vector3.m_z;
        }

        void Clamp(T minimum, T maximum)
        {
#if defined(_DEBUG)
            if (!vector3.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

#if defined(_DEBUG)
            if (!MyMath::IsValid(minimum) ||
                !MyMath::IsValid(maximum))
            {
                LOG_ERROR();
            }
#endif //#if defined(_DEBUG)

            MyMath::Clamp(m_x, m_y, m_z, minimum, maximum);
        }

        T Length() const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return MyMath::Magnitude(m_x, m_y, m_z);
        }

        T Distance(const CVector3& vector3) const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)
            
#if defined(_DEBUG)
            if (!vector3.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return MyMath::EuclideanDistance(m_x, m_y, m_z, vector3.m_x, vector3.m_y, vector3.m_z);
        }

        T Distance(T x, T y, T z) const
        {
#if defined(_DEBUG)
            if (!MyMath::IsValid(x) ||
                !MyMath::IsValid(y) ||
                !MyMath::IsValid(z))
            {
                LOG_ERROR();
            }
#endif //#if defined(_DEBUG)

#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return MyMath::EuclideanDistance(m_x, m_y, m_z, x, y, z);
        }

        T DistanceToSegment(const CVector3& s0, const CVector3& s1) const
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

            T pPointer[3] = { m_x, m_y, m_z },
                s0Pointer[3] = { s0.m_x, s0.m_y, s0.m_z },
                s1Pointer[3] = { s1.m_x, s1.m_y, s1.m_z };

            return MyMath::distance::PointToSegment3(pPointer, s0Pointer, s1Pointer);
        }

        CVector3<T>& Normalize(T epsilon = MyMath::Epsilon<T>())
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            T magnitude = MyMath::Magnitude(m_x, m_y, m_z);

            if (magnitude > epsilon)
            {
                m_x /= magnitude;
                m_y /= magnitude;
                m_z /= magnitude;
            }
            else
            {
                m_x = 0;
                m_y = 0;
                m_z = 0;
            }

            return (*this);
        }

        CVector3<T> Normalize(T epsilon = MyMath::Epsilon<T>()) const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            T magnitude = MyMath::Magnitude(m_x, m_y, m_z);

            CVector3<T> vector3;

            if (magnitude > epsilon)
                vector3 /= magnitude;
            else
                vector3.Set(0);

            return vector3;
        }

        T Dot(const CVector3& vector3) const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)
            
#if defined(_DEBUG)
            if (!vector3.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return MyMath::DotProduct(m_x, m_y, m_z, vector3.m_x, vector3.m_y, vector3.m_z);
        }

        CVector3 Cross(const CVector3& vector3) const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)
            
#if defined(_DEBUG)
            if (!vector3.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return CVector3(m_y * vector3.m_z - m_z * vector3.m_y, m_z * vector3.m_x - m_x * vector3.m_z, m_x * vector3.m_y - m_y * vector3.m_x);
        }

        CVector3& Cross(const CVector3& thumb, const CVector3& index)
        {
#if defined(_DEBUG)
            if (!thumb.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)
            
#if defined(_DEBUG)
            if (!index.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            m_y = thumb.m_y * index.m_z - thumb.m_z * index.m_y;
            m_z = thumb.m_z * index.m_x - thumb.m_x * index.m_z;
            m_x = thumb.m_x * index.m_y - thumb.m_y * index.m_x;

            return (*this);
        }

        // DEGREES
        T Angle(const CVector3& vector3) const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            T uvDot = Dot(vector3),
                uNorm = Length(),
                vNorm = vector3.Length();

            if (MyMath::IsZero(uNorm))
            {
                LOG_ERROR();

                return 0;
            }

            if (MyMath::IsZero(vNorm))
            {
                LOG_ERROR();

                return 0;
            }

            T angleCos = uvDot / uNorm / vNorm,
                angle = 0;

            // Is that possible?
            if (angleCos <= -1)
                angle = 180;
            // Is that possible?
            else if (1 <= angleCos)
                angle = 0;
            else
                angle = MyMath::RadiansToDegrees(acos(angleCos));

            return angle;
        }

        bool Equals(const CVector3& vector3, T epsilon = MyMath::Epsilon<T>()) const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

#if defined(_DEBUG)
            if (!vector3.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return MyMath::IsClose(m_x, vector3.m_x, epsilon) &&
                MyMath::IsClose(m_y, vector3.m_y, epsilon) &&
                MyMath::IsClose(m_z, vector3.m_z, epsilon);
        }

        bool IsValid() const
        {
            return true;
        }

    protected:
        T m_x;
        T m_y;
        T m_z;

        T m_invalidValue;
    };

        template <>
    inline bool CVector3<double>::IsValid() const
    {
        return MyMath::IsValid(m_x) &&
            MyMath::IsValid(m_y) &&
            MyMath::IsValid(m_z);
    }

    template <>
    inline bool CVector3<float>::IsValid() const
    {
        return MyMath::IsValid(m_x) &&
            MyMath::IsValid(m_y) &&
            MyMath::IsValid(m_z);
    }
}; // my

#endif // #if !defined(VECTOR_3_INCLUDED)

