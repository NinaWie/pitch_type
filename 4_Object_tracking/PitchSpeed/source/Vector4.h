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

#if !defined(VECTOR_4_INCLUDED)
#define VECTOR_4_INCLUDED

#include "Vector3.h"

namespace my {
    template <typename T>
    class CVector4
    {
    public:
        typedef T ValueType;

        CVector4()
            : m_x(my::Null<T>()),
            m_y(my::Null<T>()),
            m_z(my::Null<T>()),
            m_w(my::Null<T>()),
            m_invalidValue(my::Null<T>())
        {
        }

        CVector4(const CVector2<T>& vector2)
            : m_x(vector2.m_x),
            m_y(vector2.m_y),
            m_z(0),
            m_w(0),
            m_invalidValue(my::Null<T>())
        {
        }

        CVector4(const CVector3<T>& vector3)
            : m_x(vector3.m_x),
            m_y(vector3.m_y),
            m_z(vector3.m_z),
            m_w(0),
            m_invalidValue(my::Null<T>())
        {
        }

        CVector4(const CVector4& vector4)
            : m_x(vector4.m_x),
            m_y(vector4.m_y),
            m_z(vector4.m_z),
            m_w(vector4.m_w),
            m_invalidValue(my::Null<T>())
        {
        }

        CVector4(T value)
            : m_x(value),
            m_y(value),
            m_z(value),
            m_w(value),
            m_invalidValue(my::Null<T>())
        {
        }

        CVector4(T x, T y, T z, T w)
            : m_x(x),
            m_y(y),
            m_z(z),
            m_w(w),
            m_invalidValue(my::Null<T>())
        {
        }

        void operator=(const CVector4& vector4)
        {
            m_x = vector4.m_x;
            m_y = vector4.m_y;
            m_z = vector4.m_z;
            m_w = vector4.m_w;
            m_invalidValue = my::Null<T>();
        }

        template <typename T2>
        void operator=(const CVector4<T2>& vector4)
        {
#if defined(_DEBUG)
            if (!vector4.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            m_x = (T)vector4.m_x();
            m_y = (T)vector4.m_y();
            m_z = (T)vector4.m_z();
            m_w = (T)vector4.m_w();
            m_invalidValue = my::Null<T>();
        }

        // UNARY NEGATION OPERATOR
        CVector4 operator-() const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return CVector4(-m_x, -m_y, -m_z, -m_w);
        }

        CVector4 operator+(const CVector4& vector4) const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

#if defined(_DEBUG)
            if (!vector4.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return CVector4(m_x + vector4.m_x, m_y + vector4.m_y, m_z + vector4.m_z, m_w + vector4.m_w);
        }

        CVector4& operator+=(const CVector4& vector4)
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

#if defined(_DEBUG)
            if (!vector4.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            m_x += vector4.m_x;
            m_y += vector4.m_y;
            m_z += vector4.m_z;
            m_w += vector4.m_w;

            return (*this);
        }

        CVector4 operator-(const CVector4& vector4) const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

#if defined(_DEBUG)
            if (!vector4.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return CVector4(m_x - vector4.m_x, m_y - vector4.m_y, m_z - vector4.m_z, m_w - vector4.m_w);
        }

        CVector4& operator-=(const CVector4& vector4)
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

#if defined(_DEBUG)
            if (!vector4.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            m_x -= vector4.m_x;
            m_y -= vector4.m_y;
            m_z -= vector4.m_z;
            m_w -= vector4.m_w;

            return (*this);
        }

        CVector4 operator*(const T scale) const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

#if defined(_DEBUG)
            if (!MyMath::IsValid(scale))
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return CVector4(m_x * scale, m_y * scale, m_z * scale, m_w * scale);
        }

        CVector4 operator*(const CVector4& vector4) const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

#if defined(_DEBUG)
            if (!vector4.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return CVector4(m_x * vector4.m_x, m_y * vector4.m_y, m_z * vector4.m_z, m_w * vector4.m_w);
        }

        CVector4& operator*=(ValueType scale)
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
            m_w *= scale;

            return (*this);
        }

        CVector4& operator*=(const CVector4& vector4)
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

#if defined(_DEBUG)
            if (!vector4.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            m_x *= vector4.m_x;
            m_y *= vector4.m_y;
            m_z *= vector4.m_z;
            m_w *= vector4.m_w;

            return (*this);
        }

        CVector4 operator/(const T scale) const
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

            return CVector4(m_x / scale, m_y / scale, m_z / scale, m_w / scale);
        }

        CVector4& operator/=(const T scale)
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
            m_w /= scale;

            return (*this);
        }

        bool operator==(const CVector4& vector4) const
        {
            return (m_x == vector4.m_x) &&
                (m_y == vector4.m_y) &&
                (m_z == vector4.m_z) &&
                (m_w == vector4.m_w);
        }

        bool operator!=(const CVector4& vector4) const
        {
            return (m_x != vector4.m_x) ||
                (m_y != vector4.m_y) ||
                (m_z != vector4.m_z) ||
                (m_w != vector4.m_w);
        }

        T operator[](int index) const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            if ((index < 0) ||
                (index > 3))
            {
                LOG_ERROR();

                return my::Null<T>();
            }

            return (&m_x)[index];
        }

        T& operator[](int index)
        {
            if ((index < 0) ||
                (index > 3))
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

        T w() const
        {
#if defined(_DEBUG)
            if (!MyMath::IsValid(m_w))
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return m_w;
        }

        T& w()
        {
            return m_w;
        }

        CVector2<T> xy() const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return CVector2<T>(m_x, m_y);
        }

        CVector2<T> yx() const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return CVector2<T>(m_y, m_x);
        }

        CVector3<T> xyz() const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return CVector3<T>(m_x, m_y, m_z);
        }

        void Set(T x, T y, T z, T w)
        {
            m_x = x;
            m_y = y;
            m_z = z;
            m_w = w;
        }

        void Set(const CVector4& vector4)
        {
            m_x = vector4.m_x;
            m_y = vector4.m_y;
            m_z = vector4.m_z;
            m_w = vector4.m_w;
        }

        void Clamp(T minimum, T maximum)
        {
#if defined(_DEBUG)
            if (!vector4.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

#if defined(_DEBUG)
            if (!MyMath::IsValid(minimum) ||
                !MyMath::IsValid(maximum))
            {
                LOG_ERROR();
            }
#endif //#if defined(_DEBUG)

            MyMath::Clamp(m_x, m_y, m_z, m_w, minimum, maximum);
        }

        T Length() const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return MyMath::Magnitude(m_x, m_y, m_z, m_w);
        }

        T Distance(const CVector4& vector4) const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

#if defined(_DEBUG)
            if (!vector4.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return MyMath::EuclideanDistance(m_x, m_y, m_z, m_w, vector4.m_x, vector4.m_y, vector4.m_z, vector4.m_w);
        }

        T Distance(T x, T y, T z, T w) const
        {
#if defined(_DEBUG)
            if (!MyMath::IsValid(x) ||
                !MyMath::IsValid(y) ||
                !MyMath::IsValid(z) ||
                !MyMath::IsValid(w))
            {
                LOG_ERROR();
            }
#endif //#if defined(_DEBUG)

#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return MyMath::EuclideanDistance(m_x, m_y, m_z, m_w, x, y, z, w);
        }

        CVector4<T>& Normalize(T epsilon = MyMath::Epsilon<T>())
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            T magnitude = MyMath::Magnitude(m_x, m_y, m_z, m_w);

            if (magnitude > epsilon)
            {
                m_x /= magnitude;
                m_y /= magnitude;
                m_z /= magnitude;
                m_w /= magnitude;
            }
            else
            {
                m_x = 0;
                m_y = 0;
                m_z = 0;
                m_w = 0;
            }

            return (*this);
        }

        T Dot(const CVector4& vector4) const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

#if defined(_DEBUG)
            if (!vector4.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return MyMath::DotProduct(m_x, m_y, m_z, m_w, vector4.m_x, vector4.m_y, vector4.m_z, vector4.m_w);
        }

        bool Equals(const CVector4& vector4, T epsilon = MyMath::Epsilon<T>()) const
        {
#if defined(_DEBUG)
            if (!IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

#if defined(_DEBUG)
            if (!vector4.IsValid())
                LOG_ERROR();
#endif //#if defined(_DEBUG)

            return MyMath::IsClose(m_x, vector4.m_x, epsilon) &&
                MyMath::IsClose(m_y, vector4.m_y, epsilon) &&
                MyMath::IsClose(m_z, vector4.m_z, epsilon) &&
                MyMath::IsClose(m_w, vector4.m_w, epsilon);
        }

        bool IsValid() const
        {
            return true;
        }

        std::ostream& operator<<(std::ostream& os) const
        {
            os << "<" << m_x << ", " << m_y << ", " << m_z << ", " << m_w << ">";

            return os;
        }

    protected:
        T m_x;
        T m_y;
        T m_z;
        T m_w;

        T m_invalidValue;
    };

    template <>
    inline bool CVector4<double>::IsValid() const
    {
        return MyMath::IsValid(m_x) &&
            MyMath::IsValid(m_y) &&
            MyMath::IsValid(m_z) &&
            MyMath::IsValid(m_w);
    }

    template <>
    inline bool CVector4<float>::IsValid() const
    {
        return MyMath::IsValid(m_x) &&
            MyMath::IsValid(m_y) &&
            MyMath::IsValid(m_z) &&
            MyMath::IsValid(m_w);
    }
}; // my

#endif // #if !defined(VECTOR_4_INCLUDED)

