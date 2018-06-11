#if !defined(VECTORS_INCLUDED)
#define VECTORS_INCLUDED

#include <iostream>

#include "MyMath.h"

namespace my {
    template <typename T>
    struct Vector2
    {
        typedef T ValueType;

        T x;
        T y;

        // ctors
        Vector2() : x(0), y(0) {};
        Vector2(T x, T y) : x(x), y(y) {};

        // utils functions
        void        set(T x, T y);
        T       length() const;                         //
        T       distance(const Vector2& vec) const;     // distance between two vectors
        Vector2&    normalize();                            //
        T       dot(const Vector2& vec) const;          // dot product
        bool        equal(const Vector2& vec, T e) const; // compare with epsilon

        // operators
        Vector2     operator-() const;                      // unary operator (negate)
        Vector2     operator+(const Vector2& rhs) const;    // add rhs
        Vector2     operator-(const Vector2& rhs) const;    // subtract rhs
        Vector2&    operator+=(const Vector2& rhs);         // add rhs and update this object
        Vector2&    operator-=(const Vector2& rhs);         // subtract rhs and update this object
        Vector2     operator*(const T scale) const;     // scale
        Vector2     operator*(const Vector2& rhs) const;    // multiply each element
        Vector2&    operator*=(const T scale);          // scale and update this object
        Vector2&    operator*=(const Vector2& rhs);         // multiply each element and update this object
        Vector2     operator/(const T scale) const;     // inverse scale
        Vector2&    operator/=(const T scale);          // scale and update this object
        bool        operator==(const Vector2& rhs) const;   // exact compare, no epsilon
        bool        operator!=(const Vector2& rhs) const;   // exact compare, no epsilon
        bool        operator<(const Vector2& rhs) const;    // comparison for sort
        T       operator[](int index) const;            // subscript operator v[0], v[1]
        T&      operator[](int index);                  // subscript operator v[0], v[1]

        friend Vector2 operator*(const T a, const Vector2 vec);
        friend std::ostream& operator<<(std::ostream& os, const Vector2& vec);
    };

    ///////////////////////////////////////////////////////////////////////////////
    // 3D vector
    ///////////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct Vector3
    {
        typedef T ValueType;

        T x;
        T y;
        T z;

        // ctors
        Vector3() : x(0), y(0), z(0) {};
        Vector3(T x, T y, T z) : x(x), y(y), z(z) {};

        // utils functions
        void        set(T x, T y, T z);
        T       length() const;                         //
        T       distance(const Vector3& vec) const;     // distance between two vectors
        Vector3&    normalize();                            //
        T       dot(const Vector3& vec) const;          // dot product
        Vector3     cross(const Vector3& vec) const;        // cross product
        bool        equal(const Vector3& vec, T e) const; // compare with epsilon

        // operators
        Vector3     operator-() const;                      // unary operator (negate)
        Vector3     operator+(const Vector3& rhs) const;    // add rhs
        Vector3     operator-(const Vector3& rhs) const;    // subtract rhs
        Vector3&    operator+=(const Vector3& rhs);         // add rhs and update this object
        Vector3&    operator-=(const Vector3& rhs);         // subtract rhs and update this object
        Vector3     operator*(const T scale) const;     // scale
        Vector3     operator*(const Vector3& rhs) const;    // multiplay each element
        Vector3&    operator*=(const T scale);          // scale and update this object
        Vector3&    operator*=(const Vector3& rhs);         // product each element and update this object
        Vector3     operator/(const T scale) const;     // inverse scale
        Vector3&    operator/=(const T scale);          // scale and update this object
        bool        operator==(const Vector3& rhs) const;   // exact compare, no epsilon
        bool        operator!=(const Vector3& rhs) const;   // exact compare, no epsilon
        bool        operator<(const Vector3& rhs) const;    // comparison for sort
        T       operator[](int index) const;            // subscript operator v[0], v[1]
        T&      operator[](int index);                  // subscript operator v[0], v[1]

        friend Vector3 operator*(const T a, const Vector3 vec);
        friend std::ostream& operator<<(std::ostream& os, const Vector3& vec);
    };



    ///////////////////////////////////////////////////////////////////////////////
    // 4D vector
    ///////////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct Vector4
    {
        typedef T ValueType;

        T x;
        T y;
        T z;
        T w;

        // ctors
        Vector4() : x(0), y(0), z(0), w(0) {};
        Vector4(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {};

        // utils functions
        void        set(T x, T y, T z, T w);
        T       length() const;                         //
        T       distance(const Vector4& vec) const;     // distance between two vectors
        Vector4&    normalize();                            //
        T       dot(const Vector4& vec) const;          // dot product
        bool        equal(const Vector4& vec, T e) const; // compare with epsilon

        // operators
        Vector4     operator-() const;                      // unary operator (negate)
        Vector4     operator+(const Vector4& rhs) const;    // add rhs
        Vector4     operator-(const Vector4& rhs) const;    // subtract rhs
        Vector4&    operator+=(const Vector4& rhs);         // add rhs and update this object
        Vector4&    operator-=(const Vector4& rhs);         // subtract rhs and update this object
        Vector4     operator*(const T scale) const;     // scale
        Vector4     operator*(const Vector4& rhs) const;    // multiply each element
        Vector4&    operator*=(const T scale);          // scale and update this object
        Vector4&    operator*=(const Vector4& rhs);         // multiply each element and update this object
        Vector4     operator/(const T scale) const;     // inverse scale
        Vector4&    operator/=(const T scale);          // scale and update this object
        bool        operator==(const Vector4& rhs) const;   // exact compare, no epsilon
        bool        operator!=(const Vector4& rhs) const;   // exact compare, no epsilon
        bool        operator<(const Vector4& rhs) const;    // comparison for sort
        T       operator[](int index) const;            // subscript operator v[0], v[1]
        T&      operator[](int index);                  // subscript operator v[0], v[1]

        friend Vector4 operator*(const T a, const Vector4 vec);
        friend std::ostream& operator<<(std::ostream& os, const Vector4& vec);
    };

    ///////////////////////////////////////////////////////////////////////////////
    // inline functions for Vector2
    ///////////////////////////////////////////////////////////////////////////////
    template <typename T>
    inline Vector2<T> Vector2<T>::operator-() const {
        return Vector2<T>(-x, -y);
    }

    template <typename T>
    inline Vector2<T> Vector2<T>::operator+(const Vector2& rhs) const {
        return Vector2<T>(x + rhs.x, y + rhs.y);
    }

    template <typename T>
    inline Vector2<T> Vector2<T>::operator-(const Vector2<T>& rhs) const {
        return Vector2<T>(x - rhs.x, y - rhs.y);
    }

    template <typename T>
    inline Vector2<T>& Vector2<T>::operator+=(const Vector2<T>& rhs) {
        x += rhs.x; y += rhs.y; return *this;
    }

    template <typename T>
    inline Vector2<T>& Vector2<T>::operator-=(const Vector2<T>& rhs) {
        x -= rhs.x; y -= rhs.y; return *this;
    }

    template <typename T>
    inline Vector2<T> Vector2<T>::operator*(const T a) const {
        return Vector2<T>(x*a, y*a);
    }

    template <typename T>
    inline Vector2<T> Vector2<T>::operator*(const Vector2<T>& rhs) const {
        return Vector2<T>(x*rhs.x, y*rhs.y);
    }

    template <typename T>
    inline Vector2<T>& Vector2<T>::operator*=(const T a) {
        x *= a; y *= a; return *this;
    }

    template <typename T>
    inline Vector2<T>& Vector2<T>::operator*=(const Vector2<T>& rhs) {
        x *= rhs.x; y *= rhs.y; return *this;
    }

    template <typename T>
    inline Vector2<T> Vector2<T>::operator/(const T a) const {
        return Vector2<T>(x / a, y / a);
    }

    template <typename T>
    inline Vector2<T>& Vector2<T>::operator/=(const T a) {
        x /= a; y /= a; return *this;
    }

    template <typename T>
    inline bool Vector2<T>::operator==(const Vector2<T>& rhs) const {
        return (x == rhs.x) && (y == rhs.y);
    }

    template <typename T>
    inline bool Vector2<T>::operator!=(const Vector2<T>& rhs) const {
        return (x != rhs.x) || (y != rhs.y);
    }

    template <typename T>
    inline bool Vector2<T>::operator<(const Vector2<T>& rhs) const {
        if (x < rhs.x) return true;
        if (x > rhs.x) return false;
        if (y < rhs.y) return true;
        if (y > rhs.y) return false;
        return false;
    }

    template <typename T>
    inline T Vector2<T>::operator[](int index) const {
        return (&x)[index];
    }

    template <typename T>
    inline T& Vector2<T>::operator[](int index) {
        return (&x)[index];
    }

    template <typename T>
    inline void Vector2<T>::set(T x, T y) {
        this->x = x; this->y = y;
    }

    template <typename T>
    inline T Vector2<T>::length() const {
        return MyMath::SquareRoot<T>(x*x + y*y);
    }

    template <typename T>
    inline T Vector2<T>::distance(const Vector2<T>& vec) const {
        return MyMath::SquareRoot<T>((vec.x - x)*(vec.x - x) + (vec.y - y)*(vec.y - y));
    }

    template <typename T>
    inline Vector2<T>& Vector2<T>::normalize() {
        T xxyy = x*x + y*y;
        if (xxyy < MyMath::Epsilon<T>())
        {
            x = 0;
            y = 0;
            return (*this);
        }

        T invLength = 1.0 / MyMath::SquareRoot<T>(xxyy);
        x *= invLength;
        y *= invLength;
        return *this;
    }

    template <typename T>
    inline T Vector2<T>::dot(const Vector2<T>& rhs) const {
        return (x*rhs.x + y*rhs.y);
    }

    template <typename T>
    inline bool Vector2<T>::equal(const Vector2<T>& rhs, T epsilon) const {
        return fabs(x - rhs.x) < epsilon && fabs(y - rhs.y) < epsilon;
    }

    template <typename T>
    inline Vector2<T> operator*(const T a, const Vector2<T> vec) {
        return Vector2<T>(a*vec.x, a*vec.y);
    }

    template <typename T>
    inline std::ostream& operator<<(std::ostream& os, const Vector2<T>& vec) {
        os << "(" << vec.x << ", " << vec.y << ")";
        return os;
    }
    // END OF VECTOR2 /////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////////
    // inline functions for Vector3
    ///////////////////////////////////////////////////////////////////////////////
    template <typename T>
    inline Vector3<T> Vector3<T>::operator-() const {
        return Vector3<T>(-x, -y, -z);
    }

    template <typename T>
    inline Vector3<T> Vector3<T>::operator+(const Vector3<T>& rhs) const {
        return Vector3<T>(x + rhs.x, y + rhs.y, z + rhs.z);
    }

    template <typename T>
    inline Vector3<T> Vector3<T>::operator-(const Vector3<T>& rhs) const {
        return Vector3<T>(x - rhs.x, y - rhs.y, z - rhs.z);
    }

    template <typename T>
    inline Vector3<T>& Vector3<T>::operator+=(const Vector3<T>& rhs) {
        x += rhs.x; y += rhs.y; z += rhs.z; return *this;
    }

    template <typename T>
    inline Vector3<T>& Vector3<T>::operator-=(const Vector3<T>& rhs) {
        x -= rhs.x; y -= rhs.y; z -= rhs.z; return *this;
    }

    template <typename T>
    inline Vector3<T> Vector3<T>::operator*(const T a) const {
        return Vector3<T>(x*a, y*a, z*a);
    }

    template <typename T>
    inline Vector3<T> Vector3<T>::operator*(const Vector3<T>& rhs) const {
        return Vector3(x*rhs.x, y*rhs.y, z*rhs.z);
    }

    template <typename T>
    inline Vector3<T>& Vector3<T>::operator*=(const T a) {
        x *= a; y *= a; z *= a; return *this;
    }

    template <typename T>
    inline Vector3<T>& Vector3<T>::operator*=(const Vector3<T>& rhs) {
        x *= rhs.x; y *= rhs.y; z *= rhs.z; return *this;
    }

    template <typename T>
    inline Vector3<T> Vector3<T>::operator/(const T a) const {
        return Vector3<T>(x / a, y / a, z / a);
    }

    template <typename T>
    inline Vector3<T>& Vector3<T>::operator/=(const T a) {
        x /= a; y /= a; z /= a; return *this;
    }

    template <typename T>
    inline bool Vector3<T>::operator==(const Vector3<T>& rhs) const {
        return (x == rhs.x) && (y == rhs.y) && (z == rhs.z);
    }

    template <typename T>
    inline bool Vector3<T>::operator!=(const Vector3<T>& rhs) const {
        return (x != rhs.x) || (y != rhs.y) || (z != rhs.z);
    }

    template <typename T>
    inline bool Vector3<T>::operator<(const Vector3<T>& rhs) const {
        if (x < rhs.x) return true;
        if (x > rhs.x) return false;
        if (y < rhs.y) return true;
        if (y > rhs.y) return false;
        if (z < rhs.z) return true;
        if (z > rhs.z) return false;
        return false;
    }

    template <typename T>
    inline T Vector3<T>::operator[](int index) const {
        return (&x)[index];
    }

    template <typename T>
    inline T& Vector3<T>::operator[](int index) {
        return (&x)[index];
    }

    template <typename T>
    inline void Vector3<T>::set(T x, T y, T z) {
        this->x = x; this->y = y; this->z = z;
    }

    template <typename T>
    inline T Vector3<T>::length() const {
        return MyMath::SquareRoot<T>(x*x + y*y + z*z);
    }

    template <typename T>
    inline T Vector3<T>::distance(const Vector3& vec) const {
        return MyMath::SquareRoot<T>((vec.x - x)*(vec.x - x) + (vec.y - y)*(vec.y - y) + (vec.z - z)*(vec.z - z));
    }

    template <typename T>
    inline Vector3<T>& Vector3<T>::normalize() {
        T xxyyzz = x*x + y*y + z*z;
        if (xxyyzz < MyMath::Epsilon<T>())
        {
            x = 0;
            y = 0;
            z = 0;
            return *this;
        }

        T invLength = 1.0 / MyMath::SquareRoot<T>(xxyyzz);
        x *= invLength;
        y *= invLength;
        z *= invLength;
        return *this;
    }

    template <typename T>
    inline T Vector3<T>::dot(const Vector3& rhs) const {
        return (x*rhs.x + y*rhs.y + z*rhs.z);
    }

    template <typename T>
    inline Vector3<T> Vector3<T>::cross(const Vector3<T>& rhs) const {
        return Vector3<T>(y*rhs.z - z*rhs.y, z*rhs.x - x*rhs.z, x*rhs.y - y*rhs.x);
    }

    template <typename T>
    inline bool Vector3<T>::equal(const Vector3<T>& rhs, T epsilon) const {
        return fabs(x - rhs.x) < epsilon && fabs(y - rhs.y) < epsilon && fabs(z - rhs.z) < epsilon;
    }

    template <typename T>
    inline Vector3<T> operator*(const T a, const Vector3<T> vec) {
        return Vector3<T>(a*vec.x, a*vec.y, a*vec.z);
    }

    template <typename T>
    inline std::ostream& operator<<(std::ostream& os, const Vector3<T>& vec) {
        os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
        return os;
    }
    // END OF VECTOR3 /////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////////
    // inline functions for Vector4
    ///////////////////////////////////////////////////////////////////////////////
    template <typename T>
    inline Vector4<T> Vector4<T>::operator-() const {
        return Vector4<T>(-x, -y, -z, -w);
    }

    template <typename T>
    inline Vector4<T> Vector4<T>::operator+(const Vector4<T>& rhs) const {
        return Vector4<T>(x + rhs.x, y + rhs.y, z + rhs.z, w + rhs.w);
    }

    template <typename T>
    inline Vector4<T> Vector4<T>::operator-(const Vector4<T>& rhs) const {
        return Vector4<T>(x - rhs.x, y - rhs.y, z - rhs.z, w - rhs.w);
    }

    template <typename T>
    inline Vector4<T>& Vector4<T>::operator+=(const Vector4<T>& rhs) {
        x += rhs.x; y += rhs.y; z += rhs.z; w += rhs.w; return *this;
    }

    template <typename T>
    inline Vector4<T>& Vector4<T>::operator-=(const Vector4<T>& rhs) {
        x -= rhs.x; y -= rhs.y; z -= rhs.z; w -= rhs.w; return *this;
    }

    template <typename T>
    inline Vector4<T> Vector4<T>::operator*(const T a) const {
        return Vector4<T>(x*a, y*a, z*a, w*a);
    }

    template <typename T>
    inline Vector4<T> Vector4<T>::operator*(const Vector4<T>& rhs) const {
        return Vector4<T>(x*rhs.x, y*rhs.y, z*rhs.z, w*rhs.w);
    }

    template <typename T>
    inline Vector4<T>& Vector4<T>::operator*=(const T a) {
        x *= a; y *= a; z *= a; w *= a; return *this;
    }

    template <typename T>
    inline Vector4<T>& Vector4<T>::operator*=(const Vector4<T>& rhs) {
        x *= rhs.x; y *= rhs.y; z *= rhs.z; w *= rhs.w; return *this;
    }

    template <typename T>
    inline Vector4<T> Vector4<T>::operator/(const T a) const {
        return Vector4<T>(x / a, y / a, z / a, w / a);
    }

    template <typename T>
    inline Vector4<T>& Vector4<T>::operator/=(const T a) {
        x /= a; y /= a; z /= a; w /= a; return *this;
    }

    template <typename T>
    inline bool Vector4<T>::operator==(const Vector4<T>& rhs) const {
        return (x == rhs.x) && (y == rhs.y) && (z == rhs.z) && (w == rhs.w);
    }

    template <typename T>
    inline bool Vector4<T>::operator!=(const Vector4<T>& rhs) const {
        return (x != rhs.x) || (y != rhs.y) || (z != rhs.z) || (w != rhs.w);
    }

    template <typename T>
    inline bool Vector4<T>::operator<(const Vector4<T>& rhs) const {
        if (x < rhs.x) return true;
        if (x > rhs.x) return false;
        if (y < rhs.y) return true;
        if (y > rhs.y) return false;
        if (z < rhs.z) return true;
        if (z > rhs.z) return false;
        if (w < rhs.w) return true;
        if (w > rhs.w) return false;
        return false;
    }

    template <typename T>
    inline T Vector4<T>::operator[](int index) const {
        return (&x)[index];
    }

    template <typename T>
    inline T& Vector4<T>::operator[](int index) {
        return (&x)[index];
    }

    template <typename T>
    inline void Vector4<T>::set(T x, T y, T z, T w) {
        this->x = x; this->y = y; this->z = z; this->w = w;
    }

    template <typename T>
    inline T Vector4<T>::length() const {
        return MyMath::SquareRoot<T>(x*x + y*y + z*z + w*w);
    }

    template <typename T>
    inline T Vector4<T>::distance(const Vector4<T>& vec) const {
        return MyMath::SquareRoot<T>((vec.x - x)*(vec.x - x) + (vec.y - y)*(vec.y - y) + (vec.z - z)*(vec.z - z) + (vec.w - w)*(vec.w - w));
    }

    template <typename T>
    inline Vector4<T>& Vector4<T>::normalize() {
        T xxyyzz = x*x + y*y + z*z;
        if (xxyyzz < 1.0E-8)
        {
            x = 0;
            y = 0;
            z = 0;
            return *this;
        }

        T invLength = 1.0 / MyMath::SquareRoot<T>(xxyyzz);
        x *= invLength;
        y *= invLength;
        z *= invLength;
        return *this;
    }

    template <typename T>
    inline T Vector4<T>::dot(const Vector4<T>& rhs) const {
        return (x*rhs.x + y*rhs.y + z*rhs.z + w*rhs.w);
    }

    template <typename T>
    inline bool Vector4<T>::equal(const Vector4<T>& rhs, T epsilon) const {
        return fabs(x - rhs.x) < epsilon && fabs(y - rhs.y) < epsilon &&
            fabs(z - rhs.z) < epsilon && fabs(w - rhs.w) < epsilon;
    }

    template <typename T>
    inline Vector4<T> operator*(const T a, const Vector4<T> vec) {
        return Vector4<T>(a*vec.x, a*vec.y, a*vec.z, a*vec.w);
    }

    template <typename T>
    inline std::ostream& operator<<(std::ostream& os, const Vector4<T>& vec) {
        os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ", " << vec.w << ")";
        return os;
    }
    // END OF VECTOR4 /////////////////////////////////////////////////////////////
}; // my

#endif //VECTORS_INCLUDED

