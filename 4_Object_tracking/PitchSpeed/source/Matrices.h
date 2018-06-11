#if !defined(MATRICES_INCLUDED)
#define MATRICES_INCLUDED
#define MATRICES_INCLUDED

#include <iostream>
#include <iomanip>

#include "Vectors.h"

///////////////////////////////////////////////////////////////////////////
// 2x2 matrix
///////////////////////////////////////////////////////////////////////////
namespace my {
    template <typename TT>
    class Matrix2
    {
    public:
        typedef TT T;

        // constructors
        Matrix2();  // init with identity
        Matrix2(const T src[4]);
        Matrix2(T m0, T m1, T m2, T m3);

        void        set(const T src[4]);
        void        set(T m0, T m1, T m2, T m3);
        void        setRow(int index, const T row[2]);
        void        setRow(int index, const Vector2<T>& v);
        void        setColumn(int index, const T col[2]);
        void        setColumn(int index, const Vector2<T>& v);

        const T* get() const;
        T       getDeterminant();

        Matrix2&    identity();
        Matrix2&    transpose();                            // transpose itself and return reference
        Matrix2&    invert();

        // operators
        Matrix2     operator+(const Matrix2& rhs) const;    // add rhs
        Matrix2     operator-(const Matrix2& rhs) const;    // subtract rhs
        Matrix2&    operator+=(const Matrix2& rhs);         // add rhs and update this object
        Matrix2&    operator-=(const Matrix2& rhs);         // subtract rhs and update this object
        Vector2<T>     operator*(const Vector2<T>& rhs) const;    // multiplication: v' = M * v
        Matrix2     operator*(const Matrix2& rhs) const;    // multiplication: M3 = M1 * M2
        Matrix2&    operator*=(const Matrix2& rhs);         // multiplication: M1' = M1 * M2
        bool        operator==(const Matrix2& rhs) const;   // exact compare, no epsilon
        bool        operator!=(const Matrix2& rhs) const;   // exact compare, no epsilon
        T       operator[](int index) const;            // subscript operator v[0], v[1]
        T&      operator[](int index);                  // subscript operator v[0], v[1]

        friend Matrix2 operator-(const Matrix2& m);                     // unary operator (-)
        friend Matrix2 operator*(T scalar, const Matrix2& m);       // pre-multiplication
        friend Vector2<T> operator*(const Vector2<T>& vec, const Matrix2& m); // pre-multiplication
        friend std::ostream& operator<<(std::ostream& os, const Matrix2& m);

    protected:

    private:
        T m[4];

    };

    ///////////////////////////////////////////////////////////////////////////
    // 3x3 matrix
    ///////////////////////////////////////////////////////////////////////////
    template <typename TT>
    class Matrix3
    {
    public:
        typedef TT T;

        // constructors
        Matrix3();  // init with identity
        Matrix3(const T src[9]);
        Matrix3(T m0, T m1, T m2,           // 1st column
            T m3, T m4, T m5,           // 2nd column
            T m6, T m7, T m8);          // 3rd column

        void        set(const T src[9]);
        void        set(T m0, T m1, T m2,   // 1st column
            T m3, T m4, T m5,   // 2nd column
            T m6, T m7, T m8);  // 3rd column
        void        setRow(int index, const T row[3]);
        void        setRow(int index, const Vector3<T>& v);
        void        setColumn(int index, const T col[3]);
        void        setColumn(int index, const Vector3<T>& v);

        const T* get() const;
        T       getDeterminant();

        Matrix3&    identity();
        Matrix3&    transpose();                            // transpose itself and return reference
        Matrix3&    invert();

        // operators
        Matrix3     operator+(const Matrix3& rhs) const;    // add rhs
        Matrix3     operator-(const Matrix3& rhs) const;    // subtract rhs
        Matrix3&    operator+=(const Matrix3& rhs);         // add rhs and update this object
        Matrix3&    operator-=(const Matrix3& rhs);         // subtract rhs and update this object
        Vector3<T>     operator*(const Vector3<T>& rhs) const;    // multiplication: v' = M * v
        Matrix3     operator*(const Matrix3& rhs) const;    // multiplication: M3 = M1 * M2
        Matrix3&    operator*=(const Matrix3& rhs);         // multiplication: M1' = M1 * M2
        bool        operator==(const Matrix3& rhs) const;   // exact compare, no epsilon
        bool        operator!=(const Matrix3& rhs) const;   // exact compare, no epsilon
        T       operator[](int index) const;            // subscript operator v[0], v[1]
        T&      operator[](int index);                  // subscript operator v[0], v[1]

        friend Matrix3 operator-(const Matrix3& m);                     // unary operator (-)
        friend Matrix3 operator*(T scalar, const Matrix3& m);       // pre-multiplication
        friend Vector3<T> operator*(const Vector3<T>& vec, const Matrix3& m); // pre-multiplication
        friend std::ostream& operator<<(std::ostream& os, const Matrix3& m);

    protected:

    private:
        T m[9];

    };

    ///////////////////////////////////////////////////////////////////////////
    // 4x4 matrix
    ///////////////////////////////////////////////////////////////////////////
    template <typename TT>
    class Matrix4
    {
    public:
        typedef TT T;

        // constructors
        Matrix4();  // init with identity
        Matrix4(const T src[16]);
        Matrix4(T m00, T m01, T m02, T m03, // 1st column
            T m04, T m05, T m06, T m07, // 2nd column
            T m08, T m09, T m10, T m11, // 3rd column
            T m12, T m13, T m14, T m15);// 4th column

        void        set(const T src[16]);
        void        set(T m00, T m01, T m02, T m03, // 1st column
            T m04, T m05, T m06, T m07, // 2nd column
            T m08, T m09, T m10, T m11, // 3rd column
            T m12, T m13, T m14, T m15);// 4th column
        void        setRow(int index, const T row[4]);
        void        setRow(int index, const Vector4<T>& v);
        void        setRow(int index, const Vector3<T>& v);
        void        setColumn(int index, const T col[4]);
        void        setColumn(int index, const Vector4<T>& v);
        void        setColumn(int index, const Vector3<T>& v);

        const T* get() const;
        const T* getTranspose();                        // return transposed matrix
        T        getDeterminant();

        Matrix4&    identity();
        Matrix4&    transpose();                            // transpose itself and return reference
        Matrix4    transpose() const;
        Matrix4&    invert();                               // check best inverse method before inverse
        Matrix4&    invertEuclidean();                      // inverse of Euclidean transform matrix
        Matrix4&    invertAffine();                         // inverse of affine transform matrix
        Matrix4&    invertProjective();                     // inverse of projective matrix using partitioning
        Matrix4&    invertGeneral();                        // inverse of generic matrix

        // transform matrix
        Matrix4&    translate(T x, T y, T z);   // translation by (x,y,z)
        Matrix4&    translate(const Vector3<T>& v);            //
        Matrix4&    rotate(T angle, const Vector3<T>& axis); // rotate angle(degree) along the given axix
        Matrix4&    rotate(T angle, T x, T y, T z);
        Matrix4&    rotateX(T angle);                   // rotate on X-axis with degree
        Matrix4&    rotateY(T angle);                   // rotate on Y-axis with degree
        Matrix4&    rotateZ(T angle);                   // rotate on Z-axis with degree
        Matrix4&    scale(T scale);                     // uniform scale
        Matrix4&    scale(T sx, T sy, T sz);    // scale by (sx, sy, sz) on each axis

        // operators
        Matrix4     operator+(const Matrix4& rhs) const;    // add rhs
        Matrix4     operator-(const Matrix4& rhs) const;    // subtract rhs
        Matrix4&    operator+=(const Matrix4& rhs);         // add rhs and update this object
        Matrix4&    operator-=(const Matrix4& rhs);         // subtract rhs and update this object
        Vector4<T>     operator*(const Vector4<T>& rhs) const;    // multiplication: v' = M * v
        Vector3<T>     operator*(const Vector3<T>& rhs) const;    // multiplication: v' = M * v
        Matrix4     operator*(const Matrix4& rhs) const;    // multiplication: M3 = M1 * M2
        Matrix4&    operator*=(const Matrix4& rhs);         // multiplication: M1' = M1 * M2
        bool        operator==(const Matrix4& rhs) const;   // exact compare, no epsilon
        bool        operator!=(const Matrix4& rhs) const;   // exact compare, no epsilon
        T       operator[](int index) const;            // subscript operator v[0], v[1]
        T&      operator[](int index);                  // subscript operator v[0], v[1]
        T& operator()(int row, int column);
        const T operator()(int row, int column) const;

        friend Matrix4 operator-(const Matrix4& m);                     // unary operator (-)
        friend Matrix4 operator*(T scalar, const Matrix4& m);       // pre-multiplication
        friend Vector3<T> operator*(const Vector3<T>& vec, const Matrix4& m); // pre-multiplication
        friend Vector4<T> operator*(const Vector4<T>& vec, const Matrix4& m); // pre-multiplication
        friend std::ostream& operator<<(std::ostream& os, const Matrix4& m);

    protected:

    private:
        T       getCofactor(T m0, T m1, T m2,
            T m3, T m4, T m5,
            T m6, T m7, T m8);

        T m[16];
        T tm[16];                                       // transpose m

    };

    ///////////////////////////////////////////////////////////////////////////
    // inline functions for Matrix2
    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    inline Matrix2<T>::Matrix2()
    {
        // initially identity matrix
        identity();
    }

    template <typename T>
    inline Matrix2<T>::Matrix2(const T src[4])
    {
        set(src);
    }
    
    template <typename T>
    inline Matrix2<T>::Matrix2(T m0, T m1, T m2, T m3)
    {
        set(m0, m1, m2, m3);
    }

    template <typename T>
    inline void Matrix2<T>::set(const T src[4])
    {
        m[0] = src[0];  m[1] = src[1];  m[2] = src[2];  m[3] = src[3];
    }
    
    template <typename T>
    inline void Matrix2<T>::set(T m0, T m1, T m2, T m3)
    {
        m[0] = m0;  m[1] = m1;  m[2] = m2;  m[3] = m3;
    }

    template <typename T>
    inline void Matrix2<T>::setRow(int index, const T row[2])
    {
        m[index] = row[0];  m[index + 2] = row[1];
    }

    template <typename T>
    inline void Matrix2<T>::setRow(int index, const Vector2<T>& v)
    {
        m[index] = v.x;  m[index + 2] = v.y;
    }

    template <typename T>
    inline void Matrix2<T>::setColumn(int index, const T col[2])
    {
        m[index * 2] = col[0];  m[index * 2 + 1] = col[1];
    }

    template <typename T>
    inline void Matrix2<T>::setColumn(int index, const Vector2<T>& v)
    {
        m[index * 2] = v.x;  m[index * 2 + 1] = v.y;
    }

    template <typename T>
    inline const T* Matrix2<T>::get() const
    {
        return m;
    }

    template <typename T>
    inline Matrix2<T>& Matrix2<T>::identity()
    {
        m[0] = m[3] = 1.0f;
        m[1] = m[2] = 0.0f;
        return *this;
    }

    template <typename T>
    inline Matrix2<T> Matrix2<T>::operator+(const Matrix2<T>& rhs) const
    {
        return Matrix2(m[0] + rhs[0], m[1] + rhs[1], m[2] + rhs[2], m[3] + rhs[3]);
    }

    template <typename T>
    inline Matrix2<T> Matrix2<T>::operator-(const Matrix2<T>& rhs) const
    {
        return Matrix2<T>(m[0] - rhs[0], m[1] - rhs[1], m[2] - rhs[2], m[3] - rhs[3]);
    }

    template <typename T>
    inline Matrix2<T>& Matrix2<T>::operator+=(const Matrix2<T>& rhs)
    {
        m[0] += rhs[0];  m[1] += rhs[1];  m[2] += rhs[2];  m[3] += rhs[3];
        return *this;
    }

    template <typename T>
    inline Matrix2<T>& Matrix2<T>::operator-=(const Matrix2<T>& rhs)
    {
        m[0] -= rhs[0];  m[1] -= rhs[1];  m[2] -= rhs[2];  m[3] -= rhs[3];
        return *this;
    }

    template <typename T>
    inline Vector2<T> Matrix2<T>::operator*(const Vector2<T>& rhs) const
    {
        return Vector2<T>(m[0] * rhs.x + m[2] * rhs.y, m[1] * rhs.x + m[3] * rhs.y);
    }

    template <typename T>
    inline Matrix2<T> Matrix2<T>::operator*(const Matrix2<T>& rhs) const
    {
        return Matrix2<T>(m[0] * rhs[0] + m[2] * rhs[1], m[1] * rhs[0] + m[3] * rhs[1],
            m[0] * rhs[2] + m[2] * rhs[3], m[1] * rhs[2] + m[3] * rhs[3]);
    }

    template <typename T>
    inline Matrix2<T>& Matrix2<T>::operator*=(const Matrix2<T>& rhs)
    {
        *this = *this * rhs;
        return *this;
    }

    template <typename T>
    inline bool Matrix2<T>::operator==(const Matrix2<T>& rhs) const
    {
        return (m[0] == rhs[0]) && (m[1] == rhs[1]) && (m[2] == rhs[2]) && (m[3] == rhs[3]);
    }

    template <typename T>
    inline bool Matrix2<T>::operator!=(const Matrix2<T>& rhs) const
    {
        return (m[0] != rhs[0]) || (m[1] != rhs[1]) || (m[2] != rhs[2]) || (m[3] != rhs[3]);
    }

    template <typename T>
    inline T Matrix2<T>::operator[](int index) const
    {
        return m[index];
    }

    template <typename T>
    inline T& Matrix2<T>::operator[](int index)
    {
        return m[index];
    }

    template <typename T>
    inline Matrix2<T> operator-(const Matrix2<T>& rhs)
    {
        return Matrix2<T>(-rhs[0], -rhs[1], -rhs[2], -rhs[3]);
    }

    template <typename T>
    inline Matrix2<T> operator*(T s, const Matrix2<T>& rhs)
    {
        return Matrix2<T>(s*rhs[0], s*rhs[1], s*rhs[2], s*rhs[3]);
    }

    template <typename T>
    inline Vector2<T> operator*(const Vector2<T>& v, const Matrix2<T>& rhs)
    {
        return Vector2<T>(v.x*rhs[0] + v.y*rhs[1], v.x*rhs[2] + v.y*rhs[3]);
    }

    template <typename T>
    inline std::ostream& operator<<(std::ostream& os, const Matrix2<T>& m)
    {
        os << std::fixed << std::setprecision(5);
        os << "[" << std::setw(10) << m[0] << " " << std::setw(10) << m[2] << "]\n"
            << "[" << std::setw(10) << m[1] << " " << std::setw(10) << m[3] << "]\n";
        os << std::resetiosflags(std::ios_base::fixed | std::ios_base::floatfield);
        return os;
    }
    // END OF MATRIX2 INLINE //////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////
    // inline functions for Matrix3
    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    inline Matrix3<T>::Matrix3()
    {
        // initially identity matrix
        identity();
    }

    template <typename T>
    inline Matrix3<T>::Matrix3(const T src[9])
    {
        set(src);
    }

    template <typename T>
    inline Matrix3<T>::Matrix3(T m0, T m1, T m2,
        T m3, T m4, T m5,
        T m6, T m7, T m8)
    {
        set(m0, m1, m2, m3, m4, m5, m6, m7, m8);
    }

    template <typename T>
    inline void Matrix3<T>::set(const T src[9])
    {
        m[0] = src[0];  m[1] = src[1];  m[2] = src[2];
        m[3] = src[3];  m[4] = src[4];  m[5] = src[5];
        m[6] = src[6];  m[7] = src[7];  m[8] = src[8];
    }

    template <typename T>
    inline void Matrix3<T>::set(T m0, T m1, T m2,
        T m3, T m4, T m5,
        T m6, T m7, T m8)
    {
        m[0] = m0;  m[1] = m1;  m[2] = m2;
        m[3] = m3;  m[4] = m4;  m[5] = m5;
        m[6] = m6;  m[7] = m7;  m[8] = m8;
    }

    template <typename T>
    inline void Matrix3<T>::setRow(int index, const T row[3])
    {
        m[index] = row[0];  m[index + 3] = row[1];  m[index + 6] = row[2];
    }

    template <typename T>
    inline void Matrix3<T>::setRow(int index, const Vector3<T>& v)
    {
        m[index] = v.x;  m[index + 3] = v.y;  m[index + 6] = v.z;
    }

    template <typename T>
    inline void Matrix3<T>::setColumn(int index, const T col[3])
    {
        m[index * 3] = col[0];  m[index * 3 + 1] = col[1];  m[index * 3 + 2] = col[2];
    }

    template <typename T>
    inline void Matrix3<T>::setColumn(int index, const Vector3<T>& v)
    {
        m[index * 3] = v.x;  m[index * 3 + 1] = v.y;  m[index * 3 + 2] = v.z;
    }

    template <typename T>
    inline const T* Matrix3<T>::get() const
    {
        return m;
    }

    template <typename T>
    inline Matrix3<T>& Matrix3<T>::identity()
    {
        m[0] = m[4] = m[8] = 1.0f;
        m[1] = m[2] = m[3] = m[5] = m[6] = m[7] = 0.0f;
        return *this;
    }

    template <typename T>
    inline Matrix3<T> Matrix3<T>::operator+(const Matrix3<T>& rhs) const
    {
        return Matrix3<T>(m[0] + rhs[0], m[1] + rhs[1], m[2] + rhs[2],
            m[3] + rhs[3], m[4] + rhs[4], m[5] + rhs[5],
            m[6] + rhs[6], m[7] + rhs[7], m[8] + rhs[8]);
    }

    template <typename T>
    inline Matrix3<T> Matrix3<T>::operator-(const Matrix3<T>& rhs) const
    {
        return Matrix3<T>(m[0] - rhs[0], m[1] - rhs[1], m[2] - rhs[2],
            m[3] - rhs[3], m[4] - rhs[4], m[5] - rhs[5],
            m[6] - rhs[6], m[7] - rhs[7], m[8] - rhs[8]);
    }

    template <typename T>
    inline Matrix3<T>& Matrix3<T>::operator+=(const Matrix3<T>& rhs)
    {
        m[0] += rhs[0];  m[1] += rhs[1];  m[2] += rhs[2];
        m[3] += rhs[3];  m[4] += rhs[4];  m[5] += rhs[5];
        m[6] += rhs[6];  m[7] += rhs[7];  m[8] += rhs[8];
        return *this;
    }

    template <typename T>
    inline Matrix3<T>& Matrix3<T>::operator-=(const Matrix3<T>& rhs)
    {
        m[0] -= rhs[0];  m[1] -= rhs[1];  m[2] -= rhs[2];
        m[3] -= rhs[3];  m[4] -= rhs[4];  m[5] -= rhs[5];
        m[6] -= rhs[6];  m[7] -= rhs[7];  m[8] -= rhs[8];
        return *this;
    }

    template <typename T>
    inline Vector3<T> Matrix3<T>::operator*(const Vector3<T>& rhs) const
    {
        return Vector3<T>(m[0] * rhs.x + m[3] * rhs.y + m[6] * rhs.z,
            m[1] * rhs.x + m[4] * rhs.y + m[7] * rhs.z,
            m[2] * rhs.x + m[5] * rhs.y + m[8] * rhs.z);
    }

    template <typename T>
    inline Matrix3<T> Matrix3<T>::operator*(const Matrix3<T>& rhs) const
    {
        return Matrix3<T>(m[0] * rhs[0] + m[3] * rhs[1] + m[6] * rhs[2], m[1] * rhs[0] + m[4] * rhs[1] + m[7] * rhs[2], m[2] * rhs[0] + m[5] * rhs[1] + m[8] * rhs[2],
            m[0] * rhs[3] + m[3] * rhs[4] + m[6] * rhs[5], m[1] * rhs[3] + m[4] * rhs[4] + m[7] * rhs[5], m[2] * rhs[3] + m[5] * rhs[4] + m[8] * rhs[5],
            m[0] * rhs[6] + m[3] * rhs[7] + m[6] * rhs[8], m[1] * rhs[6] + m[4] * rhs[7] + m[7] * rhs[8], m[2] * rhs[6] + m[5] * rhs[7] + m[8] * rhs[8]);
    }

    template <typename T>
    inline Matrix3<T>& Matrix3<T>::operator*=(const Matrix3<T>& rhs)
    {
        *this = *this * rhs;
        return *this;
    }

    template <typename T>
    inline bool Matrix3<T>::operator==(const Matrix3<T>& rhs) const
    {
        return (m[0] == rhs[0]) && (m[1] == rhs[1]) && (m[2] == rhs[2]) &&
            (m[3] == rhs[3]) && (m[4] == rhs[4]) && (m[5] == rhs[5]) &&
            (m[6] == rhs[6]) && (m[7] == rhs[7]) && (m[8] == rhs[8]);
    }

    template <typename T>
    inline bool Matrix3<T>::operator!=(const Matrix3<T>& rhs) const
    {
        return (m[0] != rhs[0]) || (m[1] != rhs[1]) || (m[2] != rhs[2]) ||
            (m[3] != rhs[3]) || (m[4] != rhs[4]) || (m[5] != rhs[5]) ||
            (m[6] != rhs[6]) || (m[7] != rhs[7]) || (m[8] != rhs[8]);
    }

    template <typename T>
    inline T Matrix3<T>::operator[](int index) const
    {
        return m[index];
    }

    template <typename T>
    inline T& Matrix3<T>::operator[](int index)
    {
        return m[index];
    }

    template <typename T>
    inline Matrix3<T> operator-(const Matrix3<T>& rhs)
    {
        return Matrix3<T>(-rhs[0], -rhs[1], -rhs[2], -rhs[3], -rhs[4], -rhs[5], -rhs[6], -rhs[7], -rhs[8]);
    }

    template <typename T>
    inline Matrix3<T> operator*(T s, const Matrix3<T>& rhs)
    {
        return Matrix3<T>(s*rhs[0], s*rhs[1], s*rhs[2], s*rhs[3], s*rhs[4], s*rhs[5], s*rhs[6], s*rhs[7], s*rhs[8]);
    }

    template <typename T>
    inline Vector3<T> operator*(const Vector3<T>& v, const Matrix3<T>& m)
    {
        return Vector3<T>(v.x*m[0] + v.y*m[1] + v.z*m[2], v.x*m[3] + v.y*m[4] + v.z*m[5], v.x*m[6] + v.y*m[7] + v.z*m[8]);
    }

    template <typename T>
    inline std::ostream& operator<<(std::ostream& os, const Matrix3<T>& m)
    {
        os << std::fixed << std::setprecision(5);
        os << "[" << std::setw(10) << m[0] << " " << std::setw(10) << m[3] << " " << std::setw(10) << m[6] << "]\n"
            << "[" << std::setw(10) << m[1] << " " << std::setw(10) << m[4] << " " << std::setw(10) << m[7] << "]\n"
            << "[" << std::setw(10) << m[2] << " " << std::setw(10) << m[5] << " " << std::setw(10) << m[8] << "]\n";
        // There is no Tfield in std::ios_base::fmtflags
        //os << std::resetiosflags(std::ios_base::fixed | std::ios_base::Tfield);
        os << std::resetiosflags(std::ios_base::fixed);
        return os;
    }
    // END OF MATRIX3 INLINE //////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////
    // inline functions for Matrix4
    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    inline Matrix4<T>::Matrix4()
    {
        // initially identity matrix
        identity();
    }

    template <typename T>
    inline Matrix4<T>::Matrix4(const T src[16])
    {
        set(src);
    }

    template <typename T>
    inline Matrix4<T>::Matrix4(T m00, T m01, T m02, T m03,
        T m04, T m05, T m06, T m07,
        T m08, T m09, T m10, T m11,
        T m12, T m13, T m14, T m15)
    {
        set(m00, m01, m02, m03, m04, m05, m06, m07, m08, m09, m10, m11, m12, m13, m14, m15);
    }

    template <typename T>
    inline void Matrix4<T>::set(const T src[16])
    {
        m[0] = src[0];  m[1] = src[1];  m[2] = src[2];  m[3] = src[3];
        m[4] = src[4];  m[5] = src[5];  m[6] = src[6];  m[7] = src[7];
        m[8] = src[8];  m[9] = src[9];  m[10] = src[10]; m[11] = src[11];
        m[12] = src[12]; m[13] = src[13]; m[14] = src[14]; m[15] = src[15];
    }

    template <typename T>
    inline void Matrix4<T>::set(T m00, T m01, T m02, T m03,
        T m04, T m05, T m06, T m07,
        T m08, T m09, T m10, T m11,
        T m12, T m13, T m14, T m15)
    {
        m[0] = m00;  m[1] = m01;  m[2] = m02;  m[3] = m03;
        m[4] = m04;  m[5] = m05;  m[6] = m06;  m[7] = m07;
        m[8] = m08;  m[9] = m09;  m[10] = m10;  m[11] = m11;
        m[12] = m12;  m[13] = m13;  m[14] = m14;  m[15] = m15;
    }

    template <typename T>
    inline void Matrix4<T>::setRow(int index, const T row[4])
    {
        m[index] = row[0];  m[index + 4] = row[1];  m[index + 8] = row[2];  m[index + 12] = row[3];
    }

    template <typename T>
    inline void Matrix4<T>::setRow(int index, const Vector4<T>& v)
    {
        m[index] = v.x;  m[index + 4] = v.y;  m[index + 8] = v.z;  m[index + 12] = v.w;
    }

    template <typename T>
    inline void Matrix4<T>::setRow(int index, const Vector3<T>& v)
    {
        m[index] = v.x;  m[index + 4] = v.y;  m[index + 8] = v.z;
    }

    template <typename T>
    inline void Matrix4<T>::setColumn(int index, const T col[4])
    {
        m[index * 4] = col[0];  m[index * 4 + 1] = col[1];  m[index * 4 + 2] = col[2];  m[index * 4 + 3] = col[3];
    }

    template <typename T>
    inline void Matrix4<T>::setColumn(int index, const Vector4<T>& v)
    {
        m[index * 4] = v.x;  m[index * 4 + 1] = v.y;  m[index * 4 + 2] = v.z;  m[index * 4 + 3] = v.w;
    }

    template <typename T>
    inline void Matrix4<T>::setColumn(int index, const Vector3<T>& v)
    {
        m[index * 4] = v.x;  m[index * 4 + 1] = v.y;  m[index * 4 + 2] = v.z;
    }

    template <typename T>
    inline const T* Matrix4<T>::get() const
    {
        return m;
    }

    template <typename T>
    inline const T* Matrix4<T>::getTranspose()
    {
        tm[0] = m[0];   tm[1] = m[4];   tm[2] = m[8];   tm[3] = m[12];
        tm[4] = m[1];   tm[5] = m[5];   tm[6] = m[9];   tm[7] = m[13];
        tm[8] = m[2];   tm[9] = m[6];   tm[10] = m[10];  tm[11] = m[14];
        tm[12] = m[3];   tm[13] = m[7];   tm[14] = m[11];  tm[15] = m[15];
        return tm;
    }

    template <typename T>
    inline Matrix4<T>& Matrix4<T>::identity()
    {
        m[0] = m[5] = m[10] = m[15] = 1.0f;
        m[1] = m[2] = m[3] = m[4] = m[6] = m[7] = m[8] = m[9] = m[11] = m[12] = m[13] = m[14] = 0.0f;
        return *this;
    }

    template <typename T>
    inline Matrix4<T>& Matrix4<T>::transpose()
    {
        std::swap(m[1], m[4]);
        std::swap(m[2], m[8]);
        std::swap(m[3], m[12]);
        std::swap(m[6], m[9]);
        std::swap(m[7], m[13]);
        std::swap(m[11], m[14]);

        return *this;
    }

    template <typename T>
    inline Matrix4<T> Matrix4<T>::transpose() const
    {
        Matrix4<T> matrix(*this);

        return matrix.transpose();
    }

    template <typename T>
    inline Matrix4<T> Matrix4<T>::operator+(const Matrix4<T>& rhs) const
    {
        return Matrix4(m[0] + rhs[0], m[1] + rhs[1], m[2] + rhs[2], m[3] + rhs[3],
            m[4] + rhs[4], m[5] + rhs[5], m[6] + rhs[6], m[7] + rhs[7],
            m[8] + rhs[8], m[9] + rhs[9], m[10] + rhs[10], m[11] + rhs[11],
            m[12] + rhs[12], m[13] + rhs[13], m[14] + rhs[14], m[15] + rhs[15]);
    }

    template <typename T>
    inline Matrix4<T> Matrix4<T>::operator-(const Matrix4<T>& rhs) const
    {
        return Matrix4(m[0] - rhs[0], m[1] - rhs[1], m[2] - rhs[2], m[3] - rhs[3],
            m[4] - rhs[4], m[5] - rhs[5], m[6] - rhs[6], m[7] - rhs[7],
            m[8] - rhs[8], m[9] - rhs[9], m[10] - rhs[10], m[11] - rhs[11],
            m[12] - rhs[12], m[13] - rhs[13], m[14] - rhs[14], m[15] - rhs[15]);
    }

    template <typename T>
    inline Matrix4<T>& Matrix4<T>::operator+=(const Matrix4<T>& rhs)
    {
        m[0] += rhs[0];   m[1] += rhs[1];   m[2] += rhs[2];   m[3] += rhs[3];
        m[4] += rhs[4];   m[5] += rhs[5];   m[6] += rhs[6];   m[7] += rhs[7];
        m[8] += rhs[8];   m[9] += rhs[9];   m[10] += rhs[10];  m[11] += rhs[11];
        m[12] += rhs[12];  m[13] += rhs[13];  m[14] += rhs[14];  m[15] += rhs[15];
        return *this;
    }

    template <typename T>
    inline Matrix4<T>& Matrix4<T>::operator-=(const Matrix4<T>& rhs)
    {
        m[0] -= rhs[0];   m[1] -= rhs[1];   m[2] -= rhs[2];   m[3] -= rhs[3];
        m[4] -= rhs[4];   m[5] -= rhs[5];   m[6] -= rhs[6];   m[7] -= rhs[7];
        m[8] -= rhs[8];   m[9] -= rhs[9];   m[10] -= rhs[10];  m[11] -= rhs[11];
        m[12] -= rhs[12];  m[13] -= rhs[13];  m[14] -= rhs[14];  m[15] -= rhs[15];
        return *this;
    }

    template <typename T>
    inline Vector4<T> Matrix4<T>::operator*(const Vector4<T>& rhs) const
    {
        return Vector4<T>(m[0] * rhs.x + m[4] * rhs.y + m[8] * rhs.z + m[12] * rhs.w,
            m[1] * rhs.x + m[5] * rhs.y + m[9] * rhs.z + m[13] * rhs.w,
            m[2] * rhs.x + m[6] * rhs.y + m[10] * rhs.z + m[14] * rhs.w,
            m[3] * rhs.x + m[7] * rhs.y + m[11] * rhs.z + m[15] * rhs.w);
    }

    template <typename T>
    inline Vector3<T> Matrix4<T>::operator*(const Vector3<T>& rhs) const
    {
        return Vector3<T>(m[0] * rhs.x + m[4] * rhs.y + m[8] * rhs.z,
            m[1] * rhs.x + m[5] * rhs.y + m[9] * rhs.z,
            m[2] * rhs.x + m[6] * rhs.y + m[10] * rhs.z);
    }

    template <typename T>
    inline Matrix4<T> Matrix4<T>::operator*(const Matrix4<T>& n) const
    {
        return Matrix4<T>(m[0] * n[0] + m[4] * n[1] + m[8] * n[2] + m[12] * n[3], m[1] * n[0] + m[5] * n[1] + m[9] * n[2] + m[13] * n[3], m[2] * n[0] + m[6] * n[1] + m[10] * n[2] + m[14] * n[3], m[3] * n[0] + m[7] * n[1] + m[11] * n[2] + m[15] * n[3],
            m[0] * n[4] + m[4] * n[5] + m[8] * n[6] + m[12] * n[7], m[1] * n[4] + m[5] * n[5] + m[9] * n[6] + m[13] * n[7], m[2] * n[4] + m[6] * n[5] + m[10] * n[6] + m[14] * n[7], m[3] * n[4] + m[7] * n[5] + m[11] * n[6] + m[15] * n[7],
            m[0] * n[8] + m[4] * n[9] + m[8] * n[10] + m[12] * n[11], m[1] * n[8] + m[5] * n[9] + m[9] * n[10] + m[13] * n[11], m[2] * n[8] + m[6] * n[9] + m[10] * n[10] + m[14] * n[11], m[3] * n[8] + m[7] * n[9] + m[11] * n[10] + m[15] * n[11],
            m[0] * n[12] + m[4] * n[13] + m[8] * n[14] + m[12] * n[15], m[1] * n[12] + m[5] * n[13] + m[9] * n[14] + m[13] * n[15], m[2] * n[12] + m[6] * n[13] + m[10] * n[14] + m[14] * n[15], m[3] * n[12] + m[7] * n[13] + m[11] * n[14] + m[15] * n[15]);
    }

    template <typename T>
    inline Matrix4<T>& Matrix4<T>::operator*=(const Matrix4<T>& rhs)
    {
        *this = *this * rhs;
        return *this;
    }

    template <typename T>
    inline bool Matrix4<T>::operator==(const Matrix4<T>& n) const
    {
        return (m[0] == n[0]) && (m[1] == n[1]) && (m[2] == n[2]) && (m[3] == n[3]) &&
            (m[4] == n[4]) && (m[5] == n[5]) && (m[6] == n[6]) && (m[7] == n[7]) &&
            (m[8] == n[8]) && (m[9] == n[9]) && (m[10] == n[10]) && (m[11] == n[11]) &&
            (m[12] == n[12]) && (m[13] == n[13]) && (m[14] == n[14]) && (m[15] == n[15]);
    }

    template <typename T>
    inline bool Matrix4<T>::operator!=(const Matrix4<T>& n) const
    {
        return (m[0] != n[0]) || (m[1] != n[1]) || (m[2] != n[2]) || (m[3] != n[3]) ||
            (m[4] != n[4]) || (m[5] != n[5]) || (m[6] != n[6]) || (m[7] != n[7]) ||
            (m[8] != n[8]) || (m[9] != n[9]) || (m[10] != n[10]) || (m[11] != n[11]) ||
            (m[12] != n[12]) || (m[13] != n[13]) || (m[14] != n[14]) || (m[15] != n[15]);
    }

    template <typename T>
    inline T Matrix4<T>::operator[](int index) const
    {
        return m[index];
    }

    template <typename T>
    inline T& Matrix4<T>::operator[](int index)
    {
        return m[index];
    }

    template <typename T>
    T& Matrix4<T>::operator()(int row, int column)
    {
        return m[4 * row + column];
    }

    template <typename T>
    const T Matrix4<T>::operator()(int row, int column) const
    {
        return m[4 * row + column];
    }

    template <typename T>
    inline Matrix4<T> operator-(const Matrix4<T>& rhs)
    {
        return Matrix4<T>(-rhs[0], -rhs[1], -rhs[2], -rhs[3], -rhs[4], -rhs[5], -rhs[6], -rhs[7], -rhs[8], -rhs[9], -rhs[10], -rhs[11], -rhs[12], -rhs[13], -rhs[14], -rhs[15]);
    }

    template <typename T>
    inline Matrix4<T> operator*(T s, const Matrix4<T>& rhs)
    {
        return Matrix4<T>(s*rhs[0], s*rhs[1], s*rhs[2], s*rhs[3], s*rhs[4], s*rhs[5], s*rhs[6], s*rhs[7], s*rhs[8], s*rhs[9], s*rhs[10], s*rhs[11], s*rhs[12], s*rhs[13], s*rhs[14], s*rhs[15]);
    }

    template <typename T>
    inline Vector4<T> operator*(const Vector4<T>& v, const Matrix4<T>& m)
    {
        return Vector4<T>(v.x*m[0] + v.y*m[1] + v.z*m[2] + v.w*m[3], v.x*m[4] + v.y*m[5] + v.z*m[6] + v.w*m[7], v.x*m[8] + v.y*m[9] + v.z*m[10] + v.w*m[11], v.x*m[12] + v.y*m[13] + v.z*m[14] + v.w*m[15]);
    }

    template <typename T>
    inline Vector3<T> operator*(const Vector3<T>& v, const Matrix4<T>& m)
    {
        return Vector3<T>(v.x*m[0] + v.y*m[1] + v.z*m[2], v.x*m[4] + v.y*m[5] + v.z*m[6], v.x*m[8] + v.y*m[9] + v.z*m[10]);
    }

    template <typename T>
    inline std::ostream& operator<<(std::ostream& os, const Matrix4<T>& m)
    {
        os << std::fixed << std::setprecision(5);
        os << "[" << std::setw(10) << m[0] << " " << std::setw(10) << m[4] << " " << std::setw(10) << m[8] << " " << std::setw(10) << m[12] << "]\n"
            << "[" << std::setw(10) << m[1] << " " << std::setw(10) << m[5] << " " << std::setw(10) << m[9] << " " << std::setw(10) << m[13] << "]\n"
            << "[" << std::setw(10) << m[2] << " " << std::setw(10) << m[6] << " " << std::setw(10) << m[10] << " " << std::setw(10) << m[14] << "]\n"
            << "[" << std::setw(10) << m[3] << " " << std::setw(10) << m[7] << " " << std::setw(10) << m[11] << " " << std::setw(10) << m[15] << "]\n";
        os << std::resetiosflags(std::ios_base::fixed | std::ios_base::floatfield);
        return os;
    }
}; // my
// END OF MATRIX4 INLINE //////////////////////////////////////////////////////
#endif //MATRICES_INCLUDED
