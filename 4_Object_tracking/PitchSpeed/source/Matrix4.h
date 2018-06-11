#if !defined(MATRIX_4_INCLUDED)
#define MATRIX_4_INCLUDED

#include <iomanip>

#include "Vector4.h"

namespace my {
    template <typename T>
    class CMatrix4
    {
    public:
        typedef T ValueType;

        CMatrix4();
        CMatrix4(const T src[16]);
        CMatrix4(T m00, T m01, T m02, T m03, // 1st column
            T m04, T m05, T m06, T m07, // 2nd column
            T m08, T m09, T m10, T m11, // 3rd column
            T m12, T m13, T m14, T m15);// 4th column

        void        Set(const T src[16]);
        void        Set(T m00, T m01, T m02, T m03, // 1st column
            T m04, T m05, T m06, T m07, // 2nd column
            T m08, T m09, T m10, T m11, // 3rd column
            T m12, T m13, T m14, T m15);// 4th column
        void        SetRow(int index, const T row[4]);
        void        SetRow(int index, const CVector4<T>& v);
        void        SetRow(int index, const CVector3<T>& v);
        void        SetColumn(int index, const T col[4]);
        void        SetColumn(int index, const CVector4<T>& v);
        void        SetColumn(int index, const CVector3<T>& v);

        const T* Get() const;
        const T* GetTranspose();                        // return transposed matrix
        T        GetDeterminant();

        CMatrix4&    Identity();
        CMatrix4&    Transpose();                            // transpose itself and return reference
        CMatrix4    Transpose() const;
        CMatrix4&    Invert();                               // check best inverse method before inverse
        CMatrix4&    InvertEuclidean();                      // inverse of Euclidean transform matrix
        CMatrix4&    InvertAffine();                         // inverse of affine transform matrix
        CMatrix4&    InvertProjective();                     // inverse of projective matrix using partitioning
        CMatrix4&    InvertGeneral();                        // inverse of generic matrix

        // transform matrix
        CMatrix4&    Translate(T x, T y, T z);   // translation by (x,y,.z())
        CMatrix4&    Translate(const CVector3<T>& v);            //
        CMatrix4&    Rotate(T angle, const CVector3<T>& axis); // rotate angle(degree) along the given axix
        CMatrix4&    Rotate(T angle, T x, T y, T z);
        CMatrix4&    RotateX(T angle);                   // rotate on X-axis with degree
        CMatrix4&    RotateY(T angle);                   // rotate on Y-axis with degree
        CMatrix4&    RotateZ(T angle);                   // rotate on Z-axis with degree
        CMatrix4&    Scale(T scale);                     // uniform scale
        CMatrix4&    Scale(T sx, T sy, T sz);    // scale by (sx, sy, sz) on each axis

        // operators
        CMatrix4     operator+(const CMatrix4& rhs) const;    // add rhs
        CMatrix4     operator-(const CMatrix4& rhs) const;    // subtract rhs
        CMatrix4&    operator+=(const CMatrix4& rhs);         // add rhs and update this object
        CMatrix4&    operator-=(const CMatrix4& rhs);         // subtract rhs and update this object
        CVector4<T>     operator*(const CVector4<T>& rhs) const;    // multiplication: v' = M * v
        CVector3<T>     operator*(const CVector3<T>& rhs) const;    // multiplication: v' = M * v
        CMatrix4     operator*(const CMatrix4& rhs) const;    // multiplication: M3 = M1 * M2
        CMatrix4&    operator*=(const CMatrix4& rhs);         // multiplication: M1' = M1 * M2
        bool        operator==(const CMatrix4& rhs) const;   // exact compare, no epsilon
        bool        operator!=(const CMatrix4& rhs) const;   // exact compare, no epsilon
        T       operator[](int index) const;            // subscript operator v[0], v[1]
        T&      operator[](int index);                  // subscript operator v[0], v[1]
        T& operator()(int row, int column);
        const T operator()(int row, int column) const;

        friend CMatrix4 operator-(const CMatrix4& m);                     // unary operator (-)
        friend CMatrix4 operator*(T scalar, const CMatrix4& m);       // pre-multiplication
        friend CVector3<T> operator*(const CVector3<T>& vec, const CMatrix4& m); // pre-multiplication
        friend CVector4<T> operator*(const CVector4<T>& vec, const CMatrix4& m); // pre-multiplication
        friend std::ostream& operator<<(std::ostream& os, const CMatrix4& m);

    protected:

    private:
        T       GetCofactor(T m0, T m1, T m2,
            T m3, T m4, T m5,
            T m6, T m7, T m8);

        T m[16];
        T tm[16];                                       // transpose m

    };

    template <typename T>
    inline CMatrix4<T>::CMatrix4()
    {
        // initially identity matrix
        Identity();
    }

    template <typename T>
    inline CMatrix4<T>::CMatrix4(const T src[16])
    {
        Set(src);
    }

    template <typename T>
    inline CMatrix4<T>::CMatrix4(T m00, T m01, T m02, T m03,
        T m04, T m05, T m06, T m07,
        T m08, T m09, T m10, T m11,
        T m12, T m13, T m14, T m15)
    {
        Set(m00, m01, m02, m03, m04, m05, m06, m07, m08, m09, m10, m11, m12, m13, m14, m15);
    }

    template <typename T>
    inline void CMatrix4<T>::Set(const T src[16])
    {
        m[0] = src[0];  m[1] = src[1];  m[2] = src[2];  m[3] = src[3];
        m[4] = src[4];  m[5] = src[5];  m[6] = src[6];  m[7] = src[7];
        m[8] = src[8];  m[9] = src[9];  m[10] = src[10]; m[11] = src[11];
        m[12] = src[12]; m[13] = src[13]; m[14] = src[14]; m[15] = src[15];
    }

    template <typename T>
    inline void CMatrix4<T>::Set(T m00, T m01, T m02, T m03,
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
    inline void CMatrix4<T>::SetRow(int index, const T row[4])
    {
        m[index] = row[0];  m[index + 4] = row[1];  m[index + 8] = row[2];  m[index + 12] = row[3];
    }

    template <typename T>
    inline void CMatrix4<T>::SetRow(int index, const CVector4<T>& v)
    {
        m[index] = v.x();  m[index + 4] = v.y();  m[index + 8] = v.z();  m[index + 12] = v.w();
    }

    template <typename T>
    inline void CMatrix4<T>::SetRow(int index, const CVector3<T>& v)
    {
        m[index] = v.x();  m[index + 4] = v.y();  m[index + 8] = v.z();
    }

    template <typename T>
    inline void CMatrix4<T>::SetColumn(int index, const T col[4])
    {
        m[index * 4] = col[0];  m[index * 4 + 1] = col[1];  m[index * 4 + 2] = col[2];  m[index * 4 + 3] = col[3];
    }

    template <typename T>
    inline void CMatrix4<T>::SetColumn(int index, const CVector4<T>& v)
    {
        m[index * 4] = v.x();  m[index * 4 + 1] = v.y();  m[index * 4 + 2] = v.z();  m[index * 4 + 3] = v.w();
    }

    template <typename T>
    inline void CMatrix4<T>::SetColumn(int index, const CVector3<T>& v)
    {
        m[index * 4] = v.x();  m[index * 4 + 1] = v.y();  m[index * 4 + 2] = v.z();
    }

    template <typename T>
    inline const T* CMatrix4<T>::Get() const
    {
        return m;
    }

    template <typename T>
    inline const T* CMatrix4<T>::GetTranspose()
    {
        tm[0] = m[0];   tm[1] = m[4];   tm[2] = m[8];   tm[3] = m[12];
        tm[4] = m[1];   tm[5] = m[5];   tm[6] = m[9];   tm[7] = m[13];
        tm[8] = m[2];   tm[9] = m[6];   tm[10] = m[10];  tm[11] = m[14];
        tm[12] = m[3];   tm[13] = m[7];   tm[14] = m[11];  tm[15] = m[15];
        return tm;
    }

    template <typename T>
    inline CMatrix4<T>& CMatrix4<T>::Identity()
    {
        m[0] = m[5] = m[10] = m[15] = 1;
        m[1] = m[2] = m[3] = m[4] = m[6] = m[7] = m[8] = m[9] = m[11] = m[12] = m[13] = m[14] = 0;
        return *this;
    }

    template <typename T>
    inline CMatrix4<T>& CMatrix4<T>::Transpose()
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
    inline CMatrix4<T> CMatrix4<T>::Transpose() const
    {
        CMatrix4<T> matrix(*this);

        return matrix.Transpose();
    }

    template <typename T>
    inline CMatrix4<T> CMatrix4<T>::operator+(const CMatrix4<T>& rhs) const
    {
        return CMatrix4(m[0] + rhs[0], m[1] + rhs[1], m[2] + rhs[2], m[3] + rhs[3],
            m[4] + rhs[4], m[5] + rhs[5], m[6] + rhs[6], m[7] + rhs[7],
            m[8] + rhs[8], m[9] + rhs[9], m[10] + rhs[10], m[11] + rhs[11],
            m[12] + rhs[12], m[13] + rhs[13], m[14] + rhs[14], m[15] + rhs[15]);
    }

    template <typename T>
    inline CMatrix4<T> CMatrix4<T>::operator-(const CMatrix4<T>& rhs) const
    {
        return CMatrix4(m[0] - rhs[0], m[1] - rhs[1], m[2] - rhs[2], m[3] - rhs[3],
            m[4] - rhs[4], m[5] - rhs[5], m[6] - rhs[6], m[7] - rhs[7],
            m[8] - rhs[8], m[9] - rhs[9], m[10] - rhs[10], m[11] - rhs[11],
            m[12] - rhs[12], m[13] - rhs[13], m[14] - rhs[14], m[15] - rhs[15]);
    }

    template <typename T>
    inline CMatrix4<T>& CMatrix4<T>::operator+=(const CMatrix4<T>& rhs)
    {
        m[0] += rhs[0];   m[1] += rhs[1];   m[2] += rhs[2];   m[3] += rhs[3];
        m[4] += rhs[4];   m[5] += rhs[5];   m[6] += rhs[6];   m[7] += rhs[7];
        m[8] += rhs[8];   m[9] += rhs[9];   m[10] += rhs[10];  m[11] += rhs[11];
        m[12] += rhs[12];  m[13] += rhs[13];  m[14] += rhs[14];  m[15] += rhs[15];
        return *this;
    }

    template <typename T>
    inline CMatrix4<T>& CMatrix4<T>::operator-=(const CMatrix4<T>& rhs)
    {
        m[0] -= rhs[0];   m[1] -= rhs[1];   m[2] -= rhs[2];   m[3] -= rhs[3];
        m[4] -= rhs[4];   m[5] -= rhs[5];   m[6] -= rhs[6];   m[7] -= rhs[7];
        m[8] -= rhs[8];   m[9] -= rhs[9];   m[10] -= rhs[10];  m[11] -= rhs[11];
        m[12] -= rhs[12];  m[13] -= rhs[13];  m[14] -= rhs[14];  m[15] -= rhs[15];
        return *this;
    }

    template <typename T>
    inline CVector4<T> CMatrix4<T>::operator*(const CVector4<T>& rhs) const
    {
        return CVector4<T>(m[0] * rhs.x() + m[4] * rhs.y() + m[8] * rhs.z() + m[12] * rhs.w(),
            m[1] * rhs.x() + m[5] * rhs.y() + m[9] * rhs.z() + m[13] * rhs.w(),
            m[2] * rhs.x() + m[6] * rhs.y() + m[10] * rhs.z() + m[14] * rhs.w(),
            m[3] * rhs.x() + m[7] * rhs.y() + m[11] * rhs.z() + m[15] * rhs.w());
    }

    template <typename T>
    inline CVector3<T> CMatrix4<T>::operator*(const CVector3<T>& rhs) const
    {
        return CVector3<T>(m[0] * rhs.x() + m[4] * rhs.y() + m[8] * rhs.z(),
            m[1] * rhs.x() + m[5] * rhs.y() + m[9] * rhs.z(),
            m[2] * rhs.x() + m[6] * rhs.y() + m[10] * rhs.z());
    }

    template <typename T>
    inline CMatrix4<T> CMatrix4<T>::operator*(const CMatrix4<T>& n) const
    {
        return CMatrix4<T>(m[0] * n[0] + m[4] * n[1] + m[8] * n[2] + m[12] * n[3], m[1] * n[0] + m[5] * n[1] + m[9] * n[2] + m[13] * n[3], m[2] * n[0] + m[6] * n[1] + m[10] * n[2] + m[14] * n[3], m[3] * n[0] + m[7] * n[1] + m[11] * n[2] + m[15] * n[3],
            m[0] * n[4] + m[4] * n[5] + m[8] * n[6] + m[12] * n[7], m[1] * n[4] + m[5] * n[5] + m[9] * n[6] + m[13] * n[7], m[2] * n[4] + m[6] * n[5] + m[10] * n[6] + m[14] * n[7], m[3] * n[4] + m[7] * n[5] + m[11] * n[6] + m[15] * n[7],
            m[0] * n[8] + m[4] * n[9] + m[8] * n[10] + m[12] * n[11], m[1] * n[8] + m[5] * n[9] + m[9] * n[10] + m[13] * n[11], m[2] * n[8] + m[6] * n[9] + m[10] * n[10] + m[14] * n[11], m[3] * n[8] + m[7] * n[9] + m[11] * n[10] + m[15] * n[11],
            m[0] * n[12] + m[4] * n[13] + m[8] * n[14] + m[12] * n[15], m[1] * n[12] + m[5] * n[13] + m[9] * n[14] + m[13] * n[15], m[2] * n[12] + m[6] * n[13] + m[10] * n[14] + m[14] * n[15], m[3] * n[12] + m[7] * n[13] + m[11] * n[14] + m[15] * n[15]);
    }

    template <typename T>
    inline CMatrix4<T>& CMatrix4<T>::operator*=(const CMatrix4<T>& rhs)
    {
        *this = *this * rhs;
        return *this;
    }

    template <typename T>
    inline bool CMatrix4<T>::operator==(const CMatrix4<T>& n) const
    {
        return (m[0] == n[0]) && (m[1] == n[1]) && (m[2] == n[2]) && (m[3] == n[3]) &&
            (m[4] == n[4]) && (m[5] == n[5]) && (m[6] == n[6]) && (m[7] == n[7]) &&
            (m[8] == n[8]) && (m[9] == n[9]) && (m[10] == n[10]) && (m[11] == n[11]) &&
            (m[12] == n[12]) && (m[13] == n[13]) && (m[14] == n[14]) && (m[15] == n[15]);
    }

    template <typename T>
    inline bool CMatrix4<T>::operator!=(const CMatrix4<T>& n) const
    {
        return (m[0] != n[0]) || (m[1] != n[1]) || (m[2] != n[2]) || (m[3] != n[3]) ||
            (m[4] != n[4]) || (m[5] != n[5]) || (m[6] != n[6]) || (m[7] != n[7]) ||
            (m[8] != n[8]) || (m[9] != n[9]) || (m[10] != n[10]) || (m[11] != n[11]) ||
            (m[12] != n[12]) || (m[13] != n[13]) || (m[14] != n[14]) || (m[15] != n[15]);
    }

    template <typename T>
    inline T CMatrix4<T>::operator[](int index) const
    {
        return m[index];
    }

    template <typename T>
    inline T& CMatrix4<T>::operator[](int index)
    {
        return m[index];
    }

    template <typename T>
    T& CMatrix4<T>::operator()(int row, int column)
    {
        return m[4 * row + column];
    }

    template <typename T>
    const T CMatrix4<T>::operator()(int row, int column) const
    {
        return m[4 * row + column];
    }

    template <typename T>
    inline CMatrix4<T> operator-(const CMatrix4<T>& rhs)
    {
        return CMatrix4<T>(-rhs[0], -rhs[1], -rhs[2], -rhs[3], -rhs[4], -rhs[5], -rhs[6], -rhs[7], -rhs[8], -rhs[9], -rhs[10], -rhs[11], -rhs[12], -rhs[13], -rhs[14], -rhs[15]);
    }

    template <typename T>
    inline CMatrix4<T> operator*(T s, const CMatrix4<T>& rhs)
    {
        return CMatrix4<T>(s*rhs[0], s*rhs[1], s*rhs[2], s*rhs[3], s*rhs[4], s*rhs[5], s*rhs[6], s*rhs[7], s*rhs[8], s*rhs[9], s*rhs[10], s*rhs[11], s*rhs[12], s*rhs[13], s*rhs[14], s*rhs[15]);
    }

    template <typename T>
    inline CVector4<T> operator*(const CVector4<T>& v, const CMatrix4<T>& m)
    {
        return CVector4<T>(v.x()*m[0] + v.y()*m[1] + v.z()*m[2] + v.w()*m[3], v.x()*m[4] + v.y()*m[5] + v.z()*m[6] + v.w()*m[7], v.x()*m[8] + v.y()*m[9] + v.z()*m[10] + v.w()*m[11], v.x()*m[12] + v.y()*m[13] + v.z()*m[14] + v.w()*m[15]);
    }

    template <typename T>
    inline CVector3<T> operator*(const CVector3<T>& v, const CMatrix4<T>& m)
    {
        return CVector3<T>(v.x()*m[0] + v.y()*m[1] + v.z()*m[2], v.x()*m[4] + v.y()*m[5] + v.z()*m[6], v.x()*m[8] + v.y()*m[9] + v.z()*m[10]);
    }

    template <typename T>
    inline my::CMatrix4<T>& my::CMatrix4<T>::RotateX(T angle)
    {
        T c = cos(MyMath::DegreesToRadians(angle));
        T s = sin(MyMath::DegreesToRadians(angle));
        T m1 = m[1], m2 = m[2],
            m5 = m[5], m6 = m[6],
            m9 = m[9], m10 = m[10],
            m13 = m[13], m14 = m[14];

        m[1] = m1 * c + m2 *-s;
        m[2] = m1 * s + m2 * c;
        m[5] = m5 * c + m6 *-s;
        m[6] = m5 * s + m6 * c;
        m[9] = m9 * c + m10*-s;
        m[10] = m9 * s + m10* c;
        m[13] = m13* c + m14*-s;
        m[14] = m13* s + m14* c;

        return *this;
    }

    template <typename T>
    inline my::CMatrix4<T>& my::CMatrix4<T>::RotateY(T angle)
    {
        T c = cos(MyMath::DegreesToRadians(angle));
        T s = sin(MyMath::DegreesToRadians(angle));
        T m0 = m[0], m2 = m[2],
            m4 = m[4], m6 = m[6],
            m8 = m[8], m10 = m[10],
            m12 = m[12], m14 = m[14];

        m[0] = m0 * c + m2 * s;
        m[2] = m0 *-s + m2 * c;
        m[4] = m4 * c + m6 * s;
        m[6] = m4 *-s + m6 * c;
        m[8] = m8 * c + m10* s;
        m[10] = m8 *-s + m10* c;
        m[12] = m12* c + m14* s;
        m[14] = m12*-s + m14* c;

        return *this;
    }

    template <typename T>
    my::CMatrix4<T>& my::CMatrix4<T>::RotateZ(T angle)
    {
        T c = cos(MyMath::DegreesToRadians(angle));
        T s = sin(MyMath::DegreesToRadians(angle));
        T m0 = m[0], m1 = m[1],
            m4 = m[4], m5 = m[5],
            m8 = m[8], m9 = m[9],
            m12 = m[12], m13 = m[13];

        m[0] = m0 * c + m1 *-s;
        m[1] = m0 * s + m1 * c;
        m[4] = m4 * c + m5 *-s;
        m[5] = m4 * s + m5 * c;
        m[8] = m8 * c + m9 *-s;
        m[9] = m8 * s + m9 * c;
        m[12] = m12* c + m13*-s;
        m[13] = m12* s + m13* c;

        return *this;
    }

    template <typename T>
    inline std::ostream& operator<<(std::ostream& os, const CMatrix4<T>& m)
    {
        os << std::fixed << std::setprecision(5);
        os << "[" << std::setw(10) << m[0] << " " << std::setw(10) << m[4] << " " << std::setw(10) << m[8] << " " << std::setw(10) << m[12] << "]\n"
            << "[" << std::setw(10) << m[1] << " " << std::setw(10) << m[5] << " " << std::setw(10) << m[9] << " " << std::setw(10) << m[13] << "]\n"
            << "[" << std::setw(10) << m[2] << " " << std::setw(10) << m[6] << " " << std::setw(10) << m[10] << " " << std::setw(10) << m[14] << "]\n"
            << "[" << std::setw(10) << m[3] << " " << std::setw(10) << m[7] << " " << std::setw(10) << m[11] << " " << std::setw(10) << m[15] << "]\n";
        os << std::resetiosflags(std::ios_base::fixed | std::ios_base::floatfield);
        return os;
    }
} // my

#endif //#if !defined(MATRIX_4_INCLUDED)

