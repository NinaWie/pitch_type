#include <cmath>
#include <algorithm>

#include "Matrices.h"

#include "Matrix4.h"

// BUG: (30-Nov-2015) There is no definition for EPSILON
#define EPSILON 10e-08
// BUG: (30-Nov-2015) There is no definition for DEG2RAD
#define DEG2RAD 0.01745329252

///////////////////////////////////////////////////////////////////////////////
// inverse 4x4 matrix
///////////////////////////////////////////////////////////////////////////////
template <>
my::CMatrix4<double>& my::CMatrix4<double>::Invert()
{
    // If the 4th row is [0,0,0,1] then it is affine matrix and
    // it has no projective transformation.
    if(m[3] == 0 && m[7] == 0 && m[11] == 0 && m[15] == 1)
        InvertAffine();
    else
    {
        InvertGeneral();
        /*@@ invertProjective() is not optimized (slower than generic one)
        if(fabs(m[0]*m[5] - m[1]*m[4]) > EPSILON)
            this->invertProjective();   // inverse using matrix partition
        else
            this->invertGeneral();      // generalized inverse
        */
    }

    return *this;
}

///////////////////////////////////////////////////////////////////////////////
// compute the inverse of 4x4 Euclidean transformation matrix
//
// Euclidean transformation is translation, rotation, and reflection.
// With Euclidean transform, only the position and orientation of the object
// will be changed. Euclidean transform does not change the shape of an object
// (no scaling). Length and angle are reserved.
//
// Use inverseAffine() if the matrix has scale and shear transformation.
//
// M = [ R | T ]
//     [ --+-- ]    (R denotes 3x3 rotation/reflection matrix)
//     [ 0 | 1 ]    (T denotes 1x3 translation matrix)
//
// y = M*x  ->  y = R*x + T  ->  x = R^-1*(y - T)  ->  x = R^T*y - R^T*T
// (R is orthogonal,  R^-1 = R^T)
//
//  [ R | T ]-1    [ R^T | -R^T * T ]    (R denotes 3x3 rotation matrix)
//  [ --+-- ]   =  [ ----+--------- ]    (T denotes 1x3 translation)
//  [ 0 | 1 ]      [  0  |     1    ]    (R^T denotes R-transpose)
///////////////////////////////////////////////////////////////////////////////
template <typename T>
my::CMatrix4<T>& my::CMatrix4<T>::InvertEuclidean()
{
    // transpose 3x3 rotation matrix part
    // | R^T | 0 |
    // | ----+-- |
    // |  0  | 1 |
    T tmp;
    tmp = m[1];  m[1] = m[4];  m[4] = tmp;
    tmp = m[2];  m[2] = m[8];  m[8] = tmp;
    tmp = m[6];  m[6] = m[9];  m[9] = tmp;

    // compute translation part -R^T * T
    // | 0 | -R^T x |
    // | --+------- |
    // | 0 |   0    |
    T x = m[12];
    T y = m[13];
    T z = m[14];
    m[12] = -(m[0] * x + m[4] * y + m[8] * z);
    m[13] = -(m[1] * x + m[5] * y + m[9] * z);
    m[14] = -(m[2] * x + m[6] * y + m[10]* z);

    // last row should be unchanged (0,0,0,1)

    return *this;
}

///////////////////////////////////////////////////////////////////////////////
// compute the inverse of a 4x4 affine transformation matrix
//
// Affine transformations are generalizations of Euclidean transformations.
// Affine transformation includes translation, rotation, reflection, scaling,
// and shearing. Length and angle are NOT preserved.
// M = [ R | T ]
//     [ --+-- ]    (R denotes 3x3 rotation/scale/shear matrix)
//     [ 0 | 1 ]    (T denotes 1x3 translation matrix)
//
// y = M*x  ->  y = R*x + T  ->  x = R^-1*(y - T)  ->  x = R^-1*y - R^-1*T
//
//  [ R | T ]-1   [ R^-1 | -R^-1 * T ]
//  [ --+-- ]   = [ -----+---------- ]
//  [ 0 | 1 ]     [  0   +     1     ]
///////////////////////////////////////////////////////////////////////////////
template <typename T>
my::CMatrix4<T>& my::CMatrix4<T>::InvertAffine()
{   
    // R^-1
    my::Matrix3<T> r(m[0],m[1],m[2], m[4],m[5],m[6], m[8],m[9],m[10]);
    
    //r.invert();
    
    {
        T determinant, invDeterminant;
        T tmp[9];
        
        const T *rm = r.get();
        
        tmp[0] = rm[4] * rm[8] - rm[5] * rm[7];
        tmp[1] = rm[2] * rm[7] - rm[1] * rm[8];
        tmp[2] = rm[1] * rm[5] - rm[2] * rm[4];
        tmp[3] = rm[5] * rm[6] - rm[3] * rm[8];
        tmp[4] = rm[0] * rm[8] - rm[2] * rm[6];
        tmp[5] = rm[2] * rm[3] - rm[0] * rm[5];
        tmp[6] = rm[3] * rm[7] - rm[4] * rm[6];
        tmp[7] = rm[1] * rm[6] - rm[0] * rm[7];
        tmp[8] = rm[0] * rm[4] - rm[1] * rm[3];
        
        // check determinant if it is 0
        determinant = rm[0] * tmp[0] + rm[1] * tmp[3] + rm[2] * tmp[6];
        if(fabs(determinant) <= MyMath::Epsilon<T>())
        {
            r.identity(); // cannot inverse, make it idenety matrix
        }
        
        // divide by the determinant
        invDeterminant = 1.0 / determinant;
        r.set(invDeterminant * tmp[0],
         invDeterminant * tmp[1],
         invDeterminant * tmp[2],
         invDeterminant * tmp[3],
         invDeterminant * tmp[4],
        invDeterminant * tmp[5],
         invDeterminant * tmp[6],
         invDeterminant * tmp[7],
         invDeterminant * tmp[8]);
    }
    
    m[0] = r[0];  m[1] = r[1];  m[2] = r[2];
    m[4] = r[3];  m[5] = r[4];  m[6] = r[5];
    m[8] = r[6];  m[9] = r[7];  m[10]= r[8];

    // -R^-1 * T
    T x = m[12];
    T y = m[13];
    T z = m[14];
    m[12] = -(r[0] * x + r[3] * y + r[6] * z);
    m[13] = -(r[1] * x + r[4] * y + r[7] * z);
    m[14] = -(r[2] * x + r[5] * y + r[8] * z);

    // last row should be unchanged (0,0,0,1)
    //m[3] = m[7] = m[11] = 0.0f;
    //m[15] = 1.0f;

    return * this;
}

///////////////////////////////////////////////////////////////////////////////
// inverse matrix using matrix partitioning (blockwise inverse)
// It devides a 4x4 matrix into 4 of 2x2 matrices. It works in case of where
// det(A) != 0. If not, use the generic inverse method
// inverse formula.
// M = [ A | B ]    A, B, C, D are 2x2 matrix blocks
//     [ --+-- ]    det(M) = |A| * |D - ((C * A^-1) * B)|
//     [ C | D ]
//
// M^-1 = [ A' | B' ]   A' = A^-1 - (A^-1 * B) * C'
//        [ ---+--- ]   B' = (A^-1 * B) * -D'
//        [ C' | D' ]   C' = -D' * (C * A^-1)
//                      D' = (D - ((C * A^-1) * B))^-1
//
// NOTE: I wrap with () if it it used more than once.
//       The matrix is invertable even if det(A)=0, so must check det(A) before
//       calling this function, and use invertGeneric() instead.
///////////////////////////////////////////////////////////////////////////////
template <typename T>
my::CMatrix4<T>& my::CMatrix4<T>::InvertProjective()
{
    // partition
    my::Matrix2<T> a(m[0], m[1], m[4], m[5]);
    my::Matrix2<T> b(m[8], m[9], m[12], m[13]);
    my::Matrix2<T> c(m[2], m[3], m[6], m[7]);
    my::Matrix2<T> d(m[10], m[11], m[14], m[15]);

    // pre-compute repeated parts
    a.Invert();             // A^-1
    my::Matrix2<T> ab = a * b;     // A^-1 * B
    my::Matrix2<T> ca = c * a;     // C * A^-1
    my::Matrix2<T> cab = ca * b;   // C * A^-1 * B
    my::Matrix2<T> dcab = d - cab; // D - C * A^-1 * B

    // check determinant if |D - C * A^-1 * B| = 0
    //NOTE: this function assumes det(A) is already checked. if |A|=0 then,
    //      cannot use this function.
    T determinant = dcab[0] * dcab[3] - dcab[1] * dcab[2];
    if(fabs(determinant) <= MyMath::Epsilon<T>())
    {
        return Identity();
    }

    // compute D' and -D'
    my::Matrix2<T> d1 = dcab;      //  (D - C * A^-1 * B)
    d1.Invert();            //  (D - C * A^-1 * B)^-1
    my::Matrix2<T> d2 = -d1;       // -(D - C * A^-1 * B)^-1

    // compute C'
    my::Matrix2<T> c1 = d2 * ca;   // -D' * (C * A^-1)

    // compute B'
    my::Matrix2<T> b1 = ab * d2;   // (A^-1 * B) * -D'

    // compute A'
    my::Matrix2<T> a1 = a - (ab * c1); // A^-1 - (A^-1 * B) * C'

    // assemble inverse matrix
    m[0] = a1[0];  m[4] = a1[2]; /*|*/ m[8] = b1[0];  m[12]= b1[2];
    m[1] = a1[1];  m[5] = a1[3]; /*|*/ m[9] = b1[1];  m[13]= b1[3];
    /*-----------------------------+-----------------------------*/
    m[2] = c1[0];  m[6] = c1[2]; /*|*/ m[10]= d1[0];  m[14]= d1[2];
    m[3] = c1[1];  m[7] = c1[3]; /*|*/ m[11]= d1[1];  m[15]= d1[3];

    return *this;
}

///////////////////////////////////////////////////////////////////////////////
// compute the inverse of a general 4x4 matrix using Cramer's Rule
// If cannot find inverse, return indentity matrix
// M^-1 = adj(M) / det(M)
///////////////////////////////////////////////////////////////////////////////
template <typename T>
my::CMatrix4<T>& my::CMatrix4<T>::InvertGeneral()
{
    // get cofactors of minor matrices
    T cofactor0 = GetCofactor(m[5],m[6],m[7], m[9],m[10],m[11], m[13],m[14],m[15]);
    T cofactor1 = GetCofactor(m[4],m[6],m[7], m[8],m[10],m[11], m[12],m[14],m[15]);
    T cofactor2 = GetCofactor(m[4],m[5],m[7], m[8],m[9], m[11], m[12],m[13],m[15]);
    T cofactor3 = GetCofactor(m[4],m[5],m[6], m[8],m[9], m[10], m[12],m[13],m[14]);

    // get determinant
    T determinant = m[0] * cofactor0 - m[1] * cofactor1 + m[2] * cofactor2 - m[3] * cofactor3;
    if(fabs(determinant) <= MyMath::Epsilon<T>())
    {
        return Identity();
    }

    // get rest of cofactors for adj(M)
    T cofactor4 = GetCofactor(m[1],m[2],m[3], m[9],m[10],m[11], m[13],m[14],m[15]);
    T cofactor5 = GetCofactor(m[0],m[2],m[3], m[8],m[10],m[11], m[12],m[14],m[15]);
    T cofactor6 = GetCofactor(m[0],m[1],m[3], m[8],m[9], m[11], m[12],m[13],m[15]);
    T cofactor7 = GetCofactor(m[0],m[1],m[2], m[8],m[9], m[10], m[12],m[13],m[14]);

    T cofactor8 = GetCofactor(m[1],m[2],m[3], m[5],m[6], m[7],  m[13],m[14],m[15]);
    T cofactor9 = GetCofactor(m[0],m[2],m[3], m[4],m[6], m[7],  m[12],m[14],m[15]);
    T cofactor10= GetCofactor(m[0],m[1],m[3], m[4],m[5], m[7],  m[12],m[13],m[15]);
    T cofactor11= GetCofactor(m[0],m[1],m[2], m[4],m[5], m[6],  m[12],m[13],m[14]);

    T cofactor12= GetCofactor(m[1],m[2],m[3], m[5],m[6], m[7],  m[9], m[10],m[11]);
    T cofactor13= GetCofactor(m[0],m[2],m[3], m[4],m[6], m[7],  m[8], m[10],m[11]);
    T cofactor14= GetCofactor(m[0],m[1],m[3], m[4],m[5], m[7],  m[8], m[9], m[11]);
    T cofactor15= GetCofactor(m[0],m[1],m[2], m[4],m[5], m[6],  m[8], m[9], m[10]);

    // build inverse matrix = adj(M) / det(M)
    // adjugate of M is the transpose of the cofactor matrix of M
    T invDeterminant = 1.0 / determinant;
    m[0] =  invDeterminant * cofactor0;
    m[1] = -invDeterminant * cofactor4;
    m[2] =  invDeterminant * cofactor8;
    m[3] = -invDeterminant * cofactor12;

    m[4] = -invDeterminant * cofactor1;
    m[5] =  invDeterminant * cofactor5;
    m[6] = -invDeterminant * cofactor9;
    m[7] =  invDeterminant * cofactor13;

    m[8] =  invDeterminant * cofactor2;
    m[9] = -invDeterminant * cofactor6;
    m[10]=  invDeterminant * cofactor10;
    m[11]= -invDeterminant * cofactor14;

    m[12]= -invDeterminant * cofactor3;
    m[13]=  invDeterminant * cofactor7;
    m[14]= -invDeterminant * cofactor11;
    m[15]=  invDeterminant * cofactor15;

    return *this;
}

///////////////////////////////////////////////////////////////////////////////
// return determinant of 4x4 matrix
///////////////////////////////////////////////////////////////////////////////
template <typename T>
T my::CMatrix4<T>::GetDeterminant()
{
    return m[0] * GetCofactor(m[5],m[6],m[7], m[9],m[10],m[11], m[13],m[14],m[15]) -
           m[1] * GetCofactor(m[4],m[6],m[7], m[8],m[10],m[11], m[12],m[14],m[15]) +
           m[2] * GetCofactor(m[4],m[5],m[7], m[8],m[9], m[11], m[12],m[13],m[15]) -
           m[3] * GetCofactor(m[4],m[5],m[6], m[8],m[9], m[10], m[12],m[13],m[14]);
}

///////////////////////////////////////////////////////////////////////////////
// compute cofactor of 3x3 minor matrix without sign
// input params are 9 elements of the minor matrix
// NOTE: The caller must know its sign.
///////////////////////////////////////////////////////////////////////////////
template <typename T>
T my::CMatrix4<T>::GetCofactor(T m0, T m1, T m2,
                           T m3, T m4, T m5,
                           T m6, T m7, T m8)
{
    return m0 * (m4 * m8 - m5 * m7) -
           m1 * (m3 * m8 - m5 * m6) +
           m2 * (m3 * m7 - m4 * m6);
}

///////////////////////////////////////////////////////////////////////////////
// translate this matrix by (x, y, z)
///////////////////////////////////////////////////////////////////////////////
template <typename T>
my::CMatrix4<T>& my::CMatrix4<T>::Translate(const my::CVector3<T>& v)
{
    return Translate(v.x, v.y, v.z);
}

template <typename T>
my::CMatrix4<T>& my::CMatrix4<T>::Translate(T x, T y, T z)
{
    m[0] += m[3] * x;   m[4] += m[7] * x;   m[8] += m[11]* x;   m[12]+= m[15]* x;
    m[1] += m[3] * y;   m[5] += m[7] * y;   m[9] += m[11]* y;   m[13]+= m[15]* y;
    m[2] += m[3] * z;   m[6] += m[7] * z;   m[10]+= m[11]* z;   m[14]+= m[15]* z;

    return *this;
}

///////////////////////////////////////////////////////////////////////////////
// uniform scale
///////////////////////////////////////////////////////////////////////////////
template <typename T>
my::CMatrix4<T>& my::CMatrix4<T>::Scale(T s)
{
    return Scale(s, s, s);
}

template <typename T>
my::CMatrix4<T>& my::CMatrix4<T>::Scale(T x, T y, T z)
{
    m[0] *= x;   m[4] *= x;   m[8] *= x;   m[12] *= x;
    m[1] *= y;   m[5] *= y;   m[9] *= y;   m[13] *= y;
    m[2] *= z;   m[6] *= z;   m[10]*= z;   m[14] *= z;
    return *this;
}


template <>
my::CMatrix4<double>& my::CMatrix4<double>::Rotate(double angle, double x, double y, double z)
{
    double c = cos(MyMath::DegreesToRadians(angle));    // cosine
    double s = sin(MyMath::DegreesToRadians(angle));    // sine
    double c1 = 1.0 - c;                // 1 - c
    double m0 = m[0], m4 = m[4], m8 = m[8], m12 = m[12],
    m1 = m[1],  m5 = m[5],  m9 = m[9],  m13= m[13],
    m2 = m[2],  m6 = m[6],  m10= m[10], m14= m[14];
    
    // build rotation matrix
    double r0 = x * x * c1 + c;
    double r1 = x * y * c1 + z * s;
    double r2 = x * z * c1 - y * s;
    double r4 = x * y * c1 - z * s;
    double r5 = y * y * c1 + c;
    double r6 = y * z * c1 + x * s;
    double r8 = x * z * c1 + y * s;
    double r9 = y * z * c1 - x * s;
    double r10 = z * z * c1 + c;
    
    // multiply rotation matrix
    m[0] = r0 * m0 + r4 * m1 + r8 * m2;
    m[1] = r1 * m0 + r5 * m1 + r9 * m2;
    m[2] = r2 * m0 + r6 * m1 + r10* m2;
    m[4] = r0 * m4 + r4 * m5 + r8 * m6;
    m[5] = r1 * m4 + r5 * m5 + r9 * m6;
    m[6] = r2 * m4 + r6 * m5 + r10* m6;
    m[8] = r0 * m8 + r4 * m9 + r8 * m10;
    m[9] = r1 * m8 + r5 * m9 + r9 * m10;
    m[10]= r2 * m8 + r6 * m9 + r10* m10;
    m[12]= r0 * m12+ r4 * m13+ r8 * m14;
    m[13]= r1 * m12+ r5 * m13+ r9 * m14;
    m[14]= r2 * m12+ r6 * m13+ r10* m14;
    
    return *this;
}

///////////////////////////////////////////////////////////////////////////////
// build a rotation matrix with given angle(degree) and rotation axis, then
// multiply it with this object
///////////////////////////////////////////////////////////////////////////////
template <>
my::CMatrix4<double>& my::CMatrix4<double>::Rotate(double angle, const my::CVector3<double>& axis)
{
    return Rotate(angle, axis.x(), axis.y(), axis.z());
}
