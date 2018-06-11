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

#include "Logger.h"

#include "ViewProjectionMatrix.h"

/**
*/
template <>
my::CViewProjectionMatrix<double>::CViewProjectionMatrix()
{
    Create();
}

/**
*/
template <>
my::CViewProjectionMatrix<double>::CViewProjectionMatrix(const CViewProjectionMatrix<double> &viewProjectionMatrix)
{
    Copy(viewProjectionMatrix);
}

/**
*/
template <>
my::CViewProjectionMatrix<double>& my::CViewProjectionMatrix<double>::operator=(const my::CViewProjectionMatrix<double> &viewProjectionMatrix)
{
    Copy(viewProjectionMatrix);

    return *this;
}

/**
*/
template <typename T>
void my::CViewProjectionMatrix<T>::LoadIdentity()
{
    m_viewMatrix.identity();
    m_projectionMatrix.identity();
}

/**
*/
template <>
bool my::CViewProjectionMatrix<double>::SetViewMatrix(my::CMatrix4<double> viewMatrix)
{
    m_viewMatrix = viewMatrix;

    return true;
}

// http://gamedev.stackexchange.com/questions/72565/3d-camera-rotation
template <>
bool my::CViewProjectionMatrix<double>::SetViewMatrix(my::CVector3<double> opticalCenter, my::CVector3<double> referencePoint, my::CVector3<double> up)
{
    // optical axis
    my::CVector3<double> z = (opticalCenter - referencePoint).Normalize(),
        x = up.Cross(z).Normalize(),
        y = z.Cross(x);

    // x0 x1 x2 t0    0  4  8 12    00 01 02 03
    // y0 y1 y2 t1    1  5  9 13    10 11 12 13
    // z0 z1 z2 t2    2  6 10 14    20 21 22 23
    //  0  0  0  1    3  7 11 15    30 31 32 33

    my::CMatrix4<double> orientation(
        x[0], y[0], z[0],  0,
        x[1], y[1], z[1],  0,
        x[2], y[2], z[2],  0,
           0,    0,    0,  1);

    my::CMatrix4<double> translation(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        -opticalCenter[0], -opticalCenter[1], -opticalCenter[2], 1);

    m_viewMatrix = orientation * translation;

    return true;
}

/**
*/
template <>
my::CMatrix4<double> my::CViewProjectionMatrix<double>::GetViewMatrix() const
{
    return m_viewMatrix;
}

/**
*/
template <>
bool my::CViewProjectionMatrix<double>::SetProjectionMatrix(my::CMatrix4<double> projectionMatrix)
{
    m_projectionMatrix = projectionMatrix;

    return true;
}

// http://www.songho.ca/opengl/gl_matrix.html
template <>
bool my::CViewProjectionMatrix<double>::SetProjectionMatrix(double leftClippingPlane, double rightClippingPlane, double bottomClippingPlane, double topClippingPlane, double nearClippingPlane, double farClippingPlane)
{
    m_projectionMatrix.Identity();

    HEALTH_CHECK(MyMath::IsZero(rightClippingPlane - leftClippingPlane), false);
    HEALTH_CHECK(MyMath::IsZero(topClippingPlane - bottomClippingPlane), false);
    HEALTH_CHECK(MyMath::IsZero(farClippingPlane - nearClippingPlane), false);

    // 0  4  8 12    00 01 02 03
    // 1  5  9 13    10 11 12 13
    // 2  6 10 14    20 21 22 23
    // 3  7 11 15    30 31 32 33

    m_projectionMatrix(0, 0) = 2.0 * nearClippingPlane / (rightClippingPlane - leftClippingPlane);
    m_projectionMatrix(1, 1) = 2.0 * nearClippingPlane / (topClippingPlane - bottomClippingPlane);
    m_projectionMatrix(2, 0) = (rightClippingPlane + leftClippingPlane) / (rightClippingPlane - leftClippingPlane);
    m_projectionMatrix(2, 1) = (topClippingPlane + bottomClippingPlane) / (topClippingPlane - bottomClippingPlane);
    m_projectionMatrix(2, 2) = -(farClippingPlane + nearClippingPlane) / (farClippingPlane - nearClippingPlane);
    m_projectionMatrix(2, 3) = -1.0;
    m_projectionMatrix(3, 2) = -(2.0 * farClippingPlane * nearClippingPlane) / (farClippingPlane - nearClippingPlane);
    m_projectionMatrix(3, 3) = 0.0;

    m_leftClippingPlane = leftClippingPlane;
    m_rightClippingPlane = rightClippingPlane;
    m_bottomClippingPlane = bottomClippingPlane;
    m_topClippingPlane = topClippingPlane;

    return true;
}

// http://www.songho.ca/opengl/gl_matrix.html
template <>
bool my::CViewProjectionMatrix<double>::SetProjectionMatrix(double fieldOfView, double aspectRatio, double nearPlane, double farPlane)
{
    // tangent of half fovY
    double tangent = tan(MyMath::DegreesToRadians(fieldOfView / 2.0)),
        // half height of near plane
        height = nearPlane * tangent,
        // half width of near plane
        width = height * aspectRatio;

    if (!SetProjectionMatrix(-width, width, -height, height, nearPlane, farPlane))
    {
        LOG_ERROR();

        return false;
    }

    m_fieldOfView = fieldOfView;
    m_aspectRatio = aspectRatio;

    return true;
}

/**
*/
template <>
my::CMatrix4<double> my::CViewProjectionMatrix<double>::GetProjectionMatrix() const
{
    return m_projectionMatrix;
}

/**
*/
template <>
my::CMatrix4<double> my::CViewProjectionMatrix<double>::GetViewProjectionMatrix() const
{
    return m_projectionMatrix * m_viewMatrix;
}

/**
*/
template <>
my::CMatrix4<double> my::CViewProjectionMatrix<double>::GetViewProjectionInverseMatrix() const
{
    return (m_projectionMatrix * m_viewMatrix).Invert();
}

/**
*/
template <>
void my::CViewProjectionMatrix<double>::SetViewport(my::int32 x, my::int32 y, my::int32 width, my::int32 height)
{
    m_viewport.Set(x, y, width, height);
}

// X, Y, WIDTH, HEIGHT
template <>
my::CVector4<my::int32> my::CViewProjectionMatrix<double>::GetViewport() const
{
    return m_viewport;
}

// http://3dengine.org/Right-up-back_from_modelview
// gluLookAt 'eye'
template <>
my::CVector3<double> my::CViewProjectionMatrix<double>::GetOpticalCenter() const
{
    // x0 x1 x2 t0    0  4  8 12    00 01 02 03
    // y0 y1 y2 t1    1  5  9 13    10 11 12 13
    // z0 z1 z2 t2    2  6 10 14    20 21 22 23
    //  0  0  0  1    3  7 11 15    30 31 32 33

    // https://www.opengl.org/discussion_boards/showthread.php/178484-Extracting-camera-position-from-a-ModelView-Matrix
    my::CMatrix4<double> viewMatrix = m_viewMatrix;

    viewMatrix.Invert();

    return my::CVector3<double>(
        viewMatrix(3, 0) / viewMatrix(3, 3),
        viewMatrix(3, 1) / viewMatrix(3, 3),
        viewMatrix(3, 2) / viewMatrix(3, 3));
}

// gluLookAt 'center' - 'eye' (normalized)
template <>
my::CVector3<double> my::CViewProjectionMatrix<double>::GetOpticalAxis()
{
    // x0 x1 x2 t0    0  4  8 12    00 01 02 03
    // y0 y1 y2 t1    1  5  9 13    10 11 12 13
    // z0 z1 z2 t2    2  6 10 14    20 21 22 23
    //  0  0  0  1    3  7 11 15    30 31 32 33

    return -my::CVector3<double>(m_viewMatrix(0, 2), m_viewMatrix(1, 2), m_viewMatrix(2, 2));
}

// ModelView 'Y'
template <>
my::CVector3<double> my::CViewProjectionMatrix<double>::GetUp()
{
    // x0 x1 x2 t0    0  4  8 12    00 01 02 03
    // y0 y1 y2 t1    1  5  9 13    10 11 12 13
    // z0 z1 z2 t2    2  6 10 14    20 21 22 23
    //  0  0  0  1    3  7 11 15    30 31 32 33

    return my::CVector3<double>(m_viewMatrix(0, 1), m_viewMatrix(1, 1), m_viewMatrix(2, 1));
}

/**
*/
template <>
bool my::CViewProjectionMatrix<double>::SetOpticalCenter(my::CVector3<double> opticalCenter)
{
    my::CVector3<double> opticalAxis = GetOpticalAxis(),
        up = GetUp();

    if (!SetViewMatrix(opticalCenter, opticalCenter + opticalAxis, up))
    {
        LOG_ERROR();

        return false;
    }

    return true;
}

/**
*/
template <>
bool my::CViewProjectionMatrix<double>::SetOpticalAxis(my::CVector3<double> opticalAxis)
{
    my::CVector3<double> opticalCenter = GetOpticalCenter(),
        up = GetUp();

    if (!SetViewMatrix(opticalCenter, opticalCenter + opticalAxis, up))
    {
        LOG_ERROR();

        return false;
    }

    return true;
}

/**
*/
template <>
bool my::CViewProjectionMatrix<double>::SetUp(my::CVector3<double> up)
{
    my::CVector3<double> opticalCenter = GetOpticalCenter(),
        opticalAxis = GetOpticalAxis();

    if (!SetViewMatrix(opticalCenter, opticalCenter + opticalAxis, up))
    {
        LOG_ERROR();

        return false;
    }

    return true;
}

// http://forums.structure.io/t/near-far-value-from-projection-matrix/3757
template <>
double my::CViewProjectionMatrix<double>::GetNearClippingPlane() const
{
    double m22 = -m_projectionMatrix(2, 2),
        m32 = -m_projectionMatrix(3, 2);

    double farClippingPlane = (2.0 * m32) / (2.0 * m22 - 2.0),
        nearClippingPlane = ((m22 - 1.0) * farClippingPlane) / (m22 + 1.0);

    if (!MyMath::IsValid(nearClippingPlane))
        nearClippingPlane = my::Null<double>();

    return nearClippingPlane;
}

// http://forums.structure.io/t/near-far-value-from-projection-matrix/3757
template <>
double my::CViewProjectionMatrix<double>::GetFarClippingPlane() const
{
    double m22 = -m_projectionMatrix(2, 2),
        m32 = -m_projectionMatrix(3, 2);

    double farClippingPlane = (2.0 * m32) / (2.0 * m22 - 2.0);

    if (!MyMath::IsValid(farClippingPlane))
        farClippingPlane = my::Null<double>();

    return farClippingPlane;
}

/**
*/
template <>
bool my::CViewProjectionMatrix<double>::SetNearClippingPlane(double nearClippingPlane)
{
    double farClippingPlane = GetFarClippingPlane();

    if ((nearClippingPlane != my::Null<double>()) &&
        (farClippingPlane != my::Null<double>()) &&
        (m_fieldOfView != my::Null<double>()) &&
        (m_aspectRatio != my::Null<double>()))
    {
        if (!SetProjectionMatrix(m_fieldOfView, m_aspectRatio, nearClippingPlane, farClippingPlane))
        {
            LOG_ERROR();

            return false;
        }
    }

    return true;
}

/**
*/
template <>
bool my::CViewProjectionMatrix<double>::SetFarClippingPlane(double farClippingPlane)
{
    double nearClippingPlane = GetNearClippingPlane();

    if ((nearClippingPlane != my::Null<double>()) &&
        (farClippingPlane != my::Null<double>()) &&
        (m_fieldOfView != my::Null<double>()) &&
        (m_aspectRatio != my::Null<double>()))
    {
        if (!SetProjectionMatrix(m_fieldOfView, m_aspectRatio, nearClippingPlane, farClippingPlane))
        {
            LOG_ERROR();

            return false;
        }
    }

    return true;
}

/**
*/
template <>
double my::CViewProjectionMatrix<double>::GetFieldOfView() const
{
    return m_fieldOfView;
}

/**
*/
template <>
bool my::CViewProjectionMatrix<double>::SetFieldOfView(double fieldOfView)
{
    m_fieldOfView = fieldOfView;

    double nearClippingPlane = GetNearClippingPlane(),
        farClippingPlane = GetFarClippingPlane();

    if ((nearClippingPlane != my::Null<double>()) &&
        (farClippingPlane != my::Null<double>()) &&
        (m_fieldOfView != my::Null<double>()) &&
        (m_aspectRatio != my::Null<double>()))
    {
        if (!SetProjectionMatrix(m_fieldOfView, m_aspectRatio, nearClippingPlane, farClippingPlane))
        {
            LOG_ERROR();

            return false;
        }
    }

    return true;
}

/**
*/
template <>
double my::CViewProjectionMatrix<double>::GetAspectRatio() const
{
    return m_aspectRatio;
}

/**
*/
template <>
bool my::CViewProjectionMatrix<double>::SetAspectRatio(double aspectRatio)
{
    m_aspectRatio = aspectRatio;

    double nearClippingPlane = GetNearClippingPlane(),
        farClippingPlane = GetFarClippingPlane();

    if ((nearClippingPlane != my::Null<double>()) &&
        (farClippingPlane != my::Null<double>()) &&
        (m_fieldOfView != my::Null<double>()) &&
        (m_aspectRatio != my::Null<double>()))
    {
        if (!SetProjectionMatrix(m_fieldOfView, m_aspectRatio, nearClippingPlane, farClippingPlane))
        {
            LOG_ERROR();

            return false;
        }
    }

    return true;
}

// http://www.opengl.org/wiki/GluProject_and_gluUnProject_code
template <>
bool my::CViewProjectionMatrix<double>::ProjectVertex(my::CVector3<double> model, my::CVector3<double>& screen) const
{
    double cache[8] = { 0 };

    // Modelview transform
    cache[0] = m_viewMatrix[0] * model[0] + m_viewMatrix[4] * model[1] + m_viewMatrix[8] * model[2] + m_viewMatrix[12];
    cache[1] = m_viewMatrix[1] * model[0] + m_viewMatrix[5] * model[1] + m_viewMatrix[9] * model[2] + m_viewMatrix[13];
    cache[2] = m_viewMatrix[2] * model[0] + m_viewMatrix[6] * model[1] + m_viewMatrix[10] * model[2] + m_viewMatrix[14];
    cache[3] = m_viewMatrix[3] * model[0] + m_viewMatrix[7] * model[1] + m_viewMatrix[11] * model[2] + m_viewMatrix[15];
    
    // Projection transform, the final row of projection matrix is 
    // always [0 0 -1 0] so we optimize for that.
    cache[4] = m_projectionMatrix[0] * cache[0] + m_projectionMatrix[4] * cache[1] + m_projectionMatrix[8] * cache[2] + m_projectionMatrix[12] * cache[3];
    cache[5] = m_projectionMatrix[1] * cache[0] + m_projectionMatrix[5] * cache[1] + m_projectionMatrix[9] * cache[2] + m_projectionMatrix[13] * cache[3];
    cache[6] = m_projectionMatrix[2] * cache[0] + m_projectionMatrix[6] * cache[1] + m_projectionMatrix[10] * cache[2] + m_projectionMatrix[14] * cache[3];
    cache[7] = -cache[2];

    // The result normalizes between -1 and 1
    if (cache[7] == 0)
        return false;

    cache[7] = 1.0 / cache[7];

    // Perspective division
    cache[4] *= cache[7];
    cache[5] *= cache[7];
    cache[6] *= cache[7];
    
    // Window coordinates
    // Map x, y to range [0, 1]
    screen[0] = (cache[4] * 0.5 + 0.5) * m_viewport[2] + m_viewport[0];
    screen[1] = (cache[5] * 0.5 + 0.5) * m_viewport[3] + m_viewport[1];
    // This is only correct when glDepthRange(0, 1)
    screen[2] = (1.0 + cache[6]) * 0.5;

    return true;
}

// http://www.codng.com/2011/02/gluunproject-for-iphoneios.html
template <>
bool my::CViewProjectionMatrix<double>::UnProjectVertex(my::CVector3<double> screen, my::CVector3<double>& world) const
{
    my::CMatrix4<double> finalMatrix = (m_projectionMatrix * m_viewMatrix).Invert();

    my::CVector4<double> screenNormalized(screen.x(), screen.y(), screen.z(), 1.0);

    // Map x and y from window coordinates
    screenNormalized[0] = (screenNormalized[0] - m_viewport[0]) / m_viewport[2];
    screenNormalized[1] = (screenNormalized[1] - m_viewport[1]) / m_viewport[3];

    // Map to range -1 to 1
    screenNormalized[0] = screenNormalized[0] * 2.0 - 1.0;
    screenNormalized[1] = screenNormalized[1] * 2.0 - 1.0;
    screenNormalized[2] = screenNormalized[2] * 2.0 - 1.0;

    screenNormalized = finalMatrix * screenNormalized;

    if (MyMath::IsZero(screenNormalized[3]))
        return false;

    world[0] = screenNormalized[0] / screenNormalized[3];
    world[1] = screenNormalized[1] / screenNormalized[3];
    world[2] = screenNormalized[2] / screenNormalized[3];

    return true;
}

/**
*/
template <>
void my::CViewProjectionMatrix<double>::Clear()
{
    Create();
}

/**
*/
template <typename T>
void my::CViewProjectionMatrix<T>::Create()
{
    m_viewMatrix.Identity();
    m_projectionMatrix.Identity();
    m_viewport.Set(0, 0, 1, 1);
    m_leftClippingPlane = my::Null<T>();
    m_rightClippingPlane = my::Null<T>();
    m_bottomClippingPlane = my::Null<T>();
    m_topClippingPlane = my::Null<T>();
    m_fieldOfView = my::Null<T>();
    m_aspectRatio = my::Null<T>();
}

/**
*/
template <typename T>
void my::CViewProjectionMatrix<T>::Copy(const CViewProjectionMatrix &viewProjectionMatrix)
{
    m_viewMatrix = viewProjectionMatrix.m_viewMatrix;
    m_projectionMatrix = viewProjectionMatrix.m_projectionMatrix;
    m_viewport = viewProjectionMatrix.m_viewport;
    m_leftClippingPlane = viewProjectionMatrix.m_leftClippingPlane;
    m_rightClippingPlane = viewProjectionMatrix.m_rightClippingPlane;
    m_bottomClippingPlane = viewProjectionMatrix.m_bottomClippingPlane;
    m_topClippingPlane = viewProjectionMatrix.m_topClippingPlane;
    m_fieldOfView = viewProjectionMatrix.m_fieldOfView;
    m_aspectRatio = viewProjectionMatrix.m_aspectRatio;
}

