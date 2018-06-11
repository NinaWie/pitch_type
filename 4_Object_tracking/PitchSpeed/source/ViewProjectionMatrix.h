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

#if !defined(VIEW_PROJECTION_MATRIX_INCLUDED)
#define VIEW_PROJECTION_MATRIX_INCLUDED

#include "Vector4.h"
#include "Matrix4.h"

namespace my {
    template <typename T>
    class CViewProjectionMatrix
    {
    public:
        typedef T ValueType;

        CViewProjectionMatrix();
        CViewProjectionMatrix(const CViewProjectionMatrix &viewProjectionMatrix);

        CViewProjectionMatrix& operator=(const CViewProjectionMatrix &viewProjectionMatrix);

        void LoadIdentity();

        bool SetViewMatrix(my::CMatrix4<T> viewMatrix);
        bool SetViewMatrix(my::CVector3<T> opticalCenter, my::CVector3<T> referencePoint, my::CVector3<T> up);
        my::CMatrix4<T> GetViewMatrix() const;

        bool SetProjectionMatrix(my::CMatrix4<T> projectionMatrix);
        bool SetProjectionMatrix(T leftClippingPlane, T rightClippingPlane, T bottomClippingPlane, T topClippingPlane, T nearClippingPlane, T farClippingPlane);
        bool SetProjectionMatrix(T fieldOfView, T aspectRatio, T nearClippingPlane, T farClippingPlane);
        my::CMatrix4<T> GetProjectionMatrix() const;

        my::CMatrix4<T> GetViewProjectionMatrix() const;
        my::CMatrix4<T> GetViewProjectionInverseMatrix() const;

        void SetViewport(my::int32 x, my::int32 y, my::int32 width, my::int32 height);
        // X, Y, WIDTH, HEIGHT
        my::CVector4<my::int32> GetViewport() const;

        // gluLookAt 'eye'
        my::CVector3<T> GetOpticalCenter() const;
        bool SetOpticalCenter(my::CVector3<T> opticalCenter);

        // gluLookAt 'center' - 'eye' (normalized)
        my::CVector3<T> GetOpticalAxis();
        bool SetOpticalAxis(my::CVector3<T> opticalAxis);

        // ModelView 'Y'
        my::CVector3<T> GetUp();
        bool SetUp(my::CVector3<T> up);

        T GetNearClippingPlane() const;
        bool SetNearClippingPlane(T nearClippingPlane);

        T GetFarClippingPlane() const;
        bool SetFarClippingPlane(T farClippingPlane);

        T GetFieldOfView() const;
        bool SetFieldOfView(T fieldOfView);

        T GetAspectRatio() const;
        bool SetAspectRatio(T aspectRatio);

        bool ProjectVertex(my::CVector3<T> model, my::CVector3<T>& screen) const;
        bool UnProjectVertex(my::CVector3<T> screen, my::CVector3<T>& world) const;

        void Clear();

    private:
        void Create();
        void Copy(const CViewProjectionMatrix &viewProjectionMatrix);

    protected:
        my::CMatrix4<T> m_viewMatrix;
        my::CMatrix4<T> m_projectionMatrix;

        // X, Y, WIDTH, HEIGHT
        my::CVector4<my::int32> m_viewport;

        // TODO: (25-Sep-2015) FIND A WAY TO DERIVE IT FROM THE MATRICES!
        T m_leftClippingPlane;
        // TODO: (25-Sep-2015) FIND A WAY TO DERIVE IT FROM THE MATRICES!
        T m_rightClippingPlane;
        // TODO: (25-Sep-2015) FIND A WAY TO DERIVE IT FROM THE MATRICES!
        T m_bottomClippingPlane;
        // TODO: (25-Sep-2015) FIND A WAY TO DERIVE IT FROM THE MATRICES!
        T m_topClippingPlane;

        // TODO: (25-Sep-2015) FIND A WAY TO DERIVE IT FROM THE MATRICES!
        T m_fieldOfView;

        // TODO: (25-Sep-2015) FIND A WAY TO DERIVE IT FROM THE MATRICES!
        T m_aspectRatio;
    };
}; // my

#endif //#if !defined(VIEW_PROJECTION_MATRIX_INCLUDED)

