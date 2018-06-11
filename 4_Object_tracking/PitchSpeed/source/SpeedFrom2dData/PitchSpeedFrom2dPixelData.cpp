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

#include <iostream>
#include <fstream>

#include <rapidjson/rapidjson.h>
#include <rapidjson/document.h>

#include "../Common.h"
#include "../Logger.h"
#include "../FileHelper.h"
#include "../ViewProjectionMatrix.h"
#include "../CommandLineArguments.h"
#include "../UnitConversion.h"
#include "../MyAnalytics.h"

#include "../GamedayServer.h"

// The type that holds the camera transform matrices (view, projection and viewport matrices).
my::CViewProjectionMatrix<double> m_viewProjectionMatrix;

// The array of pitch 2D coordinates (input).
std::vector<my::CVector3<double> > m_inputPixelArray;
// The array of pitch 3D coordinates (output).
std::vector<my::CVector3<double> > m_outputPointArray;

// The identifier of the pitch in the MLBAM database.
std::string m_sportvisionPitchId = my::Null<std::string>();
// The speed of the pitch in the MLBAM database.
double m_sportvisionReleaseSpeed = my::Null<double>();
// The pitcher handedness.
std::string m_pitcherThrows = "R";

// Prints the error messages (if any) and exits.
int OnDestroy()
{
    return EXIT_FAILURE;
}

// Loads the camera transform from a JSON file and fills the transform matrices (the "m_viewProjectionMatrix" structure). The JSON files should be given in the Umpire Evaluation Tool standard for camera files.
bool LoadCameraTransformFromJson(std::string cameraTransformFileName)
{
    std::string jsonString;

    my::file::GetFileAsString(cameraTransformFileName, jsonString);

    rapidjson::Document document;

    HEALTH_CHECK(document.Parse<0>(jsonString.c_str()).HasParseError(), false);

    HEALTH_CHECK(!document.IsObject(), false);

    my::int32 viewport[4] = { 0 };

    double opticalCenter[3] = { 0 },
        opticalAxis[3] = { 0 },
        upVector[3] = { 0 };

    double fieldOfView = my::Null<double>(),
        nearClippingPlane = my::Null<double>(),
        farClippingPlane = my::Null<double>(),
        aspectRatio = my::Null<double>();

    // viewport

    HEALTH_CHECK(!document.HasMember("viewport"), false);

    const rapidjson::Value& viewportParameterArrayHandle = document["viewport"];

    HEALTH_CHECK(!viewportParameterArrayHandle.IsArray(), false);
    HEALTH_CHECK(viewportParameterArrayHandle.Size() != 4, false);

    for (rapidjson::SizeType viewportParameterIndex = 0; viewportParameterIndex < viewportParameterArrayHandle.Size(); ++viewportParameterIndex)
    {
        const rapidjson::Value& viewportParameterHandle = viewportParameterArrayHandle[viewportParameterIndex];

        HEALTH_CHECK(!viewportParameterHandle.IsInt(), false);

        viewport[viewportParameterIndex] = viewportParameterHandle.GetInt();
    }

    // optical center

    HEALTH_CHECK(!document.HasMember("optical_center"), false);

    const rapidjson::Value& opticalCenterHandle = document["optical_center"];

    HEALTH_CHECK(!opticalCenterHandle.IsArray(), false);
    HEALTH_CHECK(opticalCenterHandle.Size() != 3, false);

    for (rapidjson::SizeType opticalCenterParameterIndex = 0; opticalCenterParameterIndex < opticalCenterHandle.Size(); ++opticalCenterParameterIndex)
    {
        const rapidjson::Value& opticalCenterParameterHandle = opticalCenterHandle[opticalCenterParameterIndex];

        HEALTH_CHECK(!opticalCenterParameterHandle.IsDouble(), false);

        opticalCenter[opticalCenterParameterIndex] = opticalCenterParameterHandle.GetDouble();
    }

    // optical axis

    HEALTH_CHECK(!document.HasMember("optical_axis"), false);

    const rapidjson::Value& opticalAxisHandle = document["optical_axis"];

    HEALTH_CHECK(!opticalAxisHandle.IsArray(), false);
    HEALTH_CHECK(opticalAxisHandle.Size() != 3, false);

    for (rapidjson::SizeType opticalAxisParameterIndex = 0; opticalAxisParameterIndex < opticalAxisHandle.Size(); ++opticalAxisParameterIndex)
    {
        const rapidjson::Value& opticalAxisParameterHandle = opticalAxisHandle[opticalAxisParameterIndex];

        HEALTH_CHECK(!opticalAxisParameterHandle.IsDouble(), false);

        opticalAxis[opticalAxisParameterIndex] = opticalAxisParameterHandle.GetDouble();
    }

    // up vector

    HEALTH_CHECK(!document.HasMember("up_vector"), false);

    const rapidjson::Value& upVectorHandle = document["up_vector"];

    HEALTH_CHECK(!upVectorHandle.IsArray(), false);
    HEALTH_CHECK(upVectorHandle.Size() != 3, false);

    for (rapidjson::SizeType upVectorParameterIndex = 0; upVectorParameterIndex < upVectorHandle.Size(); ++upVectorParameterIndex)
    {
        const rapidjson::Value& upVectorParameterHandle = upVectorHandle[upVectorParameterIndex];

        HEALTH_CHECK(!upVectorParameterHandle.IsDouble(), false);

        upVector[upVectorParameterIndex] = upVectorParameterHandle.GetDouble();
    }

    // field of view

    HEALTH_CHECK(!document.HasMember("field_of_view"), false);

    const rapidjson::Value& fieldOfViewHandle = document["field_of_view"];

    HEALTH_CHECK(!fieldOfViewHandle.IsDouble(), false);

    fieldOfView = fieldOfViewHandle.GetDouble();

    // near clipping plane

    HEALTH_CHECK(!document.HasMember("near_clipping_plane"), false);

    const rapidjson::Value& nearClippingPlaneHandle = document["near_clipping_plane"];

    HEALTH_CHECK(!nearClippingPlaneHandle.IsNumber(), false);

    nearClippingPlane = nearClippingPlaneHandle.GetDouble();

    // far clipping plane

    HEALTH_CHECK(!document.HasMember("far_clipping_plane"), false);

    const rapidjson::Value& farClippingPlaneHandle = document["far_clipping_plane"];

    HEALTH_CHECK(!farClippingPlaneHandle.IsNumber(), false);

    farClippingPlane = farClippingPlaneHandle.GetDouble();

    // aspect ratio

    HEALTH_CHECK(!document.HasMember("aspect_ratio"), false);

    const rapidjson::Value& aspectRatioHandle = document["aspect_ratio"];

    HEALTH_CHECK(!aspectRatioHandle.IsDouble(), false);

    aspectRatio = aspectRatioHandle.GetDouble();

    // ADAPTOR FOR GLULOOKAT
    double center[3] = { opticalCenter[0] + opticalAxis[0], opticalCenter[1] + opticalAxis[1], opticalCenter[2] + opticalAxis[2] };

    if (!m_viewProjectionMatrix.SetViewMatrix(my::CVector3<double>(opticalCenter[0], opticalCenter[1], opticalCenter[2]), my::CVector3<double>(center[0], center[1], center[2]), my::CVector3<double>(upVector[0], upVector[1], upVector[2])))
    {
        LOG_ERROR();

        return false;
    }

    if (!m_viewProjectionMatrix.SetProjectionMatrix(fieldOfView, aspectRatio, nearClippingPlane, farClippingPlane))
    {
        LOG_ERROR();

        return false;
    }

    m_viewProjectionMatrix.SetViewport(viewport[0], viewport[1], viewport[2], viewport[3]);

    return true;
}

// Prints the transform matrices (debugging only).
void PrintTransformMatrices()
{
    std::cout << "VIEW MATRIX" << std::endl;

    my::CMatrix4<double> vm = m_viewProjectionMatrix.GetViewMatrix();
    
    std::cout << std::setprecision(15) << vm[0] << " " << vm[1] << " " << vm[2] << " " << vm[3] << " " << std::endl
        << vm[4] << " " << vm[5] << " " << vm[6] << " " << vm[7] << " " << std::endl
        << vm[8] << " " << vm[9] << " " << vm[10] << " " << vm[11] << " " << std::endl
        << vm[12] << " " << vm[13] << " " << vm[14] << " " << vm[15] << " " << std::endl;
    
    std::cout << "PROJECTION MATRIX" << std::endl;
    
    my::CMatrix4<double> pm = m_viewProjectionMatrix.GetProjectionMatrix();
    
    std::cout << std::setprecision(15) << pm[0] << " " << pm[1] << " " << pm[2] << " " << pm[3] << " " << std::endl
        << pm[4] << " " << pm[5] << " " << pm[6] << " " << pm[7] << " " << std::endl
        << pm[8] << " " << pm[9] << " " << pm[10] << " " << pm[11] << " " << std::endl
        << pm[12] << " " << pm[13] << " " << pm[14] << " " << pm[15] << " " << std::endl;
    
    std::cout << "VIEW-PROJECTION MATRIX" << std::endl;
    
    my::CMatrix4<double> vpm = m_viewProjectionMatrix.GetViewProjectionMatrix();
    
    std::cout << std::setprecision(15) << vpm[0] << " " << vpm[1] << " " << vpm[2] << " " << vpm[3] << " " << std::endl
        << vpm[4] << " " << vpm[5] << " " << vpm[6] << " " << vpm[7] << " " << std::endl
        << vpm[8] << " " << vpm[9] << " " << vpm[10] << " " << vpm[11] << " " << std::endl
        << vpm[12] << " " << vpm[13] << " " << vpm[14] << " " << vpm[15] << " " << std::endl;
    
    std::cout << "VIEW-PROJECTION-INVERSE MATRIX" << std::endl;
    
    my::CMatrix4<double> vpmi = m_viewProjectionMatrix.GetViewProjectionInverseMatrix();
    
    std::cout << std::setprecision(15) << vpmi[0] << " " << vpmi[1] << " " << vpmi[2] << " " << vpmi[3] << " " << std::endl
        << vpmi[4] << " " << vpmi[5] << " " << vpmi[6] << " " << vpmi[7] << " " << std::endl
        << vpmi[8] << " " << vpmi[9] << " " << vpmi[10] << " " << vpmi[11] << " " << std::endl
        << vpmi[12] << " " << vpmi[13] << " " << vpmi[14] << " " << vpmi[15] << " " << std::endl;
}

// Loads the array of pixels from a JSON file and fills the array of pitch 2D coordinates (the "m_inputPixelArray" structure).
bool LoadInputPixels(std::string pixelsFileName, std::string cameraName)
{
    std::cout << "OPENING: " << pixelsFileName << std::endl;

    m_inputPixelArray.clear();

    m_sportvisionPitchId = my::Null<std::string>();

    std::string jsonString;

    my::file::GetFileAsString(pixelsFileName, jsonString);

    rapidjson::Document document;

    HEALTH_CHECK(document.Parse<0>(jsonString.c_str()).HasParseError(), false);

    HEALTH_CHECK(!document.IsObject(), false);

    HEALTH_CHECK(!document.HasMember(cameraName.c_str()), false);

    const rapidjson::Value& linkHandle = document["link"];

    HEALTH_CHECK(!linkHandle.IsString(), false);

    std::string link = linkHandle.GetString();

    if (link.find_last_of("/") != std::string::npos)
        m_sportvisionPitchId = link.substr(link.find_last_of("/") + 1, std::string("??????_??????").size());

    const rapidjson::Value& cameraHandle = document[cameraName.c_str()];

    HEALTH_CHECK(!cameraHandle.IsObject(), false);

    for (my::int32 frameIndex = 11; frameIndex <= 21; ++frameIndex)
    {
        std::string frameName = "frame "
            + my::NumberToString(frameIndex)
            + ".0";

        my::CVector3<double> point;

        if (cameraHandle.HasMember(frameName.c_str()))
        {
            const rapidjson::Value& frameHandle = cameraHandle[frameName.c_str()];

            HEALTH_CHECK(!frameHandle.IsObject(), false);

            const rapidjson::Value& xHandle = frameHandle["x"];

            HEALTH_CHECK(!xHandle.IsNumber(), false);

            const rapidjson::Value& yHandle = frameHandle["y"];

            HEALTH_CHECK(!yHandle.IsNumber(), false);

            point.Set(xHandle.GetDouble(), yHandle.GetDouble(), 0);
        }
        
        m_inputPixelArray.push_back(point);
    }

    return true;
}

// Loads the information about the pitches of this game from the public MLBAM server.
bool LoadPitchFx(std::string gameDirectoryName)
{
    m_sportvisionReleaseSpeed = my::Null<double>();

    m_pitcherThrows = "R";

    if (my::IsNull(m_sportvisionPitchId))
        return true;

    std::vector<mlb::io::CPitch> pitchArray = mlb::io::CGamedayServer().GetPitchArray(gameDirectoryName);

    for (std::vector<mlb::io::CPitch>::const_iterator pitchIterator = pitchArray.begin(); pitchIterator != pitchArray.end(); ++pitchIterator)
    {
        if (pitchIterator->GetSportvisionPitchId() == m_sportvisionPitchId)
        {
            m_sportvisionReleaseSpeed = UnitConversion::FeetPerSecondToMilesPerHour(pitchIterator->GetSpeedAt50Feet());

            m_pitcherThrows = pitchIterator->GetPitcherThrows();
        }
    }

    return true;
}

// Computes the inverse camera transform from a 2D point (a point in the camera view frustum) to a 3D point (a point in the world).
bool UnProject(my::CVector3<double> screen, my::CVector3<double>& world)
{
    my::CVector4<my::int32> viewport(m_viewProjectionMatrix.GetViewport());

    my::int32 width = viewport[2],
        height = viewport[3];

    HEALTH_CHECK(my::IsNull(width), false);
    HEALTH_CHECK(my::IsNull(height), false);

    my::CMatrix4<double> vpmi = m_viewProjectionMatrix.GetViewProjectionInverseMatrix();

    my::CVector4<double> screenNormalized(screen.x(), (double)height - screen.y(), screen.z(), 1.0);

    // Map x,y from [0, width/height] to [0,1]
    screenNormalized[0] = screenNormalized[0] / width;
    screenNormalized[1] = screenNormalized[1] / height;

    // Map from [0,1] to [-1, 1]
    screenNormalized[0] = screenNormalized[0] * 2.0 - 1.0;
    screenNormalized[1] = screenNormalized[1] * 2.0 - 1.0;
    screenNormalized[2] = screenNormalized[2] * 2.0 - 1.0;

    double cache[4] = { 0 };

    cache[0] = vpmi(0, 0) * screenNormalized.x() + vpmi(1, 0) * screenNormalized.y() + vpmi(2, 0) * screenNormalized.z() + vpmi(3, 0);
    cache[1] = vpmi(0, 1) * screenNormalized.x() + vpmi(1, 1) * screenNormalized.y() + vpmi(2, 1) * screenNormalized.z() + vpmi(3, 1);
    cache[2] = vpmi(0, 2) * screenNormalized.x() + vpmi(1, 2) * screenNormalized.y() + vpmi(2, 2) * screenNormalized.z() + vpmi(3, 2);
    cache[3] = vpmi(0, 3) * screenNormalized.x() + vpmi(1, 3) * screenNormalized.y() + vpmi(2, 3) * screenNormalized.z() + vpmi(3, 3);

    if (MyMath::IsZero(screenNormalized[3]))
        return false;

    world[0] = cache[0] / cache[3];
    world[1] = cache[1] / cache[3];
    world[2] = cache[2] / cache[3];

    return true;
}

// Computes the 3D points associated to input pixel array.
bool UnProject(std::vector<my::CVector3<double> > inputPixelArray, std::vector<my::CVector3<double> >& outputPointArray)
{
    // The goal is to compute, for each input pixel, the intersection between (1) the ray starting at the center of projection of the camera and going through the pixel over the camera's near plane and (2) a plane defined by a point on the tip of the home plate ((0, 0, 0)) and a normal vector aligned in the 3rd base to the 1st base direction ((50, 0, 0)). The plane may be seen as the plane where all the pitch coordinates lie. 
    double normal[3] = { 50.0, 0.0, 0.0 },
        point[3] = { 0.0, 0.0, 0.0 },
        intersection[3] = { 0 };

    // A trick to improve the results of the 2D to 3D conversion. The pitcher does not throw the ball exactly over the supporting plane (i. e., exactly over the line that connects the pitching mound to the tip of the home plate), but a little bit to the side of the supporting plane, depending on the pitcher's handedness. These values, "2.001332573" for right-handed pitchers and "-1.919243909" for left-handed pitchers are the averages of the release positions for all pitches on the MLBAM database. They "rotate" the plane a little bit to place it closer to the actual position where the pitcher released the ball.
    if (m_pitcherThrows == "R")
        normal[1] = 2.001332573;
    else
        normal[1] = -1.919243909;

    MyMath::Normalize3(normal);

    outputPointArray.clear();

    // For each of the pixels, ...
    for (std::vector<my::CVector3<double> >::const_iterator inputPixelIterator = inputPixelArray.begin(); inputPixelIterator != inputPixelArray.end(); ++inputPixelIterator)
    {
        my::CVector3<double> outputPoint;

        if (inputPixelIterator->IsValid())
        {
            my::CVector3<double> world;

            double s0[3] = { 0 };

            // ... computes the "origin" of the ray (the projection of the pixel in the near plane), ...
            UnProject(my::CVector3<double>(inputPixelIterator->x(), inputPixelIterator->y(), 0), world);

            s0[0] = world.x();
            s0[1] = world.y();
            s0[2] = world.z();

            double s1[3] = { 0 };

            // ...  and the "destination" of the ray (the projection of the pixel in the far plane). The ray is then represented as a line segment [s0, s1] that goes through the view frustum.
            UnProject(my::CVector3<double>(inputPixelIterator->x(), inputPixelIterator->y(), 1), world);

            s1[0] = world.x();
            s1[1] = world.y();
            s1[2] = world.z();

            // Computes the intersection between the line segment and the plane. Whatever the pixel, there should be a valid intersection.
            if (MyMath::SegmentPlaneIntersection3(s0, s1, normal, point, intersection) != 1)
            {
                LOG_ERROR();

                return false;
            }

            outputPoint.Set(intersection[0], intersection[1], intersection[2]);
        }

        outputPointArray.push_back(outputPoint);
    }

    return true;
}

// Exports the moments of the error distribution to a file.
bool ExportErrorDistribution(std::string errorDistributionFileName)
{
    std::ifstream errorDistributionFileStream(errorDistributionFileName);

    HEALTH_CHECK(!errorDistributionFileStream.is_open(), false);

    char csvString[4069] = { 0 };

    // HEADER
    errorDistributionFileStream.getline(csvString, 4096);

    std::vector<double> errorArray;

    while (!errorDistributionFileStream.eof())
    {
        errorDistributionFileStream.getline(csvString, 4096);

        std::vector<std::string> columnAsStringArray;

        boost::split(columnAsStringArray, csvString, boost::is_any_of(";\n"));

        if (columnAsStringArray.size() > 3)
        {
            double sportvisionSpeedAt50Feet = my::Null<double>(),
                estimatedSpeedAt50Feet = my::Null<double>();

            //sportvision_pitch_id

            //sportvision_speed_at_50_feet
            if (!columnAsStringArray[1].empty() &&
                !my::IsNull(columnAsStringArray[1]))
            {
                sportvisionSpeedAt50Feet = my::StringToNumber<double>(columnAsStringArray[1]);
            }

            //linear_regression_start_speed
            if (!columnAsStringArray[2].empty() &&
                !my::IsNull(columnAsStringArray[2]))
            {
                estimatedSpeedAt50Feet = my::StringToNumber<double>(columnAsStringArray[2]);
            }

            //linear_regression_error
            //s0;s1;s2;s3;s4;s5;s6;s7;s8;s9;

            if (!my::IsNull(sportvisionSpeedAt50Feet) &&
                !my::IsNull(estimatedSpeedAt50Feet))
            {
                double error = fabs(sportvisionSpeedAt50Feet - estimatedSpeedAt50Feet);

                errorArray.push_back(error);
            }
        }
    }

    if (!errorArray.empty())
    {
        my::analytics::CMomentsOfDistribution momentsOfDistribution;

        if (!my::analytics::Moment(errorArray, momentsOfDistribution))
        {
            LOG_ERROR();

            return false;
        }

        std::string distributionFileName = errorDistributionFileName + "_DISTRIBUTION.csv";

        std::ofstream distributionFileStream(distributionFileName);

        HEALTH_CHECK(!distributionFileStream.is_open(), false);

        distributionFileStream << "number_of_scores" << ";" << momentsOfDistribution.GetNumberOfScoresInSample() << std::endl
            << "mean" << ";" << momentsOfDistribution.GetMean() << std::endl
            << "stddev" << ";" << momentsOfDistribution.GetStandardDeviation() << std::endl
            << "0th quartile" << ";" << momentsOfDistribution.Get0thQuartile() << std::endl
            << "1st quartile" << ";" << momentsOfDistribution.Get1stQuartile() << std::endl
            << "2nd quartile" << ";" << momentsOfDistribution.Get2ndQuartile() << std::endl
            << "3rd quartile" << ";" << momentsOfDistribution.Get3rdQuartile() << std::endl
            << "4th quartile" << ";" << momentsOfDistribution.Get4thQuartile() << std::endl
            << "IQR" << ";" << momentsOfDistribution.Get3rdQuartile() - momentsOfDistribution.Get1stQuartile() << std::endl
            << "IQR/stddev" << ";" << (momentsOfDistribution.Get3rdQuartile() - momentsOfDistribution.Get1stQuartile()) / momentsOfDistribution.GetStandardDeviation() << std::endl;
    }

    return true;
}

int main(int argumentCount, char **argumentArray)
{
    my::CCommandLineArguments commandLineArguments;

    // IO

    commandLineArguments.AddParameter("CAMERA_CALIBRATION");
    commandLineArguments.AddParameter("CAMERA_A_2D_COORDS");
    commandLineArguments.AddParameter("CAMERA_B_2D_COORDS");
    commandLineArguments.AddParameter("GAMEDAY_URL");
    commandLineArguments.AddParameter("CLEAR_OUTPUT_CSV_FILE");
    commandLineArguments.AddParameter("OUTPUT_CSV_FILE");
    commandLineArguments.AddParameter("DISTRIBUTION_OF_OUTPUT_CSV_FILE");

    if (!commandLineArguments.Initialize(argumentCount, argumentArray))
    {
        commandLineArguments.Print();

        LOG_ERROR();

        return OnDestroy();
    }

    if (commandLineArguments.HasParameter("CAMERA_CALIBRATION"))
    {
        if (!LoadCameraTransformFromJson(commandLineArguments.ToString("CAMERA_CALIBRATION")))
        {
            LOG_ERROR();

            return OnDestroy();
        }
    }

    if (commandLineArguments.HasParameter("CAMERA_A_2D_COORDS"))
    {
        if (!LoadInputPixels(commandLineArguments.ToString("CAMERA_A_2D_COORDS"), "camera_a"))
        {
            LOG_ERROR();

            return OnDestroy();
        }
    }

    if (commandLineArguments.HasParameter("CAMERA_B_2D_COORDS"))
    {
        if (!LoadInputPixels(commandLineArguments.ToString("CAMERA_B_2D_COORDS"), "camera_b"))
        {
            LOG_ERROR();

            return OnDestroy();
        }
    }

    if (commandLineArguments.HasParameter("GAMEDAY_URL"))
    {
        if (!LoadPitchFx(commandLineArguments.ToString("GAMEDAY_URL")))
        {
            LOG_ERROR();

            return OnDestroy();
        }
    }

    //PrintTransformMatrices();

    // COMPUTES THE PITCH SPEED AS A LINEAR REGRESSION OF THE INSTANTANEOUS SPEEDS OF THE 3D POINTS.

    double linearRegressionReleaseSpeed = my::Null<double>(),
        linearRegressionError = my::Null<double>();

    // The array of instantaneous speeds (one for each 3D point).
    std::vector<double> yArray;

    if (!m_inputPixelArray.empty())
    {
        if (!UnProject(m_inputPixelArray, m_outputPointArray))
        {
            LOG_ERROR();

            return OnDestroy();
        }

        my::CVector3<double> lastPoint;

        // Time between frames, given in seconds (assuming 30 FPS)
        double dtInSec = (1000.0 / 30.0) / 1000.0,
            // The time elapsed from the last frame.
            timeElapsedInSec = 0;

        // The array of frame indices (one for each 3D point).
        std::vector<double> xArray;

        my::int32 frameIndex = 0;

        for (std::vector<my::CVector3<double> >::const_iterator outputPointIterator = m_outputPointArray.begin(); outputPointIterator != m_outputPointArray.end(); ++outputPointIterator)
        {
            if (outputPointIterator->IsValid())
            {
                if (lastPoint.IsValid())
                {
                    double speedInMph = UnitConversion::FeetPerSecondToMilesPerHour(lastPoint.Distance(*outputPointIterator) / timeElapsedInSec);

                    timeElapsedInSec = 0;

                    // BUG: (23-Feb-2018) REMOVES PHYSICALLY IMPOSSIBLE SPEEDS!
                    if ((speedInMph > 60.0) &&
                        (speedInMph < 105.0))
                    {
                        xArray.push_back(frameIndex);
                        yArray.push_back(speedInMph);
                    }
                }

                lastPoint = (*outputPointIterator);
            }

            // BUG: (20-Feb-2018) IF THERE IS A VALID POINT IN THE PAST, THE TIME TO THAT POINT SHOULD BE INCREASED; IF NOT, LET IT BE.
            if (lastPoint.IsValid())
                timeElapsedInSec += dtInSec;

            ++frameIndex;
        }

        if (xArray.size() > 1)
        {
            int n = (int)xArray.size();

            double a = 0,
                b = 0,
                r = 0;

            if (!MyMath::LinearRegression(xArray.data(), yArray.data(), n, &a, &b, &r))
            {
                LOG_ERROR();

                return OnDestroy();
            }

            linearRegressionReleaseSpeed = a;
            linearRegressionError = r;
        }
    }

    // WRITES THE OUTPUT FILE

    if (commandLineArguments.HasParameter("CLEAR_OUTPUT_CSV_FILE"))
    {
        std::ofstream outputFileName(commandLineArguments.ToString("CLEAR_OUTPUT_CSV_FILE"));

        if (outputFileName.is_open())
        {
            outputFileName << "sportvision_pitch_id;";
            outputFileName << "sportvision_speed_at_50_feet;";
            outputFileName << "linear_regression_start_speed;";
            outputFileName << "linear_regression_error;";
            outputFileName << "s0;s1;s2;s3;s4;s5;s6;s7;s8;s9;";
            outputFileName << std::endl;
        }
    }

    if (commandLineArguments.HasParameter("OUTPUT_CSV_FILE"))
    {
        std::ofstream outputFileName(commandLineArguments.ToString("OUTPUT_CSV_FILE"), std::ofstream::out | std::ofstream::app);

        if (outputFileName.is_open())
        {
            //sportvision_pitch_id
            if (!my::IsNull(m_sportvisionPitchId))
                outputFileName << m_sportvisionPitchId;
            outputFileName << ";";

            //sportvision_speed_at_50_feet
            if (!my::IsNull(m_sportvisionReleaseSpeed))
                outputFileName << m_sportvisionReleaseSpeed;
            outputFileName << ";";

            //linear_regression_start_speed
            if (!my::IsNull(linearRegressionReleaseSpeed))
                outputFileName << linearRegressionReleaseSpeed;
            outputFileName << ";";

            //linear_regression_error
            if (!my::IsNull(linearRegressionError))
                outputFileName << linearRegressionError;
            outputFileName << ";";

            //s0;s1;s2;s3;s4;s5;s6;s7;s8;s9;
            for (my::int32 i = 0; i < 10; ++i)
            {
                if (yArray.size() > i)
                    outputFileName << yArray[i];
                outputFileName << ";";
            }

            outputFileName << std::endl;
        }
    }

    if (commandLineArguments.HasParameter("DISTRIBUTION_OF_OUTPUT_CSV_FILE"))
    {
        if (!ExportErrorDistribution(commandLineArguments.ToString("DISTRIBUTION_OF_OUTPUT_CSV_FILE")))
        {
            LOG_ERROR();

            return OnDestroy();
        }
    }

    return EXIT_SUCCESS;
}
