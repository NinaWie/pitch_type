set(SPEEDFROM2DDATA_INCLUDE
../CommandLineArguments.h
../Common.h
../FileHelper.h
../GamedayServer.h
../Logger.h
../Matrices.h
../Matrix4.h
../MomentsOfDistribution.h
../MyAnalytics.h
../MyMath.h
../Pitch.h
../PitchFxIo.h
../Timestamp.h
../UnitConversion.h
../Vector2.h
../Vector3.h
../Vector4.h
../Vectors.h
../ViewProjectionMatrix.h
)

set(SPEEDFROM2DDATA_SOURCE
../CommandLineArguments.cpp
../FileHelper.cpp
../GamedayServer.cpp
../Matrices.cpp
../Matrix4.cpp
../MyAnalytics.cpp
../Pitch.cpp
../PitchFxIo.cpp
../Timestamp.cpp
../UnitConversion.cpp
../ViewProjectionMatrix.cpp
PitchSpeedFrom2dPixelData.cpp
)

add_executable (SpeedFrom2dData ${SPEEDFROM2DDATA_INCLUDE} ${SPEEDFROM2DDATA_SOURCE})

target_link_libraries (SpeedFrom2dData LINK_PUBLIC ${Boost_LIBRARIES} ${CURL_LIBRARIES} ${PUGIXML_LIBRARY})


