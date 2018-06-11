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

#include <boost/filesystem.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <pugixml.hpp>

#include "Common.h"
#include "Logger.h"
#include "FileHelper.h"
#include "UnitConversion.h"

#include "Pitch.h"

#include "PitchFxIo.h"

/**
*/
mlb::io::CPitchFxIo::CPitchFxIo()
{
    Create();
}

/**
*/
mlb::io::CPitchFxIo::CPitchFxIo(const CPitchFxIo& pitchFx)
{
    Copy(pitchFx);
}

/**
*/
mlb::io::CPitchFxIo::~CPitchFxIo()
{
}

/**
*/
void mlb::io::CPitchFxIo::operator=(const CPitchFxIo& pitchFx)
{
    Copy(pitchFx);
}
//
///**
//*/
//bool mlb::io::CPitchFxIo::LoadFromCsv(const std::string& inputName)
//{
//	long fileSize;
//	boost::shared_array<char> fileBuffer;
//	char *fileIterator;
//	std::string token;
//	int fileFieldIndex;
//	std::string fileHeader;
//	int fileVersion;
//	mlb::io::CPitch pitch;
//
//	m_pitchArray.clear();
//
//	if (inputName.empty())
//	{
//		LOG_ERROR();
//
//		return false;
//	}
//
//    fileSize = my::file::GetFileAsString(inputName, fileBuffer);
//
//	if (fileSize == 0)
//	{
//		LOG_ERROR();
//
//		return false;
//	}
//
//	fileIterator = fileBuffer.get();
//	token.clear();
//	fileFieldIndex = 0;
//
//	// Eats file header.
//	do {
//		--fileSize;
//
//		if (*fileIterator != '\n')
//			fileHeader += *fileIterator;
//	} while (*fileIterator++ != '\n');
//
//	fileVersion = GetCsvFileVersion(fileHeader);
//
//	if (fileVersion == 0)
//	{
//        CCsvInputFile csvInputFile;
//
//        if (!csvInputFile.FromString(fileIterator, fileSize, ';', 47))
//        {
//            LOG_ERROR();
//
//            return false;
//        }
//
//        while (!csvInputFile.isEmpty())
//        {
//            mlb::io::CPitch pitch;
//
//            //GAME_PK
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetGamePrimaryKey(atoi(token.c_str()));
//
//            //GAME_ID
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//            {
//                // 2014/04/04/cinmlb-nynmlb-1 -> 2014_04_04_cinmlb_nynmlb_1
//                boost::replace_all(token, "/", "_");
//                boost::replace_all(token, "-", "_");
//
//                pitch.SetMlbGameString(token);
//            }
//
//            //SV_PITCH_ID
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetSportvisionPitchId(token);
//
//            //SEQUENCE_NUMBER
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetSequenceNumber(atoi(token.c_str()));
//				
//            //AT_BAT_NUMBER
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetGameAtbatNumber(atoi(token.c_str()));
//				
//            //PITCH_NUMBER
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetPitchNumber(atoi(token.c_str()));
//				
//            //INNING
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetInning(atoi(token.c_str()));
//
//            //TOP_INNING_SW
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetTopOfInning(token == "Y");
//				
//            //EVENT_NUMBER
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetEventNumber(atoi(token.c_str()));
//				
//            //EVENT_TYPE
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetPitchDescription(mlb::GetPitchDescriptionTypeFromString(token));
//				
//            //BATTER_ID
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetBatterMlbId(atoi(token.c_str()));
//				
//            //BATTER
//            csvInputFile.Pop();
//
//            //BAT_SIDE
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetBatterSide(token);
//				
//            //PITCHER_ID
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetPitcherMlbId(atoi(token.c_str()));
//				
//            //PITCHER
//            csvInputFile.Pop();
//
//            //SZ_TOP
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetStrikeZoneTop(atof(token.c_str()));
//				
//            //SZ_BOT
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetStrikeZoneBottom(atof(token.c_str()));
//				
//            //PITCH_START_TIME
//            csvInputFile.Pop();
//
//            //INITIAL_SPEED
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetSpeedAt50Feet(UnitConversion::MilesPerHour2FeetPerSecond(atof(token.c_str())));
//				
//            //INIT_POS_X
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//            {
//				const mlb::FloatType *releasePosition = pitch.GetPositionAt50Feet();
//
//				HEALTH_CHECK(!releasePosition, false);
//
//				pitch.SetPositionAt50Feet(atof(token.c_str()), releasePosition[1], releasePosition[2]);
//			}
//
//            //INIT_POS_Y
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//			{
//				const mlb::FloatType *releasePosition = pitch.GetPositionAt50Feet();
//
//				HEALTH_CHECK(!releasePosition, false);
//
//				pitch.SetPositionAt50Feet(releasePosition[0], atof(token.c_str()), releasePosition[2]);
//			}
//
//            //INIT_POS_Z
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//            {
//				const mlb::FloatType *releasePosition = pitch.GetPositionAt50Feet();
//
//				HEALTH_CHECK(!releasePosition, false);
//
//				pitch.SetPositionAt50Feet(releasePosition[0], releasePosition[1], atof(token.c_str()));
//			}
//
//            //INIT_VEL_X
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//            {
//				const mlb::FloatType *releaseVelocity = pitch.GetVelocityAt50Feet();
//
//				HEALTH_CHECK(!releaseVelocity, false);
//
//				pitch.SetVelocityAt50Feet(atof(token.c_str()), releaseVelocity[1], releaseVelocity[2]);
//			}
//
//            //INIT_VEL_Y
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//            {
//				const mlb::FloatType *releaseVelocity = pitch.GetVelocityAt50Feet();
//
//				HEALTH_CHECK(!releaseVelocity, false);
//
//				pitch.SetVelocityAt50Feet(releaseVelocity[0], atof(token.c_str()), releaseVelocity[2]);
//			}
//
//            //INIT_VEL_Z
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//            {
//				const mlb::FloatType *releaseVelocity = pitch.GetVelocityAt50Feet();
//
//				HEALTH_CHECK(!releaseVelocity, false);
//
//				pitch.SetVelocityAt50Feet(releaseVelocity[0], releaseVelocity[1], atof(token.c_str()));
//			}
//
//            //INIT_ACCEL_X
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//            {
//				const mlb::FloatType *releaseAcceleration = pitch.GetAccelerationAt50Feet();
//
//				HEALTH_CHECK(!releaseAcceleration, false);
//
//				pitch.SetAccelerationAt50Feet(atof(token.c_str()), releaseAcceleration[1], releaseAcceleration[2]);
//			}
//
//            //INIT_ACCEL_Y
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//            {
//				const mlb::FloatType *releaseAcceleration = pitch.GetAccelerationAt50Feet();
//
//				HEALTH_CHECK(!releaseAcceleration, false);
//
//				pitch.SetAccelerationAt50Feet(releaseAcceleration[0], atof(token.c_str()), releaseAcceleration[2]);
//			}
//
//            //INIT_ACCEL_Z
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//            {
//				const mlb::FloatType *releaseAcceleration = pitch.GetAccelerationAt50Feet();
//
//				HEALTH_CHECK(!releaseAcceleration, false);
//
//				pitch.SetAccelerationAt50Feet(releaseAcceleration[0], releaseAcceleration[1], atof(token.c_str()));
//			}
//
//            //PLATE_SPEED
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetZoneSpeed(UnitConversion::MilesPerHour2FeetPerSecond(atof(token.c_str())));
//
//            //PLATE_X
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//            {
//				const mlb::FloatType *zonePosition = pitch.GetZonePosition();
//
//				HEALTH_CHECK(!zonePosition, false);
//
//				pitch.SetZonePosition(atof(token.c_str()), zonePosition[1], zonePosition[2]);
//			}
//
//            //PLATE_Y
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//            {
//				const mlb::FloatType *zonePosition = pitch.GetZonePosition();
//
//				HEALTH_CHECK(!zonePosition, false);
//
//				pitch.SetZonePosition(zonePosition[0], atof(token.c_str()), zonePosition[2]);
//			}
//
//            //PLATE_Z
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//            {
//				const mlb::FloatType *zonePosition = pitch.GetZonePosition();
//
//				HEALTH_CHECK(!zonePosition, false);
//
//				pitch.SetZonePosition(zonePosition[0], zonePosition[1], atof(token.c_str()));
//			}
//
//            //PITCH_ZONE
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetPitchZone(atoi(token.c_str()));
//
//            //UMPIRE_CALL
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetUmpireCallType(mlb::GetUmpireCallTypeFromString(token));
//
//            //UC_NAME
//            csvInputFile.Pop();
//
//            //BREAK_X
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetHorizontalBreak(UnitConversion::Inch2Feet(atof(token.c_str())));
//
//            //BREAK_Z
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetVerticalBreak(UnitConversion::Inch2Feet(atof(token.c_str())));
//
//            //PFX
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetDistanceToNoSpinningPitch(atof(token.c_str()));
//
//            //PITCH_TYPE
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetPitchTypeCode(mlb::GetPitchTypeFromString(token));
//
//            //TYPE_CONFIDENCE
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetPitchTypeConfidence(atof(token.c_str()));
//
//            //LOP_ERROR_AVG
//            csvInputFile.Pop();
//
//			//LOP_ERROR_AT_PLATE
//            csvInputFile.Pop();
//
//            //RES_ERROR_VECTOR_SIZE
//            csvInputFile.Pop();
//
//            //GAME_DATE
//            csvInputFile.Pop();
//
//            //GAME_NBR
//            csvInputFile.Pop();
//
//            //YEAR
//            csvInputFile.Pop();
//
//            //GAME_TYPE
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetGameType(token);
//
//            m_pitchArray.push_back(pitch);
//        }
//	}
//	else if (fileVersion == 1)
//	{
//        CCsvInputFile csvInputFile;
//
//        if (!csvInputFile.FromString(fileIterator, fileSize, ';', 39))
//        {
//            LOG_ERROR();
//
//            return false;
//        }
//
//        while (!csvInputFile.isEmpty())
//        {
//            mlb::io::CPitch pitch;
//
//            //GAME_PK
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetGamePrimaryKey(atoi(token.c_str()));
//
//            //GAME_ID
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//            {
//                // 2014/04/04/cinmlb-nynmlb-1 -> 2014_04_04_cinmlb_nynmlb_1
//                boost::replace_all(token, "/", "_");
//                boost::replace_all(token, "-", "_");
//
//                pitch.SetMlbGameString(token);
//            }
//
//            //SV_PITCH_ID
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetSportvisionPitchId(token);
//
//            //EVENT#
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetEventNumber(atoi(token.c_str()));
//
//            //AT_BAT_NUMBER
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetGameAtbatNumber(atoi(token.c_str()));
//				
//            //PITCH_NUMBER
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetPitchNumber(atoi(token.c_str()));
//				
//            //INNING
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetInning(atoi(token.c_str()));
//				
//            //TOP_INNING_SW
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetTopOfInning(token == "T");
//				
//            //EVENT_TYPE
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetPitchDescription(mlb::GetPitchDescriptionTypeFromString(token));
//				
//            //RESULT_TYPE
//            csvInputFile.Pop();
//
//            //BATTER_ID
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetBatterMlbId(atoi(token.c_str()));
//				
//            //BATTER
//            csvInputFile.Pop();
//
//            //BAT_SIDE
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetBatterSide(token);
//				
//            //PITCHER_ID
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetPitcherMlbId(atoi(token.c_str()));
//				
//            //PITCHER
//            csvInputFile.Pop();
//
//            //SZ_TOP
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetStrikeZoneTop(atof(token.c_str()));
//				
//            //SZ_BOT
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetStrikeZoneBottom(atof(token.c_str()));
//				
//            //INITIAL_SPEED
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//				pitch.SetSpeedAt50Feet(UnitConversion::MilesPerHour2FeetPerSecond(atof(token.c_str())));
//
//            //INIT_POS_X
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//            {
//                const mlb::FloatType *releasePosition = pitch.GetPositionAt50Feet();
//
//                HEALTH_CHECK(!releasePosition, false);
//
//                pitch.SetPositionAt50Feet(atof(token.c_str()), releasePosition[1], releasePosition[2]);
//            }
//
//            //INIT_POS_Y
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//            {
//				const mlb::FloatType *releasePosition = pitch.GetPositionAt50Feet();
//
//				HEALTH_CHECK(!releasePosition, false);
//
//				pitch.SetPositionAt50Feet(releasePosition[0], atof(token.c_str()), releasePosition[2]);
//			}
//
//            //INIT_POS_Z
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//            {
//				const mlb::FloatType *releasePosition = pitch.GetPositionAt50Feet();
//
//				HEALTH_CHECK(!releasePosition, false);
//
//				pitch.SetPositionAt50Feet(releasePosition[0], releasePosition[1], atof(token.c_str()));
//			}
//
//            //INIT_VEL_X
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//            {
//				const mlb::FloatType *releaseVelocity = pitch.GetVelocityAt50Feet();
//
//				HEALTH_CHECK(!releaseVelocity, false);
//
//				pitch.SetVelocityAt50Feet(atof(token.c_str()), releaseVelocity[1], releaseVelocity[2]);
//			}
//
//            //INIT_VEL_Y
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//			{
//				const mlb::FloatType *releaseVelocity = pitch.GetVelocityAt50Feet();
//
//				HEALTH_CHECK(!releaseVelocity, false);
//
//				pitch.SetVelocityAt50Feet(releaseVelocity[0], atof(token.c_str()), releaseVelocity[2]);
//			}
//
//            //INIT_VEL_Z
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//            {
//				const mlb::FloatType *releaseVelocity = pitch.GetVelocityAt50Feet();
//
//				HEALTH_CHECK(!releaseVelocity, false);
//
//				pitch.SetVelocityAt50Feet(releaseVelocity[0], releaseVelocity[1], atof(token.c_str()));
//			}
//
//            //INIT_ACCEL_X
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//            {
//				const mlb::FloatType *releaseAcceleration = pitch.GetAccelerationAt50Feet();
//
//				HEALTH_CHECK(!releaseAcceleration, false);
//
//				pitch.SetAccelerationAt50Feet(atof(token.c_str()), releaseAcceleration[1], releaseAcceleration[2]);
//			}
//
//            //INIT_ACCEL_Y
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//            {
//				const mlb::FloatType *releaseAcceleration = pitch.GetAccelerationAt50Feet();
//
//				HEALTH_CHECK(!releaseAcceleration, false);
//
//				pitch.SetAccelerationAt50Feet(releaseAcceleration[0], atof(token.c_str()), releaseAcceleration[2]);
//			}
//
//            //INIT_ACCEL_Z
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//            {
//				const mlb::FloatType *releaseAcceleration = pitch.GetAccelerationAt50Feet();
//
//				HEALTH_CHECK(!releaseAcceleration, false);
//
//				pitch.SetAccelerationAt50Feet(releaseAcceleration[0], releaseAcceleration[1], atof(token.c_str()));
//			}
//
//            //PLATE_SPEED
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//				pitch.SetZoneSpeed(UnitConversion::MilesPerHour2FeetPerSecond(atof(token.c_str())));
//
//            //PLATE_X
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//            {
//		        const mlb::FloatType *zonePosition = pitch.GetZonePosition();
//
//		        HEALTH_CHECK(!zonePosition, false);
//
//		        pitch.SetZonePosition(atof(token.c_str()), zonePosition[1], zonePosition[2]);
//			}
//
//            //PLATE_Y
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//            {
//				const mlb::FloatType *zonePosition = pitch.GetZonePosition();
//
//				HEALTH_CHECK(!zonePosition, false);
//
//				pitch.SetZonePosition(zonePosition[0], atof(token.c_str()), zonePosition[2]);
//			}
//
//            //PLATE_Z
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//            {
//				const mlb::FloatType *zonePosition = pitch.GetZonePosition();
//
//				HEALTH_CHECK(!zonePosition, false);
//
//				pitch.SetZonePosition(zonePosition[0], zonePosition[1], atof(token.c_str()));
//			}
//
//            //PITCH_ZONE
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetPitchZone(atoi(token.c_str()));
//
//            //UMPIRE_CALL
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetUmpireCallType(mlb::GetUmpireCallTypeFromString(token));
//
//            //BREAK_X
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetHorizontalBreak(UnitConversion::Inch2Feet(atof(token.c_str())));
//
//            //BREAK_Z
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetVerticalBreak(UnitConversion::Inch2Feet(atof(token.c_str())));
//
//            //PFX
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetDistanceToNoSpinningPitch(atof(token.c_str()));
//
//            //PITCH_TYPE
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetPitchTypeCode(mlb::GetPitchTypeFromString(token));
//
//            //TYPE_CONFIDENCE
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            if (!token.empty())
//                pitch.SetPitchTypeConfidence(atof(token.c_str()));
//
//            //LOP_ERROR_AVG
//            csvInputFile.Pop();
//
//            m_pitchArray.push_back(pitch);
//        }
//	}
//    // TESTING: (06-Jun-2015) New integrated CSV IO.
//    else if (fileVersion == 2)
//    {
//        CCsvInputFile csvInputFile;
//
//        if (!csvInputFile.FromString(fileIterator, fileSize, ';', 44))
//        {
//            LOG_ERROR();
//
//            return false;
//        }
//
//        while (!csvInputFile.isEmpty())
//        {
//            mlb::io::CPitch pitch;
//
//            //GAME_PK
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//            if (!token.empty())
//                pitch.SetGamePrimaryKey(atoi(token.c_str()));
//
//            //GAME_ID
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            boost::replace_all(token, "/", "_");
//            boost::replace_all(token, "-", "_");
//
//            pitch.SetMlbGameString(token);
//            
//            //SV_PITCH_ID
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//            pitch.SetSportvisionPitchId(token);
//
//            //SEQUENCE_NUMBER
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            //AT_BAT_NUMBER
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//            if (!token.empty())
//                pitch.SetGameAtbatNumber(atoi(token.c_str()));
//
//            //PITCH_NUMBER
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//            if (!token.empty())
//                pitch.SetPitchNumber(atoi(token.c_str()));
//            
//            //EVENT_NUMBER
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//            if (!token.empty())
//                pitch.SetEventNumber(atoi(token.c_str()));
//            
//            //EVENT_TYPE
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//            pitch.SetPitchDescription(mlb::GetPitchDescriptionTypeFromString(token));
//            
//            //EVENT_GROUP
//            csvInputFile.Pop();
//
//            //INNING
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//            if (!token.empty())
//                pitch.SetInning(atoi(token.c_str()));
//            
//            //TOP_INNING_SW
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//            pitch.SetTopOfInning(token == "Y");
//            
//            //BATTER_ID
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//            if (!token.empty())
//                pitch.SetBatterMlbId(atoi(token.c_str()));
//            
//            //BATTER
//            csvInputFile.Pop();
//
//            //BAT_SIDE
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//            pitch.SetBatterSide(token);
//            
//            //PITCHER_ID
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//            if (!token.empty())
//                pitch.SetPitcherMlbId(atoi(token.c_str()));
//            
//            //PITCHER
//            csvInputFile.Pop();
//
//            //SZ_TOP
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//            if (!token.empty())
//                pitch.SetStrikeZoneTop(atof(token.c_str()));
//            
//            //SZ_BOT
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//            if (!token.empty())
//                pitch.SetStrikeZoneBottom(atof(token.c_str()));
//            
//            //INITIAL_SPEED
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//            if (!token.empty())
//                pitch.SetSpeedAt50Feet(UnitConversion::MilesPerHour2FeetPerSecond(atof(token.c_str())));
//            
//            //INIT_POS_X
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//            if (!token.empty())
//            {
//                const mlb::FloatType *releasePosition = pitch.GetPositionAt50Feet();
//
//                HEALTH_CHECK(!releasePosition, false);
//
//                pitch.SetPositionAt50Feet(atof(token.c_str()), releasePosition[1], releasePosition[2]);
//            }
//
//            //INIT_POS_Y
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//            if (!token.empty())
//            {
//                const mlb::FloatType *releasePosition = pitch.GetPositionAt50Feet();
//
//                HEALTH_CHECK(!releasePosition, false);
//
//                pitch.SetPositionAt50Feet(releasePosition[0], atof(token.c_str()), releasePosition[2]);
//            }
//
//            //INIT_POS_Z
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//            if (!token.empty())
//            {
//                const mlb::FloatType *releasePosition = pitch.GetPositionAt50Feet();
//
//                HEALTH_CHECK(!releasePosition, false);
//
//                pitch.SetPositionAt50Feet(releasePosition[0], releasePosition[1], atof(token.c_str()));
//            }
//
//            //INIT_VEL_X
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//            if (!token.empty())
//            {
//                const mlb::FloatType *releaseVelocity = pitch.GetVelocityAt50Feet();
//
//                HEALTH_CHECK(!releaseVelocity, false);
//
//                pitch.SetVelocityAt50Feet(atof(token.c_str()), releaseVelocity[1], releaseVelocity[2]);
//            }
//
//            //INIT_VEL_Y
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//            if (!token.empty())
//            {
//                const mlb::FloatType *releaseVelocity = pitch.GetVelocityAt50Feet();
//
//                HEALTH_CHECK(!releaseVelocity, false);
//
//                pitch.SetVelocityAt50Feet(releaseVelocity[0], atof(token.c_str()), releaseVelocity[2]);
//            }
//
//            //INIT_VEL_Z
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//            if (!token.empty())
//            {
//                const mlb::FloatType *releaseVelocity = pitch.GetVelocityAt50Feet();
//
//                HEALTH_CHECK(!releaseVelocity, false);
//
//                pitch.SetVelocityAt50Feet(releaseVelocity[0], releaseVelocity[1], atof(token.c_str()));
//            }
//
//            //INIT_ACCEL_X
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//            if (!token.empty())
//            {
//                const mlb::FloatType *releaseAcceleration = pitch.GetAccelerationAt50Feet();
//
//                HEALTH_CHECK(!releaseAcceleration, false);
//
//                pitch.SetAccelerationAt50Feet(atof(token.c_str()), releaseAcceleration[1], releaseAcceleration[2]);
//            }
//
//            //INIT_ACCEL_Y
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//            if (!token.empty())
//            {
//                const mlb::FloatType *releaseAcceleration = pitch.GetAccelerationAt50Feet();
//
//                HEALTH_CHECK(!releaseAcceleration, false);
//
//                pitch.SetAccelerationAt50Feet(releaseAcceleration[0], atof(token.c_str()), releaseAcceleration[2]);
//            }
//
//            //INIT_ACCEL_Z
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//            if (!token.empty())
//            {
//                const mlb::FloatType *releaseAcceleration = pitch.GetAccelerationAt50Feet();
//
//                HEALTH_CHECK(!releaseAcceleration, false);
//
//                pitch.SetAccelerationAt50Feet(releaseAcceleration[0], releaseAcceleration[1], atof(token.c_str()));
//            }
//
//            //PLATE_SPEED
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//            if (!token.empty())
//                pitch.SetZoneSpeed(UnitConversion::MilesPerHour2FeetPerSecond(atof(token.c_str())));
//            
//            //PLATE_X
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//            if (!token.empty())
//            {
//                const mlb::FloatType *zonePosition = pitch.GetZonePosition();
//
//                HEALTH_CHECK(!zonePosition, false);
//
//                pitch.SetZonePosition(atof(token.c_str()), zonePosition[1], zonePosition[2]);
//            }
//
//            //PLATE_Y
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//            if (!token.empty())
//            {
//                const mlb::FloatType *zonePosition = pitch.GetZonePosition();
//
//                HEALTH_CHECK(!zonePosition, false);
//
//                pitch.SetZonePosition(zonePosition[0], atof(token.c_str()), zonePosition[2]);
//            }
//            
//            //PLATE_Z
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//            if (!token.empty())
//            {
//                const mlb::FloatType *zonePosition = pitch.GetZonePosition();
//
//                HEALTH_CHECK(!zonePosition, false);
//
//                pitch.SetZonePosition(zonePosition[0], zonePosition[1], atof(token.c_str()));
//            }
//            
//            //PITCH_ZONE
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//            if (!token.empty())
//                pitch.SetPitchZone(atoi(token.c_str()));
//            
//            //BREAK_X
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//            if (!token.empty())
//                pitch.SetHorizontalBreak(UnitConversion::Inch2Feet(atof(token.c_str())));
//            
//            //BREAK_Z
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//            if (!token.empty())
//                pitch.SetVerticalBreak(UnitConversion::Inch2Feet(atof(token.c_str())));
//            
//            //NASTY_FACTOR
//            csvInputFile.Pop();
//
//            //PITCH_TYPE
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//            pitch.SetPitchTypeCode(mlb::GetPitchTypeFromString(token));
//            
//            //TYPE_CONFIDENCE
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//            if (!token.empty())
//                pitch.SetPitchTypeConfidence(atof(token.c_str()));
//            
//            //FLAG_CD
//            csvInputFile.Pop();
//
//            //PITCH_START_TIME
//            csvInputFile.Pop();
//
//            //UMPIRE_CALL
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//            pitch.SetUmpireCallType(mlb::GetUmpireCallTypeFromString(token));
//            
//            //UMP_CALL
//            csvInputFile.Pop();
//
//            //LOP_ERROR_AVG
//            csvInputFile.Pop();
//
//            //LOP_ERROR_AT_PLATE
//            HEALTH_CHECK(!csvInputFile.ReadValue(token), false);
//
//            m_pitchArray.push_back(pitch);
//        }
//    }
//    else
//	{
//		LOG_ERROR();
//
//		return false;
//	}
//
//	std::sort(m_pitchArray.begin(), m_pitchArray.end());
//
//	//// TESTING: (21-Jan-2015) Get addional data from Gameday server.
//	//int pitchCount = (int)m_pitchArray.size(),
//	//	pitchIndex = 0;
//
//	//for (std::vector<mlb::io::CPitch>::iterator pitchIterator = m_pitchArray.begin(); pitchIterator != m_pitchArray.end(); ++pitchIterator)
//	//{
//	//	if ((pitchIterator->GetReleaseTime() == mlb::Null<INT64>()) &&
//	//		!pitchIterator->GetMlbGameString().empty() &&
//	//		(pitchIterator->GetInning() != mlb::Null<INT32>()) &&
//	//		!pitchIterator->GetSportvisionPitchId().empty())
//	//	{
//	//		// DEBUG ONLY!
//	//		std::cout << "(GD2 -> PITCHf/x (" << pitchIndex++ << "/" << pitchCount << ")) " << pitchIterator->GetMlbGameString() << " " << pitchIterator->GetInning() << " " << pitchIterator->GetTopOfInning() << "... ";
//
//	//		mlb::io::CPitch pitch = LoadFromSportvisionPitchId(pitchIterator->GetMlbGameString(), pitchIterator->GetInning(), pitchIterator->GetTopOfInning(), pitchIterator->GetSportvisionPitchId());
//
//	//		if (pitch.GetReleaseTime() != mlb::Null<INT64>())
//	//		{
//	//			pitchIterator->SetReleaseTime(pitch.GetReleaseTime());
//	//			pitchIterator->SetSpinRate(pitch.GetSpinRate());
//
//	//			// DEBUG ONLY!
//	//			std::cout << " ok." << std::endl;
//	//		}
//	//		else
//	//			// DEBUG ONLY!
//	//			std::cout << " failed." << std::endl;
//	//	}
//	//}
//
//	return true;
//}

///**
//*/
//int mlb::io::CPitchFxIo::GetCsvFileVersion(std::string fileHeader) const
//{
//	std::string fileHeaderVersion0 = "GAME_PK;GAME_ID;SV_PITCH_ID;SEQUENCE_NUMBER;AT_BAT_NUMBER;PITCH_NUMBER;INNING;TOP_INNING_SW;EVENT_NUMBER;EVENT_TYPE;BATTER_ID;BATTER;BAT_SIDE;PITCHER_ID;PITCHER;SZ_TOP;SZ_BOT;PITCH_START_TIME;INITIAL_SPEED;INIT_POS_X;INIT_POS_Y;INIT_POS_Z;INIT_VEL_X;INIT_VEL_Y;INIT_VEL_Z;INIT_ACCEL_X;INIT_ACCEL_Y;INIT_ACCEL_Z;PLATE_SPEED;PLATE_X;PLATE_Y;PLATE_Z;PITCH_ZONE;UMPIRE_CALL;UC_NAME;BREAK_X;BREAK_Z;PFX;PITCH_TYPE;TYPE_CONFIDENCE;LOP_ERROR_AVG;LOP_ERROR_AT_PLATE;RES_ERROR_VECTOR_SIZE;GAME_DATE;GAME_NBR;YEAR;GAME_TYPE";
//    std::string fileHeaderVersion1 = "GAME_PK;GAME_ID;SV_PITCH_ID;EVENT#;AB#;PITCH#;INN;T/B;EVENT_TYPE;RESULT_TYPE;BATTER_ID;BATTER;BATS;PITCHER_ID;PITCHER;SZ_TOP;SZ_BOT;INITIAL_SPEED;INIT_POS_X;INIT_POS_Y;INIT_POS_Z;INIT_VEL_X;INIT_VEL_Y;INIT_VEL_Z;INIT_ACCEL_X;INIT_ACCEL_Y;INIT_ACCEL_Z;PLATE_SPEED;PLATE_X;PLATE_Y;PLATE_Z;PITCH_ZONE;UMP_CALL;BREAK_X;BREAK_Z;PFX;PITCH_TYPE;TYPE_CONFIDENCE;LOP_ERROR_AVG";
//    std::string fileHeaderVersion2 = "GAME_PK;GAME_ID;SV_PITCH_ID;SEQUENCE_NUMBER;AT_BAT_NUMBER;PITCH_NUMBER;EVENT_NUMBER;EVENT_TYPE;EVENT_GROUP;INNING;TOP_INNING_SW;BATTER_ID;BATTER;BAT_SIDE;PITCHER_ID;PITCHER;SZ_TOP;SZ_BOT;INITIAL_SPEED;INIT_POS_X;INIT_POS_Y;INIT_POS_Z;INIT_VEL_X;INIT_VEL_Y;INIT_VEL_Z;INIT_ACCEL_X;INIT_ACCEL_Y;INIT_ACCEL_Z;PLATE_SPEED;PLATE_X;PLATE_Y;PLATE_Z;PITCH_ZONE;BREAK_X;BREAK_Z;NASTY_FACTOR;PITCH_TYPE;TYPE_CONFIDENCE;FLAG_CD;PITCH_START_TIME;UMPIRE_CALL;UMP_CALL;LOP_ERROR_AVG;LOP_ERROR_AT_PLATE";
//
//	boost::trim_right(fileHeader);
//
//	if (fileHeader == fileHeaderVersion0)
//		return 0;
//    else if (fileHeader == fileHeaderVersion1)
//        return 1;
//    else if (fileHeader == fileHeaderVersion2)
//        return 2;
//
//	return -1;
//}
//
///**
//*/
//mlb::io::CPitch mlb::io::CPitchFxIo::LoadFromSportvisionPitchId(const std::string& mlbGameString, INT32 inningIndex, bool topOfInning, const std::string& sportvisionPitchId) const
//{
//	mlb::io::CPitch emptyObject;
//
//	HEALTH_CHECK(mlbGameString.empty(), emptyObject);
//
//	std::string inningHalfName = "top";
//
//	if (!topOfInning)
//		inningHalfName = "bottom";
//
//	std::string urlPrefix = mlb::io::CGamedayServer().GetGameUrl(mlbGameString);
//
//    std::string xmlResponse;
//
//    if (!my::file::GetUrlAsString(urlPrefix + "inning/inning_" + boost::lexical_cast<std::string>(inningIndex)+".xml", xmlResponse))
//    {
//        LOG_ERROR();
//
//        return emptyObject;
//    }
//
//	pugi::xml_document xmlDocument;
//	pugi::xml_parse_result xmlDocumentLog;
//
//	xmlDocumentLog = xmlDocument.load(xmlResponse.c_str());
//
//	if (!xmlDocumentLog)
//	{
//		LOG_MESSAGE(xmlDocumentLog.description());
//        LOG_MESSAGE(std::string("Error offset: ") + boost::lexical_cast<std::string>(xmlDocumentLog.offset));
//
//		return emptyObject;
//	}
//
//	std::vector<mlb::io::CPitch> pitchArray;
//
//	pugi::xml_node atBatArray = xmlDocument.child("inning").child(inningHalfName.c_str());
//
//	for (pugi::xml_node atBatIterator = atBatArray.first_child(); atBatIterator; atBatIterator = atBatIterator.next_sibling())
//	{
//        if (std::string(atBatIterator.name()) == "atbat")
//        {
//            mlb::io::CPitch pitchHeader;
//
//            pitchHeader.SetMlbGameString(mlbGameString);
//            pitchHeader.SetInning(inningIndex);
//
//            pitchHeader.SetTopOfInning(topOfInning);
//
//            if (!atBatIterator.attribute("num").empty())
//                pitchHeader.SetGameAtbatNumber(atBatIterator.attribute("num").as_int());
//
//            if (!atBatIterator.attribute("pitcher").empty())
//                pitchHeader.SetPitcherMlbId(atBatIterator.attribute("pitcher").as_int());
//
//            if (!atBatIterator.attribute("p_throws").empty())
//                pitchHeader.SetPitcherThrows(atBatIterator.attribute("p_throws").as_string());
//
//            if (!atBatIterator.attribute("batter").empty())
//                pitchHeader.SetBatterMlbId(atBatIterator.attribute("batter").as_int());
//
//            if (!atBatIterator.attribute("stand").empty())
//                pitchHeader.SetBatterSide(atBatIterator.attribute("stand").as_string());
//
//            INT32 prePitchBallCount = 0,
//                prePitchStrikeCount = 0;
//
//            for (pugi::xml_node pitchIterator = atBatIterator.first_child(); pitchIterator; pitchIterator = pitchIterator.next_sibling())
//            {
//                if (std::string(pitchIterator.name()) == "pitch")
//                {
//                    mlb::io::CPitch pitch = pitchHeader;
//
//                    //des = "Ball"
//                    if (!pitchIterator.attribute("des").empty())
//                        pitch.SetPitchDescription(mlb::GetPitchDescriptionTypeFromString(pitchIterator.attribute("des").as_string()));
//
//                    //des_es = "Bola mala"
//                    //id = "8"
//                    if (!pitchIterator.attribute("id").empty())
//                        pitch.SetPitchNumber(pitchIterator.attribute("id").as_int());
//
//                    //type = "B"
//                    if (!pitchIterator.attribute("type").empty())
//                        pitch.SetUmpireCallType(mlb::GetUmpireCallTypeFromString(pitchIterator.attribute("type").as_string()));
//
//                    //tfs = "231142"
//                    //tfs_zulu = "2014-04-22T23:11:42Z"
//                    std::string startTfsZulu = pitchIterator.attribute("tfs_zulu").as_string();
//
//                    if (startTfsZulu.empty())
//                    {
//                        LOG_ERROR();
//
//                        return emptyObject;
//                    }
//
//                    if (startTfsZulu.back() == 'Z')
//                        startTfsZulu.pop_back();
//
//                    pitch.SetReleaseTime(CUnixEpoch(startTfsZulu, "%Y-%m-%d %H:%M:%S").Get());
//
//                    //x = "74.68"
//                    //y = "172.69"
//                    //sv_id = "140422_191225"
//                    if (!pitchIterator.attribute("sv_id").empty())
//                        pitch.SetSportvisionPitchId(pitchIterator.attribute("sv_id").as_string());
//
//                    //play_guid="4a9feac9-12e4-476d-ab55-8bc14973dbe6"
//                    if (!pitchIterator.attribute("play_guid").empty())
//                        pitch.SetMeasurementId(pitchIterator.attribute("play_guid").as_string());
//
//                    //start_speed = "90.5"
//                    if (!pitchIterator.attribute("start_speed").empty())
//                        pitch.SetSpeedAt50Feet(UnitConversion::MilesPerHour2FeetPerSecond(pitchIterator.attribute("start_speed").as_double()));
//
//                    //end_speed = "82.8"
//                    if (!pitchIterator.attribute("end_speed").empty())
//                        pitch.SetZoneSpeed(UnitConversion::MilesPerHour2FeetPerSecond(pitchIterator.attribute("end_speed").as_double()));
//
//                    //sz_top = "3.27"
//                    if (!pitchIterator.attribute("sz_top").empty())
//                        pitch.SetStrikeZoneTop(pitchIterator.attribute("sz_top").as_float());
//
//                    //sz_bot = "1.48"
//                    if (!pitchIterator.attribute("sz_bot").empty())
//                        pitch.SetStrikeZoneBottom(pitchIterator.attribute("sz_bot").as_float());
//
//                    //pfx_x = "-10.72"
//                    if (!pitchIterator.attribute("pfx_x").empty())
//                        pitch.SetHorizontalBreak(pitchIterator.attribute("pfx_x").as_float());
//
//                    //pfx_z = "6.52"
//                    if (!pitchIterator.attribute("pfx_x").empty())
//                        pitch.SetVerticalBreak(pitchIterator.attribute("pfx_x").as_float());
//
//                    // break_angle
//                    if (!pitchIterator.attribute("break_angle").empty())
//                        pitch.SetBreakAngle(pitchIterator.attribute("break_angle").as_float());
//
//                    // break_length
//                    if (!pitchIterator.attribute("break_length").empty())
//                        pitch.SetBreakLength(pitchIterator.attribute("break_length").as_float());
//
//                    //px = "0.753"
//                    if (!pitchIterator.attribute("px").empty())
//                    {
//                        const mlb::FloatType *zonePosition = pitch.GetZonePosition();
//
//                        HEALTH_CHECK(!zonePosition, emptyObject);
//
//                        pitch.SetZonePosition(pitchIterator.attribute("px").as_float(), 1.41666666666662, zonePosition[2]);
//                    }
//
//                    //pz = "1.391"
//                    if (!pitchIterator.attribute("pz").empty())
//                    {
//                        const mlb::FloatType *zonePosition = pitch.GetZonePosition();
//
//                        HEALTH_CHECK(!zonePosition, emptyObject);
//
//                        pitch.SetZonePosition(zonePosition[0], 1.41666666666662, pitchIterator.attribute("pz").as_float());
//                    }
//
//                    //x0 = "-1.254"
//                    if (!pitchIterator.attribute("x0").empty())
//                    {
//                        const mlb::FloatType *releasePosition = pitch.GetPositionAt50Feet();
//
//                        HEALTH_CHECK(!releasePosition, emptyObject);
//
//                        pitch.SetPositionAt50Feet(pitchIterator.attribute("x0").as_float(), releasePosition[1], releasePosition[2]);
//                    }
//
//                    //y0 = "50.0"
//                    if (!pitchIterator.attribute("y0").empty())
//                    {
//                        const mlb::FloatType *releasePosition = pitch.GetPositionAt50Feet();
//
//                        HEALTH_CHECK(!releasePosition, emptyObject);
//
//                        pitch.SetPositionAt50Feet(releasePosition[0], pitchIterator.attribute("y0").as_float(), releasePosition[2]);
//                    }
//
//                    //z0 = "5.486"
//                    if (!pitchIterator.attribute("z0").empty())
//                    {
//                        const mlb::FloatType *releasePosition = pitch.GetPositionAt50Feet();
//
//                        HEALTH_CHECK(!releasePosition, emptyObject);
//
//                        pitch.SetPositionAt50Feet(releasePosition[0], releasePosition[1], pitchIterator.attribute("z0").as_float());
//                    }
//
//                    //vx0 = "8.838"
//                    if (!pitchIterator.attribute("vx0").empty())
//                    {
//                        const mlb::FloatType *releaseVelocity = pitch.GetVelocityAt50Feet();
//
//                        HEALTH_CHECK(!releaseVelocity, emptyObject);
//
//                        pitch.SetVelocityAt50Feet(pitchIterator.attribute("vx0").as_float(), releaseVelocity[1], releaseVelocity[2]);
//                    }
//
//                    //vy0 = "-132.222"
//                    if (!pitchIterator.attribute("vy0").empty())
//                    {
//                        const mlb::FloatType *releaseVelocity = pitch.GetVelocityAt50Feet();
//
//                        HEALTH_CHECK(!releaseVelocity, emptyObject);
//
//                        pitch.SetVelocityAt50Feet(releaseVelocity[0], pitchIterator.attribute("vy0").as_float(), releaseVelocity[2]);
//                    }
//
//                    //vz0 = "-6.684"
//                    if (!pitchIterator.attribute("vz0").empty())
//                    {
//                        const mlb::FloatType *releaseVelocity = pitch.GetVelocityAt50Feet();
//
//                        HEALTH_CHECK(!releaseVelocity, emptyObject);
//
//                        pitch.SetVelocityAt50Feet(releaseVelocity[0], releaseVelocity[1], pitchIterator.attribute("vz0").as_float());
//                    }
//
//                    //ax = "-18.817"
//                    if (!pitchIterator.attribute("ax").empty())
//                    {
//                        const mlb::FloatType *releaseAcceleration = pitch.GetAccelerationAt50Feet();
//
//                        HEALTH_CHECK(!releaseAcceleration, emptyObject);
//
//                        pitch.SetAccelerationAt50Feet(pitchIterator.attribute("ax").as_float(), releaseAcceleration[1], releaseAcceleration[2]);
//                    }
//
//                    //ay = "30.385"
//                    if (!pitchIterator.attribute("ay").empty())
//                    {
//                        const mlb::FloatType *releaseAcceleration = pitch.GetAccelerationAt50Feet();
//
//                        HEALTH_CHECK(!releaseAcceleration, emptyObject);
//
//                        pitch.SetAccelerationAt50Feet(releaseAcceleration[0], pitchIterator.attribute("ay").as_float(), releaseAcceleration[2]);
//                    }
//
//                    //az = "-20.652"
//                    if (!pitchIterator.attribute("az").empty())
//                    {
//                        const mlb::FloatType *releaseAcceleration = pitch.GetAccelerationAt50Feet();
//
//                        HEALTH_CHECK(!releaseAcceleration, emptyObject);
//
//                        pitch.SetAccelerationAt50Feet(releaseAcceleration[0], releaseAcceleration[1], pitchIterator.attribute("az").as_float());
//                    }
//
//                    //break_y = "23.8"
//                    //break_angle = "37.3"
//                    //break_length = "6.4"
//                    //pitch_type = "FT"
//                    if (!pitchIterator.attribute("pitch_type").empty())
//                        pitch.SetPitchTypeCode(mlb::GetPitchTypeFromString(pitchIterator.attribute("pitch_type").as_string()));
//
//                    //type_confidence = ".910"
//                    if (!pitchIterator.attribute("type_confidence").empty())
//                        pitch.SetPitchTypeConfidence(pitchIterator.attribute("type_confidence").as_float());
//
//                    //zone = "14" 
//                    if (!pitchIterator.attribute("zone").empty())
//                        pitch.SetPitchZone(pitchIterator.attribute("zone").as_int());
//
//                    //nasty = "71"
//                    //spin_dir = "238.519"
//                    //if (!pitchIterator.attribute("spin_dir").empty())
//                    //	pitch.SetSpinDirection(pitchIterator.attribute("spin_dir").as_float());
//
//                    //spin_rate = "2420.972"
//                    if (!pitchIterator.attribute("spin_rate").empty())
//                        pitch.SetSpinRate(pitchIterator.attribute("spin_rate").as_float());
//
//                    //cc = ""
//                    //mt = ""
//
//                    if (!pitchIterator.attribute("on_1b").empty())
//                        pitch.SetPlayerOn1stMlbId(pitchIterator.attribute("on_1b").as_int());
//                    if (!pitchIterator.attribute("on_2b").empty())
//                        pitch.SetPlayerOn2ndMlbId(pitchIterator.attribute("on_2b").as_int());
//                    if (!pitchIterator.attribute("on_3b").empty())
//                        pitch.SetPlayerOn3rdMlbId(pitchIterator.attribute("on_3b").as_int());
//
//                    pitch.SetPrePitchBalls(prePitchBallCount);
//                    pitch.SetPrePitchBalls(prePitchStrikeCount);
//
//                    switch (pitch.GetUmpireCallType()) {
//                        // Ball - Called
//                    case B:
//                        // Ball - Hit by Pitch
//                    case H:
//                        // Ball - Pitchout
//                    case P:
//                        // Ball - Intentional
//                    case I:
//                        ++prePitchBallCount;
//                        break;
//
//                        // Strike - Called
//                    case C:
//                        // Strike - Foul
//                    case F:
//                        // Strike - Foul Bunt
//                    case L:
//                        // Strike - Missed Bunt
//                    case M:
//                        // Strike - Swinging
//                    case S:
//                        // Strike - Foul Tip
//                    case T:
//                        // Strike - Swinging Blocked
//                    case W:
//                        ++prePitchStrikeCount;
//                        break;
//
//                        // Hit Into Play - No Out(s)
//                    case D:
//                        // Hit Into Play - Run(s)
//                    case E:
//                        // Hit Into Play - Out(s)
//                    case X:
//                        break;
//
//                    case UNKNOWN_UMPIRE_CALL:
//                        LOG_MESSAGE("Unknown umpire call found.");
//                        break;
//                            
//                        default:
//                            LOG_ERROR();
//                            break;
//                    }
//
//                    pitchArray.push_back(pitch);
//                }
//            }
//
//            // BUG: (16-Dec-2014) Actions (<action>) may be children of innning halfs (top and bottom).
//            if (!pitchArray.empty())
//            {
//                if (!atBatIterator.attribute("b").empty())
//                {
//                    if (prePitchBallCount != atBatIterator.attribute("b").as_int())
//                        LOG_MESSAGE("Pre-pitch balls mismatch.");
//                }
//
//                if (!atBatIterator.attribute("s").empty())
//                {
//                    if (prePitchStrikeCount != atBatIterator.attribute("s").as_int())
//                        LOG_MESSAGE("Pre-pitch strikes mismatch.");
//                }
//
//                if (!atBatIterator.attribute("des").empty())
//                {
//                    pitchArray.back().SetAtBatDescription(atBatIterator.attribute("des").as_string());
//                }
//
//                if (!atBatIterator.attribute("event").empty())
//                    pitchArray.back().SetAtBatEvent(mlb::GetAtBatEventTypeFromString(atBatIterator.attribute("event").as_string()));
//            }
//        }
//	}
//
//	for (std::vector<mlb::io::CPitch>::const_iterator pitchIterator = pitchArray.begin(); pitchIterator != pitchArray.end(); ++pitchIterator)
//	{
//		if (pitchIterator->GetSportvisionPitchId() == sportvisionPitchId)
//			return (*pitchIterator);
//	}
//
//	return emptyObject;
//}

// Url on http://gd2.mlb.com/components/game/mlb/
bool mlb::io::CPitchFxIo::LoadFromGameDirectory(const std::string& gameDirectory)
{
    HEALTH_CHECK(gameDirectory.empty(), false);

    m_pitchArray.clear();

    std::string xmlResponse;

    my::file::GetUrlAsString(gameDirectory + "inning/inning_all.xml", xmlResponse);

    HEALTH_CHECK(xmlResponse.empty(), false);
    HEALTH_CHECK(xmlResponse.find("404 Not Found") != std::string::npos, false);
    HEALTH_CHECK(xmlResponse.find("Error (404)") != std::string::npos, false);

    pugi::xml_document xmlDocument;
    pugi::xml_parse_result xmlDocumentLog;

    xmlDocumentLog = xmlDocument.load(xmlResponse.c_str());

    if (!xmlDocumentLog)
    {
        LOG_MESSAGE(xmlDocumentLog.description());
        LOG_MESSAGE(std::string("Error offset: ") + boost::lexical_cast<std::string>(xmlDocumentLog.offset));

        return false;
    }

    // GAME->INNING

    pugi::xml_node inningArray = xmlDocument.child("game");

    my::int32 inningIndex = my::Null<my::int32>();

    std::vector<std::string> inningHalfArray;

    inningHalfArray.push_back("top");
    inningHalfArray.push_back("bottom");

    for (pugi::xml_node inningIterator = inningArray.first_child(); inningIterator; inningIterator = inningIterator.next_sibling())
    {
        if (!inningIterator.attribute("num").empty())
            inningIndex = inningIterator.attribute("num").as_int();

        HEALTH_CHECK(my::IsNull(inningIndex), false);

        for (std::vector<std::string>::const_iterator inningHalfIterator = inningHalfArray.begin(); inningHalfIterator != inningHalfArray.end(); ++inningHalfIterator)
        {
            pugi::xml_node atBatArray = inningIterator.child(inningHalfIterator->c_str());

            for (pugi::xml_node atBatIterator = atBatArray.first_child(); atBatIterator; atBatIterator = atBatIterator.next_sibling())
            {
                if (std::string(atBatIterator.name()) == "atbat")
                {
                    mlb::io::CPitch header;

                    // game_primary_key

                    // mlb_game_string
                    // inning
                    // top_of_inning
                    // game_at_bat_number
                    // pitcher_mlb_id

                    // pitcher_throws
                    if (!atBatIterator.attribute("p_throws").empty())
                        header.SetPitcherThrows(atBatIterator.attribute("p_throws").as_string());

                    // batter_mlb_id
                    // batter_side
                    // release_speed
                    // sequence_number
                    // pitch_event_type
                    // release_position_x
                    // release_position_y
                    // release_position_z
                    // effective_release_speed
                    // release_horizontal_angle
                    // release_vertical_angle
                    // zone_time
                    // game_type
                    // release_backspin
                    // release_sidespin
                    // release_gyrospin
                    // distance_to_no_spinning_pitch
                    // pre_pitch_outs
                    // at_bat_event
                    // extension
                    // spin_axis_angle
                    // home_time_zone

                    for (pugi::xml_node pitchIterator = atBatIterator.first_child(); pitchIterator; pitchIterator = pitchIterator.next_sibling())
                    {
                        if (std::string(pitchIterator.name()) == "pitch")
                        {
                            mlb::io::CPitch pitch = header;

                            // pitch_description
                            // pitch_number
                            // umpire_call_type

                            // sportvision_pitch_id
                            if (!pitchIterator.attribute("sv_id").empty())
                                pitch.SetSportvisionPitchId(pitchIterator.attribute("sv_id").as_string());

                            // release_time
                            // event_number
                            // measurement_id

                            // speed_at_50_feet
                            if (!pitchIterator.attribute("start_speed").empty())
                                pitch.SetSpeedAt50Feet(UnitConversion::MilesPerHour2FeetPerSecond(pitchIterator.attribute("start_speed").as_double()));

                            // zone_speed
                            // strike_zone_top
                            // strike_zone_bottom
                            // horizontal_break
                            // break_angle
                            // break_length
                            // zone_position_x
                            // zone_position_z
                            // position_at_50_feet_x
                            // position_at_50_feet_y
                            //vx0 = "8.838"
                            //vy0 = "-132.222"
                            //vz0 = "-6.684"
                            //ax = "-18.817"
                            //ay = "30.385"
                            //az = "-20.652"
                            // pitch_type_code
                            // pitch_type_confidence
                            // pitch_zone
                            // spin_dir
                            // spin_rate
                            // player_on_1st_mlb_id
                            // player_on_2nd_mlb_id
                            // player_on_3rd_mlb_id
                            // pre_pitch_balls
                            // pre_pitch_strikes

                            m_pitchArray.push_back(pitch);
                        }
                    }
                }
            }
        }
    }

    if (m_pitchArray.empty())
        return true;

	return true;
}
//
//// Url on http://gd2.mlb.com/components/game/mlb/
//bool mlb::io::CPitchFxIo::LoadFromGameDirectory(const std::string& gameDirectory, my::int32 inning)
//{
//    HEALTH_CHECK(gameDirectory.empty(), false);
//
//    m_pitchArray.clear();
//
//    std::string url = gameDirectory + "inning/inning_" + boost::lexical_cast<std::string>(inning) + ".xml";
//
//    std::string xmlResponse;
//
//    if (!my::file::GetUrlAsString(url, xmlResponse))
//    {
//        LOG_ERROR();
//
//        return false;
//    }
//
//    HEALTH_CHECK(xmlResponse.empty(), false);
//    
//    if (xmlResponse.find("404 Not Found") != std::string::npos)
//    {
//        LOG_MESSAGE("The URL " + url + " answered with a '404 Not Found'.");
//
//        return false;
//    }
//    
//    HEALTH_CHECK(xmlResponse.find("Error (404)") != std::string::npos, false);
//
//    pugi::xml_document xmlDocument;
//    pugi::xml_parse_result xmlDocumentLog;
//
//    xmlDocumentLog = xmlDocument.load(xmlResponse.c_str());
//
//    if (!xmlDocumentLog)
//    {
//        LOG_MESSAGE(xmlDocumentLog.description());
//        LOG_MESSAGE(std::string("Error offset: ") + boost::lexical_cast<std::string>(xmlDocumentLog.offset));
//
//        return false;
//    }
//
//    pugi::xml_node inningIterator = xmlDocument.child("inning");
//
//    std::vector<std::string> inningHalfArray;
//
//    inningHalfArray.push_back("top");
//    inningHalfArray.push_back("bottom");
//
//    for (std::vector<std::string>::const_iterator inningHalfIterator = inningHalfArray.begin(); inningHalfIterator != inningHalfArray.end(); ++inningHalfIterator)
//    {
//        pugi::xml_node atBatArray = inningIterator.child(inningHalfIterator->c_str());
//
//        for (pugi::xml_node atBatIterator = atBatArray.first_child(); atBatIterator; atBatIterator = atBatIterator.next_sibling())
//        {
//            if (std::string(atBatIterator.name()) == "atbat")
//            {
//                mlb::io::CPitch header;
//
//                // game_primary_key
//
//                // mlb_game_string
//                // (BEGIN OF) DEBUG ONLY! (12-Nov-2015) TESTING OF THE AUDITING TOOL!
//                if (gameDirectory == "https://dl.dropboxusercontent.com/u/18051244/2015_08_11_anamlb_chamlb_1/")
//                    header.SetMlbGameString("2015_08_11_anamlb_chamlb_1");
//                else
//                // (END OF) DEBUG ONLY! (12-Nov-2015) TESTING OF THE AUDITING TOOL!
//                header.SetMlbGameString(mlb::io::CGamedayServer().GetMlbGameStringFromGameDirectory(gameDirectory));
//
//                // inning
//                header.SetInning(inning);
//
//                // top_of_inning
//                header.SetTopOfInning((*inningHalfIterator) == "top" ? true : false);
//
//                // game_at_bat_number
//                if (!atBatIterator.attribute("num").empty())
//                    header.SetGameAtbatNumber(atBatIterator.attribute("num").as_int());
//
//                // pitcher_mlb_id
//                if (!atBatIterator.attribute("pitcher").empty())
//                    header.SetPitcherMlbId(atBatIterator.attribute("pitcher").as_int());
//
//                // pitcher_throws
//                if (!atBatIterator.attribute("p_throws").empty())
//                    header.SetPitcherThrows(atBatIterator.attribute("p_throws").as_string());
//
//                // batter_mlb_id
//                if (!atBatIterator.attribute("batter").empty())
//                    header.SetBatterMlbId(atBatIterator.attribute("batter").as_int());
//
//                // batter_side
//                if (!atBatIterator.attribute("stand").empty())
//                    header.SetBatterSide(atBatIterator.attribute("stand").as_string());
//
//                // release_speed
//                // sequence_number
//                // pitch_event_type
//                // release_position_x
//                // release_position_y
//                // release_position_z
//                // effective_release_speed
//                // release_horizontal_angle
//                // release_vertical_angle
//                // zone_time
//                // game_type
//                // release_backspin
//                // release_sidespin
//                // release_gyrospin
//                // distance_to_no_spinning_pitch
//                // pre_pitch_outs
//                // at_bat_event
//                // extension
//                // spin_axis_angle
//                // home_time_zone
//
//                my::int32 prePitchBallCount = 0,
//                    prePitchStrikeCount = 0;
//                // BUG: (25-Mar-2016) INCONSISTENT GAMEDAY DATA
//                // http://gd2.mlb.com/components/game/mlb/year_2015/month_03/day_04/gid_2015_03_04_detmlb_balmlb_1/inning/inning_all.xml
//                // <atbat num="30" b="0" s="0" event="Flyout" ... <pitch des="In play, out(s)" id="172" type="X"
//                // http://gd2.mlb.com/components/game/mlb/year_2015/month_03/day_01/gid_2015_03_01_utabbc_phimlb_1/inning/inning_all.xml
//                // <atbat num="4" b="0" s="1" event="Flyout" <pitch des="In play, out(s)" id="23" type="X"
//                bool inPlayOutGamedayError = false;
//
//                for (pugi::xml_node pitchIterator = atBatIterator.first_child(); pitchIterator; pitchIterator = pitchIterator.next_sibling())
//                {
//                    if (std::string(pitchIterator.name()) == "pitch")
//                    {
//                        mlb::io::CPitch pitch = header;
//
//                        // pitch_description
//                        if (!pitchIterator.attribute("des").empty())
//                            pitch.SetPitchDescription(mlb::GetPitchDescriptionTypeFromString(pitchIterator.attribute("des").as_string()));
//
//                        // pitch_number
//                        if (!pitchIterator.attribute("id").empty())
//                            pitch.SetPitchNumber(pitchIterator.attribute("id").as_int());
//
//                        // umpire_call_type
//                        if (!pitchIterator.attribute("type").empty())
//                            pitch.SetUmpireCallType(mlb::GetUmpireCallTypeFromString(pitchIterator.attribute("type").as_string()));
//
//                        // sportvision_pitch_id
//                        if (!pitchIterator.attribute("sv_id").empty())
//                            pitch.SetSportvisionPitchId(pitchIterator.attribute("sv_id").as_string());
//
//                        // release_time
//                        if (pitch.GetSportvisionPitchId() != mlb::Null<std::string>())
//                        {
//                            my::CTimestamp timestamp(mlb::io::CGamedayServer().GetTimestampFromSportvisionPitchId(pitch.GetSportvisionPitchId()));
//
//                            pitch.SetReleaseTime(timestamp.ToMicrosecondsFromEpoch());
//                        }
//
//                        std::string tfsZulu = pitchIterator.attribute("tfs_zulu").as_string();
//
//                        if (tfsZulu.empty())
//                        {
//                            LOG_ERROR();
//
//                            return false;
//                        }
//
//                        if (tfsZulu.back() == 'Z')
//                            tfsZulu.pop_back();
//
//                        // TODO: (25-Mar-2016) REPLACE DEPRECATED TIMESTAMP DATA TYPE   
//                        if (pitch.GetReleaseTime() == mlb::Null<INT64>())
//                        {
//                            pitch.SetReleaseTime(CUnixEpoch(tfsZulu, "%Y-%m-%d %H:%M:%S").Get());
//                            //pitch.SetReleaseTime(my::CTimestamp(tfsZulu).ToMicrosecondsFromEpoch());
//                        }
//
//                        // event_number
//                        if (!pitchIterator.attribute("event_num").empty())
//                            pitch.SetEventNumber(pitchIterator.attribute("event_num").as_int());
//
//                        // measurement_id
//                        if (!pitchIterator.attribute("play_guid").empty())
//                            pitch.SetMeasurementId(pitchIterator.attribute("play_guid").as_string());
//
//                        // speed_at_50_feet
//                        if (!pitchIterator.attribute("start_speed").empty())
//                            pitch.SetSpeedAt50Feet(UnitConversion::MilesPerHour2FeetPerSecond(pitchIterator.attribute("start_speed").as_double()));
//
//                        // zone_speed
//                        if (!pitchIterator.attribute("end_speed").empty())
//                            pitch.SetZoneSpeed(UnitConversion::MilesPerHour2FeetPerSecond(pitchIterator.attribute("end_speed").as_double()));
//
//                        // strike_zone_top
//                        if (!pitchIterator.attribute("sz_top").empty())
//                            pitch.SetStrikeZoneTop(pitchIterator.attribute("sz_top").as_float());
//
//                        // strike_zone_bottom
//                        if (!pitchIterator.attribute("sz_bot").empty())
//                            pitch.SetStrikeZoneBottom(pitchIterator.attribute("sz_bot").as_float());
//
//                        // horizontal_break
//                        if (!pitchIterator.attribute("pfx_x").empty())
//                            pitch.SetHorizontalBreak(pitchIterator.attribute("pfx_x").as_float());
//
//                        // vertical_break
//                        if (!pitchIterator.attribute("pfx_x").empty())
//                            pitch.SetVerticalBreak(pitchIterator.attribute("pfx_x").as_float());
//
//                        // break_angle
//                        if (!pitchIterator.attribute("break_angle").empty())
//                            pitch.SetBreakAngle(pitchIterator.attribute("break_angle").as_float());
//
//                        // break_length
//                        if (!pitchIterator.attribute("break_length").empty())
//                            pitch.SetBreakLength(pitchIterator.attribute("break_length").as_float());
//
//                        // zone_position_x
//                        if (!pitchIterator.attribute("px").empty())
//                        {
//                            const double *zonePosition = pitch.GetZonePosition();
//
//                            HEALTH_CHECK(!zonePosition, false);
//
//                            pitch.SetZonePosition(pitchIterator.attribute("px").as_float(), UnitConversion::Inch2Feet(17.0), zonePosition[2]);
//                        }
//
//                        // zone_position_z
//                        if (!pitchIterator.attribute("pz").empty())
//                        {
//                            const double *zonePosition = pitch.GetZonePosition();
//
//                            HEALTH_CHECK(!zonePosition, false);
//
//                            pitch.SetZonePosition(zonePosition[0], UnitConversion::Inch2Feet(17.0), pitchIterator.attribute("pz").as_float());
//                        }
//
//                        // position_at_50_feet_x
//                        if (!pitchIterator.attribute("x0").empty())
//                        {
//                            const double *releasePosition = pitch.GetPositionAt50Feet();
//
//                            HEALTH_CHECK(!releasePosition, false);
//
//                            pitch.SetPositionAt50Feet(pitchIterator.attribute("x0").as_float(), releasePosition[1], releasePosition[2]);
//                        }
//
//                        // position_at_50_feet_y
//                        if (!pitchIterator.attribute("y0").empty())
//                        {
//                            const double *releasePosition = pitch.GetPositionAt50Feet();
//
//                            HEALTH_CHECK(!releasePosition, false);
//
//                            pitch.SetPositionAt50Feet(releasePosition[0], pitchIterator.attribute("y0").as_float(), releasePosition[2]);
//                        }
//
//                        // position_at_50_feet_z
//                        if (!pitchIterator.attribute("z0").empty())
//                        {
//                            const double *releasePosition = pitch.GetPositionAt50Feet();
//
//                            HEALTH_CHECK(!releasePosition, false);
//
//                            pitch.SetPositionAt50Feet(releasePosition[0], releasePosition[1], pitchIterator.attribute("z0").as_float());
//                        }
//
//                        //vx0 = "8.838"
//                        if (!pitchIterator.attribute("vx0").empty())
//                        {
//                            const double *releaseVelocity = pitch.GetVelocityAt50Feet();
//
//                            HEALTH_CHECK(!releaseVelocity, false);
//
//                            pitch.SetVelocityAt50Feet(pitchIterator.attribute("vx0").as_float(), releaseVelocity[1], releaseVelocity[2]);
//                        }
//
//                        //vy0 = "-132.222"
//                        if (!pitchIterator.attribute("vy0").empty())
//                        {
//                            const double *releaseVelocity = pitch.GetVelocityAt50Feet();
//
//                            HEALTH_CHECK(!releaseVelocity, false);
//
//                            pitch.SetVelocityAt50Feet(releaseVelocity[0], pitchIterator.attribute("vy0").as_float(), releaseVelocity[2]);
//                        }
//
//                        //vz0 = "-6.684"
//                        if (!pitchIterator.attribute("vz0").empty())
//                        {
//                            const double *releaseVelocity = pitch.GetVelocityAt50Feet();
//
//                            HEALTH_CHECK(!releaseVelocity, false);
//
//                            pitch.SetVelocityAt50Feet(releaseVelocity[0], releaseVelocity[1], pitchIterator.attribute("vz0").as_float());
//                        }
//
//                        //ax = "-18.817"
//                        if (!pitchIterator.attribute("ax").empty())
//                        {
//                            const double *releaseAcceleration = pitch.GetAccelerationAt50Feet();
//
//                            HEALTH_CHECK(!releaseAcceleration, false);
//
//                            pitch.SetAccelerationAt50Feet(pitchIterator.attribute("ax").as_float(), releaseAcceleration[1], releaseAcceleration[2]);
//                        }
//
//                        //ay = "30.385"
//                        if (!pitchIterator.attribute("ay").empty())
//                        {
//                            const double *releaseAcceleration = pitch.GetAccelerationAt50Feet();
//
//                            HEALTH_CHECK(!releaseAcceleration, false);
//
//                            pitch.SetAccelerationAt50Feet(releaseAcceleration[0], pitchIterator.attribute("ay").as_float(), releaseAcceleration[2]);
//                        }
//
//                        //az = "-20.652"
//                        if (!pitchIterator.attribute("az").empty())
//                        {
//                            const double *releaseAcceleration = pitch.GetAccelerationAt50Feet();
//
//                            HEALTH_CHECK(!releaseAcceleration, false);
//
//                            pitch.SetAccelerationAt50Feet(releaseAcceleration[0], releaseAcceleration[1], pitchIterator.attribute("az").as_float());
//                        }
//
//                        // pitch_type_code
//                        if (!pitchIterator.attribute("pitch_type").empty())
//                            pitch.SetPitchTypeCode(mlb::GetPitchTypeFromString(pitchIterator.attribute("pitch_type").as_string()));
//
//                        // pitch_type_confidence
//                        if (!pitchIterator.attribute("type_confidence").empty())
//                            pitch.SetPitchTypeConfidence(pitchIterator.attribute("type_confidence").as_float());
//
//                        // pitch_zone
//                        if (!pitchIterator.attribute("zone").empty())
//                            pitch.SetPitchZone(pitchIterator.attribute("zone").as_int());
//
//                        // spin_rate
//                        if (!pitchIterator.attribute("spin_rate").empty())
//                            pitch.SetSpinRate(pitchIterator.attribute("spin_rate").as_float());
//
//                        // player_on_1st_mlb_id
//                        if (!pitchIterator.attribute("on_1b").empty())
//                            pitch.SetPlayerOn1stMlbId(pitchIterator.attribute("on_1b").as_int());
//                        // player_on_2nd_mlb_id
//                        if (!pitchIterator.attribute("on_2b").empty())
//                            pitch.SetPlayerOn2ndMlbId(pitchIterator.attribute("on_2b").as_int());
//                        // player_on_3rd_mlb_id
//                        if (!pitchIterator.attribute("on_3b").empty())
//                            pitch.SetPlayerOn3rdMlbId(pitchIterator.attribute("on_3b").as_int());
//
//                        // pre_pitch_balls
//                        if (!my::IsNull(prePitchBallCount))
//                            pitch.SetPrePitchBalls(prePitchBallCount);
//                        // pre_pitch_strikes
//                        if (!my::IsNull(prePitchStrikeCount))
//                            pitch.SetPrePitchBalls(prePitchStrikeCount);
//
//                        switch (pitch.GetPitchDescription()) {
//                        case mlb::AUTOMATIC_BALL_PITCH_DESCRIPTION:
//                        case mlb::BALL_PITCH_DESCRIPTION:
//                        case mlb::BALL_IN_DIRT_PITCH_DESCRIPTION:
//                            if (!my::IsNull(prePitchBallCount))
//                                ++prePitchBallCount;
//                            break;
//
//                            // BUG: (25-Mar-2016) INCONSISTENT GAMEDAY DATA
//                            // http://gd2.mlb.com/components/game/mlb/year_2015/month_03/day_04/gid_2015_03_04_detmlb_balmlb_1/inning/inning_all.xml
//                            // <atbat num="30" b="0" s="0" event="Flyout" ... <pitch des="In play, out(s)" id="172" type="X"
//                            // http://gd2.mlb.com/components/game/mlb/year_2015/month_03/day_01/gid_2015_03_01_utabbc_phimlb_1/inning/inning_all.xml
//                            // <atbat num="4" b="0" s="1" event="Flyout" <pitch des="In play, out(s)" id="23" type="X"
//                        case mlb::IN_PLAY_OUT_PITCH_DESCRIPTION:
//                            inPlayOutGamedayError = true;
//
//                        case mlb::AUTOMATIC_STRIKE_PITCH_DESCRIPTION:
//                        case mlb::CALLED_STRIKE_PITCH_DESCRIPTION:
//                        case mlb::IN_PLAY_NO_OUT_PITCH_DESCRIPTION:
//                        case mlb::IN_PLAY_RUN_PITCH_DESCRIPTION:
//                        case mlb::SWINGING_STRIKE_PITCH_DESCRIPTION:
//                        case mlb::SWINGING_STRIKE_BLOCKED_PITCH_DESCRIPTION:
//                        case mlb::FOUL_TIP_PITCH_DESCRIPTION:
//                        case mlb::MISSED_BUNT_PITCH_DESCRIPTION:
//                            if (!my::IsNull(prePitchStrikeCount) &&
//                                (prePitchStrikeCount < 3))
//                            {
//                                ++prePitchStrikeCount;
//                            }
//                            break;
//
//                        case mlb::FOUL_RUNNER_GOING_PITCH_DESCRIPTION:
//                        case mlb::FOUL_BUNT_PITCH_DESCRIPTION:
//                        case mlb::FOUL_PITCH_DESCRIPTION:
//                            if (!my::IsNull(prePitchStrikeCount) &&
//                                (prePitchStrikeCount < 2))
//                            {
//                                ++prePitchStrikeCount;
//                            }
//                            break;
//
//                        case mlb::UNKNOWN_PITCH_DESCRIPTION:
//                            prePitchBallCount = my::Null<my::int32>();
//                            prePitchStrikeCount = my::Null<my::int32>();
//                            break;
//
//                        default:
//                            // Nothing by now
//                            break;
//                        }
//
//                        m_pitchArray.push_back(pitch);
//                    }
//                }
//
//                // ANNOYING MESSAGE (25-Feb-2016)
//                if ((!atBatIterator.attribute("b").empty()) &&
//                    !my::IsNull(prePitchBallCount))
//                {
//                    if (prePitchBallCount != atBatIterator.attribute("b").as_int())
//                        LOG_MESSAGE("Pre-pitch balls mismatch.");
//                }
//
//                // ANNOYING MESSAGE (25-Feb-2016)
//                if ((!atBatIterator.attribute("s").empty()) &&
//                    !my::IsNull(prePitchStrikeCount))
//                {
//                    my::int32 atBatPrePitchStrikeCount = atBatIterator.attribute("s").as_int();
//
//                    if (!inPlayOutGamedayError)
//                    {
//                        if (prePitchStrikeCount != atBatPrePitchStrikeCount)
//                            LOG_MESSAGE("Pre-pitch strikes mismatch.");
//                    }
//                    // BUG: (25-Mar-2016) INCONSISTENT GAMEDAY DATA
//                    // http://gd2.mlb.com/components/game/mlb/year_2015/month_03/day_04/gid_2015_03_04_detmlb_balmlb_1/inning/inning_all.xml
//                    // <atbat num="30" b="0" s="0" event="Flyout" ... <pitch des="In play, out(s)" id="172" type="X"
//                    // http://gd2.mlb.com/components/game/mlb/year_2015/month_03/day_01/gid_2015_03_01_utabbc_phimlb_1/inning/inning_all.xml
//                    // <atbat num="4" b="0" s="1" event="Flyout" <pitch des="In play, out(s)" id="23" type="X"
//                    else
//                    {
//                        if ((prePitchStrikeCount != atBatPrePitchStrikeCount) &&
//                            ((prePitchStrikeCount - 1) != atBatPrePitchStrikeCount))
//                        {
//                            LOG_MESSAGE("Pre-pitch strikes mismatch.");
//                        }
//                    }
//                }
//
//                if (!atBatIterator.attribute("des").empty())
//                    m_pitchArray.back().SetAtBatDescription(atBatIterator.attribute("des").as_string());
//
//                if (!atBatIterator.attribute("event").empty())
//                    m_pitchArray.back().SetAtBatEvent(mlb::GetAtBatEventTypeFromString(atBatIterator.attribute("event").as_string()));
//            }
//        }
//    }
//
//    return true;
//}

/**
*/
std::vector<mlb::io::CPitch> mlb::io::CPitchFxIo::GetPitchArray() const
{
    return m_pitchArray;
}

//// GID (2011_08_02_balmlb_kcamlb_1)
//bool mlb::io::CPitchFxIo::LoadFromMlbGameString(const std::string& mlbGameString)
//{
//    HEALTH_CHECK(mlbGameString.size() != std::string("2011_08_02_balmlb_kcamlb_1").size(), false);
//
//    std::string url = mlb::io::CGamedayServer().GetGameUrl(mlbGameString);
//
//    if (!LoadFromGameDirectory(url))
//    {
//        LOG_ERROR();
//
//        return false;
//    }
//
//    return true;
//}

/**
*/
void mlb::io::CPitchFxIo::Create()
{
	m_pitchArray.clear();
}

/**
*/
void mlb::io::CPitchFxIo::Copy(const CPitchFxIo& pitchFx)
{
	m_pitchArray = pitchFx.m_pitchArray;
}


