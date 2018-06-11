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

#include <boost/algorithm/string.hpp>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

#include "Logger.h"
#include "MyMath.h"

#include "Pitch.h"

/**
*/
mlb::io::CPitch::CPitch() 
{
    Create();
}

/**
*/
mlb::io::CPitch::CPitch(const CPitch& pitch) 
{
    Copy(pitch);
}

/**
*/
void mlb::io::CPitch::operator=(const CPitch& pitch) 
{
    Copy(pitch);
}

///**
//*/
//my::int32 mlb::io::CPitch::GetGamePrimaryKey() const
//{
//	return m_gamePrimaryKey;
//}
//
///**
//*/
//void mlb::io::CPitch::SetGamePrimaryKey(my::int32 gamePrimaryKey)
//{
//	m_gamePrimaryKey = gamePrimaryKey;
//}

/**
*/
std::string mlb::io::CPitch::GetMlbGameString() const
{
    return m_mlbGameString;
}

/**
*/
void mlb::io::CPitch::SetMlbGameString(std::string mlbGameString)
{
    m_mlbGameString = mlbGameString;
}

/**
*/
std::string mlb::io::CPitch::GetSportvisionPitchId() const
{
	return m_sportvisionPitchId;
}

/**
*/
void mlb::io::CPitch::SetSportvisionPitchId(std::string sportvisionPitchId)
{
	m_sportvisionPitchId = sportvisionPitchId;
}

///**
//*/
//mlb::PITCH_DESCRIPTION_TYPE mlb::io::CPitch::GetPitchDescription() const
//{
//	return m_pitchDescriptionType;
//}
//
///**
//*/
//void mlb::io::CPitch::SetPitchDescription(mlb::PITCH_DESCRIPTION_TYPE pitchDescriptionType)
//{
//	m_pitchDescriptionType = pitchDescriptionType;
//}
//
///**
//*/
//void mlb::io::CPitch::SetPitchDescriptionFromString(std::string pitchDescriptionString)
//{
//	m_pitchDescriptionType = mlb::GetPitchDescriptionTypeFromString(pitchDescriptionString);
//}
//
///**
//*/
//my::int32 mlb::io::CPitch::GetSequenceNumber() const
//{
//	return m_sequenceNumber;
//}
//
///**
//*/
//void mlb::io::CPitch::SetSequenceNumber(my::int32 sequenceNumber)
//{
//	m_sequenceNumber = sequenceNumber;
//}
//
///**
//*/
//my::int32 mlb::io::CPitch::GetGameAtbatNumber() const
//{
//	return m_gameAtBatNumber;
//}
//
///**
//*/
//void mlb::io::CPitch::SetGameAtbatNumber(my::int32 gameAtBatNumber)
//{
//	m_gameAtBatNumber = gameAtBatNumber;
//}
//
///**
//*/
//my::int32 mlb::io::CPitch::GetPitchNumber() const
//{
//	return m_pitchNumber;
//}
//
///**
//*/
//void mlb::io::CPitch::SetPitchNumber(my::int32 pitchNumber)
//{
//	m_pitchNumber = pitchNumber;
//}
//
///**
//*/
//my::int32 mlb::io::CPitch::GetAtbatPitchNumber() const
//{
//    return m_atBatPitchNumber;
//}
//
///**
//*/
//void mlb::io::CPitch::SetAtbatPitchNumber(my::int32 atBatPitchNumber)
//{
//    m_atBatPitchNumber = atBatPitchNumber;
//}
//
///**
//*/
//my::int32 mlb::io::CPitch::GetInning() const
//{
//	return m_inning;
//}
//
///**
//*/
//void mlb::io::CPitch::SetInning(my::int32 inning)
//{
//	m_inning = inning;
//}
//
///**
//*/
//bool mlb::io::CPitch::GetTopOfInning() const
//{
//	return m_topOfInning;
//}
//
///**
//*/
//void mlb::io::CPitch::SetTopOfInning(bool topOfInning)
//{
//	m_topOfInning = topOfInning;
//}
//
///**
//*/
//void mlb::io::CPitch::SetTopOfInningFromInt(my::int32 topOfInning)
//{
//	m_topOfInning = (topOfInning == 1);
//}
//
///**
//*/
//my::int32 mlb::io::CPitch::GetEventNumber() const
//{
//	return m_eventNumber;
//}
//
///**
//*/
//void mlb::io::CPitch::SetEventNumber(my::int32 eventNumber)
//{
//	m_eventNumber = eventNumber;
//}
//
///**
//*/
//mlb::PITCH_EVENT_TYPE mlb::io::CPitch::GetPitchEventType() const
//{
//	return m_pitchEventType;
//}
//
///**
//*/
//void mlb::io::CPitch::SetPitchEventType(mlb::PITCH_EVENT_TYPE pitchEventType)
//{
//	m_pitchEventType = pitchEventType;
//}
//
///**
//*/
//void mlb::io::CPitch::SetPitchEventTypeFromString(std::string pitchEventString)
//{
//	m_pitchEventType = mlb::GetPitchEventTypeFromString(pitchEventString);
//}
//
///**
//*/
//my::int32 mlb::io::CPitch::GetPrePitchBalls() const
//{
//	return m_prePitchBalls;
//}
//
///**
//*/
//void mlb::io::CPitch::SetPrePitchBalls(my::int32 prePitchBalls)
//{
//	m_prePitchBalls = prePitchBalls;
//}
//
///**
//*/
//my::int32 mlb::io::CPitch::GetPrePitchStrikes() const
//{
//	return m_prePitchStrikes;
//}
//
///**
//*/
//void mlb::io::CPitch::SetPrePitchStrikes(my::int32 prePitchStrikes)
//{
//	m_prePitchStrikes = prePitchStrikes;
//}
//
///**
//*/
//my::int32 mlb::io::CPitch::GetPrePitchOuts() const
//{
//    return m_prePitchOuts;
//}
//
///**
//*/
//void mlb::io::CPitch::SetPrePitchOuts(my::int32 prePitchOuts)
//{
//    m_prePitchOuts = prePitchOuts;
//}
//
///**
//*/
//my::int32 mlb::io::CPitch::GetPostPitchBalls() const
//{
//	return m_postPitchBalls;
//}
//
///**
//*/
//void mlb::io::CPitch::SetPostPitchBalls(my::int32 postPitchBalls)
//{
//	m_postPitchBalls = postPitchBalls;
//}
//
///**
//*/
//my::int32 mlb::io::CPitch::GetPostPitchStrikes() const
//{
//	return m_postPitchStrikes;
//}
//
///**
//*/
//void mlb::io::CPitch::SetPostPitchStrikes(my::int32 postPitchStrikes)
//{
//	m_postPitchStrikes = postPitchStrikes;
//}
//
///**
//*/
//my::int32 mlb::io::CPitch::GetPostPitchOuts() const
//{
//    return m_postPitchOuts;
//}
//
///**
//*/
//void mlb::io::CPitch::SetPostPitchOuts(my::int32 postPitchOuts)
//{
//    m_postPitchOuts = postPitchOuts;
//}
//
///**
//*/
//my::int32 mlb::io::CPitch::GetBatterMlbId() const
//{
//	return m_batterMlbId;
//}
//
///**
//*/
//void mlb::io::CPitch::SetBatterMlbId(my::int32 batterMlbId)
//{
//	m_batterMlbId = batterMlbId;
//}
//
///**
//*/
//std::string mlb::io::CPitch::GetBatterSide() const
//{
//	return m_batterSide;
//}
//
///**
//*/
//void mlb::io::CPitch::SetBatterSide(std::string batterSide)
//{
//	m_batterSide = batterSide;
//}
//
///**
//*/
//my::int32 mlb::io::CPitch::GetPitcherMlbId() const
//{
//	return m_pitcherMlbId;
//}
//
///**
//*/
//void mlb::io::CPitch::SetPitcherMlbId(my::int32 pitcherMlbId)
//{
//	m_pitcherMlbId = pitcherMlbId;
//}

/**
*/
std::string mlb::io::CPitch::GetPitcherThrows() const
{
	return m_pitcherThrows;
}

/**
*/
void mlb::io::CPitch::SetPitcherThrows(std::string pitcherThrows)
{
	m_pitcherThrows = pitcherThrows;
}

///**
//*/
//my::int32 mlb::io::CPitch::GetUmpireMlbId() const
//{
//    return m_umpireMlbId;
//}
//
///**
//*/
//void mlb::io::CPitch::SetUmpireMlbId(my::int32 umpireMlbId)
//{
//    m_umpireMlbId = umpireMlbId;
//}
//
///**
//*/
//const double *mlb::io::CPitch::GetReleasePosition() const
//{
//	return m_releasePosition;
//}
//
///**
//*/
//void mlb::io::CPitch::SetReleasePosition(double x, double y, double z)
//{
//	MyMath::Assign(x, y, z, m_releasePosition);
//}
//
///**
//*/
//const double *mlb::io::CPitch::GetReleaseVelocity() const
//{
//	return m_releaseVelocity;
//}
//
///**
//*/
//void mlb::io::CPitch::SetReleaseVelocity(double vx, double vy, double vz)
//{
//	MyMath::Assign(vx, vy, vz, m_releaseVelocity);
//}
//
///**
//*/
//const double *mlb::io::CPitch::GetReleaseAcceleration() const
//{
//	return m_releaseAcceleration;
//}
//
///**
//*/
//void mlb::io::CPitch::SetReleaseAcceleration(double ax, double ay, double az)
//{
//	MyMath::Assign(ax, ay, az, m_releaseAcceleration);
//}
//
///**
//*/
//double mlb::io::CPitch::GetReleaseSpeed() const
//{
//    return m_releaseSpeed;
//}
//
///**
//*/
//void mlb::io::CPitch::SetReleaseSpeed(double releaseSpeed)
//{
//    m_releaseSpeed = releaseSpeed;
//}
//
///**
//*/
//double mlb::io::CPitch::GetEffectiveReleaseSpeed() const
//{
//    return m_effectiveReleaseSpeed;
//}
//
///**
//*/
//void mlb::io::CPitch::SetEffectiveReleaseSpeed(double effectiveReleaseSpeed)
//{
//    m_effectiveReleaseSpeed = effectiveReleaseSpeed;
//}
//
///**
//*/
//double mlb::io::CPitch::GetReleaseHorizontalAngle() const
//{
//    return m_releaseHorizontalAngle;
//}
//
//// Launch horizontal angle in degrees, relative to the line from the center of the rubber to the tip of home plate at which the pitch was released. Positive towards 3B.
//void mlb::io::CPitch::SetReleaseHorizontalAngle(double horizontalAngleInDegrees)
//{
//    m_releaseHorizontalAngle = horizontalAngleInDegrees;
//}
//
///**
//*/
//double mlb::io::CPitch::GetReleaseVerticalAngle() const
//{
//    return m_releaseVerticalAngle;
//}
//
//// Launch vertical angle in degrees; positive is up, negative is down.
//void mlb::io::CPitch::SetReleaseVerticalAngle(double verticalAngleInDegrees)
//{
//    m_releaseVerticalAngle = verticalAngleInDegrees;
//}
//
///**
//*/
//const double *mlb::io::CPitch::GetPositionAt50Feet() const
//{
//	return m_positionAt50Feet;
//}
//
///**
//*/
//void mlb::io::CPitch::SetPositionAt50Feet(double x, double y, double z)
//{
//	MyMath::Assign(x, y, z, m_positionAt50Feet);
//}
//
///**
//*/
//const double *mlb::io::CPitch::GetVelocityAt50Feet() const
//{
//	return m_velocityAt50Feet;
//}
//
///**
//*/
//void mlb::io::CPitch::SetVelocityAt50Feet(double vx, double vy, double vz)
//{
//	MyMath::Assign(vx, vy, vz, m_velocityAt50Feet);
//}
//
///**
//*/
//const double *mlb::io::CPitch::GetAccelerationAt50Feet() const
//{
//	return m_accelerationAt50Feet;
//}
//
///**
//*/
//void mlb::io::CPitch::SetAccelerationAt50Feet(double ax, double ay, double az)
//{
//	MyMath::Assign(ax, ay, az, m_accelerationAt50Feet);
//}

/**
*/
double mlb::io::CPitch::GetSpeedAt50Feet() const
{
	return m_speedAt50Feet;
}

/**
*/
void mlb::io::CPitch::SetSpeedAt50Feet(double speedAt50Feet)
{
	m_speedAt50Feet = speedAt50Feet;
}

///**
//*/
//const double *mlb::io::CPitch::GetZonePosition() const
//{
//	return m_zonePosition;
//}
//
///**
//*/
//void mlb::io::CPitch::SetZonePosition(double x, double y, double z)
//{
//	MyMath::Assign(x, y, z, m_zonePosition);
//}
//
///**
//*/
//const double *mlb::io::CPitch::GetZoneVelocity() const
//{
//	return m_zoneVelocity;
//}
//
///**
//*/
//void mlb::io::CPitch::SetZoneVelocity(double vx, double vy, double vz)
//{
//	MyMath::Assign(vx, vy, vz, m_zoneVelocity);
//}
//
///**
//*/
//const double *mlb::io::CPitch::GetZoneAcceleration() const
//{
//	return m_zoneAcceleration;
//}
//
///**
//*/
//void mlb::io::CPitch::SetZoneAcceleration(double ax, double ay, double az)
//{
//	MyMath::Assign(ax, ay, az, m_zoneAcceleration);
//}
//
///**
//*/
//double mlb::io::CPitch::GetZoneSpeed() const
//{
//	return m_zoneSpeed;
//}
//
///**
//*/
//void mlb::io::CPitch::SetZoneSpeed(double zoneSpeed)
//{
//	m_zoneSpeed = zoneSpeed;
//}
//
///**
//*/
//double mlb::io::CPitch::GetHorizontalBreak() const
//{
//	return m_horizontalBreak;
//}
//
///**
//*/
//void mlb::io::CPitch::SetHorizontalBreak(double horizontalBreak)
//{
//	m_horizontalBreak = horizontalBreak;
//}
//
///**
//*/
//double mlb::io::CPitch::GetVerticalBreak() const
//{
//	return m_verticalBreak;
//}
//
///**
//*/
//void mlb::io::CPitch::SetVerticalBreak(double verticalBreak)
//{
//	m_verticalBreak = verticalBreak;
//}
//
///**
//*/
//double mlb::io::CPitch::GetVerticalBreakInduced() const
//{
//    return m_verticalBreakInduced;
//}
//
///**
//*/
//void mlb::io::CPitch::SetVerticalBreakInduced(double verticalBreakInduced)
//{
//    m_verticalBreakInduced = verticalBreakInduced;
//}
//
//// The angle, in degrees, from vertical to the straight line path from the release point to where the pitch crossed the front of home plate, as seen from the catcher's/umpire's perspective.
//double mlb::io::CPitch::GetBreakAngle() const
//{
//    return m_breakAngle;
//}
//
//// The angle, in degrees, from vertical to the straight line path from the release point to where the pitch crossed the front of home plate, as seen from the catcher's/umpire's perspective.
//void mlb::io::CPitch::SetBreakAngle(double breakAngle)
//{
//    m_breakAngle = breakAngle;
//}
//
//// The measurement of the greatest distance, in inches, between the trajectory of the pitch at any point between the release point and the front of home plate, and the straight line path from the release point and the front of home plate, per the MLB Gameday team. John Walsh's article "In Search of the Sinker" has a good illustration of this parameter.
//double mlb::io::CPitch::GetBreakLength() const
//{
//    return m_breakLength;
//}
//
//// The measurement of the greatest distance, in inches, between the trajectory of the pitch at any point between the release point and the front of home plate, and the straight line path from the release point and the front of home plate, per the MLB Gameday team. John Walsh's article "In Search of the Sinker" has a good illustration of this parameter.
//void mlb::io::CPitch::SetBreakLength(double breakLength)
//{
//    m_breakLength = breakLength;
//}
//
///**
//*/
//mlb::PITCH_TYPE_CODE mlb::io::CPitch::GetPitchTypeCode() const
//{
//	return m_pitchTypeCode;
//}
//
///**
//*/
//void mlb::io::CPitch::SetPitchTypeCode(mlb::PITCH_TYPE_CODE pitchTypeCode)
//{
//	m_pitchTypeCode = pitchTypeCode;
//}
//
///**
//*/
//void mlb::io::CPitch::SetPitchTypeCodeFromString(std::string pitchTypeCodeString)
//{
//	m_pitchTypeCode = mlb::GetPitchTypeFromString(pitchTypeCodeString);
//}
//
///**
//*/
//my::int64 mlb::io::CPitch::GetReleaseTime() const
//{
//	return m_releaseTime;
//}
//
///**
//*/
//void mlb::io::CPitch::SetReleaseTime(my::int64 releaseTime)
//{
//	m_releaseTime = releaseTime;
//}
//
///**
//*/
//my::int64 mlb::io::CPitch::GetZoneTime() const
//{
//	return m_zoneTime;
//}
//
///**
//*/
//void mlb::io::CPitch::SetZoneTime(my::int64 zoneTime)
//{
//    m_zoneTime = zoneTime;
//}
//
///**
//*/
//my::CTimestamp mlb::io::CPitch::GetTipOfTheHomePlateTimestamp() const
//{
//    return m_tipOfTheHomePlateTimestamp;
//}
//
//// LEGACY (TIMESTAMP FROM TIME)
//void mlb::io::CPitch::SetTipOfTheHomePlateTimestamp(my::int64 tipOfTheHomePlateTime)
//{
//    m_tipOfTheHomePlateTimestamp.Set(tipOfTheHomePlateTime);
//}
//
///**
//*/
//void mlb::io::CPitch::SetTipOfTheHomePlateTimestamp(my::CTimestamp tipOfTheHomePlateTimestamp)
//{
//    m_tipOfTheHomePlateTimestamp = tipOfTheHomePlateTimestamp;
//}
//
///**
//*/
//std::string mlb::io::CPitch::GetGameType() const
//{
//	return m_gameType;
//}
//
///**
//*/
//void mlb::io::CPitch::SetGameType(std::string gameType)
//{
//	m_gameType = gameType;
//}
//
///**
//*/
//double mlb::io::CPitch::GetStrikeZoneTop() const
//{
//	return m_strikeZoneTop;
//}
//
///**
//*/
//void mlb::io::CPitch::SetStrikeZoneTop(double strikeZoneTop)
//{
//	m_strikeZoneTop = strikeZoneTop;
//}
//
///**
//*/
//double mlb::io::CPitch::GetStrikeZoneBottom() const
//{
//	return m_strikeZoneBottom;
//}
//
///**
//*/
//void mlb::io::CPitch::SetStrikeZoneBottom(double strikeZoneBottom)
//{
//	m_strikeZoneBottom = strikeZoneBottom;
//}
//
///**
//*/
//double mlb::io::CPitch::GetPitchTypeConfidence() const
//{
//	return m_pitchTypeConfidence;
//}
//
///**
//*/
//void mlb::io::CPitch::SetPitchTypeConfidence(double pitchTypeConfidence)
//{
//	m_pitchTypeConfidence = pitchTypeConfidence;
//}
//
///**
//*/
//double mlb::io::CPitch::GetSpinRate() const
//{
//	return m_spinRate;
//}
//
///**
//*/
//void mlb::io::CPitch::SetSpinRate(double spinRate)
//{
//	m_spinRate = spinRate;
//}
//
///**
//*/
//const double *mlb::io::CPitch::GetSpinAxis() const
//{
//	return m_spinAxis;
//}
//
///**
//*/
//void mlb::io::CPitch::SetSpinAxis(double x, double y, double z)
//{
//	MyMath::Assign(x, y, z, m_spinAxis);
//}
//
//// Backspin (pitch) in rpm; positive for backspin (upward movement); negative for topspin (downward movement).
//double mlb::io::CPitch::GetReleaseBackspin() const
//{
//    return m_releaseBackspinRpm;
//}
//
//// Backspin (pitch) in rpm; positive for backspin (upward movement); negative for topspin (downward movement).
//void mlb::io::CPitch::SetReleaseBackspin(double releaseBackspin)
//{
//    m_releaseBackspinRpm = releaseBackspin;
//}
//
//// Sidespin (yaw) in rpm; positive for pitch movement to pitcher's left and batted ball movement toward LF.
//double mlb::io::CPitch::GetReleaseSidespin() const
//{
//    return m_releaseSidespinRpm;
//}
//
//// Sidespin (yaw) in rpm; positive for pitch movement to pitcher's left and batted ball movement toward LF.
//void mlb::io::CPitch::SetReleaseSidespin(double releaseSidespin)
//{
//    m_releaseSidespinRpm = releaseSidespin;
//}
//
//// Gyrospin (roll) in rpm.
//double mlb::io::CPitch::GetReleaseGyrospin() const
//{
//    return m_releaseGyrospinRpm;
//}
//
//// Gyrospin (roll) in rpm.
//void mlb::io::CPitch::SetReleaseGyrospin(double releaseGyrospin)
//{
//    m_releaseGyrospinRpm = releaseGyrospin;
//}
//
///**
//*/
//double mlb::io::CPitch::GetDistanceToNoSpinningPitch() const
//{
//	return m_distanceToNoSpinningPitch;
//}
//
///**
//*/
//void mlb::io::CPitch::SetDistanceToNoSpinningPitch(double distanceToNoSpinningPitch)
//{
//	m_distanceToNoSpinningPitch = distanceToNoSpinningPitch;
//}
//
///**
//*/
//my::int32 mlb::io::CPitch::GetPitchZone() const
//{
//	return m_pitchZone;
//}
//
///**
//*/
//void mlb::io::CPitch::SetPitchZone(my::int32 pitchZone)
//{
//	m_pitchZone = pitchZone;
//}
//
///**
//*/
//mlb::UMPIRE_CALL_TYPE_CODE mlb::io::CPitch::GetUmpireCallType() const
//{
//	return m_umpireCallType;
//}
//
///**
//*/
//void mlb::io::CPitch::SetUmpireCallType(mlb::UMPIRE_CALL_TYPE_CODE umpireCallType)
//{
//	m_umpireCallType = umpireCallType;
//}
//
///**
//*/
//void mlb::io::CPitch::SetUmpireCallTypeFromString(std::string umpireCallTypeString)
//{
//	m_umpireCallType = mlb::GetUmpireCallTypeFromString(umpireCallTypeString);
//}
//
///**
//*/
//std::string mlb::io::CPitch::GetAtBatDescription() const
//{
//	return m_atBatDescription;
//}
//
///**
//*/
//void mlb::io::CPitch::SetAtBatDescription(std::string atBatDescription)
//{
//	m_atBatDescription = atBatDescription;
//}
//
///**
//*/
//mlb::AT_BAT_EVENT_TYPE mlb::io::CPitch::GetAtBatEvent() const
//{
//	return m_atBatEvent;
//}
//
///**
//*/
//void mlb::io::CPitch::SetAtBatEvent(mlb::AT_BAT_EVENT_TYPE atBatEvent)
//{
//	m_atBatEvent = atBatEvent;
//}
//
///**
//*/
//void mlb::io::CPitch::SetAtBatEventFromString(std::string atBatEventString)
//{
//	m_atBatEvent = mlb::GetAtBatEventTypeFromString(atBatEventString);
//}
//
///**
//*/
//my::int32 mlb::io::CPitch::GetPlayerOn1stMlbId() const
//{
//	return m_playerOn1stMlbId;
//}
//
///**
//*/
//void mlb::io::CPitch::SetPlayerOn1stMlbId(my::int32 playerOn1stMlbId)
//{
//	m_playerOn1stMlbId = playerOn1stMlbId;
//}
//
///**
//*/
//my::int32 mlb::io::CPitch::GetPlayerOn2ndMlbId() const
//{
//	return m_playerOn2ndMlbId;
//}
//
///**
//*/
//void mlb::io::CPitch::SetPlayerOn2ndMlbId(my::int32 playerOn2ndMlbId)
//{
//	m_playerOn2ndMlbId = playerOn2ndMlbId;
//}
//
///**
//*/
//my::int32 mlb::io::CPitch::GetPlayerOn3rdMlbId() const
//{
//	return m_playerOn3rdMlbId;
//}
//
///**
//*/
//void mlb::io::CPitch::SetPlayerOn3rdMlbId(my::int32 playerOn3rdMlbId)
//{
//	m_playerOn3rdMlbId = playerOn3rdMlbId;
//}
//
///**
//*/
//void mlb::io::CPitch::AddtTargetPosition(const mlb::io::CTargetPosition& targetPosition)
//{
//	m_targetPositionArray.push_back(targetPosition);
//}
//
///**
//*/
//std::vector<mlb::io::CTargetPosition> mlb::io::CPitch::GetTargetPositionArray() const
//{
//	return m_targetPositionArray;
//}
//
///**
//*/
//void mlb::io::CPitch::ClearTargetPositionArray()
//{
//	m_targetPositionArray.clear();
//}
//
///**
//*/
//double mlb::io::CPitch::GetExtension() const
//{
//	return m_extension;
//}
//
///**
//*/
//void mlb::io::CPitch::SetExtension(double extension)
//{
//	m_extension = extension;
//}
//
///**
//*/
//double mlb::io::CPitch::GetSpinAxisAngle() const
//{
//	return m_spinAxisAngle;
//}
//
///**
//*/
//void mlb::io::CPitch::SetSpinAxisAngle(double spinAxisAngle)
//{
//	m_spinAxisAngle = spinAxisAngle;
//}
//
//// DEBUG ONLY! (05-Jan-2015)
//std::string mlb::io::CPitch::GetMeasurementId() const
//{
//	return m_measurementId;
//}
//
//// DEBUG ONLY! (05-Jan-2015)
//void mlb::io::CPitch::SetMeasurementId(std::string measurementId)
//{
//	m_measurementId = measurementId;
//}
//
///**
//*/
//my::int64 mlb::io::CPitch::GetHomeTimeZone() const
//{
//    return m_homeTimeZone;
//}
//
///**
//*/
//void mlb::io::CPitch::SetHomeTimeZone(my::int64 homeTimeZone)
//{
//    m_homeTimeZone = homeTimeZone;
//}
//
///**
//*/
//mlb::PITCHING_POSITION_TYPE_CODE mlb::io::CPitch::GetPitchingPositionType() const
//{
//    return m_pitchingPositionType;
//}
//
///**
//*/
//void mlb::io::CPitch::SetPitchingPositionType(mlb::PITCHING_POSITION_TYPE_CODE pitchingPositionType)
//{
//    m_pitchingPositionType = pitchingPositionType;
//}
//
///**
//*/
//my::CVector3<double> mlb::io::CPitch::GetLastMeasuredPosition() const
//{
//    return m_lastMeasuredPosition;
//}
//
///**
//*/
//void mlb::io::CPitch::SetLastMeasuredPosition(double x, double y, double z)
//{
//    m_lastMeasuredPosition.Set(x, y, z);
//}
//
///**
//*/
//std::string mlb::io::CPitch::ToCsv(std::string delimiter) const
//{
//	CCsvOutputFile csvFile;
//	std::string emptyString;
//
//	HEALTH_CHECK(!csvFile.WriteValue(GetGamePrimaryKey()), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(GetMlbGameString()), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(GetSportvisionPitchId()), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(mlb::GetPitchDescriptionTypeString(GetPitchDescription())), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(GetSequenceNumber()), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(GetGameAtbatNumber()), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(GetPitchNumber()), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(GetInning()), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(GetTopOfInning()), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(GetEventNumber()), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(mlb::GetPitchEventTypeString(GetPitchEventType())), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(GetPrePitchBalls()), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(GetPrePitchStrikes()), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(GetPostPitchBalls()), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(GetPostPitchStrikes()), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(GetBatterMlbId()), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(GetBatterSide()), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(GetPitcherMlbId()), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(GetPitcherThrows()), emptyString);
//	HEALTH_CHECK(!csvFile.WriteBuffer(GetReleasePosition(), 3), emptyString);
//	HEALTH_CHECK(!csvFile.WriteBuffer(GetReleaseVelocity(), 3), emptyString);
//	HEALTH_CHECK(!csvFile.WriteBuffer(GetReleaseAcceleration(), 3), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(GetReleaseSpeed()), emptyString);
//	HEALTH_CHECK(!csvFile.WriteBuffer(GetPositionAt50Feet(), 3), emptyString);
//	HEALTH_CHECK(!csvFile.WriteBuffer(GetVelocityAt50Feet(), 3), emptyString);
//	HEALTH_CHECK(!csvFile.WriteBuffer(GetAccelerationAt50Feet(), 3), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(GetSpeedAt50Feet()), emptyString);
//	HEALTH_CHECK(!csvFile.WriteBuffer(GetZonePosition(), 3), emptyString);
//	HEALTH_CHECK(!csvFile.WriteBuffer(GetZoneVelocity(), 3), emptyString);
//	HEALTH_CHECK(!csvFile.WriteBuffer(GetZoneAcceleration(), 3), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(GetZoneSpeed()), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(GetHorizontalBreak()), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(GetVerticalBreak()), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(mlb::GetPitchTypeString(GetPitchTypeCode())), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(GetReleaseTime()), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(GetZoneTime()), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(GetGameType()), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(GetStrikeZoneTop()), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(GetStrikeZoneBottom()), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(GetPitchTypeConfidence()), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(GetSpinRate()), emptyString);
//	HEALTH_CHECK(!csvFile.WriteBuffer(GetSpinAxis(), 3), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(GetDistanceToNoSpinningPitch()), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(GetPitchZone()), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(mlb::GetUmpireCallTypeString(GetUmpireCallType())), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(GetPrePitchOuts()), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(GetAtBatDescription()), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(mlb::GetAtBatEventTypeString(GetAtBatEvent())), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(GetPlayerOn1stMlbId()), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(GetPlayerOn2ndMlbId()), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(GetPlayerOn3rdMlbId()), emptyString);
//	HEALTH_CHECK(!csvFile.WriteValue(GetExtension()), emptyString);
//    HEALTH_CHECK(!csvFile.WriteValue(GetSpinAxisAngle()), emptyString);
//    HEALTH_CHECK(!csvFile.WriteValue(GetMeasurementId()), emptyString);
//
//	// TRICKY: (22-Dec-2014)
//	std::vector<mlb::io::CTargetPosition> targetPositionArray = GetTargetPositionArray();
//	HEALTH_CHECK(!csvFile.WriteValue((my::int32)targetPositionArray.size()), emptyString);
//	for (std::vector<mlb::io::CTargetPosition>::const_iterator targetPositionIterator = targetPositionArray.begin(); targetPositionIterator != targetPositionArray.end(); ++targetPositionIterator)
//	{
//		HEALTH_CHECK(!csvFile.WriteValue(targetPositionIterator->ToCsv(delimiter)), emptyString);
//	}
//
//	return csvFile.ToString(delimiter);
//}
//
///**
//*/
//bool mlb::io::CPitch::FromCsv(std::string csvString, std::string delimiter)
//{
//	CCsvInputFile csvFile;
//
//	if (!csvFile.FromString(csvString, delimiter))
//	{
//		LOG_ERROR();
//
//		return false;
//	}
//
//	if (!FromCsv(csvFile))
//	{
//		LOG_ERROR();
//
//		return false;
//	}
//
//	if (!csvFile.isEmpty())
//		LOG_ERROR();
//
//	return true;
//}
//
///**
//*/
//bool mlb::io::CPitch::FromCsv(CCsvInputFile& csvFile)
//{
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetGamePrimaryKey, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetMlbGameString, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetSportvisionPitchId, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetPitchDescriptionFromString, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetSequenceNumber, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetGameAtbatNumber, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetPitchNumber, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetInning, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetTopOfInningFromInt, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetEventNumber, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetPitchEventTypeFromString, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetPrePitchBalls, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetPrePitchStrikes, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetPostPitchBalls, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetPostPitchStrikes, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetBatterMlbId, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetBatterSide, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetPitcherMlbId, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetPitcherThrows, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetReleasePosition, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetReleaseVelocity, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetReleaseAcceleration, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetReleaseSpeed, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetPositionAt50Feet, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetVelocityAt50Feet, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetAccelerationAt50Feet, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetSpeedAt50Feet, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetZonePosition, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetZoneVelocity, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetZoneAcceleration, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetZoneSpeed, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetHorizontalBreak, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetVerticalBreak, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetPitchTypeCodeFromString, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetReleaseTime, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetZoneTime, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetGameType, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetStrikeZoneTop, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetStrikeZoneBottom, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetPitchTypeConfidence, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetSpinRate, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetSpinAxis, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetDistanceToNoSpinningPitch, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetPitchZone, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetUmpireCallTypeFromString, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetPrePitchOuts, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetAtBatDescription, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetAtBatEventFromString, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetPlayerOn1stMlbId, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetPlayerOn2ndMlbId, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetPlayerOn3rdMlbId, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetExtension, *this), false);
//	HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetSpinAxisAngle, *this), false);
//    HEALTH_CHECK(!csvFile.ReadValue(&mlb::io::CPitch::SetMeasurementId, *this), false);
//
//	// TRICKY: (22-Dec-2014)
//	my::int32 targetPositionCount = 0;
//	if (!csvFile.ReadValue(targetPositionCount))
//	{
//		LOG_ERROR();
//		return false;
//	}
//	for (my::int32 targetPositionIndex = 0; targetPositionIndex < targetPositionCount; ++targetPositionIndex)
//	{
//		mlb::io::CTargetPosition targetPosition;
//		if (!targetPosition.FromCsv(csvFile))
//		{
//			LOG_ERROR();
//			return false;
//		}
//		AddtTargetPosition(targetPosition);
//	}
//
//	return true;
//}
//
//template <>
//void mlb::io::CPitch::ToJson(rapidjson::Writer<rapidjson::StringBuffer>& jsonWriter) const
//{
//#define JSON_WRITE_BOOLEAN(NAME, ATTRIBUTE) jsonWriter.String(NAME);\
//    jsonWriter.Bool(ATTRIBUTE);\
//    
//#define JSON_WRITE_my::int32(NAME, ATTRIBUTE, NULL_VALUE) if (ATTRIBUTE != NULL_VALUE)\
//{\
//    jsonWriter.String(NAME);\
//    jsonWriter.Int(ATTRIBUTE);\
//}
//
//#define JSON_WRITE_my::int64(NAME, ATTRIBUTE, NULL_VALUE) if (ATTRIBUTE != NULL_VALUE)\
//{\
//    jsonWriter.String(NAME);\
//    jsonWriter.my::int64(ATTRIBUTE);\
//}
//    
//#define JSON_WRITE_DOUBLE(NAME, ATTRIBUTE, NULL_VALUE) if (ATTRIBUTE != NULL_VALUE)\
//{\
//    jsonWriter.String(NAME);\
//    jsonWriter.Double(ATTRIBUTE);\
//}
//    
//#define JSON_WRITE_STDSTRING(NAME, ATTRIBUTE, NULL_VALUE)\
//if (!ATTRIBUTE.empty() && (ATTRIBUTE != NULL_VALUE))\
//{\
//    jsonWriter.String(NAME);\
//    jsonWriter.String(ATTRIBUTE.c_str());\
//}
//    
//#define JSON_WRITE_DOUBLE_ARRAY(NAME, ATTRIBUTE, SIZE, NULL_VALUE)\
//    if (ATTRIBUTE &&\
//        (SIZE > 0) &&\
//        (ATTRIBUTE[0] != NULL_VALUE))\
//    {\
//        jsonWriter.String(NAME);\
//        jsonWriter.StartArray();\
//        for (my::int32 i = 0; i < SIZE; ++i)\
//            jsonWriter.Double(ATTRIBUTE[i]);\
//        jsonWriter.EndArray();\
//    }
//
//#define JSON_WRITE_DOUBLE_VECTOR3(NAME, ATTRIBUTE) if (ATTRIBUTE.IsValid())\
//            {\
//        jsonWriter.String(NAME);\
//        jsonWriter.StartArray();\
//        jsonWriter.Double(ATTRIBUTE.x());\
//        jsonWriter.Double(ATTRIBUTE.y());\
//        jsonWriter.Double(ATTRIBUTE.z());\
//        jsonWriter.EndArray();\
//            }
//
//    jsonWriter.StartObject();
//
//    JSON_WRITE_my::int32("game_primary_key", m_gamePrimaryKey, mlb::Null<my::int32>());
//    JSON_WRITE_STDSTRING("mlb_game_string", m_mlbGameString, mlb::Null<std::string>());
//    JSON_WRITE_STDSTRING("sportvision_pitch_id", m_sportvisionPitchId, mlb::Null<std::string>());
//    JSON_WRITE_STDSTRING("description", mlb::GetPitchDescriptionTypeString(m_pitchDescriptionType), mlb::GetPitchDescriptionTypeString(mlb::UNKNOWN_PITCH_DESCRIPTION));
//    JSON_WRITE_my::int32("sequence_number", m_sequenceNumber, mlb::Null<my::int32>());
//    JSON_WRITE_my::int32("game_at_bat_number", m_gameAtBatNumber, mlb::Null<my::int32>());
//    JSON_WRITE_my::int32("pitch_number", m_pitchNumber, mlb::Null<my::int32>());
//    JSON_WRITE_my::int32("at_bat_pitch_number", m_atBatPitchNumber, my::Null<my::int32>());
//    JSON_WRITE_my::int32("inning", m_inning, mlb::Null<my::int32>());
//    JSON_WRITE_BOOLEAN("top_of_inning", m_topOfInning);
//    JSON_WRITE_my::int32("event_number", m_eventNumber, mlb::Null<my::int32>());
//    JSON_WRITE_STDSTRING("event", mlb::GetPitchEventTypeString(m_pitchEventType), mlb::GetPitchEventTypeString(mlb::UNKNOWN_PITCH_EVENT));
//    JSON_WRITE_my::int32("pre_pitch_balls", m_prePitchBalls, mlb::Null<my::int32>());
//    JSON_WRITE_my::int32("pre_pitch_strikes", m_prePitchStrikes, mlb::Null<my::int32>());
//    JSON_WRITE_my::int32("pre_pitch_outs", m_prePitchOuts, my::Null<my::int32>());
//    JSON_WRITE_my::int32("post_pitch_balls", m_postPitchBalls, mlb::Null<my::int32>());
//    JSON_WRITE_my::int32("post_pitch_strikes", m_postPitchStrikes, mlb::Null<my::int32>());
//    JSON_WRITE_my::int32("post_pitch_outs", m_postPitchOuts, my::Null<my::int32>());
//    JSON_WRITE_my::int32("batter_mlb_id", m_batterMlbId, mlb::Null<my::int32>());
//    JSON_WRITE_STDSTRING("batter_side", m_batterSide, mlb::Null<std::string>());
//    JSON_WRITE_my::int32("pitcher_mlb_id", m_pitcherMlbId, mlb::Null<my::int32>());
//    JSON_WRITE_STDSTRING("pitcher_throws", m_pitcherThrows, mlb::Null<std::string>());
//    JSON_WRITE_my::int32("umpire_mlb_id", m_umpireMlbId, my::Null<my::int32>());
//    JSON_WRITE_DOUBLE_ARRAY("release_position", m_releasePosition, 3, mlb::Null<double>());
//    JSON_WRITE_DOUBLE_ARRAY("release_velocity", m_releaseVelocity, 3, mlb::Null<double>());
//    JSON_WRITE_DOUBLE_ARRAY("release_acceleration", m_releaseAcceleration, 3, mlb::Null<double>());
//    JSON_WRITE_DOUBLE("release_speed", m_releaseSpeed, my::Null<double>());
//    JSON_WRITE_DOUBLE("effective_release_speed", m_effectiveReleaseSpeed, my::Null<double>());
//    JSON_WRITE_DOUBLE("release_horizontal_angle", m_releaseHorizontalAngle, my::Null<double>());
//    JSON_WRITE_DOUBLE("release_vertical_angle", m_releaseVerticalAngle, my::Null<double>());
//    JSON_WRITE_DOUBLE_ARRAY("position_at_50_feet", m_positionAt50Feet, 3, mlb::Null<double>());
//    JSON_WRITE_DOUBLE_ARRAY("velocity_at_50_feet", m_velocityAt50Feet, 3, mlb::Null<double>());
//    JSON_WRITE_DOUBLE_ARRAY("acceleration_at_50_feet", m_accelerationAt50Feet, 3, mlb::Null<double>());
//    JSON_WRITE_DOUBLE("speed_at_50_feet", m_speedAt50Feet, mlb::Null<double>());
//    JSON_WRITE_DOUBLE_ARRAY("zone_position", m_zonePosition, 3, mlb::Null<double>());
//    JSON_WRITE_DOUBLE_ARRAY("zone_velocity", m_zoneVelocity, 3, mlb::Null<double>());
//    JSON_WRITE_DOUBLE_ARRAY("zone_acceleration", m_zoneAcceleration, 3, mlb::Null<double>());
//    JSON_WRITE_DOUBLE("zone_speed", m_zoneSpeed, mlb::Null<double>());
//    JSON_WRITE_DOUBLE("horizontal_break", m_horizontalBreak, mlb::Null<double>());
//    JSON_WRITE_DOUBLE("vertical_break", m_verticalBreak, mlb::Null<double>());
//    JSON_WRITE_DOUBLE("vertical_break_induced", m_verticalBreakInduced, mlb::Null<double>());
//    JSON_WRITE_DOUBLE("break_angle", m_breakAngle, my::Null<double>());
//    JSON_WRITE_DOUBLE("break_length", m_breakLength, my::Null<double>());
//    JSON_WRITE_STDSTRING("type", mlb::GetPitchTypeString(m_pitchTypeCode), mlb::GetPitchTypeString(mlb::UNKNOWN_PITCH_TYPE));
//    JSON_WRITE_my::int64("release_time", m_releaseTime, mlb::Null<my::int64>());
//    JSON_WRITE_my::int64("zone_time", m_zoneTime, my::Null<my::int64>());
//    JSON_WRITE_STDSTRING("game_type", m_gameType, mlb::Null<std::string>());
//    JSON_WRITE_DOUBLE("strike_zone_top", m_strikeZoneTop, mlb::Null<double>());
//    JSON_WRITE_DOUBLE("strike_zone_bottom", m_strikeZoneBottom, mlb::Null<double>());
//    JSON_WRITE_DOUBLE("pitch_type_confidence", m_pitchTypeConfidence, mlb::Null<double>());
//    JSON_WRITE_DOUBLE("spin_rate", m_spinRate, my::Null<double>());
//    JSON_WRITE_DOUBLE_ARRAY("spin_axis", m_spinAxis, 3, mlb::Null<double>());
//    JSON_WRITE_DOUBLE("backspin", m_releaseBackspinRpm, my::Null<double>());
//    JSON_WRITE_DOUBLE("sidespin", m_releaseSidespinRpm, my::Null<double>());
//    JSON_WRITE_DOUBLE("gyrospin", m_releaseGyrospinRpm, my::Null<double>());
//    JSON_WRITE_DOUBLE("distance_to_non_spinning_pitch", m_distanceToNoSpinningPitch, mlb::Null<double>());
//    JSON_WRITE_my::int32("pitch_zone", m_pitchZone, mlb::Null<my::int32>());
//    JSON_WRITE_STDSTRING("umpire_call", mlb::GetUmpireCallTypeString(m_umpireCallType), mlb::GetUmpireCallTypeString(mlb::UNKNOWN_UMPIRE_CALL));
//    JSON_WRITE_STDSTRING("at_bat_description", m_atBatDescription, mlb::Null<std::string>());
//    JSON_WRITE_STDSTRING("at_bat_event", mlb::GetAtBatEventTypeString(m_atBatEvent), mlb::GetAtBatEventTypeString(mlb::UNKNOWN_AT_BAT_EVENT));
//    JSON_WRITE_my::int32("player_on_1st_mlb_id", m_playerOn1stMlbId, mlb::Null<my::int32>());
//    JSON_WRITE_my::int32("player_on_2nd_mlb_id", m_playerOn2ndMlbId, mlb::Null<my::int32>());
//    JSON_WRITE_my::int32("player_on_3rd_mlb_id", m_playerOn3rdMlbId, mlb::Null<my::int32>());
//    JSON_WRITE_DOUBLE("extension", m_extension, my::Null<double>());
//    JSON_WRITE_DOUBLE("spin_axis_angle", m_spinAxisAngle, my::Null<double>());
//    JSON_WRITE_STDSTRING("statcast_play_id", m_measurementId, mlb::Null<std::string>());
//    JSON_WRITE_my::int64("home_time_zone", m_homeTimeZone, my::Null<my::int64>());
//    JSON_WRITE_STDSTRING("pitching_position", mlb::GetPitchingPositionTypeString(m_pitchingPositionType), mlb::GetPitchingPositionTypeString(mlb::Null<mlb::PITCHING_POSITION_TYPE_CODE>()));
//    JSON_WRITE_DOUBLE_VECTOR3("last_measured_position", m_lastMeasuredPosition);
//
//    jsonWriter.EndObject();
//}
//
///**
//*/
//bool mlb::io::CPitch::IsValid() const
//{
//    return m_releaseTime != mlb::Null<my::int64>();
//}

///**
//*/
//void mlb::io::CPitch::Clear()
//{
//	Create();
//}

/**
*/
void mlb::io::CPitch::Create()
{
	//m_gamePrimaryKey = mlb::Null<my::int32>();
	m_mlbGameString = my::Null<std::string>();
	m_sportvisionPitchId = my::Null<std::string>();
	//m_pitchDescriptionType = mlb::UNKNOWN_PITCH_DESCRIPTION;
	//m_sequenceNumber = mlb::Null<my::int32>();
	//m_gameAtBatNumber = mlb::Null<my::int32>();
	//m_pitchNumber = mlb::Null<my::int32>();
 //   m_atBatPitchNumber = my::Null<my::int32>();
	//m_inning = mlb::Null<my::int32>();
	//m_topOfInning = true;
	//m_eventNumber = mlb::Null<my::int32>();
	//m_pitchEventType = mlb::UNKNOWN_PITCH_EVENT;
	//m_prePitchBalls = mlb::Null<my::int32>();
	//m_prePitchStrikes = mlb::Null<my::int32>();
 //   m_prePitchOuts = my::Null<my::int32>();
 //   m_postPitchBalls = mlb::Null<my::int32>();
	//m_postPitchStrikes = mlb::Null<my::int32>();
 //   m_postPitchOuts = my::Null<my::int32>();
 //   m_batterMlbId = mlb::Null<my::int32>();
	//m_batterSide = mlb::Null<std::string>();
	//m_pitcherMlbId = mlb::Null<my::int32>();
	m_pitcherThrows = my::Null<std::string>();
 //   m_umpireMlbId = my::Null<my::int32>();
	//MyMath::Assign(mlb::Null<double>(), mlb::Null<double>(), mlb::Null<double>(), m_releasePosition);
	//MyMath::Assign(mlb::Null<double>(), mlb::Null<double>(), mlb::Null<double>(), m_releaseVelocity);
	//MyMath::Assign(mlb::Null<double>(), mlb::Null<double>(), mlb::Null<double>(), m_releaseAcceleration);
 //   // BUG: (19-Jun-2015) There may be NULL (invalid) or 0 (valid) pitch speeds. 
 //   m_releaseSpeed = my::Null<double>();
 //   // BUG: (17-Jun-2015) There may be NULL (invalid) or 0 (valid) perceived pitch speeds. 
 //   m_effectiveReleaseSpeed = my::Null<double>();
 //   m_releaseHorizontalAngle = my::Null<double>();
 //   m_releaseVerticalAngle = my::Null<double>();
	//MyMath::Assign(mlb::Null<double>(), mlb::Null<double>(), mlb::Null<double>(), m_positionAt50Feet);
	//MyMath::Assign(mlb::Null<double>(), mlb::Null<double>(), mlb::Null<double>(), m_velocityAt50Feet);
	//MyMath::Assign(mlb::Null<double>(), mlb::Null<double>(), mlb::Null<double>(), m_accelerationAt50Feet);
	m_speedAt50Feet = my::Null<double>();
	//MyMath::Assign(mlb::Null<double>(), mlb::Null<double>(), mlb::Null<double>(), m_zonePosition);
	//MyMath::Assign(mlb::Null<double>(), mlb::Null<double>(), mlb::Null<double>(), m_zoneVelocity);
	//MyMath::Assign(mlb::Null<double>(), mlb::Null<double>(), mlb::Null<double>(), m_zoneAcceleration);
	//m_zoneSpeed = mlb::Null<double>();
	//m_horizontalBreak = mlb::Null<double>();
 //   m_verticalBreak = mlb::Null<double>();
 //   m_verticalBreakInduced = mlb::Null<double>();
 //   // The angle, in degrees, from vertical to the straight line path from the release point to where the pitch crossed the front of home plate, as seen from the catcher's/umpire's perspective.
 //   m_breakAngle = my::Null<double>();
 //   // The measurement of the greatest distance, in inches, between the trajectory of the pitch at any point between the release point and the front of home plate, and the straight line path from the release point and the front of home plate, per the MLB Gameday team. John Walsh's article "In Search of the Sinker" has a good illustration of this parameter.
 //   m_breakLength = my::Null<double>();
 //   m_pitchTypeCode = mlb::UNKNOWN_PITCH_TYPE;
	//m_releaseTime = mlb::Null<my::int64>();
	//m_zoneTime = my::Null<my::int64>();
 //   // INITIALIZED AT CONSTRUCTOR
 //   //m_tipOfTheHomePlateTimestamp
	//m_gameType = mlb::Null<std::string>();
	//m_strikeZoneTop = mlb::Null<double>();
	//m_strikeZoneBottom = mlb::Null<double>();
	//m_pitchTypeConfidence = mlb::Null<double>();
 //   // BUG: (17-Jun-2015) There may be NULL (invalid) or 0 (valid) spin rates. 
	//m_spinRate = my::Null<double>();
	//MyMath::Assign(mlb::Null<double>(), mlb::Null<double>(), mlb::Null<double>(), m_spinAxis);
 //   m_releaseBackspinRpm = my::Null<double>();
 //   m_releaseSidespinRpm = my::Null<double>();
 //   m_releaseGyrospinRpm = my::Null<double>();
	//m_distanceToNoSpinningPitch = mlb::Null<double>();
	//m_pitchZone = mlb::Null<my::int32>();
	//m_umpireCallType = mlb::UNKNOWN_UMPIRE_CALL;
	//m_atBatDescription = mlb::Null<std::string>();
	//m_atBatEvent = mlb::UNKNOWN_AT_BAT_EVENT;
	//m_playerOn1stMlbId = mlb::Null<my::int32>();
	//m_playerOn2ndMlbId = mlb::Null<my::int32>();
	//m_playerOn3rdMlbId = mlb::Null<my::int32>();
 //   // BUG: (17-Jun-2015) There may be NULL (invalid) or 0 (valid) spin rates. 
 //   m_extension = my::Null<double>();
	//m_targetPositionArray.clear();
 //   m_spinAxisAngle = my::Null<double>();
 //   // DEBUG ONLY! (05-Jan-2015)
	//m_measurementId = mlb::Null<std::string>();
 //   m_homeTimeZone = my::Null<my::int64>();
 //   m_pitchingPositionType = mlb::Null<mlb::PITCHING_POSITION_TYPE_CODE>();
 //   m_lastMeasuredPosition.Set(my::Null<double>());
}

/**
*/
void mlb::io::CPitch::Copy(const CPitch& pitch) 
{
	//m_gamePrimaryKey = pitch.m_gamePrimaryKey;
	m_mlbGameString = pitch.m_mlbGameString;
	m_sportvisionPitchId = pitch.m_sportvisionPitchId;
	//m_pitchDescriptionType = pitch.m_pitchDescriptionType;
	//m_sequenceNumber = pitch.m_sequenceNumber;
	//m_gameAtBatNumber = pitch.m_gameAtBatNumber;
	//m_pitchNumber = pitch.m_pitchNumber;
 //   m_atBatPitchNumber = pitch.m_atBatPitchNumber;
	//m_inning = pitch.m_inning;
	//m_topOfInning = pitch.m_topOfInning;
	//m_eventNumber = pitch.m_eventNumber;
	//m_pitchEventType = pitch.m_pitchEventType;
	//m_prePitchBalls = pitch.m_prePitchBalls;
	//m_prePitchStrikes = pitch.m_prePitchStrikes;
 //   m_prePitchOuts = pitch.m_prePitchOuts;
 //   m_postPitchBalls = pitch.m_postPitchBalls;
	//m_postPitchStrikes = pitch.m_postPitchStrikes;
 //   m_postPitchOuts = pitch.m_postPitchOuts;
 //   m_batterMlbId = pitch.m_batterMlbId;
	//m_batterSide = pitch.m_batterSide;
	//m_pitcherMlbId = pitch.m_pitcherMlbId;
	m_pitcherThrows = pitch.m_pitcherThrows;
 //   m_umpireMlbId = pitch.m_umpireMlbId;
 //   MyMath::Assign3(pitch.m_releasePosition, m_releasePosition);
	//MyMath::Assign3(pitch.m_releaseVelocity, m_releaseVelocity);
	//MyMath::Assign3(pitch.m_releaseAcceleration, m_releaseAcceleration);
	//m_releaseSpeed = pitch.m_releaseSpeed;
 //   m_effectiveReleaseSpeed = pitch.m_effectiveReleaseSpeed;
 //   m_releaseHorizontalAngle = pitch.m_releaseHorizontalAngle;
 //   m_releaseVerticalAngle = pitch.m_releaseVerticalAngle;
 //   MyMath::Assign3(pitch.m_positionAt50Feet, m_positionAt50Feet);
	//MyMath::Assign3(pitch.m_velocityAt50Feet, m_velocityAt50Feet);
	//MyMath::Assign3(pitch.m_accelerationAt50Feet, m_accelerationAt50Feet);
	m_speedAt50Feet = pitch.m_speedAt50Feet;
	//MyMath::Assign3(pitch.m_zonePosition, m_zonePosition);
	//MyMath::Assign3(pitch.m_zoneVelocity, m_zoneVelocity);
	//MyMath::Assign3(pitch.m_zoneAcceleration, m_zoneAcceleration);
	//m_zoneSpeed = pitch.m_zoneSpeed;
	//m_horizontalBreak = pitch.m_horizontalBreak;
 //   m_verticalBreak = pitch.m_verticalBreak;
 //   m_verticalBreakInduced = pitch.m_verticalBreakInduced;
 //   // The angle, in degrees, from vertical to the straight line path from the release point to where the pitch crossed the front of home plate, as seen from the catcher's/umpire's perspective.
 //   m_breakAngle = pitch.m_breakAngle;
 //   // The measurement of the greatest distance, in inches, between the trajectory of the pitch at any point between the release point and the front of home plate, and the straight line path from the release point and the front of home plate, per the MLB Gameday team. John Walsh's article "In Search of the Sinker" has a good illustration of this parameter.
 //   m_breakLength = pitch.m_breakLength;
 //   m_pitchTypeCode = pitch.m_pitchTypeCode;
	//m_releaseTime = pitch.m_releaseTime;
	//m_zoneTime = pitch.m_zoneTime;
 //   m_tipOfTheHomePlateTimestamp = pitch.m_tipOfTheHomePlateTimestamp;
	//m_gameType = pitch.m_gameType;
	//m_strikeZoneTop = pitch.m_strikeZoneTop;
	//m_strikeZoneBottom = pitch.m_strikeZoneBottom;
	//m_pitchTypeConfidence = pitch.m_pitchTypeConfidence;
	//m_spinRate = pitch.m_spinRate;
	//MyMath::Assign3(pitch.m_spinAxis, m_spinAxis);
 //   m_releaseBackspinRpm = pitch.m_releaseBackspinRpm;
 //   m_releaseSidespinRpm = pitch.m_releaseSidespinRpm;
 //   m_releaseGyrospinRpm = pitch.m_releaseGyrospinRpm;
	//m_distanceToNoSpinningPitch = pitch.m_distanceToNoSpinningPitch;
	//m_pitchZone = pitch.m_pitchZone;
	//m_umpireCallType = pitch.m_umpireCallType;
	//m_atBatDescription = pitch.m_atBatDescription;
	//m_atBatEvent = pitch.m_atBatEvent;
	//m_playerOn1stMlbId = pitch.m_playerOn1stMlbId;
	//m_playerOn2ndMlbId = pitch.m_playerOn2ndMlbId;
	//m_playerOn3rdMlbId = pitch.m_playerOn3rdMlbId;
	//m_extension = pitch.m_extension;
	//m_targetPositionArray = pitch.m_targetPositionArray;
	//m_spinAxisAngle = pitch.m_spinAxisAngle;
	//// DEBUG ONLY! (05-Jan-2015)
	//m_measurementId = pitch.m_measurementId;
 //   m_homeTimeZone = pitch.m_homeTimeZone;
 //   m_pitchingPositionType = pitch.m_pitchingPositionType;
 //   m_lastMeasuredPosition = pitch.m_lastMeasuredPosition;
}

