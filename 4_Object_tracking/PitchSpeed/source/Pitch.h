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

#if !defined(PITCH_INCLUDED)
#define PITCH_INCLUDED

#include "Timestamp.h"

namespace mlb {
	namespace io {
		class CPitch 
		{
		public:
			CPitch();
			CPitch(const CPitch& pitch);

			void operator=(const CPitch& pitch);

			bool operator<(const CPitch& pitch) const;

			//my::int32 GetGamePrimaryKey() const;
			//void SetGamePrimaryKey(my::int32 gamePrimaryKey);
		
			std::string GetMlbGameString() const;
			void SetMlbGameString(std::string mlbGameString);

		    std::string GetSportvisionPitchId() const;
			void SetSportvisionPitchId(std::string sportvisionPitchId);

			//my::int32 GetSequenceNumber() const;
			//void SetSequenceNumber(my::int32 sequenceNumber);

			//my::int32 GetGameAtbatNumber() const;
			//void SetGameAtbatNumber(my::int32 gameAtBatNumber);

			//my::int32 GetPitchNumber() const;
			//void SetPitchNumber(my::int32 pitchNumber);

   //         my::int32 GetAtbatPitchNumber() const;
   //         void SetAtbatPitchNumber(my::int32 atBatPitchNumber);

   //         my::int32 GetInning() const;
			//void SetInning(my::int32 inning);

			//bool GetTopOfInning() const;
			//void SetTopOfInning(bool topOfInning);
			//void SetTopOfInningFromInt(my::int32 topOfInning);

			//my::int32 GetEventNumber() const;
			//void SetEventNumber(my::int32 eventNumber);

			//my::int32 GetPrePitchBalls() const;
			//void SetPrePitchBalls(my::int32 prePitchBalls);

			//my::int32 GetPrePitchStrikes() const;
			//void SetPrePitchStrikes(my::int32 prePitchStrikes);

   //         my::int32 GetPrePitchOuts() const;
   //         void SetPrePitchOuts(my::int32 prePitchOuts);

   //         my::int32 GetPostPitchBalls() const;
			//void SetPostPitchBalls(my::int32 postPitchBalls);

			//my::int32 GetPostPitchStrikes() const;
			//void SetPostPitchStrikes(my::int32 postPitchStrikes);

   //         my::int32 GetPostPitchOuts() const;
   //         void SetPostPitchOuts(my::int32 postPitchOuts);

   //         my::int32 GetBatterMlbId() const;
			//void SetBatterMlbId(my::int32 batterMlbId);

			//std::string GetBatterSide() const;
			//void SetBatterSide(std::string batterSide);
		
			//my::int32 GetPitcherMlbId() const;
			//void SetPitcherMlbId(my::int32 pitcherMlbId);

			std::string GetPitcherThrows() const;
			virtual void SetPitcherThrows(std::string pitcherThrows);

   //         my::int32 GetUmpireMlbId() const;
   //         void SetUmpireMlbId(my::int32 umpireMlbId);

   //         const double *GetReleasePosition() const;
			//virtual void SetReleasePosition(double x, double y, double z);

			//const double *GetReleaseVelocity() const;
			//virtual void SetReleaseVelocity(double vx, double vy, double vz);

			//const double *GetReleaseAcceleration() const;
			//void SetReleaseAcceleration(double ax, double ay, double az);

   //         // FEET/SEC
   //         double GetReleaseSpeed() const;
   //         // FEET/SEC
   //         void SetReleaseSpeed(double releaseSpeed);

   //         // TODO: (24-Mar-2015) Temporary attribute.
   //         double GetEffectiveReleaseSpeed() const;
   //         // TODO: (24-Mar-2015) Temporary attribute.
   //         void SetEffectiveReleaseSpeed(double EffectiveReleaseSpeed);

   //         double GetReleaseHorizontalAngle() const;
   //         // Launch horizontal angle in degrees, relative to the line from the center of the rubber to the tip of home plate at which the pitch was released. Positive towards 3B.
   //         void SetReleaseHorizontalAngle(double horizontalAngleInDegrees);

   //         double GetReleaseVerticalAngle() const;
   //         // Launch vertical angle in degrees; positive is up, negative is down.
   //         void SetReleaseVerticalAngle(double verticalAngleInDegrees);

   //         const double *GetPositionAt50Feet() const;
			//void SetPositionAt50Feet(double x, double y, double z);

			//const double *GetVelocityAt50Feet() const;
			//void SetVelocityAt50Feet(double vx, double vy, double vz);

			//const double *GetAccelerationAt50Feet() const;
			//void SetAccelerationAt50Feet(double ax, double ay, double az);

			double GetSpeedAt50Feet() const;
			void SetSpeedAt50Feet(double speedAt50Feet);

			//const double *GetZonePosition() const;
			//void SetZonePosition(double x, double y, double z);

			//const double *GetZoneVelocity() const;
			//void SetZoneVelocity(double vx, double vy, double vz);

			//const double *GetZoneAcceleration() const;
			//void SetZoneAcceleration(double ax, double ay, double az);

			//double GetZoneSpeed() const;
			//void SetZoneSpeed(double zoneSpeed);

   //         double GetHorizontalBreak() const;
			//void SetHorizontalBreak(double horizontalBreak);

   //         double GetVerticalBreak() const;
   //         void SetVerticalBreak(double verticalBreak);

   //         double GetVerticalBreakInduced() const;
   //         void SetVerticalBreakInduced(double verticalBreakInduced);

   //         // The angle, in degrees, from vertical to the straight line path from the release point to where the pitch crossed the front of home plate, as seen from the catcher's/umpire's perspective.
   //         double GetBreakAngle() const;
   //         // The angle, in degrees, from vertical to the straight line path from the release point to where the pitch crossed the front of home plate, as seen from the catcher's/umpire's perspective.
   //         void SetBreakAngle(double breakAngle);
   //         
   //         // The measurement of the greatest distance, in inches, between the trajectory of the pitch at any point between the release point and the front of home plate, and the straight line path from the release point and the front of home plate, per the MLB Gameday team. John Walsh's article "In Search of the Sinker" has a good illustration of this parameter.
   //         double GetBreakLength() const;
   //         // The measurement of the greatest distance, in inches, between the trajectory of the pitch at any point between the release point and the front of home plate, and the straight line path from the release point and the front of home plate, per the MLB Gameday team. John Walsh's article "In Search of the Sinker" has a good illustration of this parameter.
   //         void SetBreakLength(double breakLength);

			//my::int64 GetReleaseTime() const;
			//void SetReleaseTime(my::int64 releaseTime);

   //         my::int64 GetZoneTime() const;
   //         void SetZoneTime(my::int64 zoneTime);

   //         my::CTimestamp GetTipOfTheHomePlateTimestamp() const;
   //         // LEGACY (TIMESTAMP FROM TIME)
   //         void SetTipOfTheHomePlateTimestamp(my::int64 tipOfTheHomePlateTime);
   //         void SetTipOfTheHomePlateTimestamp(my::CTimestamp tipOfTheHomePlateTimestamp);

   //         std::string GetGameType() const;
			//void SetGameType(std::string gameType);

   //         // The top of the strike zone is defined in the Major Leagues Official Rules as a horizontal line at the midpoint between the top of the batter's shoulders and the top of the uniform pants. 
			//double GetStrikeZoneTop() const;
			//void SetStrikeZoneTop(double strikeZoneTop);

   //         // The bottom of the strike zone is a line at the hollow beneath the kneecap.
			//double GetStrikeZoneBottom() const;
			//void SetStrikeZoneBottom(double strikeZoneBottom);

			//double GetPitchTypeConfidence() const;
			//void SetPitchTypeConfidence(double pitchTypeConfidence);

   //         // DEPRECATED: (23-Feb-2016)
			//virtual double GetSpinRate() const;
   //         // DEPRECATED: (23-Feb-2016)
   //         virtual void SetSpinRate(double spinRate);

   //         // DEPRECATED: (23-Feb-2016)
   //         const double *GetSpinAxis() const;
   //         // DEPRECATED: (23-Feb-2016)
   //         void SetSpinAxis(double x, double y, double z);

   //         // Backspin (pitch) in rpm; positive for backspin (upward movement); negative for topspin (downward movement).
   //         double GetReleaseBackspin() const;
   //         // Backspin (pitch) in rpm; positive for backspin (upward movement); negative for topspin (downward movement).
   //         void SetReleaseBackspin(double releaseBackspin);

   //         // Sidespin (yaw) in rpm; positive for pitch movement to pitcher's left and batted ball movement toward LF.
   //         double GetReleaseSidespin() const;
   //         // Backspin (pitch) in rpm; positive for backspin (upward movement); negative for topspin (downward movement).
   //         void SetReleaseSidespin(double releaseSidespin);

   //         // Gyrospin (roll) in rpm.
   //         double GetReleaseGyrospin() const;
   //         // Backspin (pitch) in rpm; positive for backspin (upward movement); negative for topspin (downward movement).
   //         void SetReleaseGyrospin(double releaseGyrospin);

			//double GetDistanceToNoSpinningPitch() const;
			//void SetDistanceToNoSpinningPitch(double distanceToNoSpinningPitch);

			//my::int32 GetPitchZone() const;
			//void SetPitchZone(my::int32 pitchZone);

			//std::string GetAtBatDescription() const;
			//void SetAtBatDescription(std::string atBatDescription);

			//my::int32 GetPlayerOn1stMlbId() const;
			//void SetPlayerOn1stMlbId(my::int32 playerOn1stMlbId);

			//my::int32 GetPlayerOn2ndMlbId() const;
			//void SetPlayerOn2ndMlbId(my::int32 playerOn2ndMlbId);

			//my::int32 GetPlayerOn3rdMlbId() const;
			//void SetPlayerOn3rdMlbId(my::int32 playerOn3rdMlbId);

			//double GetExtension() const;
			//void SetExtension(double extension);

   //         // The axis of rotation for the ball at release given as an angle that reflects how the spin will influence the ball trajectory. Pure back-spin is 180 degrees, pure side-spin that pulls the ball to the 1B side is 90 degrees, pure-side spin that pulls the ball to the 3B side is 270 degress, and pure top-spin is 0 or 360 degrees.
			//double GetSpinAxisAngle() const;
			//virtual void SetSpinAxisAngle(double spinAxisAngle);

			//// "playId" from StatCast.
			//std::string GetMeasurementId() const;
			//// "playId" from StatCast.
			//void SetMeasurementId(std::string measurementId);

   //         my::int64 GetHomeTimeZone() const;
   //         void SetHomeTimeZone(my::int64 homeTimeZone);

   //         my::CVector3<double> GetLastMeasuredPosition() const;
   //         void SetLastMeasuredPosition(double x, double y, double z);

   //         // BASED ON RELEASE TIME
   //         virtual bool IsValid() const;

   //         void Clear();

        private:
			void Create();

			void Copy(const CPitch& pitch);

		protected:
			//// GAME_PK (required) A unique integer that identifies each game. 
			//// The GAME_PK remains static even if the game is postponed and 
			//// rescheduled.
			//my::int32 m_gamePrimaryKey;

			// GAME_ID (required) The alphanumeric code that identifies the game 
			// data, visiting team, home team, level of play and game number 
			// (in the event of a doubleheader). The GAME_ID will change if the 
			// game is postponed and rescheduled 
			// Format : "yyyy / mm / dd / visxxx - homxxx - #" where: 
			// "vis" = visiting team abbreviation 
			// "hom" = home team abbreviation 
			// "xxx" = sport_code 
			// Example: The game_id for Texas at Boston game on July 18, 2008 is:
			// "2008 / 07 / 18 / texmlb - bosmlb - 1"
			std::string m_mlbGameString;

			// SV_PITCH_ID A unique value within a game that identifies the game 
			// date and time when the pitch was recorded 
			// The format is yymmdd_hhmmss, where yymmdd is always the game date 
			// and hhmmss is the timestamp in local military time.
			// Example: the SV_PITCH_ID for a pitch recorded at 7:04:35 p.m. 
			// local time on April 9 is : 080409_190435 
			// Note that because the timestamp may cross midnight, but the game 
			// date will be unchanged, this value should NOT be used for sorting 
			// purposes
			std::string m_sportvisionPitchId;

			//// SEQUENCE_NUMBER An integer that identifies in the order in which 
			//// each pitch occurred during the game SEQUENCE_NUMBER will be unique 
			//// on a per - game basis
			//my::int32 m_sequenceNumber;

			//// AT_BAT_NUMBER (required) An integer used to determine the 
			//// sequence of at - bats during the game.
			//my::int32 m_gameAtBatNumber;

			//// PITCH_NUMBER An integer that determines the pitch sequence during 
			//// an individual plate appearance
			//my::int32 m_pitchNumber;

   //         my::int32 m_atBatPitchNumber;

			//// INNING (required) A integer indicating the inning when the pitch 
			//// occurred
			//my::int32 m_inning;

			//// TOP_INNING_SW (required) Indicates whether the event occurred in 
			//// the top(Y) or bottom(N) of the inning.
			//bool m_topOfInning;

			//// EVENT_NUMBER (required) An integer representing the sequence of 
			//// events for that game.
			//my::int32 m_eventNumber;

			//// PBP_NUMBER The event_number for the end result of a single plate 
			//// appearance. The field will be null with the exception of events 
			//// indicating runner movement (SB, CS, WP, etc.) or the last pitch 
			//// of the plate appearance.

			//// EVENT_RESULT The event_type for the end result of a single plate 
			//// appearance. The field will be null with the exception of events 
			//// indicating runner movement (SB, CS, WP, etc.) or the last pitch 
			//// of the plate appearance, but will include the result for each plate 
			//// appearance regardless of the outcome.

			//// PRE_BALLS The number of Balls in the count before the pitch.
			//my::int32 m_prePitchBalls;

			//// PRE_STRIKES The number of Strikes in the count before the pitch.
			//my::int32 m_prePitchStrikes;

   //         my::int32 m_prePitchOuts;

   //         // POST_BALLS The number of Balls in the count resulting from the 
			//// pitch.
			//my::int32 m_postPitchBalls;

			//// POST_STRIKES The number of Strikes in the count resulting from the 
			//// pitch.
			//my::int32 m_postPitchStrikes;

   //         my::int32 m_postPitchOuts;

   //         // BATTER_ID (required) The unique identifier number of the batter 
			//// when the event occurred.
			//my::int32 m_batterMlbId;

			//// BAT_SIDE (required) The actual batting side ("L" or "R") of the 
			//// batter when the event occurred.
			//std::string m_batterSide;

			//// PITCHER_ID (required) The unique identifier number of the pitcher 
			//// when the event occurred.
			//my::int32 m_pitcherMlbId;

			// PITCHER_THROWS (required) The pitching hand("L" or "R") of the 
			// pitcher when the event occur.
			std::string m_pitcherThrows;
		
   //         my::int32 m_umpireMlbId;
   //         
   //         // INIT_POS_X / INIT_POS_Y / INIT_POS_Z Coordinate location of the 
			//// ball at the point it was released from the pitcher's hand on the 
			//// x /y /z axis (time = 0).
			//double m_releasePosition[3];

			//// INIT_VEL_X / INIT_VEL_Y / INIT_VEL_Z Velocity of the ball from 
			//// x / y /z axis.
			//double m_releaseVelocity[3];

			//// INIT_ACCEL_X / INIT_ACCEL_X / INIT_ACCEL_X Ball acceleration on 
			//// x / y / z axis.
			//double m_releaseAcceleration[3];

			//double m_releaseSpeed;

   //         // TODO: (24-Mar-2015) Temporary attribute.
   //         double m_effectiveReleaseSpeed;
   //         
   //         double m_releaseHorizontalAngle;
   //         double m_releaseVerticalAngle;

   //         // Pfx compatibility.
			//double m_positionAt50Feet[3];

			//// Pfx compatibility.
			//double m_velocityAt50Feet[3];

			//// Pfx compatibility.
			//double m_accelerationAt50Feet[3];

			// INTIAL_SPEED The speed in MPH of the ball at 50 feet in front of 
			// home plate.
			double m_speedAt50Feet;

			//// PLATE_X / PLATE_Y / PLATE_Z Horizontal / 0 / Vertical position in 
			//// feet of the ball as it crosses the front axis of home plate.
			//double m_zonePosition[3];

			//double m_zoneVelocity[3];

			//double m_zoneAcceleration[3];

			//// PLATE_SPEED The speed in MPH of the bass as it crosses the front 
			//// edge of home plate (0, 0 in the x axis).
			//double m_zoneSpeed;

			//// BREAK_X Horizontal movement of the ball in feet.
			//double m_horizontalBreak;

			//// BREAK_Z Vertical movement of the ball in feet.
			//double m_verticalBreak;

   //         double m_verticalBreakInduced;
   //         
   //         // The angle, in degrees, from vertical to the straight line path from the release point to where the pitch crossed the front of home plate, as seen from the catcher's/umpire's perspective.
   //         double m_breakAngle;

   //         // The measurement of the greatest distance, in inches, between the trajectory of the pitch at any point between the release point and the front of home plate, and the straight line path from the release point and the front of home plate, per the MLB Gameday team. John Walsh's article "In Search of the Sinker" has a good illustration of this parameter.
   //         double m_breakLength;

			//// PITCH_NAME Full name of pitch.

			//// TIME_STAMP Time stamp when the data was captured.
			//my::int64 m_releaseTime;

   //         my::int64 m_zoneTime;

   //         my::CTimestamp m_tipOfTheHomePlateTimestamp;

			//// GAME_DATE (required) The date on which the game occurred.

			//// GAME_NBR (required) Always "1" except for the second game of a 
			//// doubleheader, which is "2"

			//// YEAR (required) The year in which the game was played.

			//// GAME_TYPE (required) Indicates whether the game is regular 
			//// season ("R"), All - Star ("A"), first - round playoff ("F"), 
			//// divisional playoff ("D"), league championship series ("L") or 
			//// World Series ("W") Postseason game_types work "backwards" from "W", 
			//// that is, the championship of any league is always game_type "W"
			//std::string m_gameType;

			//// SZ_TOP Distance in feet from the ground to the top of the batter 
			//// box.
			//double m_strikeZoneTop;

			//// SZ_BOTTOM Distance in feet from the ground to the bottom of the 
			//// batter box.
			//double m_strikeZoneBottom;

			//// PITCH_TYPE_CONFIDENCE The output value of the pitch type's output 
			//// neuron for the pitch.
			//double m_pitchTypeConfidence;

   //         // DEPRECATED: (23-Feb-2016)
   //         // The rate of spin of a pitch.
			//double m_spinRate;

   //         // SPIN DIRECTION Calculation of the spin direction of a pitch Calculation sql:
			//// round((case when p.init_accel_x < 0 then((atan((p.init_accel_x + 32.174) / p.init_accel_x) * 57.297) + 270) else ((atan((p.init_accel_z + 32.174) / p.init_accel_x) * 57.297) + 90) end), 3)
			////double m_spinDirection;

   //         // DEPRECATED: (23-Feb-2016)
   //         double m_spinAxis[3];

   //         // Backspin (pitch) in rpm; positive for backspin (upward movement); negative for topspin (downward movement).
   //         double m_releaseBackspinRpm;
   //         // Sidespin (yaw) in rpm; positive for pitch movement to pitcher's left and batted ball movement toward LF.
   //         double m_releaseSidespinRpm;
   //         // Gyrospin (roll) in rpm.
   //         double m_releaseGyrospinRpm;
   //         
   //         // PFX The distance between the location of the actual pitch, and the 
			//// calculated location of a ball thrown by the pitcher in the same way 
			//// but with no spin; this is the amount of 'movement' the pitcher 
			//// applies to the pitch. A faster, straighter pitch like a fastball 
			//// will have a higher PFX value than a slower, breaking ball like a 
			//// curverball, which will have a higher break. Calculation sql:
			//// round(sqrt((p.break_x * p.break_x) + (p.break_z * p.break_z)), 3)
			//double m_distanceToNoSpinningPitch;

			//my::int32 m_pitchZone;

			////LOP_ERROR_AVG
			////LOP_ERROR_AT_PLATE
			////RES_ERROR_VECTOR_SIZE

			//std::string m_atBatDescription;
			//
			//my::int32 m_playerOn1stMlbId;
			//my::int32 m_playerOn2ndMlbId;
			//my::int32 m_playerOn3rdMlbId;

			//// The pitcher extension at the time of release. Measured as the distance in the Y coordinate from the front of the pitching rubber at which the ball was released.
			//double m_extension;

			//std::vector<mlb::io::CTargetPosition> m_targetPositionArray;

			//// The axis of rotation for the ball at release given as an angle that reflects how the spin will influence the ball trajectory. Pure back-spin is 180 degrees, pure side-spin that pulls the ball to the 1B side is 90 degrees, pure-side spin that pulls the ball to the 3B side is 270 degress, and pure top-spin is 0 or 360 degrees. 
			//double m_spinAxisAngle;

			//// "playId" from StatCast.
			//std::string m_measurementId;

   //         my::int64 m_homeTimeZone;

   //         my::CVector3<double> m_lastMeasuredPosition;
        };
	} //namespace io
} //namespace mlb

#endif // #if !defined(PITCH_INCLUDED)

