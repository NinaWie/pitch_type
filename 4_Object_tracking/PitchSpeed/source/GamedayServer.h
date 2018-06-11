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

#if !defined(GAMEDAY_SERVER_INCLUDED)
#define GAMEDAY_SERVER_INCLUDED

#include <map>
#include <vector>

#include "Common.h"
//#include <Timestamp.h>

//#include "Game.h"
//#include "Player.h"
#include "Pitch.h"
//#include "Venue.h"
//#include "Team.h"

namespace mlb {
    namespace io {
        class CGamedayServer
        {
        public:
	        CGamedayServer();

			std::string GetServerUrl() const;

            std::string GetGameUrl(const std::string& mlbGameString) const;
   //         std::string GetGameUrl(INT32 gamePrimaryKey);

   //         // Web-based 
   //         mlb::io::CPlayer GetPlayerByMlbGameString(std::string mlbGameString, my::int32 targetMlbId);
   //         // Web-based (local files for mlbGameString)
   //         mlb::io::CPlayer GetPlayerByGame(my::int32 gamePrimaryKey, my::int32 targetMlbId);
   //         // Local (local files) and Web-based
   //         std::vector<mlb::io::CPlayer> GetPlayerArray(my::int32 gamePrimaryKey);
   //         // Web-based 
   //         std::vector<mlb::io::CPlayer> GetPlayerArray(std::string mlbGameString);

   //         // Web-based (get game directory from local tables)
   //         mlb::io::CGame GetGameFromGamePrimaryKey(INT32 gamePrimaryKey);
   //         // Web-based (Gameday Server)
   //         mlb::io::CGame GetGameFromMlbGameString(std::string mlbGameString) const;

   //         // Web-based, but mlbGameString is read from Local data (local files) if timestamp is not available
   //         std::vector<mlb::io::CPitch> GetPitchArray(my::int32 gamePrimaryKey, my::CTimestamp timestamp = my::Null<my::CTimestamp>());
   //         // Web-based, but mlbGameString is read from Local data (local files)
   //         std::vector<mlb::io::CPitch> GetPitchArray(my::int32 gamePrimaryKey, my::int32 inning);
            // Web-based
            std::vector<mlb::io::CPitch> GetPitchArray(std::string mlbGameString) const;

            //// Local data (local files)
            //std::string GetMlbGameStringFromGamePrimaryKey(INT32 gamePrimaryKey);

            //// Local data (local files)
            //INT32 GetGamePrimaryKeyFromMlbGameString(std::string mlbGameString);
            //// Local data (local files)
            //// Directory: full game URL on http://gd2.mlb.com
            //INT32 GetGamePrimaryKeyFromGameDirectory(std::string gameDirectory);
            //std::vector<my::int32> GetGamePrimaryKeyArrayFromYear(my::int32 year);

            //// Directory: full game URL on http://gd2.mlb.com
            //my::CTimestamp GetTimestampFromGameDirectory(std::string gameDirectory) const;
            //my::CTimestamp GetTimestampFromMlbGameString(std::string mlbGameString) const;
            //// SV_PITCH_ID 
            //// A unique value within a game that identifies the game date and time when the pitch was recorded. 
            //// The format is yymmdd_hhmmss, where yymmdd is always the game date and hhmmss is the timestamp in local military time. 
            //// Example: the SV_PITCH_ID for a pitch recorded at 7:04:35 p.m. local time on April 9 is:
            //// 080409_190435 
            //// Note that because the timestamp may cross midnight, but the game date will be unchanged, this value should NOT be used for sorting purposes.
            //my::CTimestamp GetTimestampFromSportvisionPitchId(std::string sportvisionPitchId) const;

            //// Local data (local files)
            //// Directory: full game URL on http://gd2.mlb.com
            //std::vector<std::string> GetGameDirectoryArray();
            //// Local data (local files)
            //// Directory: full game URL on http://gd2.mlb.com
            //std::vector<std::string> GetGameDirectoryArrayFromDate(INT32 year, INT32 month, INT32 day);

            //// Web-based (Gameday Server)
            //std::vector<mlb::io::CGame> GetGameArrayFromDate(my::int32 year, my::int32 month, my::int32 day) const;

            //// Web-based (Gameday Server)
            //std::vector<std::string> GetStatCastGameUrlArrayFromDate(my::int32 year, my::int32 month, my::int32 day) const;
            //// Web-based (Gameday Server)
            //std::vector<std::string> GetStatCastGameUrlArrayFromDate(my::int32 year, my::int32 month) const;

            //// Directory: full game URL on http://gd2.mlb.com
            //std::string GetMlbGameStringFromGameDirectory(std::string gameDirectory) const;

            //// Web-based
            //// Directory: full game URL on http://gd2.mlb.com
            //CVenue GetVenue(std::string gameDirectory) const;
            //// Local data (local files)
            //CVenue GetVenueFromId(INT32 venueId);

            //// Local data (local files)
            //CTeam GetTeamFromCode(std::string code);
            //// Local data (local files)
            //CTeam GetTeamFromId(INT32 id);

            ////// DEBUG ONLY! (22-Oct-2015) TEAM INFORMATION DATALOAD
            ////struct TEAMS {
            ////    TEAMS()
            ////        :away_name_abbrev(),
            ////        away_code(),
            ////        away_team_id(-1),
            ////        away_team_city(),
            ////        away_team_name(),
            ////        home_name_abbrev(),
            ////        home_code(),
            ////        home_team_id(-1),
            ////        home_team_city(),
            ////        home_team_name()
            ////    { }
            ////    std::string away_name_abbrev;
            ////    std::string away_code;
            ////    int away_team_id;
            ////    std::string away_team_city;
            ////    std::string away_team_name;
            ////    std::string home_name_abbrev;
            ////    std::string home_code;
            ////    int home_team_id;
            ////    std::string home_team_city;
            ////    std::string home_team_name;
            ////};
            ////TEAMS GetTeams(std::string gameDirectory) const;

            //bool Update();

        private:
			std::string GetLeague(const std::string& mlbGameString) const;
			std::string GetGameDataDirectory(const std::string& mlbGameString) const;
			//
			//bool UpdateYearlyDirectoryArray();
   //         bool UpdateMonthlyDirectoryArray();
   //         bool UpdateDailyDirectoryArray();

   //         bool UpdateGameDirectoryArray();
   //         bool UpdateGamePrimaryKeyArray();
   //         
   //         bool UpdateStatCastGameUrlArray();

   //         bool UpdateGameArray();

   //         bool LoadLastUpdateTime();
   //         bool UpdateLastUpdateTime();

   //         bool LoadVenueArray();

   //         template < typename Container >
   //         bool LoadDataUrlContainer(const std::string& fileName, Container& dataUrlContainer);
   //         template < typename Container >
   //         bool SaveDataUrlContainer(const std::string& fileName, const Container& dataUrlContainer) const;

            void Create();

        protected:
			std::string m_serverUrl;

   //         std::vector<std::string> m_yearlyDirectoryArray;
   //         std::vector<std::string> m_yearlyDirectoryToUpdateArray;

   //         std::vector<std::string> m_monthlyDirectoryArray;
   //         std::vector<std::string> m_monthlyDirectoryToUpdateArray;

   //         std::vector<std::string> m_dailyDirectoryArray;
   //         std::vector<std::string> m_dailyDirectoryToUpdateArray;

   //         std::vector<std::string> m_gameDirectoryArray;
   //         std::vector<std::string> m_gameDirectoryToUpdateArray;

   //         std::map<std::string, INT32> m_gameDirectoryToPrimaryKeyMap;
   //         std::map<std::string, INT32> m_gameDirectoryToPrimaryKeyToUpdateMap;

   //         std::map<std::string, std::string> m_gameDirectoryToStatCastGameUrlMap;
   //         
   //         std::vector<mlb::io::CGame> m_gameArray;

   //         std::vector<mlb::io::CVenue> m_venueArray;

   //         INT64 m_lastUpdateTime;

   //         std::vector<CTeam> m_teamArray;
        };
    }; //namespace io
} //namespace mlb 

#endif // #if !defined(GAMEDAY_SERVER_INCLUDED)

