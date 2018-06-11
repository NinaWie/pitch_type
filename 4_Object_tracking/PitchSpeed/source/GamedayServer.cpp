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

#include <fstream>
#include <iostream>
#include <set>

#include <boost/lexical_cast.hpp>

#include <pugixml.hpp>
#include <rapidjson/rapidjson.h>
#include <rapidjson/document.h>

#include "Logger.h"
#include "Common.h"
#include "MyMath.h"
#include "FileHelper.h"
#include "PitchFxIo.h"

#include "GamedayServer.h"

///**
//*/
//template < >
//bool mlb::io::CGamedayServer::LoadDataUrlContainer(const std::string& fileName, std::vector<std::string>& dataUrlArray)
//{
//    std::ifstream fileStream;
//
//    dataUrlArray.clear();
//
//    fileStream.open(fileName);
//
//    if (!fileStream.is_open())
//        return true;
//
//    for (std::string line; std::getline(fileStream, line);)
//    {
//        if (!line.empty())
//            dataUrlArray.push_back(line);
//    }
//
//    return true;
//}
//
///**
//*/
//template < >
//bool mlb::io::CGamedayServer::LoadDataUrlContainer(const std::string& fileName, std::map<std::string, INT32>& dataUrlArray)
//{
//    std::ifstream fileStream;
//
//    dataUrlArray.clear();
//
//    fileStream.open(fileName);
//
//    if (!fileStream.is_open())
//        return true;
//
//    while (!fileStream.eof())
//    {
//        std::string token;
//        INT32 integer = 0;
//
//        fileStream >> token;
//
//        if (!token.empty())
//        {
//            if (fileStream.eof())
//            {
//                LOG_ERROR();
//
//                return false;
//            }
//
//            fileStream >> integer;
//
//            dataUrlArray[token] = integer;
//        }
//    }
//
//    return true;
//}
//
///**
//*/
//template < >
//bool mlb::io::CGamedayServer::LoadDataUrlContainer(const std::string& fileName, std::map<std::string, std::string>& dataUrlArray)
//{
//    std::ifstream fileStream;
//
//    dataUrlArray.clear();
//
//    fileStream.open(fileName);
//
//    if (!fileStream.is_open())
//        return true;
//
//    while (!fileStream.eof())
//    {
//        std::string keyToken,
//            valueToken;
//
//        fileStream >> keyToken;
//
//        if (!keyToken.empty())
//        {
//            if (fileStream.eof())
//            {
//                LOG_ERROR();
//
//                return false;
//            }
//
//            fileStream >> valueToken;
//
//            dataUrlArray[keyToken] = valueToken;
//        }
//    }
//
//    return true;
//}
//
///**
//*/
//template < >
//bool mlb::io::CGamedayServer::SaveDataUrlContainer(const std::string& fileName, const std::vector<std::string>& dataUrlArray) const
//{
//    std::ofstream fileStream;
//
//    fileStream.open(fileName);
//
//    if (!fileStream.is_open())
//    {
//        LOG_ERROR();
//
//        return false;
//    }
//
//    for (std::vector<std::string>::const_iterator dataUrlIterator = dataUrlArray.begin(); dataUrlIterator != dataUrlArray.end(); ++dataUrlIterator)
//    {
//        fileStream << (*dataUrlIterator) << std::endl;
//    }
//
//    return true;
//}
//
///**
//*/
//template < >
//bool mlb::io::CGamedayServer::SaveDataUrlContainer(const std::string& fileName, const std::map<std::string, INT32>& dataUrlArray) const
//{
//    std::ofstream fileStream;
//
//    fileStream.open(fileName);
//
//    if (!fileStream.is_open())
//    {
//        LOG_ERROR();
//
//        return false;
//    }
//
//    for (std::map<std::string, INT32>::const_iterator dataUrlIterator = dataUrlArray.begin(); dataUrlIterator != dataUrlArray.end(); ++dataUrlIterator)
//    {
//        fileStream << dataUrlIterator->first << " " << dataUrlIterator->second << std::endl;
//    }
//
//    return true;
//}
//
///**
//*/
//template < >
//bool mlb::io::CGamedayServer::SaveDataUrlContainer(const std::string& fileName, const std::map<std::string, std::string>& dataUrlArray) const
//{
//    std::ofstream fileStream;
//
//    fileStream.open(fileName);
//
//    if (!fileStream.is_open())
//    {
//        LOG_ERROR();
//
//        return false;
//    }
//
//    for (std::map<std::string, std::string>::const_iterator dataUrlIterator = dataUrlArray.begin(); dataUrlIterator != dataUrlArray.end(); ++dataUrlIterator)
//    {
//        fileStream << dataUrlIterator->first << " " << dataUrlIterator->second << std::endl;
//    }
//
//    return true;
//}

/**
*/
mlb::io::CGamedayServer::CGamedayServer()
{
    Create();
}

/**
*/
std::string mlb::io::CGamedayServer::GetServerUrl() const
{
	return m_serverUrl;
}

/**
*/
std::string mlb::io::CGamedayServer::GetGameUrl(const std::string& mlbGameString) const
{
	std::string urlPrefix = GetServerUrl() + GetLeague(mlbGameString) + GetGameDataDirectory(mlbGameString);

	return urlPrefix;
}

///**
//*/
//std::string mlb::io::CGamedayServer::GetGameUrl(INT32 gamePrimaryKey)
//{
//    if (m_gameDirectoryToPrimaryKeyMap.empty())
//    {
//        if (!LoadDataUrlContainer("./data/tables/gameday_game_primary_keys.txt", m_gameDirectoryToPrimaryKeyMap))
//        {
//            LOG_ERROR();
//
//            return my::Null<std::string>();
//        }
//    }
//
//    for (std::map<std::string, INT32>::const_iterator gameDirectoryToPrimaryKeyIterator = m_gameDirectoryToPrimaryKeyMap.begin(); gameDirectoryToPrimaryKeyIterator != m_gameDirectoryToPrimaryKeyMap.end(); ++gameDirectoryToPrimaryKeyIterator)
//    {
//        if (gameDirectoryToPrimaryKeyIterator->second == gamePrimaryKey)
//        {
//            return gameDirectoryToPrimaryKeyIterator->first;
//        }
//    }
//
//    return my::Null<std::string>();
//}
//
///**
//*/
//mlb::io::CPlayer mlb::io::CGamedayServer::GetPlayerByMlbGameString(std::string mlbGameString, my::int32 targetMlbId)
//{
//    mlb::io::CPlayer emptyObject;
//
//    HEALTH_CHECK(my::IsNull(mlbGameString), emptyObject);
//    HEALTH_CHECK(my::IsNull(targetMlbId), emptyObject);
//
//    std::string gameDirectory = GetGameUrl(mlbGameString);
//
//    HEALTH_CHECK(gameDirectory == mlb::Null<std::string>(), emptyObject);
//
//    // PITCHER OR BATTER?
//    std::string xmlResponse;
//
//    my::file::GetUrlAsString(gameDirectory + "pitchers/" + my::NumberToString(targetMlbId) + ".xml", xmlResponse);
//
//    if (xmlResponse.empty() ||
//        (xmlResponse.find("404 Not Found") != std::string::npos) ||
//        (xmlResponse.find("Error (404)") != std::string::npos))
//    {
//        my::file::GetUrlAsString(gameDirectory + "batters/" + my::NumberToString(targetMlbId) + ".xml", xmlResponse);
//    }
//
//    HEALTH_CHECK(xmlResponse.empty(), emptyObject);
//    HEALTH_CHECK(xmlResponse.find("404 Not Found") != std::string::npos, emptyObject);
//    HEALTH_CHECK(xmlResponse.find("Error (404)") != std::string::npos, emptyObject);
//
//    pugi::xml_document xmlDocument;
//    pugi::xml_parse_result xmlDocumentLog = xmlDocument.load(xmlResponse.c_str());
//
//    if (!xmlDocumentLog)
//    {
//        LOG_MESSAGE(xmlDocumentLog.description());
//        LOG_MESSAGE("Error offset: " + boost::lexical_cast<std::string>(xmlDocumentLog.offset));
//
//        return emptyObject;
//    }
//
//    pugi::xml_node playerIterator = xmlDocument.child("Player");
//
//    mlb::io::CPlayer player;
//
//    if (!playerIterator.attribute("team").empty())
//        player.SetTeamAbbreviation(playerIterator.attribute("team").as_string());
//
//    if (!playerIterator.attribute("id").empty())
//        player.SetMlbId(playerIterator.attribute("id").as_int());
//
//    if (!playerIterator.attribute("pos").empty())
//        player.SetPrimaryPosition(mlb::GetPlayerIdFromString(playerIterator.attribute("pos").as_string()));
//
//    // type="pitcher" 
//
//    if (!playerIterator.attribute("first_name").empty())
//        player.SetFirstName(playerIterator.attribute("first_name").as_string());
//
//    if (!playerIterator.attribute("last_name").empty())
//        player.SetLastName(playerIterator.attribute("last_name").as_string());
//
//    if (!playerIterator.attribute("jersey_number").empty())
//        player.SetJerseyNumber(playerIterator.attribute("jersey_number").as_int());
//
//    // height="6-4" 
//    // weight="215" 
//
//    if (!playerIterator.attribute("bats").empty())
//        player.SetBats(playerIterator.attribute("bats").as_string());
//
//    if (!playerIterator.attribute("throws").empty())
//        player.SetThrows(playerIterator.attribute("throws").as_string());
//
//    // dob="07/13/1991"
//
//    return player;
//}
//
///**
//*/
//mlb::io::CPlayer mlb::io::CGamedayServer::GetPlayerByGame(my::int32 gamePrimaryKey, my::int32 targetMlbId)
//{
//    mlb::io::CPlayer emptyObject;
//
//    HEALTH_CHECK(my::IsNull(gamePrimaryKey), emptyObject);
//    HEALTH_CHECK(my::IsNull(targetMlbId), emptyObject);
//
//    std::string gameDirectory = GetGameUrl(gamePrimaryKey);
//
//    HEALTH_CHECK(gameDirectory == mlb::Null<std::string>(), emptyObject);
//
//    // PITCHER OR BATTER?
//    std::string xmlResponse;
//    
//    my::file::GetUrlAsString(gameDirectory + "pitchers/" + my::NumberToString(targetMlbId) + ".xml", xmlResponse);
//
//    if (xmlResponse.empty() ||
//        (xmlResponse.find("404 Not Found") != std::string::npos) ||
//        (xmlResponse.find("Error (404)") != std::string::npos))
//    {
//        my::file::GetUrlAsString(gameDirectory + "batters/" + my::NumberToString(targetMlbId) + ".xml", xmlResponse);
//    }
//
//    HEALTH_CHECK(xmlResponse.empty(), emptyObject);
//    HEALTH_CHECK(xmlResponse.find("404 Not Found") != std::string::npos, emptyObject);
//    HEALTH_CHECK(xmlResponse.find("Error (404)") != std::string::npos, emptyObject);
//
//    pugi::xml_document xmlDocument;
//    pugi::xml_parse_result xmlDocumentLog = xmlDocument.load(xmlResponse.c_str());
//
//    if (!xmlDocumentLog)
//    {
//        LOG_MESSAGE(xmlDocumentLog.description());
//        LOG_MESSAGE("Error offset: " + boost::lexical_cast<std::string>(xmlDocumentLog.offset));
//
//        return emptyObject;
//    }
//
//    pugi::xml_node playerIterator = xmlDocument.child("Player");
//
//    mlb::io::CPlayer player;
//
//    if (!playerIterator.attribute("team").empty())
//        player.SetTeamAbbreviation(playerIterator.attribute("team").as_string());
//
//    if (!playerIterator.attribute("id").empty())
//        player.SetMlbId(playerIterator.attribute("id").as_int());
//
//    if (!playerIterator.attribute("pos").empty())
//        player.SetPrimaryPosition(mlb::GetPlayerIdFromString(playerIterator.attribute("pos").as_string()));
//
//    // type="pitcher" 
//
//    if (!playerIterator.attribute("first_name").empty())
//        player.SetFirstName(playerIterator.attribute("first_name").as_string());
//
//    if (!playerIterator.attribute("last_name").empty())
//        player.SetLastName(playerIterator.attribute("last_name").as_string());
//
//    if (!playerIterator.attribute("jersey_number").empty())
//        player.SetJerseyNumber(playerIterator.attribute("jersey_number").as_int());
//
//    // height="6-4" 
//    // weight="215" 
//
//    if (!playerIterator.attribute("bats").empty())
//        player.SetBats(playerIterator.attribute("bats").as_string());
//
//    if (!playerIterator.attribute("throws").empty())
//        player.SetThrows(playerIterator.attribute("throws").as_string());
//
//    // dob="07/13/1991"
//
//    return player;
//}
//
///**
//*/
//std::vector<mlb::io::CPlayer> mlb::io::CGamedayServer::GetPlayerArray(my::int32 gamePrimaryKey)
//{
//    std::vector<mlb::io::CPlayer> playerArray;
//
//    std::string gameDirectory = GetGameUrl(gamePrimaryKey);
//
//    HEALTH_CHECK(gameDirectory == mlb::Null<std::string>(), playerArray);
//
//    std::string xmlResponse;
//
//    // FROM FILE
//
//    std::string fileName = "./data/players/" 
//        + boost::lexical_cast<std::string>(gamePrimaryKey)
//        + ".xml";
//
//    my::file::GetFileAsString(fileName, xmlResponse);
//
//    // FROM URL
//
//    if (xmlResponse.empty())
//    {
//        my::file::GetUrlAsString(gameDirectory + "players.xml", xmlResponse);
//
//        // (BEGIN OF) DEBUG ONLY! (26-Feb-2016) LOCAL COPY
//        if (!xmlResponse.empty())
//        {
//            std::ofstream fileStream(fileName);
//
//            if (fileStream.is_open())
//                fileStream << xmlResponse;
//        }
//        // (END OF) DEBUG ONLY! (26-Feb-2016) LOCAL COPY
//    }
//
//    HEALTH_CHECK(xmlResponse.empty(), playerArray);
//
//    pugi::xml_document xmlDocument;
//    pugi::xml_parse_result xmlDocumentLog = xmlDocument.load(xmlResponse.c_str());
//
//    if (!xmlDocumentLog)
//    {
//        LOG_MESSAGE(xmlDocumentLog.description());
//        LOG_MESSAGE("Error offset: " + boost::lexical_cast<std::string>(xmlDocumentLog.offset));
//
//        return playerArray;
//    }
//
//    pugi::xml_node gameElementArray = xmlDocument.child("game");
//
//    for (pugi::xml_node elementIterator = gameElementArray.first_child(); elementIterator; elementIterator = elementIterator.next_sibling())
//    {
//        if (std::string(elementIterator.name()) == "team")
//        {
//            for (pugi::xml_node playerIterator = elementIterator.first_child(); playerIterator; playerIterator = playerIterator.next_sibling())
//            {
//                if (std::string(playerIterator.name()) == "player")
//                {
//                    mlb::io::CPlayer player;
//
//                    if (!playerIterator.attribute("id").empty())
//                        player.SetMlbId(playerIterator.attribute("id").as_int());
//
//                    if (!playerIterator.attribute("num").empty())
//                        player.SetJerseyNumber(playerIterator.attribute("num").as_int());
//
//                    if (!playerIterator.attribute("boxname").empty())
//                        player.SetFullName(playerIterator.attribute("boxname").as_string());
//                    
//                    if (!playerIterator.attribute("last").empty())
//                        player.SetLastName(playerIterator.attribute("last").as_string());
//                    
//                    if (!playerIterator.attribute("first").empty())
//                        player.SetFirstName(playerIterator.attribute("first").as_string());
//
//                    if (!playerIterator.attribute("team_abbrev").empty())
//                        player.SetTeamAbbreviation(playerIterator.attribute("team_abbrev").as_string());
//
//                    if (!playerIterator.attribute("team_id").empty())
//                        player.SetTeamId(playerIterator.attribute("team_id").as_int());
//
//                    if (!playerIterator.attribute("rl").empty())
//                        player.SetThrows(playerIterator.attribute("rl").as_string());
//
//                    if (!playerIterator.attribute("bats").empty())
//                        player.SetBats(playerIterator.attribute("bats").as_string());
//
//                    // position = "3B" 
//                    // current_position = "LF" 
//                    // status = "A" 
//                    // team_abbrev = "CLE" 
//                    // team_id = "114" 
//                    // bat_order = "9" 
//                    
//                    if (!playerIterator.attribute("game_position").empty())
//                        player.SetPrimaryPosition(mlb::GetPlayerIdFromString(playerIterator.attribute("game_position").as_string()));
//
//                    // avg = ".000" 
//                    // hr = "0" 
//                    // rbi = "0"
//
//                    playerArray.push_back(player);
//                }
//            }
//        }
//    }
//
//    return playerArray;
//}
//
///**
//*/
//std::vector<mlb::io::CPlayer> mlb::io::CGamedayServer::GetPlayerArray(std::string mlbGameString)
//{
//    std::vector<mlb::io::CPlayer> playerArray;
//
//    std::string gameDirectory = GetGameUrl(mlbGameString);
//
//    HEALTH_CHECK(gameDirectory == mlb::Null<std::string>(), playerArray);
//
//    std::string xmlResponse;
//
//    // FROM FILE
//
//    std::string fileName = "./data/players/"
//        + boost::lexical_cast<std::string>(mlbGameString)
//        +".xml";
//
//    my::file::GetFileAsString(fileName, xmlResponse);
//
//    // FROM URL
//
//    if (xmlResponse.empty())
//    {
//        my::file::GetUrlAsString(gameDirectory + "players.xml", xmlResponse);
//
//        // (BEGIN OF) DEBUG ONLY! (26-Feb-2016) LOCAL COPY
//        if (!xmlResponse.empty())
//        {
//            std::ofstream fileStream(fileName);
//
//            if (fileStream.is_open())
//                fileStream << xmlResponse;
//        }
//        // (END OF) DEBUG ONLY! (26-Feb-2016) LOCAL COPY
//    }
//
//    HEALTH_CHECK(xmlResponse.empty(), playerArray);
//
//    pugi::xml_document xmlDocument;
//    pugi::xml_parse_result xmlDocumentLog = xmlDocument.load(xmlResponse.c_str());
//
//    if (!xmlDocumentLog)
//    {
//        LOG_MESSAGE(xmlDocumentLog.description());
//        LOG_MESSAGE("Error offset: " + boost::lexical_cast<std::string>(xmlDocumentLog.offset));
//
//        return playerArray;
//    }
//
//    pugi::xml_node gameElementArray = xmlDocument.child("game");
//
//    for (pugi::xml_node elementIterator = gameElementArray.first_child(); elementIterator; elementIterator = elementIterator.next_sibling())
//    {
//        if (std::string(elementIterator.name()) == "team")
//        {
//            for (pugi::xml_node playerIterator = elementIterator.first_child(); playerIterator; playerIterator = playerIterator.next_sibling())
//            {
//                if (std::string(playerIterator.name()) == "player")
//                {
//                    mlb::io::CPlayer player;
//
//                    if (!playerIterator.attribute("id").empty())
//                        player.SetMlbId(playerIterator.attribute("id").as_int());
//
//                    if (!playerIterator.attribute("num").empty())
//                        player.SetJerseyNumber(playerIterator.attribute("num").as_int());
//
//                    if (!playerIterator.attribute("boxname").empty())
//                        player.SetFullName(playerIterator.attribute("boxname").as_string());
//
//                    if (!playerIterator.attribute("last").empty())
//                        player.SetLastName(playerIterator.attribute("last").as_string());
//
//                    if (!playerIterator.attribute("first").empty())
//                        player.SetFirstName(playerIterator.attribute("first").as_string());
//
//                    if (!playerIterator.attribute("team_abbrev").empty())
//                        player.SetTeamAbbreviation(playerIterator.attribute("team_abbrev").as_string());
//
//                    if (!playerIterator.attribute("team_id").empty())
//                        player.SetTeamId(playerIterator.attribute("team_id").as_int());
//
//                    if (!playerIterator.attribute("rl").empty())
//                        player.SetThrows(playerIterator.attribute("rl").as_string());
//
//                    if (!playerIterator.attribute("bats").empty())
//                        player.SetBats(playerIterator.attribute("bats").as_string());
//
//                    // position = "3B" 
//                    // current_position = "LF" 
//                    // status = "A" 
//                    // team_abbrev = "CLE" 
//                    // team_id = "114" 
//                    // bat_order = "9" 
//
//                    if (!playerIterator.attribute("game_position").empty())
//                        player.SetPrimaryPosition(mlb::GetPlayerIdFromString(playerIterator.attribute("game_position").as_string()));
//
//                    // avg = ".000" 
//                    // hr = "0" 
//                    // rbi = "0"
//
//                    playerArray.push_back(player);
//                }
//            }
//        }
//    }
//
//    return playerArray;
//}
//
///**
//*/
//mlb::io::CGame mlb::io::CGamedayServer::GetGameFromGamePrimaryKey(INT32 gamePrimaryKey)
//{
//    std::string gameUrl = GetGameUrl(gamePrimaryKey);
//
//    mlb::io::CGame game;
//
//    game.SetGamePrimaryKey(gamePrimaryKey);
//    game.SetGamedayUrl(gameUrl);
//    //game.SetStatCastUrl
//
//    // LINESCORE
//
//    std::string fileBuffer;
//    
//    my::file::GetUrlAsString(gameUrl + "linescore.json", fileBuffer);
//
//    rapidjson::Document document;
//    document.Parse<0>(fileBuffer.c_str());
//
//    if (document.IsObject() &&
//        document.HasMember("data"))
//    {
//        const rapidjson::Value& dataHandle = document["data"];
//
//        if (dataHandle.HasMember("game"))
//        {
//            const rapidjson::Value& gameHandle = dataHandle["game"];
//
//            // time_date
//            if (gameHandle.HasMember("time_date"))
//            {
//                const rapidjson::Value& timeDateHandle = gameHandle["time_date"];
//
//                // 2015/04/10 7:05
//                if (timeDateHandle.IsString())
//                    game.SetTime(CUnixEpoch(timeDateHandle.GetString(), "%Y/%m/%d").Get());
//            }
//
//            // home_team_id
//            if (gameHandle.HasMember("home_team_id"))
//            {
//                const rapidjson::Value& homeTeamIdHandle = gameHandle["home_team_id"];
//
//                if (homeTeamIdHandle.IsInt())
//                    game.SetHomeTeamId(homeTeamIdHandle.GetInt());
//                else if (homeTeamIdHandle.IsString())
//                    game.SetHomeTeamId(atoi(homeTeamIdHandle.GetString()));
//            }
//
//            // home_name_abbrev
//            if (gameHandle.HasMember("home_name_abbrev"))
//            {
//                const rapidjson::Value& homeNameAbbrevHandle = gameHandle["home_name_abbrev"];
//
//                // 2015/04/10 7:05
//                if (homeNameAbbrevHandle.IsString())
//                    game.SetHomeTeamNameAbbreviation(homeNameAbbrevHandle.GetString());
//            }
//
//            // home_team_runs
//            if (gameHandle.HasMember("home_team_runs"))
//            {
//                const rapidjson::Value& homeTeamRunsHandle = gameHandle["home_team_runs"];
//
//                if (homeTeamRunsHandle.IsInt())
//                    game.SetHomeTeamRuns(homeTeamRunsHandle.GetInt());
//                else if (homeTeamRunsHandle.IsString())
//                    game.SetHomeTeamRuns(atoi(homeTeamRunsHandle.GetString()));
//            }
//
//            // away_team_id
//            if (gameHandle.HasMember("away_team_id"))
//            {
//                const rapidjson::Value& awayTeamIdHandle = gameHandle["away_team_id"];
//
//                if (awayTeamIdHandle.IsInt())
//                    game.SetAwayTeamId(awayTeamIdHandle.GetInt());
//                else if (awayTeamIdHandle.IsString())
//                    game.SetAwayTeamId(atoi(awayTeamIdHandle.GetString()));
//            }
//
//            // away_name_abbrev
//            if (gameHandle.HasMember("away_name_abbrev"))
//            {
//                const rapidjson::Value& awayNameAbbrevHandle = gameHandle["away_name_abbrev"];
//
//                // 2015/04/10 7:05
//                if (awayNameAbbrevHandle.IsString())
//                    game.SetAwayTeamNameAbbreviation(awayNameAbbrevHandle.GetString());
//            }
//
//            // away_team_runs
//            if (gameHandle.HasMember("away_team_runs"))
//            {
//                const rapidjson::Value& awayTeamRunsHandle = gameHandle["away_team_runs"];
//
//                if (awayTeamRunsHandle.IsInt())
//                    game.SetAwayTeamRuns(awayTeamRunsHandle.GetInt());
//                else if (awayTeamRunsHandle.IsString())
//                    game.SetAwayTeamRuns(atoi(awayTeamRunsHandle.GetString()));
//            }
//
//            // venue_id
//            if (gameHandle.HasMember("venue_id"))
//            {
//                const rapidjson::Value& venueIdHandle = gameHandle["venue_id"];
//
//                if (venueIdHandle.IsInt())
//                    game.SetVenueId(venueIdHandle.GetInt());
//                else if (venueIdHandle.IsString())
//                    game.SetVenueId(atoi(venueIdHandle.GetString()));
//            }
//
//            // venue
//            if (gameHandle.HasMember("venue"))
//            {
//                const rapidjson::Value& venueNameHandle = gameHandle["venue"];
//
//                // Fenway Park
//                if (venueNameHandle.IsString())
//                    game.SetVenueName(venueNameHandle.GetString());
//            }
//
//            // home_time
//            if (gameHandle.HasMember("home_time") &&
//                (game.GetTime() != mlb::Null<INT64>()))
//            {
//                const rapidjson::Value& homeTimeHandle = gameHandle["home_time"];
//
//                // 7:05
//                if (homeTimeHandle.IsString())
//                {
//                    CUnixEpoch unixEpoch(game.GetTime());
//                    
//                    std::string homeTimeString = homeTimeHandle.GetString();
//
//                    if (homeTimeString.find(":") != std::string::npos)
//                    {
//                        std::string hourString = homeTimeString.substr(0, homeTimeString.find(":")),
//                            minuteString = homeTimeString.substr(homeTimeString.find(":") + 1);
//
//                        if (!hourString.empty() &&
//                            !minuteString.empty())
//                        {
//                            INT32 hour = atoi(hourString.c_str()),
//                                minute = atoi(minuteString.c_str());
//
//                            // TESTING: (08-May-2015) AM/PM?
//                            //if (hour < 12)
//                            //    hour += 12;
//
//                            game.SetHomeTime(CUnixEpoch(unixEpoch.GetYear(), unixEpoch.GetMonth(), unixEpoch.GetDay(), hour, minute).Get());
//                        }
//                    }
//                }
//            }
//
//            // time_zone_hm_lg
//            if (gameHandle.HasMember("time_zone_hm_lg"))
//            {
//                const rapidjson::Value& homeTimeZoneHandle = gameHandle["time_zone_hm_lg"];
//
//                // -4
//                if (homeTimeZoneHandle.IsString())
//                {
//                    TRY_CATCH_BLOCK(game.SetHomeTimeZone(boost::lexical_cast<INT64>(homeTimeZoneHandle.GetString())), mlb::io::CGame());
//                }
//            }
//        }
//    }
//
//    return game;
//}
//
///**
//*/
//mlb::io::CGame mlb::io::CGamedayServer::GetGameFromMlbGameString(std::string mlbGameString) const
//{
//    std::string gameUrl = GetGameUrl(mlbGameString);
//
//    mlb::io::CGame game;
//
//    game.SetGamedayUrl(gameUrl);
//    //game.SetStatCastUrl
//
//    // LINESCORE
//
//    std::string fileBuffer;
//
//    my::file::GetUrlAsString(gameUrl + "linescore.json", fileBuffer);
//
//    rapidjson::Document document;
//    document.Parse<0>(fileBuffer.c_str());
//
//    if (document.IsObject() &&
//        document.HasMember("data"))
//    {
//        const rapidjson::Value& dataHandle = document["data"];
//
//        if (dataHandle.HasMember("game"))
//        {
//            const rapidjson::Value& gameHandle = dataHandle["game"];
//
//            // game_pk
//            if (gameHandle.HasMember("game_pk"))
//            {
//                const rapidjson::Value& gamePkHandle = gameHandle["game_pk"];
//
//                // 346834
//                if (gamePkHandle.IsString())
//                {
//                    std::string gamePrimaryKeyString = gamePkHandle.GetString();
//
//                    if (!gamePrimaryKeyString.empty())
//                        game.SetGamePrimaryKey(atoi(gamePrimaryKeyString.c_str()));
//                }
//            }
//
//            // time_date
//            if (gameHandle.HasMember("time_date"))
//            {
//                const rapidjson::Value& timeDateHandle = gameHandle["time_date"];
//
//                // 2015/04/10 7:05
//                if (timeDateHandle.IsString())
//                    game.SetTime(CUnixEpoch(timeDateHandle.GetString(), "%Y/%m/%d").Get());
//            }
//
//            // home_team_id
//            if (gameHandle.HasMember("home_team_id"))
//            {
//                const rapidjson::Value& homeTeamIdHandle = gameHandle["home_team_id"];
//
//                if (homeTeamIdHandle.IsInt())
//                    game.SetHomeTeamId(homeTeamIdHandle.GetInt());
//                else if (homeTeamIdHandle.IsString())
//                    game.SetHomeTeamId(atoi(homeTeamIdHandle.GetString()));
//            }
//
//            // home_name_abbrev
//            if (gameHandle.HasMember("home_name_abbrev"))
//            {
//                const rapidjson::Value& homeNameAbbrevHandle = gameHandle["home_name_abbrev"];
//
//                // 2015/04/10 7:05
//                if (homeNameAbbrevHandle.IsString())
//                    game.SetHomeTeamNameAbbreviation(homeNameAbbrevHandle.GetString());
//            }
//
//            // home_team_runs
//            if (gameHandle.HasMember("home_team_runs"))
//            {
//                const rapidjson::Value& homeTeamRunsHandle = gameHandle["home_team_runs"];
//
//                if (homeTeamRunsHandle.IsInt())
//                    game.SetHomeTeamRuns(homeTeamRunsHandle.GetInt());
//                else if (homeTeamRunsHandle.IsString())
//                    game.SetHomeTeamRuns(atoi(homeTeamRunsHandle.GetString()));
//            }
//
//            // away_team_id
//            if (gameHandle.HasMember("away_team_id"))
//            {
//                const rapidjson::Value& awayTeamIdHandle = gameHandle["away_team_id"];
//
//                if (awayTeamIdHandle.IsInt())
//                    game.SetAwayTeamId(awayTeamIdHandle.GetInt());
//                else if (awayTeamIdHandle.IsString())
//                    game.SetAwayTeamId(atoi(awayTeamIdHandle.GetString()));
//            }
//
//            // away_name_abbrev
//            if (gameHandle.HasMember("away_name_abbrev"))
//            {
//                const rapidjson::Value& awayNameAbbrevHandle = gameHandle["away_name_abbrev"];
//
//                // 2015/04/10 7:05
//                if (awayNameAbbrevHandle.IsString())
//                    game.SetAwayTeamNameAbbreviation(awayNameAbbrevHandle.GetString());
//            }
//
//            // away_team_runs
//            if (gameHandle.HasMember("away_team_runs"))
//            {
//                const rapidjson::Value& awayTeamRunsHandle = gameHandle["away_team_runs"];
//
//                if (awayTeamRunsHandle.IsInt())
//                    game.SetAwayTeamRuns(awayTeamRunsHandle.GetInt());
//                else if (awayTeamRunsHandle.IsString())
//                    game.SetAwayTeamRuns(atoi(awayTeamRunsHandle.GetString()));
//            }
//
//            // venue_id
//            if (gameHandle.HasMember("venue_id"))
//            {
//                const rapidjson::Value& venueIdHandle = gameHandle["venue_id"];
//
//                if (venueIdHandle.IsInt())
//                    game.SetVenueId(venueIdHandle.GetInt());
//                else if (venueIdHandle.IsString())
//                    game.SetVenueId(atoi(venueIdHandle.GetString()));
//            }
//
//            // venue
//            if (gameHandle.HasMember("venue"))
//            {
//                const rapidjson::Value& venueNameHandle = gameHandle["venue"];
//
//                // Fenway Park
//                if (venueNameHandle.IsString())
//                    game.SetVenueName(venueNameHandle.GetString());
//            }
//
//            // home_time
//            if (gameHandle.HasMember("home_time") &&
//                (game.GetTime() != mlb::Null<INT64>()))
//            {
//                const rapidjson::Value& homeTimeHandle = gameHandle["home_time"];
//
//                // 7:05
//                if (homeTimeHandle.IsString())
//                {
//                    CUnixEpoch unixEpoch(game.GetTime());
//
//                    std::string homeTimeString = homeTimeHandle.GetString();
//
//                    if (homeTimeString.find(":") != std::string::npos)
//                    {
//                        std::string hourString = homeTimeString.substr(0, homeTimeString.find(":")),
//                            minuteString = homeTimeString.substr(homeTimeString.find(":") + 1);
//
//                        if (!hourString.empty() &&
//                            !minuteString.empty())
//                        {
//                            INT32 hour = atoi(hourString.c_str()),
//                                minute = atoi(minuteString.c_str());
//
//                            // TESTING: (08-May-2015) AM/PM?
//                            //if (hour < 12)
//                            //    hour += 12;
//
//                            game.SetHomeTime(CUnixEpoch(unixEpoch.GetYear(), unixEpoch.GetMonth(), unixEpoch.GetDay(), hour, minute).Get());
//                        }
//                    }
//                }
//            }
//
//            // time_zone_hm_lg
//            if (gameHandle.HasMember("time_zone_hm_lg"))
//            {
//                const rapidjson::Value& homeTimeZoneHandle = gameHandle["time_zone_hm_lg"];
//
//                // -4
//                if (homeTimeZoneHandle.IsString())
//                {
//                    TRY_CATCH_BLOCK(game.SetHomeTimeZone(boost::lexical_cast<INT64>(homeTimeZoneHandle.GetString())), mlb::io::CGame());
//                }
//            }
//        }
//    }
//
//    return game;
//}
//
//// Web-based, but mlbGameString is read from Local data (local files) if timestamp is not available
//std::vector<mlb::io::CPitch> mlb::io::CGamedayServer::GetPitchArray(my::int32 gamePrimaryKey, my::CTimestamp timestamp)
//{
//    std::string gameDirectory = my::Null<std::string>();
//    
//    if (my::IsNull(timestamp))
//        gameDirectory = GetGameUrl(gamePrimaryKey);
//    else
//    {
//        // (BEGIN OF) TESTING: (10-May-2017) GET STRING FROM GDX ITSELF
//        // BUG: (09-May-2017) ?????
//        timestamp.AddHours(-12);
//
//        my::int32 year = timestamp.GetYear(),
//            month = timestamp.GetMonth(),
//            day = timestamp.GetDay();
//
//        std::vector<mlb::io::CGame> gameArray = GetGameArrayFromDate(year, month, day);
//
//        for (std::vector<mlb::io::CGame>::const_iterator gameIterator = gameArray.begin(); gameIterator != gameArray.end(); ++gameIterator)
//        {
//            if (gameIterator->GetGamePrimaryKey() == gamePrimaryKey)
//            {
//                std::string mlbGameString = gameIterator->GetMlbGameString();
//
//                if (!my::IsNull(mlbGameString))
//                    gameDirectory = GetGameUrl(mlbGameString);
//            }
//        }
//        // (END OF) TESTING: (10-May-2017) GET STRING FROM GDX ITSELF
//    }
//
//    std::vector<mlb::io::CPitch> emptyObject;
//
//    HEALTH_CHECK(my::IsNull(gameDirectory), emptyObject);
//
//    if (gameDirectory == my::Null<std::string>())
//    {
//        LOG_MESSAGE("Failed to compute the Gameday Server url from the game primary key " + boost::lexical_cast<std::string>(gamePrimaryKey) + ".");
//
//        return emptyObject;
//    }
//
//    mlb::io::CPitchFxIo pitchFxIo;
//
//    if (!pitchFxIo.LoadFromGameDirectory(gameDirectory))
//    {
//        LOG_ERROR();
//
//        return emptyObject;
//    }
//
//    std::vector<mlb::io::CPitch> pitchArray = pitchFxIo.GetPitchArray();
//
//    if (!pitchArray.empty() &&
//        (pitchArray.front().GetGamePrimaryKey() == mlb::Null<INT32>()))
//    {
//        for (std::vector<mlb::io::CPitch>::iterator pitchIterator = pitchArray.begin(); pitchIterator != pitchArray.end(); ++pitchIterator)
//            pitchIterator->SetGamePrimaryKey(gamePrimaryKey);
//    }
//
//    return pitchArray;
//}
//
///**
//*/
//std::vector<mlb::io::CPitch> mlb::io::CGamedayServer::GetPitchArray(my::int32 gamePrimaryKey, my::int32 inning)
//{
//    std::string mlbGameString = GetGameUrl(gamePrimaryKey);
//
//    std::vector<mlb::io::CPitch> emptyObject;
//
//    if (mlbGameString == my::Null<std::string>())
//    {
//        LOG_MESSAGE("Failed to compute the Gameday Server URL from the game primary key " + boost::lexical_cast<std::string>(gamePrimaryKey) + ".");
//
//        return emptyObject;
//    }
//
//    mlb::io::CPitchFxIo pitchFxIo;
//
//    if (!pitchFxIo.LoadFromGameDirectory(mlbGameString, inning))
//    {
//        LOG_ERROR();
//
//        return emptyObject;
//    }
//
//    std::vector<mlb::io::CPitch> pitchArray = pitchFxIo.GetPitchArray();
//
//    if (!pitchArray.empty() &&
//        (pitchArray.front().GetGamePrimaryKey() == mlb::Null<INT32>()))
//    {
//        for (std::vector<mlb::io::CPitch>::iterator pitchIterator = pitchArray.begin(); pitchIterator != pitchArray.end(); ++pitchIterator)
//            pitchIterator->SetGamePrimaryKey(gamePrimaryKey);
//    }
//
//    return pitchArray;
//}

/**
*/
std::vector<mlb::io::CPitch> mlb::io::CGamedayServer::GetPitchArray(std::string mlbGameString) const
{
    std::string gameDirectory = GetGameUrl(mlbGameString);

    std::vector<mlb::io::CPitch> emptyObject;

    if (my::IsNull(gameDirectory))
    {
        LOG_MESSAGE("Failed to compute the Gameday Server url from the MLB game string " + mlbGameString + ".");

        return emptyObject;
    }

    mlb::io::CPitchFxIo pitchFxIo;

    if (!pitchFxIo.LoadFromGameDirectory(gameDirectory))
    {
        LOG_ERROR();

        return emptyObject;
    }

    return pitchFxIo.GetPitchArray();
}

///**
//*/
//std::string mlb::io::CGamedayServer::GetMlbGameStringFromGamePrimaryKey(INT32 gamePrimaryKey)
//{
//    if (m_gameDirectoryToPrimaryKeyMap.empty())
//    {
//        if (!LoadDataUrlContainer("./data/tables/gameday_game_primary_keys.txt", m_gameDirectoryToPrimaryKeyMap))
//        {
//            LOG_ERROR();
//
//            return mlb::Null<std::string>();
//        }
//    }
//
//    for (std::map<std::string, INT32>::const_iterator gameDirectoryToPrimaryKeyIterator = m_gameDirectoryToPrimaryKeyMap.begin(); gameDirectoryToPrimaryKeyIterator != m_gameDirectoryToPrimaryKeyMap.end(); ++gameDirectoryToPrimaryKeyIterator)
//    {
//        if (gameDirectoryToPrimaryKeyIterator->second == gamePrimaryKey)
//        {
//            std::string mlbGameString = gameDirectoryToPrimaryKeyIterator->first;
//
//            HEALTH_CHECK(mlbGameString.find("gid_") == std::string::npos, 0);
//
//            mlbGameString = mlbGameString.substr(mlbGameString.find("gid_") + 4);
//
//            HEALTH_CHECK(mlbGameString.find("/") == std::string::npos, 0);
//
//            mlbGameString = mlbGameString.substr(0, mlbGameString.find("/"));
//
//            return mlbGameString;
//        }
//    }
//
//    return mlb::Null<std::string>();
//}
//
///**
//*/
//INT32 mlb::io::CGamedayServer::GetGamePrimaryKeyFromMlbGameString(std::string mlbGameString)
//{
//    if (m_gameDirectoryToPrimaryKeyMap.empty())
//    {
//        if (!LoadDataUrlContainer("./data/tables/gameday_game_primary_keys.txt", m_gameDirectoryToPrimaryKeyMap))
//            return mlb::Null<INT32>();
//    }
//
//    for (std::map<std::string, INT32>::const_iterator gameDirectoryToPrimaryKeyIterator = m_gameDirectoryToPrimaryKeyMap.begin(); gameDirectoryToPrimaryKeyIterator != m_gameDirectoryToPrimaryKeyMap.end(); ++gameDirectoryToPrimaryKeyIterator)
//    {
//        if (gameDirectoryToPrimaryKeyIterator->first.find(mlbGameString) != std::string::npos)
//            return gameDirectoryToPrimaryKeyIterator->second;
//    }
//
//    return mlb::Null<INT32>();
//}
//
///**
//*/
//INT32 mlb::io::CGamedayServer::GetGamePrimaryKeyFromGameDirectory(std::string gameDirectory)
//{
//    if (m_gameDirectoryToPrimaryKeyMap.empty())
//    {
//        if (!LoadDataUrlContainer("./data/tables/gameday_game_primary_keys.txt", m_gameDirectoryToPrimaryKeyMap))
//            return mlb::Null<INT32>();
//    }
//
//    my::AddTrailingSlash(gameDirectory);
//
//    for (std::map<std::string, INT32>::const_iterator gameDirectoryToPrimaryKeyIterator = m_gameDirectoryToPrimaryKeyMap.begin(); gameDirectoryToPrimaryKeyIterator != m_gameDirectoryToPrimaryKeyMap.end(); ++gameDirectoryToPrimaryKeyIterator)
//    {
//        if (gameDirectoryToPrimaryKeyIterator->first == gameDirectory)
//            return gameDirectoryToPrimaryKeyIterator->second;
//    }
//
//    return mlb::Null<INT32>();
//}
//
///**
//*/
//std::vector<my::int32> mlb::io::CGamedayServer::GetGamePrimaryKeyArrayFromYear(my::int32 year)
//{
//    std::vector<my::int32> emptyObject;
//
//    if (m_gameDirectoryToPrimaryKeyMap.empty())
//    {
//        if (!LoadDataUrlContainer("./data/tables/gameday_game_primary_keys.txt", m_gameDirectoryToPrimaryKeyMap))
//        {
//            LOG_ERROR();
//
//            return emptyObject;
//        }
//    }
//
//    std::vector<my::int32> gamePrimaryKeyArray;
//
//    for (std::map<std::string, INT32>::const_iterator gameDirectoryToPrimaryKeyIterator = m_gameDirectoryToPrimaryKeyMap.begin(); gameDirectoryToPrimaryKeyIterator != m_gameDirectoryToPrimaryKeyMap.end(); ++gameDirectoryToPrimaryKeyIterator)
//    {
//        my::CTimestamp timestamp = GetTimestampFromGameDirectory(gameDirectoryToPrimaryKeyIterator->first);
//
//        if (timestamp.GetYear() == year)
//            gamePrimaryKeyArray.push_back(gameDirectoryToPrimaryKeyIterator->second);
//    }
//
//    return gamePrimaryKeyArray;
//}
//
//// Directory: full game URL on http://gd2.mlb.com
//my::CTimestamp mlb::io::CGamedayServer::GetTimestampFromGameDirectory(std::string gameDirectory) const
//{
//    std::string yearPrefix = "year_",
//        monthPrefix = "month_",
//        dayPrefix = "day_";
//
//    std::string::size_type yearPosition = gameDirectory.find(yearPrefix),
//        monthPosition = gameDirectory.find(monthPrefix),
//        dayPosition = gameDirectory.find(dayPrefix);
//
//    HEALTH_CHECK(yearPosition == std::string::npos, my::Null<my::CTimestamp>());
//    HEALTH_CHECK(monthPosition == std::string::npos, my::Null<my::CTimestamp>());
//    HEALTH_CHECK(dayPosition == std::string::npos, my::Null<my::CTimestamp>());
//
//    // "http://gd2.mlb.com/components/game/mlb/year_2014/month_02/day_27/gid_2014_02_27_clemlb_cinmlb_1/"
//    std::string yearString = gameDirectory.substr(yearPosition + yearPrefix.size(), 4),
//        monthString = gameDirectory.substr(monthPosition + monthPrefix.size(), 2),
//        dayString = gameDirectory.substr(dayPosition + dayPrefix.size(), 2);
//
//    INT64 year = my::Null<INT64>(),
//        month = my::Null<INT64>(),
//        day = my::Null<INT64>();
//
//    try
//    {
//        year = boost::lexical_cast<INT64>(yearString);
//        month = boost::lexical_cast<INT64>(monthString);
//        day = boost::lexical_cast<INT64>(dayString);
//    }
//    catch (...)
//    {
//        LOG_ERROR();
//
//        return my::Null<my::CTimestamp>();
//    }
//
//    return my::CTimestamp(year, month, day);
//}
//
///**
//*/
//my::CTimestamp mlb::io::CGamedayServer::GetTimestampFromMlbGameString(std::string mlbGameString) const
//{
//    HEALTH_CHECK(mlbGameString.size() != 26, my::Null<my::CTimestamp>());
//
//    // "2014_02_27_clemlb_cinmlb_1"
//    std::string yearString = mlbGameString.substr(0, 4),
//        monthString = mlbGameString.substr(5, 2),
//        dayString = mlbGameString.substr(8, 2);
//
//    INT64 year = my::Null<INT64>(),
//        month = my::Null<INT64>(),
//        day = my::Null<INT64>();
//
//    try 
//    {
//        year = boost::lexical_cast<INT64>(yearString);
//        month = boost::lexical_cast<INT64>(monthString);
//        day = boost::lexical_cast<INT64>(dayString);
//    }
//    catch (...)
//    {
//        LOG_ERROR();
//
//        return my::Null<my::CTimestamp>();
//    }
//
//    return my::CTimestamp(year, month, day);
//}
//
//// SV_PITCH_ID 
//// A unique value within a game that identifies the game date and time when the pitch was recorded. 
//// The format is yymmdd_hhmmss, where yymmdd is always the game date and hhmmss is the timestamp in local military time. 
//// Example: the SV_PITCH_ID for a pitch recorded at 7:04:35 p.m. local time on April 9 is:
//// 080409_190435 
//// Note that because the timestamp may cross midnight, but the game date will be unchanged, this value should NOT be used for sorting purposes.
//my::CTimestamp mlb::io::CGamedayServer::GetTimestampFromSportvisionPitchId(std::string sportvisionPitchId) const
//{
//    if (sportvisionPitchId.size() != 13)
//    {
//        LOG_ERROR();
//
//        return false;
//    }
//
//    std::string yearString = sportvisionPitchId.substr(0, 2),
//        monthString = sportvisionPitchId.substr(2, 2),
//        dayString = sportvisionPitchId.substr(4, 2),
//        hoursString = sportvisionPitchId.substr(7, 2),
//        minutesString = sportvisionPitchId.substr(9, 2),
//        secondsString = sportvisionPitchId.substr(11, 2);
//
//    HEALTH_CHECK(yearString.size() != 2, false);
//    HEALTH_CHECK(monthString.size() != 2, false);
//    HEALTH_CHECK(dayString.size() != 2, false);
//    HEALTH_CHECK(hoursString.size() != 2, false);
//    HEALTH_CHECK(minutesString.size() != 2, false);
//    HEALTH_CHECK(secondsString.size() != 2, false);
//
//    my::int32 year = 0,
//        month = 0,
//        day = 0,
//        hours = 0,
//        minutes = 0,
//        seconds = 0;
//
//    try {
//        year = boost::lexical_cast<my::int32>(yearString);
//
//        if (year < 26)
//            year += 2000;
//        else
//            year += 1900;
//
//        month = boost::lexical_cast<my::int32>(monthString);
//        day = boost::lexical_cast<my::int32>(dayString);
//        hours = boost::lexical_cast<my::int32>(hoursString);
//        minutes = boost::lexical_cast<my::int32>(minutesString);
//        seconds = boost::lexical_cast<my::int32>(secondsString);
//    }
//    catch (...) {
//        LOG_ERROR();
//
//        return false;
//    }
//
//    return my::CTimestamp(year, month, day, hours, minutes, seconds);
//}
//
///**
//*/
//std::vector<std::string> mlb::io::CGamedayServer::GetGameDirectoryArray()
//{
//    std::vector<std::string> emptyObject;
//
//    if (m_gameDirectoryArray.empty())
//    {
//        if (!LoadDataUrlContainer("./data/tables/gameday_game_data_urls.txt", m_gameDirectoryArray))
//        {
//            LOG_ERROR();
//
//            return emptyObject;
//        }
//    }
//
//    return m_gameDirectoryArray;
//}
//
///**
//*/
//std::vector<std::string> mlb::io::CGamedayServer::GetGameDirectoryArrayFromDate(INT32 year, INT32 month, INT32 day)
//{
//    std::vector<std::string> emptyObject;
//
//    if (m_gameDirectoryArray.empty())
//    {
//        if (!LoadDataUrlContainer("./data/tables/gameday_game_data_urls.txt", m_gameDirectoryArray))
//        {
//            LOG_ERROR();
//
//            return emptyObject;
//        }
//    }
//
//    std::vector<std::string> mlbGameStringArray;
//
//    for (std::vector<std::string>::const_iterator gameDirectoryIterator = m_gameDirectoryArray.begin(); gameDirectoryIterator != m_gameDirectoryArray.end(); ++gameDirectoryIterator)
//    {
//        my::CTimestamp timestamp = GetTimestampFromGameDirectory(*gameDirectoryIterator);
//
//        if ((timestamp.GetYear() == year) &&
//            (timestamp.GetMonth() == month) &&
//            (timestamp.GetDay() == day))
//        {
//            mlbGameStringArray.push_back(*gameDirectoryIterator);
//        }
//    }
//
//    return mlbGameStringArray;
//}
//
//// Web-based (Gameday Server)
//std::vector<mlb::io::CGame> mlb::io::CGamedayServer::GetGameArrayFromDate(my::int32 year, my::int32 month, my::int32 day) const
//{
//    std::vector<mlb::io::CGame> emptyObject;
//
//    HEALTH_CHECK(my::IsNull(year), emptyObject);
//    HEALTH_CHECK(year < 1900, emptyObject);
//    HEALTH_CHECK(my::IsNull(month), emptyObject);
//    HEALTH_CHECK(month < 1, emptyObject);
//    HEALTH_CHECK(month > 12, emptyObject);
//    HEALTH_CHECK(my::IsNull(day), emptyObject);
//    HEALTH_CHECK(day < 1, emptyObject);
//    HEALTH_CHECK(day > 31, emptyObject);
//
//    std::string yearString = my::NumberToString(year),
//        monthString = my::NumberToString(month),
//        dayString = my::NumberToString(day);
//
//    if (monthString.size() == 1)
//        monthString = "0" + monthString;
//
//    if (dayString.size() == 1)
//        dayString = "0" + dayString;
//
//    std::vector<mlb::io::CGame> gameArray;
//
//    std::string dayGridUrl = "http://gd2.mlb.com/components/game/mlb/year_"
//        + yearString
//        + "/month_"
//        + monthString
//        + "/day_"
//        + dayString
//        + "/grid.json";
//
//    //// DEBUG ONLY! (07-Jun-2016)
//    //std::cout << "LOADING " << dayGridUrl << std::endl;
//
//    std::string fileBuffer;
//    
//    my::file::GetUrlAsString(dayGridUrl, fileBuffer);
//
//    if (!fileBuffer.empty() &&
//        (fileBuffer.find("404 Not Found") == std::string::npos) &&
//        (fileBuffer.find("Error (404)") == std::string::npos))
//    {
//        rapidjson::Document document;
//
//        HEALTH_CHECK(document.Parse<0>(fileBuffer.c_str()).HasParseError(), emptyObject);
//        HEALTH_CHECK(!document.IsObject(), emptyObject);
//
//        if (document.IsObject() &&
//            document.HasMember("data"))
//        {
//            const rapidjson::Value& dataHandle = document["data"];
//
//            if (dataHandle.HasMember("games"))
//            {
//                const rapidjson::Value& gamesHandle = dataHandle["games"];
//
//                if (gamesHandle.HasMember("game"))
//                {
//                    const rapidjson::Value& gameArrayHandle = gamesHandle["game"];
//
//                    if (gameArrayHandle.IsArray())
//                    {
//                        for (rapidjson::SizeType gameIndex = 0; gameIndex < gameArrayHandle.Size(); ++gameIndex)
//                        {
//                            const rapidjson::Value& gameIterator = gameArrayHandle[gameIndex];
//
//                            HEALTH_CHECK(!gameIterator.IsObject(), emptyObject);
//
//                            mlb::io::CGame game;
//
//                            my::CTimestamp gameTimestamp(my::Null<my::CTimestamp>());
//
//                            // "game_type":"R"
//                            // "game_nbr":"1"
//                            // "double_header_sw":"N"
//                            // "away_team_name":"Reds"
//
//                            // "id":"2016/05/16/cinmlb-clemlb-1"
//                            if (gameIterator.HasMember("id"))
//                            {
//                                const rapidjson::Value& idHandle = gameIterator["id"];
//
//                                std::string id = idHandle.GetString();
//
//                                boost::replace_all(id, "/", "_");
//                                boost::replace_all(id, "-", "_");
//
//                                if (id.size() == 26)
//                                {
//                                    game.SetMlbGameString(id);
//
//                                    gameTimestamp = GetTimestampFromMlbGameString(id);
//                                }
//                            }
//
//                            // "home_name_abbrev":"CLE"
//                            if (gameIterator.HasMember("home_name_abbrev"))
//                            {
//                                const rapidjson::Value& homeTeamNameAbbreviationHandle = gameIterator["home_name_abbrev"];
//
//                                std::string homeTeamNameAbbreviation = homeTeamNameAbbreviationHandle.GetString();
//
//                                if (!homeTeamNameAbbreviation.empty())
//                                    game.SetHomeTeamNameAbbreviation(homeTeamNameAbbreviation);
//                            }
//
//                            // "media_state":"media_archive"
//                            // "top_inning":"Y"
//                            // "home_team_name":"Indians"
//                            // "ind":"F"
//
//                            // "venue_id":"5"
//                            if (gameIterator.HasMember("venue_id"))
//                            {
//                                const rapidjson::Value& venueIdHandle = gameIterator["venue_id"];
//
//                                std::string venueIdString = venueIdHandle.GetString();
//
//                                if (!venueIdString.empty())
//                                {
//                                    my::int32 venueId = boost::lexical_cast<my::int32>(venueIdString);
//
//                                    game.SetVenueId(venueId);
//                                }
//                            }
//
//                            // "gameday_sw":"P"
//
//                            // "away_team_id":"113"
//                            if (gameIterator.HasMember("away_team_id"))
//                            {
//                                const rapidjson::Value& awayTeamIdHandle = gameIterator["away_team_id"];
//
//                                std::string awayTeamIdString = awayTeamIdHandle.GetString();
//
//                                if (!awayTeamIdString.empty())
//                                {
//                                    my::int32 awayTeamId = boost::lexical_cast<my::int32>(awayTeamIdString);
//
//                                    game.SetAwayTeamId(awayTeamId);
//                                }
//                            }
//
//                            // "home_score":"15"
//                            if (gameIterator.HasMember("home_score"))
//                            {
//                                const rapidjson::Value& homeTeamRunsHandle = gameIterator["home_score"];
//
//                                std::string homeTeamRunsString = homeTeamRunsHandle.GetString();
//
//                                if (!homeTeamRunsString.empty())
//                                {
//                                    my::int32 homeTeamRuns = boost::lexical_cast<my::int32>(homeTeamRunsString);
//
//                                    game.SetHomeTeamRuns(homeTeamRuns);
//                                }
//                            }
//
//                            // "status":"Final"
//                            // "home_code":"cle"
//
//                            // "away_score":"6"
//                            if (gameIterator.HasMember("away_score"))
//                            {
//                                const rapidjson::Value& awayTeamRunsHandle = gameIterator["away_score"];
//
//                                std::string awayTeamRunsString = awayTeamRunsHandle.GetString();
//
//                                if (!awayTeamRunsString.empty())
//                                {
//                                    my::int32 awayTeamRuns = boost::lexical_cast<my::int32>(awayTeamRunsString);
//
//                                    game.SetAwayTeamRuns(awayTeamRuns);
//                                }
//                            }
//
//                            // "inning":"9"
//
//                            // "game_pk":"447438"
//                            if (gameIterator.HasMember("game_pk"))
//                            {
//                                const rapidjson::Value& gamePkHandle = gameIterator["game_pk"];
//
//                                my::int32 gamePrimaryKey = my::Null<my::int32>();
//
//                                if (gamePkHandle.IsInt())
//                                    gamePrimaryKey = gamePkHandle.GetInt();
//                                else if (gamePkHandle.IsString())
//                                    gamePrimaryKey = atoi(gamePkHandle.GetString());
//
//                                if (!my::IsNull(gamePrimaryKey))
//                                {
//                                    game.SetGamePrimaryKey(gamePrimaryKey);
//
//                                    std::string statCastUrl = "https://statsapi.mlb.com/api/v1/game/"
//                                        + my::NumberToString(gamePrimaryKey)
//                                        + "/guids";
//
//                                    game.SetStatCastUrl(statCastUrl);
//                                }
//                            }
//
//                            // "away_name_abbrev":"CIN"
//                            if (gameIterator.HasMember("away_name_abbrev"))
//                            {
//                                const rapidjson::Value& awayTeamNameAbbreviationHandle = gameIterator["away_name_abbrev"];
//
//                                std::string awayTeamNameAbbreviation = awayTeamNameAbbreviationHandle.GetString();
//
//                                if (!awayTeamNameAbbreviation.empty())
//                                    game.SetAwayTeamNameAbbreviation(awayTeamNameAbbreviation);
//                            }
//
//                            // "venue":"Progressive Field"
//                            if (gameIterator.HasMember("venue"))
//                            {
//                                const rapidjson::Value& venueNameHandle = gameIterator["venue"];
//
//                                std::string venueName = venueNameHandle.GetString();
//
//                                if (!venueName.empty())
//                                    game.SetVenueName(venueName);
//                            }
//
//                            // "home_file_code":"cle"
//                            // "away_file_code":"cin"
//
//                            // "event_time":"6:10 PM"
//                            if (gameIterator.HasMember("event_time"))
//                            {
//                                const rapidjson::Value& eventTimeHandle = gameIterator["event_time"];
//
//                                std::string eventTimeString = eventTimeHandle.GetString();
//
//                                if (eventTimeString.size() == 7)
//                                {
//                                    std::string hoursString = eventTimeString.substr(0, eventTimeString.find_first_of(":")),
//                                        minutesString = eventTimeString.substr(eventTimeString.find_first_of(":") + 1, 2);
//
//                                    my::int32 hours = 0,
//                                        minutes = 0;
//
//                                    try {
//                                        hours = boost::lexical_cast<my::int32>(hoursString);
//
//                                        if (eventTimeString.find("PM") != std::string::npos)
//                                            hours += 12;
//
//                                        minutes = boost::lexical_cast<my::int32>(minutesString);
//                                    }
//                                    catch (...) {
//                                        LOG_ERROR();
//                                    }
//
//                                    if (gameTimestamp != my::Null<my::CTimestamp>())
//                                    {
//                                        gameTimestamp.AddHours(hours);
//                                        gameTimestamp.AddMinutes(minutes);
//                                    }
//                                }
//                            }
//
//                            // "home_team_id":"114"
//                            if (gameIterator.HasMember("home_team_id"))
//                            {
//                                const rapidjson::Value& homeTeamIdHandle = gameIterator["home_team_id"];
//
//                                std::string homeTeamIdString = homeTeamIdHandle.GetString();
//
//                                if (!homeTeamIdString.empty())
//                                {
//                                    my::int32 homeTeamId = boost::lexical_cast<my::int32>(homeTeamIdString);
//
//                                    game.SetHomeTeamId(homeTeamId);
//                                }
//                            }
//
//                            // "calendar_event_id":"14-447438-2016-05-16"
//                            // "group":"MLB"
//                            // "tbd_flag":"N"
//                            // "away_code":"cin"
//
//                            if (gameTimestamp != my::Null<my::CTimestamp>())
//                                game.SetTime(gameTimestamp.ToMicrosecondsFromEpoch());
//
//                            gameArray.push_back(game);
//                        }
//                    }
//                    else if (gameArrayHandle.IsObject())
//                    {
//                        mlb::io::CGame game;
//
//                        my::CTimestamp gameTimestamp(my::Null<my::CTimestamp>());
//
//                        // "game_type":"R"
//                        // "game_nbr":"1"
//                        // "double_header_sw":"N"
//                        // "away_team_name":"Reds"
//
//                        // "id":"2016/05/16/cinmlb-clemlb-1"
//                        if (gameArrayHandle.HasMember("id"))
//                        {
//                            const rapidjson::Value& idHandle = gameArrayHandle["id"];
//
//                            std::string id = idHandle.GetString();
//
//                            boost::replace_all(id, "/", "_");
//                            boost::replace_all(id, "-", "_");
//
//                            if (id.size() == 26)
//                            {
//                                game.SetMlbGameString(id);
//
//                                gameTimestamp = GetTimestampFromMlbGameString(id);
//                            }
//                        }
//
//                        // "home_name_abbrev":"CLE"
//                        if (gameArrayHandle.HasMember("home_name_abbrev"))
//                        {
//                            const rapidjson::Value& homeTeamNameAbbreviationHandle = gameArrayHandle["home_name_abbrev"];
//
//                            std::string homeTeamNameAbbreviation = homeTeamNameAbbreviationHandle.GetString();
//
//                            if (!homeTeamNameAbbreviation.empty())
//                                game.SetHomeTeamNameAbbreviation(homeTeamNameAbbreviation);
//                        }
//
//                        // "media_state":"media_archive"
//                        // "top_inning":"Y"
//                        // "home_team_name":"Indians"
//                        // "ind":"F"
//
//                        // "venue_id":"5"
//                        if (gameArrayHandle.HasMember("venue_id"))
//                        {
//                            const rapidjson::Value& venueIdHandle = gameArrayHandle["venue_id"];
//
//                            std::string venueIdString = venueIdHandle.GetString();
//
//                            if (!venueIdString.empty())
//                            {
//                                my::int32 venueId = boost::lexical_cast<my::int32>(venueIdString);
//
//                                game.SetVenueId(venueId);
//                            }
//                        }
//
//                        // "gameday_sw":"P"
//
//                        // "away_team_id":"113"
//                        if (gameArrayHandle.HasMember("away_team_id"))
//                        {
//                            const rapidjson::Value& awayTeamIdHandle = gameArrayHandle["away_team_id"];
//
//                            std::string awayTeamIdString = awayTeamIdHandle.GetString();
//
//                            if (!awayTeamIdString.empty())
//                            {
//                                my::int32 awayTeamId = boost::lexical_cast<my::int32>(awayTeamIdString);
//
//                                game.SetAwayTeamId(awayTeamId);
//                            }
//                        }
//
//                        // "home_score":"15"
//                        if (gameArrayHandle.HasMember("home_score"))
//                        {
//                            const rapidjson::Value& homeTeamRunsHandle = gameArrayHandle["home_score"];
//
//                            std::string homeTeamRunsString = homeTeamRunsHandle.GetString();
//
//                            if (!homeTeamRunsString.empty())
//                            {
//                                my::int32 homeTeamRuns = boost::lexical_cast<my::int32>(homeTeamRunsString);
//
//                                game.SetHomeTeamRuns(homeTeamRuns);
//                            }
//                        }
//
//                        // "status":"Final"
//                        // "home_code":"cle"
//
//                        // "away_score":"6"
//                        if (gameArrayHandle.HasMember("away_score"))
//                        {
//                            const rapidjson::Value& awayTeamRunsHandle = gameArrayHandle["away_score"];
//
//                            std::string awayTeamRunsString = awayTeamRunsHandle.GetString();
//
//                            if (!awayTeamRunsString.empty())
//                            {
//                                my::int32 awayTeamRuns = boost::lexical_cast<my::int32>(awayTeamRunsString);
//
//                                game.SetAwayTeamRuns(awayTeamRuns);
//                            }
//                        }
//
//                        // "inning":"9"
//
//                        // "game_pk":"447438"
//                        if (gameArrayHandle.HasMember("game_pk"))
//                        {
//                            const rapidjson::Value& gamePkHandle = gameArrayHandle["game_pk"];
//
//                            my::int32 gamePrimaryKey = my::Null<my::int32>();
//
//                            if (gamePkHandle.IsInt())
//                                gamePrimaryKey = gamePkHandle.GetInt();
//                            else if (gamePkHandle.IsString())
//                                gamePrimaryKey = atoi(gamePkHandle.GetString());
//
//                            if (!my::IsNull(gamePrimaryKey))
//                            {
//                                game.SetGamePrimaryKey(gamePrimaryKey);
//
//                                std::string statCastUrl = "https://statsapi.mlb.com/api/v1/game/"
//                                    + my::NumberToString(gamePrimaryKey)
//                                    + "/guids";
//
//                                game.SetStatCastUrl(statCastUrl);
//                            }
//                        }
//
//                        // "away_name_abbrev":"CIN"
//                        if (gameArrayHandle.HasMember("away_name_abbrev"))
//                        {
//                            const rapidjson::Value& awayTeamNameAbbreviationHandle = gameArrayHandle["away_name_abbrev"];
//
//                            std::string awayTeamNameAbbreviation = awayTeamNameAbbreviationHandle.GetString();
//
//                            if (!awayTeamNameAbbreviation.empty())
//                                game.SetAwayTeamNameAbbreviation(awayTeamNameAbbreviation);
//                        }
//
//                        // "venue":"Progressive Field"
//                        if (gameArrayHandle.HasMember("venue"))
//                        {
//                            const rapidjson::Value& venueNameHandle = gameArrayHandle["venue"];
//
//                            std::string venueName = venueNameHandle.GetString();
//
//                            if (!venueName.empty())
//                                game.SetVenueName(venueName);
//                        }
//
//                        // "home_file_code":"cle"
//                        // "away_file_code":"cin"
//
//                        // "event_time":"6:10 PM"
//                        if (gameArrayHandle.HasMember("event_time"))
//                        {
//                            const rapidjson::Value& eventTimeHandle = gameArrayHandle["event_time"];
//
//                            std::string eventTimeString = eventTimeHandle.GetString();
//
//                            if (eventTimeString.size() == 7)
//                            {
//                                std::string hoursString = eventTimeString.substr(0, eventTimeString.find_first_of(":")),
//                                    minutesString = eventTimeString.substr(eventTimeString.find_first_of(":") + 1, 2);
//
//                                my::int32 hours = 0,
//                                    minutes = 0;
//
//                                try {
//                                    hours = boost::lexical_cast<my::int32>(hoursString);
//
//                                    if (eventTimeString.find("PM") != std::string::npos)
//                                        hours += 12;
//
//                                    minutes = boost::lexical_cast<my::int32>(minutesString);
//                                }
//                                catch (...) {
//                                    LOG_ERROR();
//                                }
//
//                                if (gameTimestamp != my::Null<my::CTimestamp>())
//                                {
//                                    gameTimestamp.AddHours(hours);
//                                    gameTimestamp.AddMinutes(minutes);
//                                }
//                            }
//                        }
//
//                        // "home_team_id":"114"
//                        if (gameArrayHandle.HasMember("home_team_id"))
//                        {
//                            const rapidjson::Value& homeTeamIdHandle = gameArrayHandle["home_team_id"];
//
//                            std::string homeTeamIdString = homeTeamIdHandle.GetString();
//
//                            if (!homeTeamIdString.empty())
//                            {
//                                my::int32 homeTeamId = boost::lexical_cast<my::int32>(homeTeamIdString);
//
//                                game.SetHomeTeamId(homeTeamId);
//                            }
//                        }
//
//                        // "calendar_event_id":"14-447438-2016-05-16"
//                        // "group":"MLB"
//                        // "tbd_flag":"N"
//                        // "away_code":"cin"
//
//                        if (gameTimestamp != my::Null<my::CTimestamp>())
//                            game.SetTime(gameTimestamp.ToMicrosecondsFromEpoch());
//
//                        gameArray.push_back(game);
//                    }
//                }
//            }
//        }
//    }
//
//    return gameArray;
//}
//
///**
//*/
//std::vector<std::string> mlb::io::CGamedayServer::GetStatCastGameUrlArrayFromDate(my::int32 year, my::int32 month, my::int32 day) const
//{
//    std::vector<std::string> emptyObject;
//
//    HEALTH_CHECK(my::IsNull(year), emptyObject);
//    HEALTH_CHECK(year < 1900, emptyObject);
//    HEALTH_CHECK(my::IsNull(month), emptyObject);
//    HEALTH_CHECK(month < 1, emptyObject);
//    HEALTH_CHECK(month > 12, emptyObject);
//    HEALTH_CHECK(my::IsNull(day), emptyObject);
//    HEALTH_CHECK(day < 1, emptyObject);
//    HEALTH_CHECK(day > 31, emptyObject);
//
//    std::string yearString = my::NumberToString(year),
//        monthString = my::NumberToString(month),
//        dayString = my::NumberToString(day);
//
//    if (monthString.size() == 1)
//        monthString = "0" + monthString;
//
//    if (dayString.size() == 1)
//        dayString = "0" + dayString;
//
//    std::set<my::int32> gamePrimaryKeySet;
//    std::vector<std::string> statCastGameUrlArray;
//
//    std::string dayGridUrl = "http://gd2.mlb.com/components/game/mlb/year_"
//        + yearString
//        + "/month_"
//        + monthString
//        + "/day_"
//        + dayString
//        + "/grid.json";
//
//    std::cout << "LOADING " << dayGridUrl << std::endl;
//
//    std::string fileBuffer;
//    
//    my::file::GetUrlAsString(dayGridUrl, fileBuffer);
//
//    if (!fileBuffer.empty() &&
//        (fileBuffer.find("404 Not Found") == std::string::npos) &&
//        (fileBuffer.find("Error (404)") == std::string::npos))
//    {
//        rapidjson::Document document;
//
//        HEALTH_CHECK(document.Parse<0>(fileBuffer.c_str()).HasParseError(), emptyObject);
//        HEALTH_CHECK(!document.IsObject(), emptyObject);
//
//        if (document.IsObject() &&
//            document.HasMember("data"))
//        {
//            const rapidjson::Value& dataHandle = document["data"];
//
//            if (dataHandle.HasMember("games"))
//            {
//                const rapidjson::Value& gamesHandle = dataHandle["games"];
//
//                if (gamesHandle.HasMember("game"))
//                {
//                    const rapidjson::Value& gameDataHandle = gamesHandle["game"];
//
//                    if (gameDataHandle.IsArray())
//                    {
//                        for (rapidjson::SizeType gameIndex = 0; gameIndex < gameDataHandle.Size(); ++gameIndex)
//                        {
//                            const rapidjson::Value& gameIterator = gameDataHandle[gameIndex];
//
//                            HEALTH_CHECK(!gameIterator.IsObject(), emptyObject);
//
//                            if (gameIterator.HasMember("game_pk"))
//                            {
//                                const rapidjson::Value& gamePkHandle = gameIterator["game_pk"];
//
//                                my::int32 gamePrimaryKey = my::Null<my::int32>();
//
//                                if (gamePkHandle.IsInt())
//                                    gamePrimaryKey = gamePkHandle.GetInt();
//                                else if (gamePkHandle.IsString())
//                                    gamePrimaryKey = atoi(gamePkHandle.GetString());
//
//                                if (!my::IsNull(gamePrimaryKey) &&
//                                    (gamePrimaryKeySet.find(gamePrimaryKey) == gamePrimaryKeySet.end()))
//                                {
//                                    std::string statCastGameUrl = "https://statsapi.mlb.com/api/v1/game/"
//                                        + my::NumberToString(gamePrimaryKey)
//                                        + "/guids";
//
//                                    statCastGameUrlArray.push_back(statCastGameUrl);
//
//                                    gamePrimaryKeySet.insert(gamePrimaryKey);
//                                }
//                            }
//                        }
//                    }
//                    else if (gameDataHandle.IsObject())
//                    {
//                        if (gameDataHandle.HasMember("game_pk"))
//                        {
//                            const rapidjson::Value& gamePkHandle = gameDataHandle["game_pk"];
//
//                            my::int32 gamePrimaryKey = my::Null<my::int32>();
//
//                            if (gamePkHandle.IsInt())
//                                gamePrimaryKey = gamePkHandle.GetInt();
//                            else if (gamePkHandle.IsString())
//                                gamePrimaryKey = atoi(gamePkHandle.GetString());
//
//                            if (!my::IsNull(gamePrimaryKey) &&
//                                (gamePrimaryKeySet.find(gamePrimaryKey) == gamePrimaryKeySet.end()))
//                            {
//                                std::string statCastGameUrl = "https://statsapi.mlb.com/api/v1/game/"
//                                    + my::NumberToString(gamePrimaryKey)
//                                    + "/guids";
//
//                                statCastGameUrlArray.push_back(statCastGameUrl);
//
//                                gamePrimaryKeySet.insert(gamePrimaryKey);
//                            }
//                        }
//                    }
//                    else
//                        LOG_ERROR();
//                }
//            }
//        }
//    }
//
//    return statCastGameUrlArray;
//}
//
///**
//*/
//std::vector<std::string> mlb::io::CGamedayServer::GetStatCastGameUrlArrayFromDate(my::int32 year, my::int32 month) const
//{
//    std::vector<std::string> emptyObject;
//
//    HEALTH_CHECK(my::IsNull(year), emptyObject);
//    HEALTH_CHECK(year < 1900, emptyObject);
//    HEALTH_CHECK(my::IsNull(month), emptyObject);
//    HEALTH_CHECK(month < 1, emptyObject);
//    HEALTH_CHECK(month > 12, emptyObject);
//
//    std::string yearString = my::NumberToString(year),
//        monthString = my::NumberToString(month);
//
//    if (monthString.size() == 1)
//        monthString = "0" + monthString;
//
//    std::set<my::int32> gamePrimaryKeySet;
//    std::vector<std::string> statCastGameUrlArray;
//
//    for (my::int32 day = 1; day <= 31; ++day)
//    {
//        std::string dayString = my::NumberToString(day);
//
//        if (dayString.size() == 1)
//            dayString = "0" + dayString;
//
//        std::string dayGridUrl = "http://gd2.mlb.com/components/game/mlb/year_"
//            + yearString
//            + "/month_"
//            + monthString
//            + "/day_"
//            + dayString
//            + "/grid.json";
//
//        std::cout << "LOADING " << dayGridUrl << std::endl;
//
//        std::string fileBuffer;
//        
//        my::file::GetUrlAsString(dayGridUrl, fileBuffer);
//
//        if (!fileBuffer.empty() &&
//            (fileBuffer.find("404 Not Found") == std::string::npos) &&
//            (fileBuffer.find("Error (404)") == std::string::npos))
//        {
//            rapidjson::Document document;
//
//            HEALTH_CHECK(document.Parse<0>(fileBuffer.c_str()).HasParseError(), emptyObject);
//            HEALTH_CHECK(!document.IsObject(), emptyObject);
//
//            if (document.IsObject() &&
//                document.HasMember("data"))
//            {
//                const rapidjson::Value& dataHandle = document["data"];
//
//                if (dataHandle.HasMember("games"))
//                {
//                    const rapidjson::Value& gamesHandle = dataHandle["games"];
//
//                    if (gamesHandle.HasMember("game"))
//                    {
//                        const rapidjson::Value& gameArrayHandle = gamesHandle["game"];
//
//                        HEALTH_CHECK(!gameArrayHandle.IsArray(), emptyObject);
//
//                        for (rapidjson::SizeType gameIndex = 0; gameIndex < gameArrayHandle.Size(); ++gameIndex)
//                        {
//                            const rapidjson::Value& gameIterator = gameArrayHandle[gameIndex];
//
//                            HEALTH_CHECK(!gameIterator.IsObject(), emptyObject);
//
//                            if (gameIterator.HasMember("game_pk"))
//                            {
//                                const rapidjson::Value& gamePkHandle = gameIterator["game_pk"];
//
//                                my::int32 gamePrimaryKey = my::Null<my::int32>();
//
//                                if (gamePkHandle.IsInt())
//                                    gamePrimaryKey = gamePkHandle.GetInt();
//                                else if (gamePkHandle.IsString())
//                                    gamePrimaryKey = atoi(gamePkHandle.GetString());
//
//                                if (!my::IsNull(gamePrimaryKey) &&
//                                    (gamePrimaryKeySet.find(gamePrimaryKey) == gamePrimaryKeySet.end()))
//                                {
//                                    std::string statCastGameUrl = "https://statsapi.mlb.com/api/v1/game/"
//                                        + my::NumberToString(gamePrimaryKey)
//                                        + "/guids";
//
//                                    statCastGameUrlArray.push_back(statCastGameUrl);
//
//                                    gamePrimaryKeySet.insert(gamePrimaryKey);
//                                }
//                            }
//                        }
//                    }
//                }
//            }
//        }
//    }
//
//    return statCastGameUrlArray;
//}
//
///**
//*/
//std::string mlb::io::CGamedayServer::GetMlbGameStringFromGameDirectory(std::string gameDirectory) const
//{
//    std::string gidPrefix = "gid_";
//
//    std::string::size_type gidPosition = gameDirectory.find(gidPrefix);
//
//    std::string emptyObject;
//
//    if (gidPosition == std::string::npos)
//    {
//        LOG_MESSAGE("Failed to find game delimiter (\"gid_\") on string " + gameDirectory + ".");
//
//        return emptyObject;
//    }
//
//    // "http://gd2.mlb.com/components/game/mlb/year_2014/month_02/day_27/gid_2014_02_27_clemlb_cinmlb_1/"
//    std::string mlbGameString = gameDirectory.substr(gidPosition + gidPrefix.size(), 26);
//
//    HEALTH_CHECK(mlbGameString.empty(), emptyObject);
//
//    return mlbGameString;
//}
//
///**
//*/
//mlb::io::CVenue mlb::io::CGamedayServer::GetVenue(std::string gameDirectory) const
//{
//    std::string fileBuffer;
//    
//    my::file::GetUrlAsString(my::AddTrailingSlash(gameDirectory) + "linescore.xml", fileBuffer);
//
//    mlb::io::CVenue emptyObject,
//        venue;
//
//    if (fileBuffer.empty() ||
//        (fileBuffer.find("404 Not Found") != std::string::npos))
//    {
//        LOG_ERROR();
//
//        return emptyObject;
//    }
//
//    pugi::xml_document xmlDocument;
//
//    pugi::xml_parse_result xmlDocumentLog = xmlDocument.load(fileBuffer.c_str());
//
//    if (!xmlDocumentLog)
//    {
//        LOG_MESSAGE(xmlDocumentLog.description());
//        LOG_MESSAGE(std::string("Error offset: ") + boost::lexical_cast<std::string>(xmlDocumentLog.offset));
//
//        LOG_ERROR();
//
//        return emptyObject;
//    }
//
//    if (xmlDocument.child("game_position").empty())
//    {
//        LOG_ERROR();
//
//        return emptyObject;
//    }
//        
//    if (!xmlDocument.child("game").attribute("venue").empty())
//        venue.SetName(xmlDocument.child("game").attribute("venue").as_string());
//    if (!xmlDocument.child("game").attribute("venue_id").empty())
//        venue.SetId(xmlDocument.child("game").attribute("venue_id").as_int());
//    if (!xmlDocument.child("game").attribute("home_code").empty())
//        venue.SetCode(xmlDocument.child("game").attribute("home_code").as_string());
//    if (!xmlDocument.child("game").attribute("home_team_name").empty())
//        venue.SetTeamName(xmlDocument.child("game").attribute("home_team_name").as_string());
//    if (!xmlDocument.child("game").attribute("home_name_abbrev").empty())
//        venue.SetTeamNameAbbreviation(xmlDocument.child("game").attribute("home_name_abbrev").as_string());
//    if (!xmlDocument.child("game").attribute("home_team_id").empty())
//        venue.SetTeamId(xmlDocument.child("game").attribute("home_team_id").as_int());
//    if (!xmlDocument.child("game").attribute("home_team_city").empty())
//        venue.SetTeamCity(xmlDocument.child("game").attribute("home_team_city").as_string());
//
//    return venue;
//}
//
///**
//*/
//mlb::io::CVenue mlb::io::CGamedayServer::GetVenueFromId(INT32 venueId)
//{
//    CVenue emptyObject;
//
//    if (m_venueArray.empty())
//    {
//        std::string fileBuffer;
//
//        if (my::file::GetFileAsString("./data/tables/venue_information.csv", fileBuffer) == 0)
//        {
//            LOG_ERROR();
//
//            return emptyObject;
//        }
//
//        CCsvInputFile csvInputFile;
//
//        if (!csvInputFile.FromString(fileBuffer, ";"))
//        {
//            LOG_ERROR();
//
//            return emptyObject;
//        }
//
//        csvInputFile.Pop();
//        csvInputFile.Pop();
//        csvInputFile.Pop();
//        csvInputFile.Pop();
//        csvInputFile.Pop();
//        csvInputFile.Pop();
//        csvInputFile.Pop();
//
//        while (!csvInputFile.isEmpty())
//        {
//            CVenue venue;
//
//            HEALTH_CHECK(!csvInputFile.ReadValue(&mlb::io::CVenue::SetName, venue), emptyObject);
//            HEALTH_CHECK(!csvInputFile.ReadValue(&mlb::io::CVenue::SetId, venue), emptyObject);
//            HEALTH_CHECK(!csvInputFile.ReadValue(&mlb::io::CVenue::SetCode, venue), emptyObject);
//            HEALTH_CHECK(!csvInputFile.ReadValue(&mlb::io::CVenue::SetTeamName, venue), emptyObject);
//            HEALTH_CHECK(!csvInputFile.ReadValue(&mlb::io::CVenue::SetTeamNameAbbreviation, venue), emptyObject);
//            HEALTH_CHECK(!csvInputFile.ReadValue(&mlb::io::CVenue::SetTeamId, venue), emptyObject);
//            HEALTH_CHECK(!csvInputFile.ReadValue(&mlb::io::CVenue::SetTeamCity, venue), emptyObject);
//
//            m_venueArray.push_back(venue);
//        }
//    }
//
//    for (std::vector<CVenue>::const_iterator venueIterator = m_venueArray.begin(); venueIterator != m_venueArray.end(); ++venueIterator)
//    {
//        if (venueIterator->GetId() == venueId)
//            return (*venueIterator);
//    }
//
//    return emptyObject;
//}
//
//// Local data (data/tables)
//mlb::io::CTeam mlb::io::CGamedayServer::GetTeamFromCode(std::string code)
//{
//    CTeam emptyObject;
//
//    if (m_teamArray.empty())
//    {
//        std::string fileBuffer;
//
//        if (my::file::GetFileAsString("./data/tables/team_information.csv", fileBuffer) == 0)
//        {
//            LOG_ERROR();
//
//            return emptyObject;
//        }
//
//        CCsvInputFile csvInputFile;
//
//        if (!csvInputFile.FromString(fileBuffer, ";"))
//        {
//            LOG_ERROR();
//
//            return emptyObject;
//        }
//
//        while (!csvInputFile.isEmpty())
//        {
//            CTeam team;
//
//            HEALTH_CHECK(!csvInputFile.ReadValue(&mlb::io::CTeam::SetNameAbbreviation, team), emptyObject);
//            HEALTH_CHECK(!csvInputFile.ReadValue(&mlb::io::CTeam::SetCode, team), emptyObject);
//            HEALTH_CHECK(!csvInputFile.ReadValue(&mlb::io::CTeam::SetId, team), emptyObject);
//            HEALTH_CHECK(!csvInputFile.ReadValue(&mlb::io::CTeam::SetCity, team), emptyObject);
//            HEALTH_CHECK(!csvInputFile.ReadValue(&mlb::io::CTeam::SetName, team), emptyObject);
//
//            m_teamArray.push_back(team);
//        }
//    }
//
//    for (std::vector<CTeam>::const_iterator teamIterator = m_teamArray.begin(); teamIterator != m_teamArray.end(); ++teamIterator)
//    {
//        if (teamIterator->GetCode() == code)
//            return *teamIterator;
//    }
//
//    return emptyObject;
//}
//
//// Local data (data/tables)
//mlb::io::CTeam mlb::io::CGamedayServer::GetTeamFromId(INT32 id)
//{
//    CTeam emptyObject;
//
//    if (m_teamArray.empty())
//    {
//        std::string fileBuffer;
//
//        if (my::file::GetFileAsString("./data/tables/team_information.csv", fileBuffer) == 0)
//        {
//            LOG_ERROR();
//
//            return emptyObject;
//        }
//
//        CCsvInputFile csvInputFile;
//
//        if (!csvInputFile.FromString(fileBuffer, ";"))
//        {
//            LOG_ERROR();
//
//            return emptyObject;
//        }
//
//        while (!csvInputFile.isEmpty())
//        {
//            CTeam team;
//
//            HEALTH_CHECK(!csvInputFile.ReadValue(&mlb::io::CTeam::SetNameAbbreviation, team), emptyObject);
//            HEALTH_CHECK(!csvInputFile.ReadValue(&mlb::io::CTeam::SetCode, team), emptyObject);
//            HEALTH_CHECK(!csvInputFile.ReadValue(&mlb::io::CTeam::SetId, team), emptyObject);
//            HEALTH_CHECK(!csvInputFile.ReadValue(&mlb::io::CTeam::SetCity, team), emptyObject);
//            HEALTH_CHECK(!csvInputFile.ReadValue(&mlb::io::CTeam::SetName, team), emptyObject);
//
//            m_teamArray.push_back(team);
//        }
//    }
//
//    for (std::vector<CTeam>::const_iterator teamIterator = m_teamArray.begin(); teamIterator != m_teamArray.end(); ++teamIterator)
//    {
//        if (teamIterator->GetId() == id)
//            return *teamIterator;
//    }
//
//    return emptyObject;
//}
//
////// DEBUG ONLY! (22-Oct-2015) TEAM INFORMATION DATALOAD
////mlb::io::CGamedayServer::TEAMS mlb::io::CGamedayServer::GetTeams(std::string gameDirectory) const
////{
////    CMyHttpGet httpGet;
////
////    std::string fileBuffer = httpGet.Get(my::AddTrailingSlash(gameDirectory) + "linescore.xml");
////
////    TEAMS emptyObject,
////        teams;
////
////    if (fileBuffer.empty() ||
////        (fileBuffer == "GameDay - 404 Not Found"))
////    {
////        LOG_ERROR();
////
////        return emptyObject;
////    }
////
////    pugi::xml_document xmlDocument;
////
////    pugi::xml_parse_result xmlDocumentLog = xmlDocument.load(fileBuffer.c_str());
////
////    if (!xmlDocumentLog)
////    {
////        LOG_ERROR();
////        LOG_MESSAGE(xmlDocumentLog.description());
////        LOG_MESSAGE("Error offset: " + xmlDocumentLog.offset);
////
////        return emptyObject;
////    }
////
////    teams.away_name_abbrev = xmlDocument.child("game").attribute("away_name_abbrev").as_string();
////    teams.away_code = xmlDocument.child("game").attribute("away_code").as_string();
////    teams.away_team_id = xmlDocument.child("game").attribute("away_team_id").as_int();
////    teams.away_team_city = xmlDocument.child("game").attribute("away_team_city").as_string();
////    teams.away_team_name = xmlDocument.child("game").attribute("away_team_name").as_string();
////    teams.home_name_abbrev = xmlDocument.child("game").attribute("home_name_abbrev").as_string();
////    teams.home_code = xmlDocument.child("game").attribute("home_code").as_string();
////    teams.home_team_id = xmlDocument.child("game").attribute("home_team_id").as_int();
////    teams.home_team_city = xmlDocument.child("game").attribute("home_team_city").as_string();
////    teams.home_team_name = xmlDocument.child("game").attribute("home_team_name").as_string();
////
////    return teams;
////}
//
///**
//*/
//bool mlb::io::CGamedayServer::Update()
//{
//    if (!LoadLastUpdateTime())
//    {
//        LOG_ERROR();
//
//        return false;
//    }
//
//    if (!UpdateYearlyDirectoryArray())
//    {
//        LOG_ERROR();
//
//        return false;
//    }
//
//    if (!UpdateMonthlyDirectoryArray())
//    {
//        LOG_ERROR();
//
//        return false;
//    }
//
//    if (!UpdateDailyDirectoryArray())
//    {
//        LOG_ERROR();
//
//        return false;
//    }
//
//    if (!UpdateGameDirectoryArray())
//    {
//        LOG_ERROR();
//
//        return false;
//    }
//    
//    if (!UpdateGamePrimaryKeyArray())
//    {
//        LOG_ERROR();
//
//        return false;
//    }
//
//    if (!UpdateStatCastGameUrlArray())
//    {
//        LOG_ERROR();
//
//        return false;
//    }
//
//    if (!UpdateGameArray())
//    {
//        LOG_ERROR();
//
//        return false;
//    }
//
//    if (!UpdateLastUpdateTime())
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
std::string mlb::io::CGamedayServer::GetLeague(const std::string& mlbGameString) const
{
	std::vector<std::string> leagueArray;
	std::string url,
		dataDir,
		league;
	std::string xmlResponse;

	// http://gd2.mlb.com/components/game/
	leagueArray.push_back("mlb/");
	leagueArray.push_back("win/");
	leagueArray.push_back("aaa/");
	leagueArray.push_back("aax/");
	leagueArray.push_back("afa/");
	leagueArray.push_back("afx/");
	leagueArray.push_back("asx/");
	leagueArray.push_back("bbc/");
	leagueArray.push_back("fps/");
	leagueArray.push_back("hsb/");
	leagueArray.push_back("ind/");
	leagueArray.push_back("int/");
	leagueArray.push_back("jml/");
	leagueArray.push_back("milb/");
	leagueArray.push_back("min/");
	leagueArray.push_back("nae/");
	leagueArray.push_back("naf/");
	leagueArray.push_back("nas/");
	leagueArray.push_back("nat/");
	leagueArray.push_back("naw/");
	leagueArray.push_back("oly/");
	leagueArray.push_back("rok/");

	dataDir = GetGameDataDirectory(mlbGameString);

	for (std::vector<std::string>::const_iterator leagueIterator = leagueArray.begin(); leagueIterator != leagueArray.end(); ++leagueIterator)
	{
		url = m_serverUrl + (*leagueIterator) + dataDir;

        my::file::GetUrlAsString(url, xmlResponse);

		if (xmlResponse.find("404 Not Found") == std::string::npos)
		{
			league = (*leagueIterator);

			break;
		}
	}

	if (league.empty())
		LOG_MESSAGE("Failed to locate the game at Gameday Server");

	return league;
}

/**
*/
std::string mlb::io::CGamedayServer::GetGameDataDirectory(const std::string& mlbGameString) const
{
	std::string::size_type tokenBegin,
		tokenEnd;
	std::string emptyString,
		yearString,
		monthString,
		dayString,
		gameDataDirectory;

	HEALTH_CHECK(mlbGameString.size() != 26, emptyString);

	tokenEnd = mlbGameString.find("_");

	HEALTH_CHECK(tokenEnd == std::string::npos, emptyString);

	yearString = mlbGameString.substr(0, tokenEnd);

	tokenBegin = tokenEnd + 1;
	tokenEnd = mlbGameString.find("_", tokenBegin);

	HEALTH_CHECK(tokenEnd == std::string::npos, emptyString);

	monthString = mlbGameString.substr(tokenBegin, tokenEnd - tokenBegin);

	tokenBegin = tokenEnd + 1;
	tokenEnd = mlbGameString.find("_", tokenBegin);

	HEALTH_CHECK(tokenEnd == std::string::npos, emptyString);

	dayString = mlbGameString.substr(tokenBegin, tokenEnd - tokenBegin);

	// "2011_08_03_tormlb_tbamlb_1" -> "year_2011/month_08/day_03/gid_2011_08_03_tormlb_tbamlb_1/"
	gameDataDirectory = "year_" + yearString
		+ "/month_" + monthString
		+ "/day_" + dayString
		+ "/gid_" + mlbGameString + "/";

	return gameDataDirectory;
}

//// Local helper
//my::CTimestamp GetTimestampFromYearlyUrl(std::string yearlyUrl)
//{
//    std::string yearPrefix = "year_";
//
//    std::string::size_type yearPosition = yearlyUrl.find(yearPrefix);
//
//    HEALTH_CHECK(yearPosition == std::string::npos, my::Null<my::CTimestamp>());
//
//    // "http://gd2.mlb.com/components/game/mlb/year_1970/"
//    std::string yearString = yearlyUrl.substr(yearPosition + yearPrefix.size(), 4);
//
//    HEALTH_CHECK(yearString.empty(), my::Null<my::CTimestamp>());
//
//    return my::CTimestamp(atoi(yearString.c_str()), 1, 1);
//}
//
///**
//*/
//bool mlb::io::CGamedayServer::UpdateYearlyDirectoryArray()
//{
//    std::string yearUrl,
//        yearResponse,
//        yearPrefix,
//        yearToken,
//        yearSuffix;
//    std::string::size_type stringPosition = 0;
//    std::string yearUrlString;
//    std::vector<std::string> yearlyDirectoryArray;
//
//    std::cout << "Updating gameday yearly data..." << std::endl;
//
//    m_yearlyDirectoryArray.clear();
//    m_yearlyDirectoryToUpdateArray.clear();
//
//    if (!LoadDataUrlContainer("./data/tables/gameday_yearly_data_urls.txt", m_yearlyDirectoryArray))
//    {
//        LOG_ERROR();
//
//        return false;
//    }
//
//    // TRICKY: This will not change till 2017.
//    CUnixEpoch currentUnixEpoch;
//    currentUnixEpoch.Now();
//
//    if (currentUnixEpoch.GetYear() > 2016)
//    {
//        yearUrl = "http://gd2.mlb.com/components/game/mlb/";
//        yearPrefix = "href=\"";
//        yearToken = "year";
//        yearSuffix = "\">";
//
//        my::file::GetUrlAsString(yearUrl, yearResponse);
//
//        if (yearResponse.empty())
//        {
//            LOG_ERROR();
//
//            return false;
//        }
//
//        while (!yearResponse.empty())
//        {
//            stringPosition = yearResponse.find(yearPrefix);
//
//            if (stringPosition != std::string::npos)
//            {
//                yearResponse = yearResponse.substr(stringPosition + yearPrefix.size());
//
//                // year_2004/
//                if ((yearResponse.size() >= 10) &&
//                    (yearResponse.substr(0, yearToken.size()) == yearToken) &&
//                    (atoi(yearResponse.substr(yearToken.size() + 1, 4).c_str()) != 0) &&
//                    (yearResponse.find(yearSuffix) == 10))
//                {
//                    yearUrlString = yearResponse.substr(0, 10);
//
//                    yearlyDirectoryArray.push_back(yearUrl + yearUrlString);
//                }
//            }
//            else
//                yearResponse.clear();
//        }
//    }
//    else
//        yearlyDirectoryArray = m_yearlyDirectoryArray;
//    
//    my::CTimestamp lastTimestamp(m_lastUpdateTime),
//        currentTimestamp(my::CTimestamp::Now());
//
//    for (std::vector<std::string>::const_iterator yearlyDirectoryIterator = yearlyDirectoryArray.begin();
//        yearlyDirectoryIterator != yearlyDirectoryArray.end(); ++yearlyDirectoryIterator)
//    {
//        my::CTimestamp yearlyDirectoryTimestamp = GetTimestampFromYearlyUrl(*yearlyDirectoryIterator);
//
//        // Add the data to be updated
//        if ((yearlyDirectoryTimestamp.GetYear() >= lastTimestamp.GetYear()) &&
//            (yearlyDirectoryTimestamp.GetYear() <= currentTimestamp.GetYear()))
//        {
//            m_yearlyDirectoryToUpdateArray.push_back(*yearlyDirectoryIterator);
//        }
//
//        if (std::find(m_yearlyDirectoryArray.begin(), m_yearlyDirectoryArray.end(), *yearlyDirectoryIterator) == m_yearlyDirectoryArray.end())
//        {
//            m_yearlyDirectoryArray.push_back(*yearlyDirectoryIterator);
//        }
//    }
//
//    std::sort(m_yearlyDirectoryArray.begin(), m_yearlyDirectoryArray.end());
//
//    SaveDataUrlContainer("./data/tables/gameday_yearly_data_urls.txt", m_yearlyDirectoryArray);
//
//    return true;
//}
//
//// Local helper
//my::CTimestamp GetTimestampFromMonthlyUrl(std::string monthlyUrl)
//{
//    std::string yearPrefix = "year_",
//        monthPrefix = "month_";
//
//    std::string::size_type yearPosition = monthlyUrl.find(yearPrefix),
//        monthPosition = monthlyUrl.find(monthPrefix);
//
//    HEALTH_CHECK(yearPosition == std::string::npos, my::Null<my::CTimestamp>());
//    HEALTH_CHECK(monthPosition == std::string::npos, my::Null<my::CTimestamp>());
//
//    // "http://gd2.mlb.com/components/game/mlb/year_1969/month_09/"
//    std::string yearString = monthlyUrl.substr(yearPosition + yearPrefix.size(), 4),
//        monthString = monthlyUrl.substr(monthPosition + monthPrefix.size(), 2);
//
//    HEALTH_CHECK(yearString.empty(), my::Null<my::CTimestamp>());
//    HEALTH_CHECK(monthString.empty(), my::Null<my::CTimestamp>());
//
//    return my::CTimestamp(atoi(yearString.c_str()), atoi(monthString.c_str()), 1);
//}
//
///**
//*/
//bool mlb::io::CGamedayServer::UpdateMonthlyDirectoryArray()
//{
//    std::string monthResponse,
//        monthPrefix,
//        monthToken,
//        monthSuffix;
//    std::string::size_type stringPosition = 0;
//    std::string monthUrlString;
//    std::vector<std::string> monthlyDirectoryArray;
//
//    std::cout << "Updating gameday monthly data..." << std::endl;
//
//    m_monthlyDirectoryArray.clear();
//    m_monthlyDirectoryToUpdateArray.clear();
//
//    if (!LoadDataUrlContainer("./data/tables/gameday_monthly_data_urls.txt", m_monthlyDirectoryArray))
//    {
//        LOG_ERROR();
//
//        return false;
//    }
//
//    monthPrefix = "href=\"";
//    monthToken = "month";
//    monthSuffix = "\">";
//
//    // Update instead of rebuild.
//    for (std::vector<std::string>::const_iterator yearUrlStringIterator = m_yearlyDirectoryToUpdateArray.begin(); yearUrlStringIterator != m_yearlyDirectoryToUpdateArray.end(); ++yearUrlStringIterator)
//    {
//        my::file::GetUrlAsString(*yearUrlStringIterator, monthResponse);
//
//        if (monthResponse.empty())
//        {
//            LOG_ERROR();
//
//            return false;
//        }
//
//        while (!monthResponse.empty())
//        {
//            stringPosition = monthResponse.find(monthPrefix);
//
//            if (stringPosition != std::string::npos)
//            {
//                monthResponse = monthResponse.substr(stringPosition + monthPrefix.size());
//
//                // month_01/
//                if ((monthResponse.size() >= 9) &&
//                    (monthResponse.substr(0, monthToken.size()) == monthToken) &&
//                    (atoi(monthResponse.substr(monthToken.size() + 1, 2).c_str()) != 0) &&
//                    (monthResponse.find(monthSuffix) == 9))
//                {
//                    monthUrlString = monthResponse.substr(0, 9);
//
//                    monthlyDirectoryArray.push_back((*yearUrlStringIterator) + monthUrlString);
//                }
//            }
//            else
//                monthResponse.clear();
//        }
//    }
//
//    my::CTimestamp lastTimestamp(m_lastUpdateTime);
//
//    for (std::vector<std::string>::const_iterator monthlyDirectoryIterator = monthlyDirectoryArray.begin();
//        monthlyDirectoryIterator != monthlyDirectoryArray.end(); ++monthlyDirectoryIterator)
//    {
//        my::CTimestamp monthlyDirectoryTimestamp = GetTimestampFromMonthlyUrl(*monthlyDirectoryIterator);
//
//        // Add the data to be updated
//        if ((monthlyDirectoryTimestamp.GetYear() >= lastTimestamp.GetYear()) &&
//            (monthlyDirectoryTimestamp.GetMonth() >= lastTimestamp.GetMonth()))
//        {
//            m_monthlyDirectoryToUpdateArray.push_back(*monthlyDirectoryIterator);
//        }
//
//        if (std::find(m_monthlyDirectoryArray.begin(), m_monthlyDirectoryArray.end(), *monthlyDirectoryIterator) == m_monthlyDirectoryArray.end())
//        {
//            m_monthlyDirectoryArray.push_back(*monthlyDirectoryIterator);
//        }
//    }
//
//    std::sort(m_monthlyDirectoryArray.begin(), m_monthlyDirectoryArray.end());
//
//    SaveDataUrlContainer("./data/tables/gameday_monthly_data_urls.txt", m_monthlyDirectoryArray);
//
//    return true;
//}
//
///**
//*/
//bool mlb::io::CGamedayServer::UpdateDailyDirectoryArray()
//{
//    std::string dayResponse,
//        dayPrefix,
//        dayToken,
//        daySuffix;
//    std::string::size_type stringPosition = 0;
//    std::string dayUrlString;
//    std::vector<std::string> dailyDirectoryArray;
//
//    std::cout << "Updating gameday daily data..." << std::endl;
//
//    m_dailyDirectoryArray.clear();
//    m_dailyDirectoryToUpdateArray.clear();
//
//    if (!LoadDataUrlContainer("./data/tables/gameday_daily_data_urls.txt", m_dailyDirectoryArray))
//    {
//        LOG_ERROR();
//
//        return false;
//    }
//
//    dayPrefix = "href=\"";
//    dayToken = "day";
//    daySuffix = "\">";
//
//    // Update instead of rebuild.
//    //for (std::vector<std::string>::const_iterator monthUrlStringIterator=m_monthlyDirectoryArray.begin(); monthUrlStringIterator!=m_monthlyDirectoryArray.end(); ++monthUrlStringIterator)
//    for (std::vector<std::string>::const_iterator monthUrlStringIterator=m_monthlyDirectoryToUpdateArray.begin(); monthUrlStringIterator!=m_monthlyDirectoryToUpdateArray.end(); ++monthUrlStringIterator)
//    {
//        my::file::GetUrlAsString(*monthUrlStringIterator, dayResponse);
//
//        if (dayResponse.empty())
//        {
//            LOG_ERROR();
//
//            return false;
//        }
//
//        while (!dayResponse.empty())
//        {
//            stringPosition = dayResponse.find(dayPrefix);
//
//            if (stringPosition != std::string::npos)
//            {
//                dayResponse = dayResponse.substr(stringPosition + dayPrefix.size());
//
//                // day_06/
//                if ((dayResponse.size() >= 7) &&
//                    (dayResponse.substr(0, dayToken.size()) == dayToken) &&
//                    (atoi(dayResponse.substr(dayToken.size() + 1, 2).c_str()) != 0) &&
//                    (dayResponse.find(daySuffix) == 7))
//                {
//                    dayUrlString = dayResponse.substr(0, 7);
//
//                    dailyDirectoryArray.push_back((*monthUrlStringIterator) + dayUrlString);
//                }
//            }
//            else
//                dayResponse.clear();
//        }
//    }
//
//    // Rebuild (?).
//    for (std::vector<std::string>::const_iterator dailyDirectoryIterator = dailyDirectoryArray.begin(); dailyDirectoryIterator != dailyDirectoryArray.end(); ++dailyDirectoryIterator)
//    {
//        // Add the data to be updated
//        m_dailyDirectoryToUpdateArray.push_back(*dailyDirectoryIterator);
//
//        // DEBUG ONLY! (13-Apr-2015)
//        std::cout << (*dailyDirectoryIterator) << std::endl;
//
//        if (std::find(m_dailyDirectoryArray.begin(), m_dailyDirectoryArray.end(), *dailyDirectoryIterator) == m_dailyDirectoryArray.end())
//            m_dailyDirectoryArray.push_back(*dailyDirectoryIterator);
//    }
//
//    std::sort(m_dailyDirectoryArray.begin(), m_dailyDirectoryArray.end());
//
//    SaveDataUrlContainer("./data/tables/gameday_daily_data_urls.txt", m_dailyDirectoryArray);
//
//    return true;
//}
//
///**
//*/
//bool mlb::io::CGamedayServer::UpdateGameDirectoryArray()
//{
//    std::string gameResponse,
//        gamePrefix,
//        gameToken,
//        gameSuffix;
//    std::string::size_type stringPosition = 0;
//    std::string gameUrlString;
//    std::vector<std::string> gameDirectoryArray;
//
//    std::cout << "Updating gameday game data..." << std::endl;
//
//    m_gameDirectoryArray.clear();
//    m_gameDirectoryToUpdateArray.clear();
//
//    if (!LoadDataUrlContainer("./data/tables/gameday_game_data_urls.txt", m_gameDirectoryArray))
//    {
//        LOG_ERROR();
//
//        return false;
//    }
//
//    gamePrefix = "href=\"";
//    gameToken = "gid_";
//    gameSuffix = "\">";
//
//    // Update instead of rebuild.
//    for (std::vector<std::string>::const_iterator dailyDirectoryIterator = m_dailyDirectoryToUpdateArray.begin(); dailyDirectoryIterator != m_dailyDirectoryToUpdateArray.end(); ++dailyDirectoryIterator)
//    {
//        my::file::GetUrlAsString(*dailyDirectoryIterator, gameResponse);
//
//        if (gameResponse.empty())
//        {
//            LOG_ERROR();
//
//            return false;
//        }
//
//        while (!gameResponse.empty())
//        {
//            stringPosition = gameResponse.find(gamePrefix);
//
//            if (stringPosition != std::string::npos)
//            {
//                gameResponse = gameResponse.substr(stringPosition + gamePrefix.size());
//
//                // gid_2011_08_16_arimlb_phimlb_1/
//                if ((gameResponse.size() >= 32) &&
//                    (gameResponse.substr(0, gameToken.size()) == gameToken) &&
//                    (atoi(gameResponse.substr(gameToken.size() + 1, 2).c_str()) != 0) &&
//                    (gameResponse.find(gameSuffix) == 31))
//                {
//                    gameUrlString = gameResponse.substr(0, 31);
//
//                    gameDirectoryArray.push_back((*dailyDirectoryIterator) + gameUrlString);
//                }
//            }
//            else
//                gameResponse.clear();
//        }
//    }
//
//    // Rebuild (?).
//    for (std::vector<std::string>::const_iterator gameDirectoryIterator = gameDirectoryArray.begin(); gameDirectoryIterator != gameDirectoryArray.end(); ++gameDirectoryIterator)
//    {
//        // Add the data to be updated
//        m_gameDirectoryToUpdateArray.push_back(*gameDirectoryIterator);
//
//        // DEBUG ONLY! (13-Apr-2015)
//        std::cout << (*gameDirectoryIterator) << std::endl;
//
//        if (std::find(m_gameDirectoryArray.begin(), m_gameDirectoryArray.end(), *gameDirectoryIterator) == m_gameDirectoryArray.end())
//            m_gameDirectoryArray.push_back(*gameDirectoryIterator);
//    }
//
//    std::sort(m_gameDirectoryArray.begin(), m_gameDirectoryArray.end());
//
//    SaveDataUrlContainer("./data/tables/gameday_game_data_urls.txt", m_gameDirectoryArray);
//
//    return true;
//}
//
///**
//*/
//bool mlb::io::CGamedayServer::UpdateGamePrimaryKeyArray()
//{
//    m_gameDirectoryToPrimaryKeyMap.clear();
//    m_gameDirectoryToPrimaryKeyToUpdateMap.clear();
//
//    if (!LoadDataUrlContainer("./data/tables/gameday_game_primary_keys.txt", m_gameDirectoryToPrimaryKeyMap))
//    {
//        LOG_ERROR();
//
//        return false;
//    }
//
//    for (std::vector<std::string>::const_iterator gameDirectoryToUpdateIterator = m_gameDirectoryToUpdateArray.begin(); gameDirectoryToUpdateIterator != m_gameDirectoryToUpdateArray.end(); ++gameDirectoryToUpdateIterator)
//    {
//        // Update all games
//        //*if (m_gameDirectoryToPrimaryKeyMap.find(*gameDirectoryToUpdateIterator) == m_gameDirectoryToPrimaryKeyMap.end())
//        {
//            std::string fileBuffer;
//            
//            my::file::GetUrlAsString((*gameDirectoryToUpdateIterator) + "linescore.json", fileBuffer);
//
//            rapidjson::Document document;
//            document.Parse<0>(fileBuffer.c_str());
//
//            if (document.IsObject() &&
//                document.HasMember("Data"))
//            {
//                const rapidjson::Value& dataHandle = document["Data"];
//
//                if (dataHandle.HasMember("boxscore"))
//                {
//                    const rapidjson::Value& boxscoreHandle = dataHandle["boxscore"];
//
//                    if (boxscoreHandle.HasMember("game_pk"))
//                    {
//                        const rapidjson::Value& gamePkHandle = boxscoreHandle["game_pk"];
//
//                        INT32 gamePk = 0;
//
//                        if (gamePkHandle.IsInt())
//                            gamePk = gamePkHandle.GetInt();
//                        else if (gamePkHandle.IsString())
//                            gamePk = atoi(gamePkHandle.GetString());
//                        
//                        if (gamePk != 0)
//                        {
//                            m_gameDirectoryToPrimaryKeyMap[*gameDirectoryToUpdateIterator] = gamePk;
//                            
//                            m_gameDirectoryToPrimaryKeyToUpdateMap[*gameDirectoryToUpdateIterator] = gamePk;
//
//                            // DEBUG ONLY! (13-Apr-2015)
//                            std::cout << gamePk << std::endl;
//                        }
//                    }
//                }
//            }
//
//            if (document.IsObject() &&
//                document.HasMember("data"))
//            {
//                const rapidjson::Value& dataHandle = document["data"];
//
//                if (dataHandle.HasMember("game"))
//                {
//                    const rapidjson::Value& gameHandle = dataHandle["game"];
//
//                    if (gameHandle.HasMember("game_pk"))
//                    {
//                        const rapidjson::Value& gamePkHandle = gameHandle["game_pk"];
//
//                        INT32 gamePk = 0;
//
//                        if (gamePkHandle.IsInt())
//                            gamePk = gamePkHandle.GetInt();
//                        else if (gamePkHandle.IsString())
//                            gamePk = atoi(gamePkHandle.GetString());
//
//                        if (gamePk != 0)
//                        {
//                            m_gameDirectoryToPrimaryKeyMap[*gameDirectoryToUpdateIterator] = gamePk;
//
//                            m_gameDirectoryToPrimaryKeyToUpdateMap[*gameDirectoryToUpdateIterator] = gamePk;
//
//                            // DEBUG ONLY! (13-Apr-2015)
//                            std::cout << gamePk << std::endl;
//                        }
//                    }
//                }
//            }
//
//            if (m_gameDirectoryToPrimaryKeyMap.find(*gameDirectoryToUpdateIterator) == m_gameDirectoryToPrimaryKeyMap.end())
//                LOG_MESSAGE("Failed to read " + (*gameDirectoryToUpdateIterator));
//        }
//    }
//
//    SaveDataUrlContainer("./data/tables/gameday_game_primary_keys.txt", m_gameDirectoryToPrimaryKeyMap);
//
//    return true;
//}
//
///**
//*/
//bool mlb::io::CGamedayServer::UpdateStatCastGameUrlArray()
//{
//    m_gameDirectoryToStatCastGameUrlMap.clear();
//
//    if (!LoadDataUrlContainer("./data/tables/gameday_statcast_game_data_urls.txt", m_gameDirectoryToStatCastGameUrlMap))
//    {
//        LOG_ERROR();
//
//        return false;
//    }
//
//    // Update instead of rebuilt.
//    for (std::map<std::string, INT32>::const_iterator gameDirectoryToPrimaryKeyIterator = m_gameDirectoryToPrimaryKeyToUpdateMap.begin(); gameDirectoryToPrimaryKeyIterator != m_gameDirectoryToPrimaryKeyToUpdateMap.end(); ++gameDirectoryToPrimaryKeyIterator)
//    {
//        if (m_gameDirectoryToStatCastGameUrlMap.find(gameDirectoryToPrimaryKeyIterator->first) == m_gameDirectoryToStatCastGameUrlMap.end())
//        {
//            std::string statCastGameDirectory = "https://statsapi.mlb.com/api/v1/game/" 
//                + boost::lexical_cast<std::string>(gameDirectoryToPrimaryKeyIterator->second)
//                + "/guids";
//
//            std::string fileBuffer;
//            
//            my::file::GetUrlAsString(statCastGameDirectory, fileBuffer, "brazilians", "garrincha62");
//
//            rapidjson::Document document;
//            document.Parse<0>(fileBuffer.c_str());
//
//            if (document.IsArray() &&
//                (document.Size() > 0))
//            {
//                const rapidjson::Value& playIterator = document[0];
//
//                if (playIterator.IsObject() &&
//                    playIterator.HasMember("rawFile"))
//                {
//                    m_gameDirectoryToStatCastGameUrlMap[gameDirectoryToPrimaryKeyIterator->first] = statCastGameDirectory;
//
//                    // DEBUG ONLY! (13-Apr-2015)
//                    std::cout << statCastGameDirectory << std::endl;
//                }
//            }
//        }
//    }
//
//    SaveDataUrlContainer("./data/tables/gameday_statcast_game_data_urls.txt", m_gameDirectoryToStatCastGameUrlMap);
//
//    return true;
//}
//
///**
//*/
//bool mlb::io::CGamedayServer::UpdateGameArray()
//{
//    m_gameArray.clear();
//
//    for (std::map<std::string, std::string>::const_iterator gameDirectoryToStatCastGameUrlIterator = m_gameDirectoryToStatCastGameUrlMap.begin(); gameDirectoryToStatCastGameUrlIterator != m_gameDirectoryToStatCastGameUrlMap.end(); ++gameDirectoryToStatCastGameUrlIterator)
//    {
//        mlb::io::CGame game;
//
//        HEALTH_CHECK(m_gameDirectoryToPrimaryKeyMap.find(gameDirectoryToStatCastGameUrlIterator->first) == m_gameDirectoryToPrimaryKeyMap.end(), false);
//
//        game.SetGamePrimaryKey(m_gameDirectoryToPrimaryKeyMap[gameDirectoryToStatCastGameUrlIterator->first]);
//        game.SetGamedayUrl(gameDirectoryToStatCastGameUrlIterator->first);
//        game.SetStatCastUrl(gameDirectoryToStatCastGameUrlIterator->second);
//        
//        // LINESCORE
//
//        std::string fileBuffer;
//        
//        my::file::GetUrlAsString(game.GetGamedayUrl() + "linescore.json", fileBuffer);
//
//        rapidjson::Document document;
//        document.Parse<0>(fileBuffer.c_str());
//
//        if (document.IsObject() &&
//            document.HasMember("data"))
//        {
//            const rapidjson::Value& dataHandle = document["data"];
//
//            if (dataHandle.HasMember("game"))
//            {
//                const rapidjson::Value& gameHandle = dataHandle["game"];
//
//                // time_date
//                if (gameHandle.HasMember("time_date"))
//                {
//                    const rapidjson::Value& timeDateHandle = gameHandle["time_date"];
//
//                    // 2015/04/10 7:05
//                    if (timeDateHandle.IsString())
//                        game.SetTime(CUnixEpoch(timeDateHandle.GetString(), "%Y/%m/%d").Get());
//                }
//
//                // home_team_id
//                if (gameHandle.HasMember("home_team_id"))
//                {
//                    const rapidjson::Value& homeTeamIdHandle = gameHandle["home_team_id"];
//
//                    if (homeTeamIdHandle.IsInt())
//                        game.SetHomeTeamId(homeTeamIdHandle.GetInt());
//                    else if (homeTeamIdHandle.IsString())
//                        game.SetHomeTeamId(atoi(homeTeamIdHandle.GetString()));
//                }
//
//                // home_name_abbrev
//                if (gameHandle.HasMember("home_name_abbrev"))
//                {
//                    const rapidjson::Value& homeNameAbbrevHandle = gameHandle["home_name_abbrev"];
//
//                    // 2015/04/10 7:05
//                    if (homeNameAbbrevHandle.IsString())
//                        game.SetHomeTeamNameAbbreviation(homeNameAbbrevHandle.GetString());
//                }
//
//                // home_team_runs
//                if (gameHandle.HasMember("home_team_runs"))
//                {
//                    const rapidjson::Value& homeTeamRunsHandle = gameHandle["home_team_runs"];
//
//                    if (homeTeamRunsHandle.IsInt())
//                        game.SetHomeTeamRuns(homeTeamRunsHandle.GetInt());
//                    else if (homeTeamRunsHandle.IsString())
//                        game.SetHomeTeamRuns(atoi(homeTeamRunsHandle.GetString()));
//                }
//
//                // away_team_id
//                if (gameHandle.HasMember("away_team_id"))
//                {
//                    const rapidjson::Value& awayTeamIdHandle = gameHandle["away_team_id"];
//
//                    if (awayTeamIdHandle.IsInt())
//                        game.SetAwayTeamId(awayTeamIdHandle.GetInt());
//                    else if (awayTeamIdHandle.IsString())
//                        game.SetAwayTeamId(atoi(awayTeamIdHandle.GetString()));
//                }
//
//                // away_name_abbrev
//                if (gameHandle.HasMember("away_name_abbrev"))
//                {
//                    const rapidjson::Value& awayNameAbbrevHandle = gameHandle["away_name_abbrev"];
//
//                    // 2015/04/10 7:05
//                    if (awayNameAbbrevHandle.IsString())
//                        game.SetAwayTeamNameAbbreviation(awayNameAbbrevHandle.GetString());
//                }
//
//                // away_team_runs
//                if (gameHandle.HasMember("away_team_runs"))
//                {
//                    const rapidjson::Value& awayTeamRunsHandle = gameHandle["away_team_runs"];
//
//                    if (awayTeamRunsHandle.IsInt())
//                        game.SetAwayTeamRuns(awayTeamRunsHandle.GetInt());
//                    else if (awayTeamRunsHandle.IsString())
//                        game.SetAwayTeamRuns(atoi(awayTeamRunsHandle.GetString()));
//                }
//            }
//
//            // DEBUG ONLY! (13-Apr-2015)
//            std::cout << game.GetHomeTeamNameAbbreviation() << " " << game.GetAwayTeamNameAbbreviation() << std::endl;
//        }
//
//        m_gameArray.push_back(game);
//    }
//
//    // JSON FILE
//
//    std::string jsonString;
//
//    // Brute-force JSON packing.
//    jsonString += "[";
//
//    for (std::vector<mlb::io::CGame>::const_iterator gameIterator = m_gameArray.begin(); gameIterator != m_gameArray.end(); ++gameIterator)
//        jsonString += gameIterator->ToJson() + ",";
//
//    if (jsonString.back() == ',')
//        jsonString.pop_back();
//
//    jsonString += "]";
//
//    std::ofstream fileStream;
//
//    fileStream.open("./data/tables/game_index.json");
//
//    if (!fileStream.is_open())
//    {
//        LOG_ERROR();
//
//        return false;
//    }
//
//    fileStream << jsonString;
//
//    return true;
//}
//
///**
//*/
//bool mlb::io::CGamedayServer::LoadLastUpdateTime()
//{
//    std::ifstream fileStream;
//
//    fileStream.open("./data/tables/gameday_last_update.txt");
//
//    if (!fileStream.is_open())
//        return true;
//
//    fileStream >> m_lastUpdateTime;
//
//    return true;
//}
//
///**
//*/
//bool mlb::io::CGamedayServer::UpdateLastUpdateTime()
//{
//    std::ofstream fileStream;
//
//    fileStream.open("./data/tables/gameday_last_update.txt");
//
//    if (!fileStream.is_open())
//    {
//        LOG_ERROR();
//
//        return false;
//    }
//
//    fileStream << CUnixEpoch(boost::posix_time::microsec_clock::universal_time()).Get();
//
//    return true;
//}

/**
*/
void mlb::io::CGamedayServer::Create()
{
	m_serverUrl = "http://gd2.mlb.com/components/game/";

	//m_yearlyDirectoryArray.clear();
 //   m_yearlyDirectoryToUpdateArray.clear();

 //   m_monthlyDirectoryArray.clear();
 //   m_monthlyDirectoryToUpdateArray.clear();

 //   m_dailyDirectoryArray.clear();
 //   m_dailyDirectoryToUpdateArray.clear();

 //   m_gameDirectoryArray.clear();
 //   m_gameDirectoryToUpdateArray.clear();

 //   m_gameDirectoryToPrimaryKeyMap.clear();
 //   m_gameDirectoryToPrimaryKeyToUpdateMap.clear();

 //   m_gameDirectoryToStatCastGameUrlMap.clear();

 //   m_gameArray.clear();

 //   m_venueArray.clear();

 //   m_lastUpdateTime = mlb::Null<INT64>();

 //   m_teamArray.clear();
}

