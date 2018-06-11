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

#include "Common.h"
#include "Logger.h"

#include "CommandLineArguments.h"

my::CCommandLineArguments::CCommandLineArguments()
{
    Create();
}

my::CCommandLineArguments::CCommandLineArguments(const CCommandLineArguments& commandLineArguments)
{
    Copy(commandLineArguments);
}

my::CCommandLineArguments::~CCommandLineArguments()
{
    for (std::vector<std::string>::const_iterator unknownArgumentIterator = m_unknownArgumentArray.begin(); unknownArgumentIterator != m_unknownArgumentArray.end(); ++unknownArgumentIterator)
    {
        LOG_MESSAGE("An unknown command line argument named '" + (*unknownArgumentIterator) + "' was found.");
    }
}

void my::CCommandLineArguments::operator = (const CCommandLineArguments& commandLineArguments)
{
    Copy(commandLineArguments);
}

bool my::CCommandLineArguments::Initialize(int argumentCount, char **argumentArray)
{
    int argumentIndex = 1;

    while (argumentIndex < argumentCount)
    {
        std::string token = argumentArray[argumentIndex];

        std::map<std::string, std::string>::const_iterator aliasToKeyIterator = m_aliasToKeyMap.find(token);

        if (aliasToKeyIterator != m_aliasToKeyMap.end())
            token = aliasToKeyIterator->second;

        if (m_keyToValueArrayMap.find(token) != m_keyToValueArrayMap.end())
        {
            if (argumentIndex == (argumentCount - 1))
            {
                LOG_MESSAGE("No value found for the key '" + token + "'.");

                return false;
            }

            ++argumentIndex;

            if ((m_keyToValueArrayMap[token].size() == 1) &&
                (m_keyToValueArrayMap[token].front() == "null"))
            {
                m_keyToValueArrayMap[token].clear();
            }

            m_keyToValueArrayMap[token].push_back(argumentArray[argumentIndex]);
        }
        else if (m_flagToValueMap.find(token) != m_flagToValueMap.end())
        {
            m_flagToValueMap[token] = true;
        }
        else
        {
            m_unknownArgumentArray.push_back(token);
        }

        ++argumentIndex;
    }

    return true;
}

bool my::CCommandLineArguments::AddParameter(std::string key, std::string help)
{
    HEALTH_CHECK(key.empty(), false);
    HEALTH_CHECK(m_keyToValueArrayMap.find(key) != m_keyToValueArrayMap.end(), false);
    HEALTH_CHECK(m_flagToValueMap.find(key) != m_flagToValueMap.end(), false);

    m_keyToValueArrayMap[key].push_back("null");

    if (!help.empty())
        m_keyToHelpMap[key] = help;

    return true;
}

bool my::CCommandLineArguments::HasParameter(std::string key) const
{
    std::map<std::string, std::vector<std::string> >::const_iterator keyToValueArrayIterator = m_keyToValueArrayMap.find(key);

    if (keyToValueArrayIterator != m_keyToValueArrayMap.end())
        return keyToValueArrayIterator->second.front() != "null";

    return false;
}

int my::CCommandLineArguments::ParameterCount(std::string key) const
{
    std::map<std::string, std::vector<std::string> >::const_iterator keyToValueArrayIterator = m_keyToValueArrayMap.find(key);

    if (keyToValueArrayIterator != m_keyToValueArrayMap.end())
        return (int)keyToValueArrayIterator->second.size();

    return 0;
}

int my::CCommandLineArguments::ToInt32(std::string key, int index) const
{
    std::map<std::string, std::vector<std::string> >::const_iterator keyToValueArrayIterator = m_keyToValueArrayMap.find(key);

    int value = INT_MAX;

    if (keyToValueArrayIterator != m_keyToValueArrayMap.end())
    {
        try
        {
            if ((keyToValueArrayIterator->second.size() > index) &&
                (keyToValueArrayIterator->second[index] != "null"))
            {
                value = atoi(keyToValueArrayIterator->second[index].c_str());
            }
        }
        catch (...)
        {
            LOG_MESSAGE("Failed to convert the command line argument under the key '" + key + "'.");
        }
    }

    return value;
}

double my::CCommandLineArguments::ToDouble(std::string key, int index) const
{
    std::map<std::string, std::vector<std::string> >::const_iterator keyToValueArrayIterator = m_keyToValueArrayMap.find(key);

    double value = DBL_MAX;

    if (keyToValueArrayIterator != m_keyToValueArrayMap.end())
    {
        try
        {
            if ((keyToValueArrayIterator->second.size() > index) &&
                (keyToValueArrayIterator->second[index] != "null"))
            {
                value = atof(keyToValueArrayIterator->second[index].c_str());
            }
        }
        catch (...)
        {
            LOG_MESSAGE("Failed to convert the command line argument under the key '" + key + "'.");
        }
    }

    return value;
}

std::string my::CCommandLineArguments::ToString(std::string key, int index)
{
    std::map<std::string, std::vector<std::string> >::const_iterator keyToValueArrayIterator = m_keyToValueArrayMap.find(key);

    if (keyToValueArrayIterator != m_keyToValueArrayMap.end())
    {
        if (keyToValueArrayIterator->second.size() > index)
            return keyToValueArrayIterator->second[index];
    }

    return "null";
}

bool my::CCommandLineArguments::AddFlag(std::string key, std::string help)
{
    HEALTH_CHECK(key.empty(), false);
    HEALTH_CHECK(m_keyToValueArrayMap.find(key) != m_keyToValueArrayMap.end(), false);
    HEALTH_CHECK(m_flagToValueMap.find(key) != m_flagToValueMap.end(), false);

    m_flagToValueMap[key] = false;

    if (!help.empty())
        m_keyToHelpMap[key] = help;

    return true;
}

bool my::CCommandLineArguments::GetFlag(std::string key) const
{
    std::map<std::string, bool>::const_iterator flagToValueIterator = m_flagToValueMap.find(key);

    if (flagToValueIterator != m_flagToValueMap.end())
        return flagToValueIterator->second;

    return false;
}

bool my::CCommandLineArguments::AddAlias(std::string key, std::string alias)
{
    HEALTH_CHECK(key.empty(), false);
    HEALTH_CHECK(alias.empty(), false);

    m_aliasToKeyMap[alias] = key;

    return true;
}

void my::CCommandLineArguments::Print() const
{
    std::cerr << "COMMAND LINE SYNTAX:" << std::endl;

    if (!m_keyToValueArrayMap.empty())
    {
        std::cout << "PARAMETER(S)" << std::endl;

        for (std::map<std::string, std::vector<std::string> >::const_iterator keyToValueArrayIterator = m_keyToValueArrayMap.begin(); keyToValueArrayIterator != m_keyToValueArrayMap.end(); ++keyToValueArrayIterator)
        {
            std::cout << "   " << keyToValueArrayIterator->first << " <value>";

            std::map<std::string, std::string>::const_iterator keyToHelpIterator = m_keyToHelpMap.find(keyToValueArrayIterator->first);

            if (keyToHelpIterator != m_keyToHelpMap.end())
                std::cout << ": " << keyToHelpIterator->second;

            std::cout << std::endl;
        }
    }

    if (!m_flagToValueMap.empty())
    {
        std::cout << "FLAG(S)" << std::endl;

        for (std::map<std::string, bool>::const_iterator flagToValueIterator = m_flagToValueMap.begin(); flagToValueIterator != m_flagToValueMap.end(); ++flagToValueIterator)
        {
            std::cout << "   " << flagToValueIterator->first;

            std::map<std::string, std::string>::const_iterator keyToHelpIterator = m_keyToHelpMap.find(flagToValueIterator->first);

            if (keyToHelpIterator != m_keyToHelpMap.end())
                std::cout << ": " << keyToHelpIterator->second;

            std::cout << std::endl;
        }
    }

    std::cerr << std::endl;
}

void my::CCommandLineArguments::Create()
{
    m_keyToValueArrayMap.clear();
    m_flagToValueMap.clear();
    m_unknownArgumentArray.clear();
    m_keyToHelpMap.clear();
}

void my::CCommandLineArguments::Copy(const CCommandLineArguments& commandLineArguments)
{
    m_keyToValueArrayMap = commandLineArguments.m_keyToValueArrayMap;
    m_flagToValueMap = commandLineArguments.m_flagToValueMap;
    m_unknownArgumentArray = commandLineArguments.m_unknownArgumentArray;
    m_keyToHelpMap = commandLineArguments.m_keyToHelpMap;
}

