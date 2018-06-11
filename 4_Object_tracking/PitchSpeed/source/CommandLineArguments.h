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

#if !defined(COMMAND_LINE_ARGUMENTS_INCLUDED)
#define COMMAND_LINE_ARGUMENTS_INCLUDED

#include <string>
#include <map>
#include <vector>

namespace my
{
    class CCommandLineArguments
    {
    public:
        CCommandLineArguments();
        CCommandLineArguments(const CCommandLineArguments& commandLineArguments);

        ~CCommandLineArguments();

        void operator=(const CCommandLineArguments& commandLineArguments);

        bool Initialize(int argumentCount, char **argumentArray);

        bool AddParameter(std::string key, std::string help = std::string());
        bool HasParameter(std::string key) const;
        int ParameterCount(std::string key) const;

        int ToInt32(std::string key, int index = 0) const;
        double ToDouble(std::string key, int index = 0) const;
        std::string ToString(std::string key, int index = 0);

        bool AddFlag(std::string key, std::string help = std::string());
        bool GetFlag(std::string key) const;

        bool AddAlias(std::string key, std::string alias);

        void Print() const;

    private:
        void Create();
        void Copy(const CCommandLineArguments& commandLineArguments);

    protected:
        std::map<std::string, std::vector<std::string> > m_keyToValueArrayMap;

        std::map<std::string, bool> m_flagToValueMap;

        std::vector<std::string> m_unknownArgumentArray;

        std::map<std::string, std::string> m_keyToHelpMap;

        std::map<std::string, std::string> m_aliasToKeyMap;
    };
}; // my

#endif // #if !defined(COMMAND_LINE_ARGUMENTS_INCLUDED)

