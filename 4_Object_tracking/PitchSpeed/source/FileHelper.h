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

#ifndef FILE_HELPER_INCLUDED
#define FILE_HELPER_INCLUDED

#include <string>

#include <boost/shared_array.hpp>

#include "Common.h"

namespace my {
    namespace file {
        // \r\n for the DOS / Windows world
        // \r for the pre - OSX Mac world
        // \n for the Unix and Unix - like world
        long GetFileAsString(std::string fileName, std::string& fileAsString, bool removeNewLine = true, bool removeCarriageReturn = true);

        long GetFileAsString(std::string fileName, boost::shared_array<char>& fileAsString);

        // URL (BODY) TO STRING
        // timeout - the maximum time in seconds that you allow the transfer operation to take.
        // TRICKY: (11-Apr-2017) RETURNS RESPONSE ON urlAsString IN CASE OF ERROR
        bool GetUrlAsString(std::string url, std::string& urlAsString, std::string user = my::Null<std::string>(), std::string password = my::Null<std::string>(), my::int32 timeout = my::Null<my::int32>());

        // STRING (JSON) TO URL 
        bool PostJsonStringToUrl(std::string url, std::string message, std::string user = my::Null<std::string>(), std::string password = my::Null<std::string>());

        // FILE TO URL 
        // TRICKY: (14-Apr-2017) RETURNS RESPONSE IN CASE OF ERROR
        std::string PutToUrl(std::string url, std::string fileName, std::string user = my::Null<std::string>(), std::string password = my::Null<std::string>());

        // URL (BODY) TO LOCAL FILE
        bool GetUrlAsFile(std::string url, std::string fileName, bool binary = true, std::string user = my::Null<std::string>(), std::string password = my::Null<std::string>());

        // http://stackoverflow.com/questions/7053538/how-do-i-encode-a-string-to-base64-using-only-boost
        std::string DecodeBase64(const std::string& valueAsString);
        // http://stackoverflow.com/questions/7053538/how-do-i-encode-a-string-to-base64-using-only-boost
        std::string EncodeBase64(const std::string& valueAsString);
        // http://stackoverflow.com/questions/7053538/how-do-i-encode-a-string-to-base64-using-only-boost
        std::string EncodeBase64(const unsigned char *valueAsArray, my::int32 length);
    }
};

#endif // FILE_HELPER_INCLUDED

