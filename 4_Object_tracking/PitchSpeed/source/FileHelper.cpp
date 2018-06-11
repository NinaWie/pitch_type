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
#include <ostream>

#include <boost/scoped_array.hpp>
// http://stackoverflow.com/questions/7053538/how-do-i-encode-a-string-to-base64-using-only-boost
#include <boost/archive/iterators/binary_from_base64.hpp>
#include <boost/archive/iterators/base64_from_binary.hpp>
#include <boost/archive/iterators/transform_width.hpp>
#include <boost/algorithm/string.hpp>

// http://nadeausoftware.com/articles/2012/01/c_c_tip_how_use_compiler_predefined_macros_detect_operating_system
#if defined(__APPLE__) && defined(__MACH__)
#include <curl/curl.h>
#else //#if defined(__APPLE__) && defined(__MACH__)
#include <curl_easy.h>
#include <curl_exception.h>
#include <curl_ios.h>
#endif //#if defined(__APPLE__) && defined(__MACH__)

#include "Logger.h"

#include "FileHelper.h"

// \r\n for the DOS\Windows world
// \r for the pre - OSX Mac world
// \n for the Unix and Unix - like world
long my::file::GetFileAsString(std::string fileName, std::string& fileAsString, bool removeNewLine, bool removeCarriageReturn)
{
    boost::shared_array<char> fileCharArray;

    long charCount = GetFileAsString(fileName, fileCharArray);

    if (charCount == 0)
        return 0;

    if (removeNewLine ||
        removeCarriageReturn)
    {
        boost::scoped_array<char> formattedFileCharArray(new char[charCount]);

        HEALTH_CHECK(!formattedFileCharArray, 0);

        // TESTING: (19-Dec-2014) Remove \r and \n?
        char *charIterator = fileCharArray.get(),
            *formattedCharIterator = formattedFileCharArray.get();

        long charIndex = charCount,
            formattedCharCount = 0;

        while (charIndex--)
        {
            bool discard = false;

            if (removeCarriageReturn &&
                ((*charIterator) == '\r'))
            {
                discard = true;
            }

            if (removeNewLine &&
                ((*charIterator) == '\n'))
            {
                discard = true;
            }

            if (!discard)
            {
                (*formattedCharIterator) = (*charIterator);

                ++formattedCharIterator;
                ++formattedCharCount;
            }

            ++charIterator;
        }

        fileAsString.assign(formattedFileCharArray.get(), formattedFileCharArray.get() + formattedCharCount);

        charCount = formattedCharCount;
    }
    else
        fileAsString.assign(fileCharArray.get(), fileCharArray.get() + charCount);

    return charCount;
}

long my::file::GetFileAsString(std::string fileName, boost::shared_array<char>& fileAsString)
{
    HEALTH_CHECK(fileName.empty(), 0);

    FILE * fileStream = 0;

    PORTABLE_FOPEN(fileStream, fileName.c_str(), "rb", 0);

    // Compute file size.
    fseek(fileStream, 0, SEEK_END);
    long fileSize = ftell(fileStream);
    rewind(fileStream);

    // Allocate memory to contain the whole file.
    fileAsString.reset(new char[fileSize]);

    if (!fileAsString)
    {
        fclose(fileStream);

        return 0;
    }

    // Copy the file into the fileBuffer.
    size_t result = fread(fileAsString.get(), 1, fileSize, fileStream);

    if (result != fileSize) 
    {
        fclose(fileStream);

        return 0;
    }

    fclose(fileStream);

    return fileSize;
}

// BUG: (04-Nov-2016) GET APPROACH ADAPTED TO APPLE
// https://curl.haxx.se/libcurl/c/getinmemory.html
struct MemoryStruct {
    std::vector<char> memory;
    size_t size;
};

// https://curl.haxx.se/libcurl/c/getinmemory.html
static size_t WriteMemoryCallback(void *contents, size_t size, size_t nmemb, void *userp)
{
    size_t realsize = size * nmemb;
    struct MemoryStruct *mem = (struct MemoryStruct *)userp;

    try {
        mem->memory.resize(mem->size + realsize + 1, 0);
    }
    catch (...) {
        LOG_ERROR();

        return false;
    }

    memcpy(&(mem->memory[mem->size]), contents, realsize);
    mem->size += realsize;
    mem->memory[mem->size] = 0;

    return realsize;
}

// URL (BODY) TO STRING
// timeout - the maximum time in seconds that you allow the transfer operation to take.
// TRICKY: (11-Apr-2017) RETURNS RESPONSE ON urlAsString IN CASE OF ERROR
bool my::file::GetUrlAsString(std::string url, std::string& urlAsString, std::string user, std::string password, my::int32 timeout)
{
    HEALTH_CHECK(url.empty(), false);

    bool isOk = true;

    // BUG: (04-Nov-2016) GET APPROACH ADAPTED TO APPLE
    // https://curl.haxx.se/libcurl/c/getinmemory.html
    CURL *curl_handle = 0;

    struct MemoryStruct chunk;

    chunk.size = 0;

    curl_global_init(CURL_GLOBAL_ALL);

    curl_handle = curl_easy_init();

    curl_easy_setopt(curl_handle, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl_handle, CURLOPT_FOLLOWLOCATION, 1L);

    if (!my::IsNull(timeout))
        curl_easy_setopt(curl_handle, CURLOPT_TIMEOUT, timeout);

    curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
    curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, (void *)&chunk);

    if (url.find("https") != std::string::npos)
    {
        curl_easy_setopt(curl_handle, CURLOPT_SSL_VERIFYHOST, 2);
        // HTTPS
        curl_easy_setopt(curl_handle, CURLOPT_SSL_VERIFYPEER, false);
    }

    curl_easy_setopt(curl_handle, CURLOPT_USERAGENT, "Mozilla/4.0 (compatible; MSIE 5.01; Windows NT 5.0)");

    if (!my::IsNull(user) &&
        !my::IsNull(password))
    {
        curl_easy_setopt(curl_handle, CURLOPT_HTTPAUTH, (long)CURLAUTH_BASIC);
        curl_easy_setopt(curl_handle, CURLOPT_USERPWD, (user + ":" + password).c_str());
    }

    CURLcode urlCode = curl_easy_perform(curl_handle);

    if (urlCode != CURLE_OK)
    {
        LOG_MESSAGE(std::string("cURL GET failed for ") + url + ": " + curl_easy_strerror(urlCode));

        // TRICKY: (11-Apr-2017) RETURNS RESPONSE ON BODY IN CASE OF ERRORS
        urlAsString = curl_easy_strerror(urlCode);

        isOk = false;
    }
    // BUG: (21-Feb-2017) 
    else if (chunk.memory.empty())
    {
        urlAsString.clear();

        isOk = false;
    }
    else
    {
        // CHUNK.MEMORY POINTS TO A MEMORY BLOCK THAT IS CHUNK.SIZE BYTES BIG AND CONTAINS THE RESPONSE
        urlAsString = chunk.memory.data();
    }

    curl_easy_cleanup(curl_handle);

    curl_global_cleanup();

    return isOk;
}

// STRING (JSON) TO URL 
bool my::file::PostJsonStringToUrl(std::string url, std::string message, std::string user, std::string password)
{
    HEALTH_CHECK(url.empty(), false);
    HEALTH_CHECK(message.empty(), false);

    bool returnValue = true;

#if defined(_MSC_VER)
    CURL *curl = curl_easy_init();

    if (!curl)
    {
        LOG_ERROR();

        return false;
    }
    
    struct curl_slist *stringList = NULL;

    // BUG: (14-Sep-2016) http://stackoverflow.com/questions/8251325/how-do-i-post-a-buffer-of-json-using-libcurl
    stringList = curl_slist_append(stringList, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, stringList);
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, message.c_str());
    // if we don't provide POSTFIELDSIZE, libcurl will strlen() by itself.
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)message.size());
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

    if (url.find("https") != std::string::npos)
    {
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2);
        curl_easy_setopt(curl, CURLOPT_USERAGENT, "Mozilla/4.0 (compatible; MSIE 5.01; Windows NT 5.0)");
        
        // This line makes it work under https.
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, false);
    }

    if (!my::IsNull(user) &&
        !my::IsNull(password))
    {
        // Tell libcurl we can use "any" auth, which lets the lib pick one, but it also costs one extra round-trip and possibly sending of all the PUT data twice!!!
        curl_easy_setopt(curl, CURLOPT_HTTPAUTH, (long)CURLAUTH_BASIC);
        curl_easy_setopt(curl, CURLOPT_USERPWD, (user + ":" + password).c_str());
    }

    CURLcode urlCode = curl_easy_perform(curl); 

    if (urlCode != CURLE_OK)
    {
        LOG_MESSAGE(std::string("cURL POST failed for ") + url + ": " + curl_easy_strerror(urlCode));

        returnValue = false;
    }

    curl_slist_free_all(stringList);

    stringList = NULL;

    curl_easy_cleanup(curl);

#else //#if defined(_MSC_VER)
    return false;
#endif //#if defined(_MSC_VER)

    return returnValue;
}

// https://curl.haxx.se/libcurl/c/httpput.html
static size_t read_callback(void *ptr, size_t size, size_t nmemb, void *stream)
{
    // In real-world cases, this would probably get this data differently as this fread() stuff is exactly what the library already would do by default internally
    size_t retcode = fread(ptr, size, nmemb, (FILE *)stream);

    return retcode;
}

// FILE TO URL 
// TRICKY: (14-Apr-2017) RETURNS RESPONSE IN CASE OF ERROR
std::string my::file::PutToUrl(std::string url, std::string fileName, std::string user, std::string password)
{
    // EMPTY IN CASE OF SUCCESS
    std::string response;

    HEALTH_CHECK(url.empty(), response);
    HEALTH_CHECK(fileName.empty(), response);

#if defined(_MSC_VER)
    CURL *curl = curl_easy_init();

    if (!curl)
    {
        LOG_ERROR();

        return "Failed to start a libcurl easy session.";
    }

    struct curl_slist *stringList = NULL;

    // BUG: (14-Sep-2016) http://stackoverflow.com/questions/8251325/how-do-i-post-a-buffer-of-json-using-libcurl
    stringList = curl_slist_append(stringList, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, stringList);

    FILE * fileStream = 0;
    
#if defined(_MSC_VER)
    if (fopen_s(&fileStream, fileName.c_str(), "rb") != 0)
    {
        LOG_ERROR();

        curl_easy_cleanup(curl);

        return "Failed to open the local file.";
    }
#else //#if defined(_MSC_VER)
    fileStream = fopen(fileName.c_str(), "rb");

    if (fileStream == NULL)
    {
        LOG_ERROR();

        curl_easy_cleanup(curl);

        return "Failed to open the local file.";
    }
#endif //#if defined(_MSC_VER)

    // Compute file size.
    fseek(fileStream, 0, SEEK_END);
    long fileSize = ftell(fileStream);
    rewind(fileStream);

    curl_easy_setopt(curl, CURLOPT_READFUNCTION, read_callback);

    curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);
    curl_easy_setopt(curl, CURLOPT_PUT, 1L);

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

    curl_easy_setopt(curl, CURLOPT_READDATA, fileStream);

    curl_easy_setopt(curl, CURLOPT_INFILESIZE_LARGE, (curl_off_t)fileSize);

    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

    if (url.find("https") != std::string::npos)
    {
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2);
        curl_easy_setopt(curl, CURLOPT_USERAGENT, "Mozilla/4.0 (compatible; MSIE 5.01; Windows NT 5.0)");

        // This line makes it work under https.
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, false);
    }

    if (!my::IsNull(user) &&
        !my::IsNull(password))
    {
        // Tell libcurl we can use "any" auth, which lets the lib pick one, but it also costs one extra round-trip and possibly sending of all the PUT data twice!!!
        curl_easy_setopt(curl, CURLOPT_HTTPAUTH, (long)CURLAUTH_BASIC);
        curl_easy_setopt(curl, CURLOPT_USERPWD, (user + ":" + password).c_str());
    }

    CURLcode urlCode = curl_easy_perform(curl);

    fclose(fileStream);

    if (urlCode != CURLE_OK)
    {
        LOG_MESSAGE(std::string("cURL PUT failed for ") + url + ": " + curl_easy_strerror(urlCode));

        response = curl_easy_strerror(urlCode);
    }

    curl_slist_free_all(stringList);

    stringList = NULL;

    curl_easy_cleanup(curl);

#else //#if defined(_MSC_VER)
    return response;
#endif //#if defined(_MSC_VER)

    return response;
}

// URL (BODY) TO LOCAL FILE
bool my::file::GetUrlAsFile(std::string url, std::string fileName, bool binary, std::string user, std::string password)
{
    HEALTH_CHECK(url.empty(), false);
    HEALTH_CHECK(fileName.empty(), false);

    bool isOk = true;

    // BUG: (04-Nov-2016) GET APPROACH ADAPTED TO APPLE
    // https://curl.haxx.se/libcurl/c/getinmemory.html
    CURL *curl_handle = 0;

    struct MemoryStruct chunk;

    chunk.size = 0;

    curl_global_init(CURL_GLOBAL_ALL);

    curl_handle = curl_easy_init();

    curl_easy_setopt(curl_handle, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl_handle, CURLOPT_FOLLOWLOCATION, 1L);

    // Send all data to this function
    curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);

    // We pass our 'chunk' struct to the callback function
    curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, (void *)&chunk);

    if (url.find("https") != std::string::npos)
    {
        curl_easy_setopt(curl_handle, CURLOPT_SSL_VERIFYHOST, 2);

        // This line makes it work under https.
        curl_easy_setopt(curl_handle, CURLOPT_SSL_VERIFYPEER, false);
    }

    // Some servers don't like requests that are made without a user-agent field, so we provide one */
    curl_easy_setopt(curl_handle, CURLOPT_USERAGENT, "Mozilla/4.0 (compatible; MSIE 5.01; Windows NT 5.0)");

    if (!my::IsNull(user) &&
        !my::IsNull(password))
    {
        // Tell libcurl we can use "any" auth, which lets the lib pick one, but it also costs one extra round-trip and possibly sending of all the PUT data twice!!!
        curl_easy_setopt(curl_handle, CURLOPT_HTTPAUTH, (long)CURLAUTH_BASIC);
        curl_easy_setopt(curl_handle, CURLOPT_USERPWD, (user + ":" + password).c_str());
    }

    CURLcode urlCode = curl_easy_perform(curl_handle);

    if (urlCode != CURLE_OK)
    {
        LOG_MESSAGE(std::string("cURL GET failed for ") + url + ": " + curl_easy_strerror(urlCode));

        isOk = false;
    }
    else
    {
        // BUG: (22-Feb-2017) NULL RESPONSE?
        if (!chunk.memory.empty() &&
            // BUG: (13-Jan-2016) DUMB SERVERS THAT DON'T RETURN ERRORS, BUT RECORD THE ERROR ON THE RESPONSE INSTEAD
            (std::string(chunk.memory.data()).find("Internal Server Error") == std::string::npos) &&
            // BUG: (01-Apr-2017) DUMB SERVERS THAT DON'T RETURN ERRORS, BUT RECORD THE ERROR ON THE RESPONSE INSTEAD
            (std::string(chunk.memory.data()).find("404 Not Found") == std::string::npos))
        {
            FILE *fileStream = NULL;

#if defined(_MSC_VER)
            if (fopen_s(&fileStream, fileName.c_str(), "wb") != 0)
                fileStream = NULL;
#else //#if defined(_MSC_VER)
            fileStream = fopen(fileName.c_str(), "wb");
#endif //#if defined(_MSC_VER)

            if (fileStream != NULL)
            {
                // Now, our chunk.memory points to a memory block that is chunk.size bytes big and contains the remote file.
                fwrite(chunk.memory.data(), sizeof(char), chunk.memory.size(), fileStream);

                fclose(fileStream);
            }
            else
            {
                LOG_ERROR();

                isOk = false;
            }
        }
        else
        {
            LOG_MESSAGE("Failed to GET url " + fileName);

            isOk = false;
        }
    }

    curl_easy_cleanup(curl_handle);

    curl_global_cleanup();

    return isOk;
}

// http://stackoverflow.com/questions/7053538/how-do-i-encode-a-string-to-base64-using-only-boost
std::string my::file::DecodeBase64(const std::string& valueAsString)
{
    using namespace boost::archive::iterators;

    using iterator = transform_width<binary_from_base64<std::string::const_iterator>, 8, 6>;

    return boost::algorithm::trim_right_copy_if(std::string(iterator(std::begin(valueAsString)), iterator(std::end(valueAsString))), [](char c) {
        return c == '\0';
    });
}

// http://stackoverflow.com/questions/7053538/how-do-i-encode-a-string-to-base64-using-only-boost
std::string my::file::EncodeBase64(const std::string& valueAsString)
{
    using namespace boost::archive::iterators;

    using iterator = base64_from_binary<transform_width<std::string::const_iterator, 6, 8>>;

    auto tmp = std::string(iterator(std::begin(valueAsString)), iterator(std::end(valueAsString)));

    return tmp.append((3 - valueAsString.size() % 3) % 3, '=');
}

// http://stackoverflow.com/questions/7053538/how-do-i-encode-a-string-to-base64-using-only-boost
std::string my::file::EncodeBase64(const unsigned char *valueAsArray, my::int32 length)
{
    using namespace boost::archive::iterators;

    using iterator = base64_from_binary<transform_width<const unsigned char *, 6, 8>>;

    auto tmp = std::string(iterator(valueAsArray), iterator(valueAsArray + length));

    return tmp.append((3 - length % 3) % 3, '=');
}

