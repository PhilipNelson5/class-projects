using System.Net;
using System.Net.Sockets;

namespace wordgame
{
    internal class Shared
    {
        /// <summary>
        ///     Parses and constructs an IPEndPoint
        ///     Provided by Dr. Stephen Clyde
        /// </summary>
        /// <param name="hostnameAndPort">a string containing an ip address and port</param>
        /// <example>"127.0.0.1:12345"</example>
        /// <returns>IPEndPoint with the ip address and port specified</returns>
        public static IPEndPoint Parse(string hostnameAndPort)
        {
            if (string.IsNullOrWhiteSpace(hostnameAndPort)) return null;

            IPEndPoint result = null;
            var tmp = hostnameAndPort.Split(':');
            if (tmp.Length == 2 && !string.IsNullOrWhiteSpace(tmp[0]) && !string.IsNullOrWhiteSpace(tmp[1]))
                result = new IPEndPoint(ParseAddress(tmp[0]), ParsePort(tmp[1]));

            return result;
        }

        /// <summary>
        ///     Parses a hostname string to an IPAddress object
        ///     Provided by Dr. Stephen Clyde
        /// </summary>
        /// <param name="hostname">string containing a valid hostname</param>
        /// <example>"127.0.0.1"</example>
        /// <returns>an IPAddress of the specified hostname</returns>
        public static IPAddress ParseAddress(string hostname)
        {
            IPAddress result = null;
            var addressList = Dns.GetHostAddresses(hostname);
            for (var i = 0; i < addressList.Length && result == null; i++)
                if (addressList[i].AddressFamily == AddressFamily.InterNetwork)
                    result = addressList[i];
            return result;
        }

        /// <summary>
        ///     Parse a port number as a string into in integer
        ///     Provided by Dr. Stephen Clyde
        /// </summary>
        /// <param name="portStr">a port number stored as a string</param>
        /// <example>"12345"</example>
        /// <returns>an int which is the result of parsing the string, zero on error</returns>
        public static int ParsePort(string portStr)
        {
            var port = 0;
            if (!string.IsNullOrWhiteSpace(portStr))
                int.TryParse(portStr, out port);
            return port;
        }
    }
}