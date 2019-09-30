using System;
using System.IO;
using System.Linq;
using System.Net;
using System.Reflection;
using System.Text;
using log4net;

namespace Messages
{
    public abstract class Message
    {
        private static readonly ILog Log =
                LogManager.GetLogger(typeof(Message));

        protected Message(short type)
        {
            MessageType = type;
        }

        public short MessageType { get; }

        public abstract byte[] Encode();

        public static Message Decode(byte[] bytes)
        {
            Message message = null;
            var type = IPAddress.NetworkToHostOrder(BitConverter.ToInt16(bytes, 0));
            Log.Info($"Received Message type {type}");
            Log.Info($"Bytes received {bytes.Aggregate("", (current, b) => current + $"{b} ")}");
            switch (type)
            {
                case 2: // Game Definition - Response to a New Game Message
                    message = GameDefMessage.Decode(bytes);
                    break;
                case 4: // Answer - Response to a Guess Message
                    message = AnswerMessage.Decode(bytes);
                    break;
                case 6: // Hint - Response to a Get Hint Message
                    message = HintMessage.Decode(bytes);
                    break;
                case 8: // Ack - Response to an Exit Message
                    message = AckMessage.Decode(bytes);
                    break;
                case 9: // Error Message
                    message = ErrorMessage.Decode(bytes);
                    break;
                case 10: // Heartbeat Message
                    message = HeartbeatMessage.Decode(bytes);
                    break;
                default:
                    Log.Info("Received unknown message type");
                    break;
            }

            return message;
        }

        //-----------------------------------------------------------
        // Helper functions for reading and writing short, int long,
        // and strings to memory streams for encoding a message.
        // Author: Dr. Stephen Clyde
        //-----------------------------------------------------------
        protected static void Write(Stream memoryStream, short value)
        {
            var bytes = BitConverter.GetBytes(IPAddress.HostToNetworkOrder(value));
            memoryStream.Write(bytes, 0, bytes.Length);
        }

        protected static void Write(Stream memoryStream, int value)
        {
            var bytes = BitConverter.GetBytes(IPAddress.HostToNetworkOrder(value));
            memoryStream.Write(bytes, 0, bytes.Length);
        }

        protected static void Write(Stream memoryStream, long value)
        {
            var bytes = BitConverter.GetBytes(IPAddress.HostToNetworkOrder(value));
            memoryStream.Write(bytes, 0, bytes.Length);
        }

        protected static void Write(Stream memoryStream, string value)
        {
            var bytes = Encoding.BigEndianUnicode.GetBytes(value);
            Write(memoryStream, (short) bytes.Length);
            memoryStream.Write(bytes, 0, bytes.Length);
        }

        protected static bool ReadBool(Stream memoryStream)
        {
            var bytes = new byte[1];
            var bytesRead = memoryStream.Read(bytes, 0, bytes.Length);
            if (bytesRead != bytes.Length)
                throw new Exception("Cannot decode a boolean from message");

            return BitConverter.ToBoolean(bytes, 0);
        }

        protected static short ReadShort(Stream memoryStream)
        {
            var bytes = new byte[2];
            var bytesRead = memoryStream.Read(bytes, 0, bytes.Length);
            if (bytesRead != bytes.Length)
                throw new Exception("Cannot decode a short integer from message");

            return IPAddress.NetworkToHostOrder(BitConverter.ToInt16(bytes, 0));
        }

        protected static int ReadInt(Stream memoryStream)
        {
            var bytes = new byte[4];
            var bytesRead = memoryStream.Read(bytes, 0, bytes.Length);
            if (bytesRead != bytes.Length)
                throw new Exception("Cannot decode an integer from message");

            return IPAddress.NetworkToHostOrder(BitConverter.ToInt32(bytes, 0));
        }

        protected static long ReadLong(Stream memoryStream)
        {
            var bytes = new byte[8];
            var bytesRead = memoryStream.Read(bytes, 0, bytes.Length);
            if (bytesRead != bytes.Length)
                throw new Exception("Cannot decode a long integer from message");

            return IPAddress.NetworkToHostOrder(BitConverter.ToInt64(bytes, 0));
        }

        protected static string ReadString(Stream memoryStream)
        {
            var result = string.Empty;
            var length = ReadShort(memoryStream);
            if (length <= 0) return result;

            var bytes = new byte[length];
            var bytesRead = memoryStream.Read(bytes, 0, bytes.Length);
            if (bytesRead != length)
                throw new Exception("Cannot decode a string from message");

            result = Encoding.BigEndianUnicode.GetString(bytes, 0, bytes.Length);
            return result;
        }
    }
}