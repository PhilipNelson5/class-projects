using System.Collections.Concurrent;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Threading;
using log4net;
using Messages;

namespace wordgame
{
    /// <summary>
    ///     Sender handles sending messages to m_sendDestination by sending
    ///     all messages that are placed in the concurrent queue m_networkSendQueue.
    ///     All messages are sent to the IPEndPoint m_sendDestination
    /// </summary>
    public class Sender
    {
        private static readonly ILog Log =
                LogManager.GetLogger(
                        MethodBase.GetCurrentMethod().DeclaringType);

        private readonly ConcurrentQueue<Message> m_networkSendQueue;
        private readonly IPEndPoint m_sendDestination;
        private readonly UdpClient m_udpClient;
        private volatile bool m_running;
        private Thread m_thread;

        /// <summary>
        ///     Constructs a sender which can send messages to an IPEndpoint
        /// </summary>
        /// <param name="udpClient">udpClient to use for sending messages</param>
        /// <param name="endpoint">destination to send messages to</param>
        /// <param name="networkSendQueue">concurrent queue of messages to monitor and send</param>
        public Sender(ref UdpClient udpClient,
                string endpoint,
                ref ConcurrentQueue<Message> networkSendQueue)
        {
            m_udpClient = udpClient;
            m_sendDestination = Shared.Parse(endpoint);
            m_networkSendQueue = networkSendQueue;
            m_running = false;
        }

        /// <summary>
        ///     Starts the process of sending messages from the concurrent send queue on a separate thread
        /// </summary>
        public void Start()
        {
            if (m_running) return;
            Log.Debug("Starting");
            m_running = true;
            m_thread = new Thread(ProcessSendQueue);
            m_thread.Start();
            Log.Info("Started");
        }

        /// <summary>
        ///     Stops the message sending thread
        /// </summary>
        public void Stop()
        {
            if (!m_running) return;
            Log.Debug("Stopping");
            m_running = false;
            m_thread.Join(1000);
            Log.Info("Stopped");
        }

        /// <summary>
        ///     Method used to send messages from the send queue to the destination IPEndPoint
        /// </summary>
        private void ProcessSendQueue()
        {
            Log.Info("Processing Send Queue");
            while (m_running)
                if (m_networkSendQueue.TryDequeue(out var message))
                {
                    var sentBytes = Send(message);
                    Log.Debug($"Sent {sentBytes} bytes");
                }
                else
                {
                    Thread.Sleep(10);
                }
        }

        /// <summary>
        ///     Helper function to send a message
        /// </summary>
        /// <param name="message">message to send</param>
        /// <returns>bytes sent</returns>
        private int Send(Message message)
        {
            Log.Info($"Sending Message Type {message.MessageType}");

            var sendBuffer = message.Encode();
            var bytesSent = m_udpClient.Send(sendBuffer, sendBuffer.Length, m_sendDestination);

            Log.Info($"Bytes Sent: {sendBuffer.Aggregate("", (acc, b) => acc + b + " ")}");

            return bytesSent;
        }
    }
}