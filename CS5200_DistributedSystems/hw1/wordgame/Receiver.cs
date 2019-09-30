using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Threading;
using log4net;
using Messages;

namespace wordgame
{
    /// <summary>
    ///     Receiver monitors the udpClient for incoming messages on all network devices,
    ///     decodes the messages and en-queues them on the concurrent m_recvQueue
    /// </summary>
    public class Receiver
    {
        private static readonly ILog Log =
                LogManager.GetLogger(
                        MethodBase.GetCurrentMethod().DeclaringType);

        private readonly ConcurrentQueue<Message> m_recvQueue;
        private readonly UdpClient m_udpClient;
        private volatile bool m_running;
        private Thread m_thread;

        /// <summary>
        ///     Constructs a receiver which listens for messages from the udpClient and en-queues
        ///     them on the concurrent recvQueue
        /// </summary>
        /// <param name="udpClient">udpClient for listening for incoming messages</param>
        /// <param name="recvQueue">concurrent queue to enqueue decoded messages</param>
        public Receiver(ref UdpClient udpClient,
                ref ConcurrentQueue<Message> recvQueue)
        {
            m_udpClient = udpClient;
            m_recvQueue = recvQueue;
            m_running = false;
        }

        /// <summary>
        ///     Start the process of listening for incoming messages on a separate thread
        /// </summary>
        public void Start()
        {
            if (m_running) return;

            Log.Debug("Starting");
            m_running = true;
            m_thread = new Thread(BeginRecv);
            m_thread.Start();
            Log.Info("Started");
        }

        /// <summary>
        ///     Stops the message receiving thread
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
        ///     Method used to listen for incoming messages, decode them and en-queue them on the
        ///     concurrent m_recvQueue. Unrecognized messages are ignore.
        /// </summary>
        public void BeginRecv()
        {
            Log.Info($"Listening on {(IPEndPoint) m_udpClient.Client.LocalEndPoint}");
            while (m_running)
            {
                var remoteEp = new IPEndPoint(IPAddress.Any, 0);
                byte[] recvBuf = null;
                try
                {
                    recvBuf = m_udpClient.Receive(ref remoteEp);
                }
                catch (SocketException)
                {
                    // Console.WriteLine("Timeout");
                }

                if (recvBuf == null)
                {
                    Thread.Sleep(10);
                    continue;
                }

                var message = Message.Decode(recvBuf);

                if (message != null)
                    m_recvQueue.Enqueue(message);
                else
                    Log.Debug("Malformed or unrecognized message, did not decode");
            }
        }
    }
}