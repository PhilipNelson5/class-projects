using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using Messages;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using wordgame;

namespace WordgameTests
{
    [TestClass]
    public class UnitTest
    {
        [TestMethod]
        public void CommunicationTestNewGame()
        {
            var sendQueue = new ConcurrentQueue<Message>();
            var recvQueue = new ConcurrentQueue<Message>();

            var localEpSender = new IPEndPoint(IPAddress.Any, 0);
            var udpClientSender = new UdpClient(localEpSender);

            var localEpReceiver = new IPEndPoint(IPAddress.Any, 0);
            var udpClientReceiver = new UdpClient(localEpReceiver) { Client = { ReceiveTimeout = 1000 } };

            var sender = new Sender(ref udpClientSender, $"127.0.0.1:{udpClientReceiver.Client.LocalEndPoint.ToString().Split(':')[1]}", ref sendQueue);
            sender.Start();

            var receiver = new Receiver(ref udpClientReceiver, ref recvQueue);
            receiver.Start();

            var newGameMessage = new NewGameMessage("A01010101", "Smith", "John", "Joe");

            sendQueue.Enqueue(newGameMessage);

            Thread.Sleep(100);

            sender.Stop();
            receiver.Stop();

            // receiver does not handle inbound new game messages
            // so it will not add it to the receive queue
            Assert.AreEqual(0, recvQueue.Count);
        }

        [TestMethod]
        public void CommunicationTestGameDef()
        {
            var sendQueue = new ConcurrentQueue<Message>();
            var recvQueue = new ConcurrentQueue<Message>();

            var localEpSender = new IPEndPoint(IPAddress.Any, 0);
            var udpClientSender = new UdpClient(localEpSender);

            var localEpReceiver = new IPEndPoint(IPAddress.Any, 0);
            var udpClientReceiver = new UdpClient(localEpReceiver) { Client = { ReceiveTimeout = 1000 } };

            var sender = new Sender(ref udpClientSender, $"127.0.0.1:{udpClientReceiver.Client.LocalEndPoint.ToString().Split(':')[1]}", ref sendQueue);
            sender.Start();

            var receiver = new Receiver(ref udpClientReceiver, ref recvQueue);
            receiver.Start();

            var gameDefMessage = new GameDefMessage(1, "_____", "word definition");

            sendQueue.Enqueue(gameDefMessage);

            Thread.Sleep(100);

            sender.Stop();
            receiver.Stop();

            if (recvQueue.TryDequeue(out var message))
            {
                Assert.AreEqual(gameDefMessage.GameId, ((GameDefMessage)message).GameId);
                Assert.AreEqual(gameDefMessage.Hint, ((GameDefMessage)message).Hint);
                Assert.AreEqual(gameDefMessage.Definition, ((GameDefMessage)message).Definition);
            }
            else
            {
                Assert.Fail();
            }
        }
    }
}