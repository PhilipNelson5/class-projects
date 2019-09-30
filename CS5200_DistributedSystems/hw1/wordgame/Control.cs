using System;
using System.Collections.Concurrent;
using System.Globalization;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Threading;
using System.Windows;
using log4net;
using Messages;

namespace wordgame
{
    /// <summary>
    ///     In the MVC pattern, Control is a controller which handles incoming messages from
    ///     the receiver and updates the View. Control also sends messages initiated in the
    ///     View. The Model is maintained by the server (points, guesses, word, hint, definition, etc...)
    ///     so the only Model maintained by Control is the current gameId.
    /// </summary>
    public class Control
    {
        private static readonly ILog Log =
                LogManager.GetLogger(
                        MethodBase.GetCurrentMethod().DeclaringType);

        private readonly ConcurrentQueue<Message> m_networkRecvQueue;
        private readonly ConcurrentQueue<Message> m_networkSendQueue;
        private readonly Receiver m_receiver;
        private readonly Sender m_sender;
        private readonly MainWindow m_theApp;
        private short m_gameId;
        private volatile bool m_running;
        private Thread m_thread;

        /// <summary>
        ///     Constructs a control object which in turn constructs and starts a sender, receiver,
        ///     and finally it's own message processing thread where received messages are handled.
        /// </summary>
        /// <param name="theApp">
        ///     a reference to the application used for logging messages
        ///     to the logTextBox and updating labels for hint, definition, score, etc...
        /// </param>
        public Control(MainWindow theApp)
        {
            Log.Info("Starting App Setup");
            m_gameId = 0;
            m_theApp = theApp;

            m_networkRecvQueue = new ConcurrentQueue<Message>();
            m_networkSendQueue = new ConcurrentQueue<Message>();

            var localEp = new IPEndPoint(IPAddress.Any, 0);
            var udpClient = new UdpClient(localEp) {Client = {ReceiveTimeout = 1000}};

            m_sender = new Sender(ref udpClient,
                    m_theApp.txt_serverIpAddr.Text,
                    ref m_networkSendQueue);

            m_receiver = new Receiver(ref udpClient, ref m_networkRecvQueue);

            m_receiver.Start();
            m_sender.Start();
            Start();

            Log.Info("Finished App Setup");
        }

        /// <summary>
        ///     Starts the process of processing received messages from the receiver on a new thread
        /// </summary>
        private void Start()
        {
            Log.Info("Starting");
            m_running = true;
            m_thread = new Thread(ProcessRecvQueue);
            m_thread.Start();
        }

        /// <summary>
        ///     Stops the sender and receiver. Stops the Control thread for processing received messages
        /// </summary>
        public void Stop()
        {
            if (!m_running) return;

            Log.Info("Stopping");
            m_running = false;
            m_sender.Stop();
            m_receiver.Stop();
            m_thread.Join(1000);
        }

        /// <summary>
        ///     Method used to process received messages and display their results to the user
        /// </summary>
        private void ProcessRecvQueue()
        {
            Log.Info("Processing Network Recv Queue");
            while (m_running)
                if (m_networkRecvQueue.TryDequeue(out var message))
                {
                    switch (message.MessageType)
                    {
                        case 2: // Game Definition - Response to a New Game Message
                            OnGameDef((GameDefMessage) message);
                            break;
                        case 4: // Answer - Response to a Guess Message
                            OnAnswer((AnswerMessage) message);
                            break;
                        case 6: // Hint - Response to a Get Hint Message
                            OnHint((HintMessage) message);
                            break;
                        case 8: // Ack - Response to an Exit Message
                            OnAck((AckMessage) message);
                            break;
                        case 9: // Error Message
                            OnError((ErrorMessage) message);
                            break;
                        case 10: // Heartbeat Message
                            OnHeartbeat((HeartbeatMessage) message);
                            break;
                    }
                }
                else
                {
                    Thread.Sleep(10);
                }
        }

        /// <summary>
        ///     Process a new game request from the user
        /// </summary>
        /// <param name="aNumber">user's A#</param>
        /// <param name="lastName">user's last name</param>
        /// <param name="firstName">user's first name</param>
        /// <param name="alias">user's public alias</param>
        public void NewGame(string aNumber, string lastName, string firstName, string alias)
        {
            m_networkSendQueue.Enqueue(new NewGameMessage(aNumber, lastName, firstName, alias));
            Log.Debug($"Enqueued new game: {aNumber}, {lastName}, {firstName}, {alias}");
        }

        /// <summary>
        ///     Process a make guess request from the user
        /// </summary>
        /// <param name="guess">string containing the user's guess</param>
        public void MakeGuess(string guess)
        {
            m_networkSendQueue.Enqueue(new GuessMessage(m_gameId, guess));
            Log.Debug($"Enqueued make guess: {guess}");
        }

        /// <summary>
        ///     Process a get hint request from the user
        /// </summary>
        public void GetHint()
        {
            m_networkSendQueue.Enqueue(new GetHintMessage(m_gameId));
            Log.Debug("Enqueued get hint");
        }

        /// <summary>
        ///     Process an exit request from the user
        /// </summary>
        public void Exit()
        {
            m_networkSendQueue.Enqueue(new ExitMessage(m_gameId));
            Log.Debug($"Enqueued exit: {m_gameId}");
        }

        /// <summary>
        ///     Process a new game def message from the server
        /// </summary>
        /// <param name="message">message from the server</param>
        private void OnGameDef(GameDefMessage message)
        {
            m_gameId = message.GameId;

            m_theApp.Dispatcher.Invoke(() =>
                    m_theApp.label_definition.Content = message.Definition
            );

            DisplayHint(message.Hint);
            Log.Debug($"Processed game def: {message.GameId}, {message.Definition}, {message.Hint}");
        }

        /// <summary>
        ///     Process an answer message from the server
        /// </summary>
        /// <param name="message">message from server</param>
        private void OnAnswer(AnswerMessage message)
        {
            if (message.Result) // Correct Guess
            {
                DisplayHint(message.Hint);
                m_theApp.Dispatcher.Invoke(() =>
                {
                    m_theApp.label_score.Content = $"Score: {message.Score}";
                    m_theApp.EnableControls(false);
                    m_theApp.label_message.Content = "correct answer";
                });
            }
            else // Incorrect Guess
            {
                DisplayHint(message.Hint);
                m_theApp.Dispatcher.Invoke(() =>
                {
                    m_theApp.txt_guess.Text = "";
                    m_theApp.label_message.Content = "wrong answer";
                });
            }

            Log.Debug($"Processed answer: {message.GameId}, {message.Result}, {message.Hint}, {message.Score}");
        }

        /// <summary>
        ///     Process a hint message from the server
        /// </summary>
        /// <param name="message">message from the server</param>
        private void OnHint(HintMessage message)
        {
            DisplayHint(message.Hint);
            Log.Debug($"Processed hint: {message.GameId}, {message.Hint}");
        }

        /// <summary>
        ///     Process an ack message from the server
        /// </summary>
        /// <param name="message">message from the server</param>
        private void OnAck(AckMessage message)
        {
            Stop();
            m_theApp.Dispatcher.Invoke(() =>
                    Application.Current.Shutdown()
            );
            Log.Debug($"Processed ack: {message.GameId}");
        }

        /// <summary>
        ///     Process an error message from the server
        /// </summary>
        /// <param name="message">message from the server</param>
        private void OnError(ErrorMessage message)
        {
            m_theApp.Dispatcher.Invoke(
                    () => m_theApp.logTextBox.AppendText($"[ERROR] {message.Error}\n")
            );
            Log.Debug($"Processed Error: {message.GameId}");
        }

        /// <summary>
        ///     Process a heartbeat message from the server
        /// </summary>
        /// <param name="message">message from the server</param>
        private void OnHeartbeat(HeartbeatMessage message)
        {
            m_networkSendQueue.Enqueue(
                    new AckMessage(message.GameId)
            );

            m_theApp.Dispatcher.Invoke(
                    () => m_theApp.logTextBox.AppendText(
                            $"Heartbeat {DateTime.Now.ToString(CultureInfo.CurrentCulture)}\n")
            );
            Log.Debug($"Processed heartbeat: {message.GameId}");
        }

        /// <summary>
        ///     Helper function to format and display a hint to the user.
        ///     Hints are given a space character between each original character.
        /// </summary>
        /// <param name="hint">hint to display</param>
        private void DisplayHint(string hint)
        {
            var hintDisp = hint.Aggregate("", (acc, c) => acc + c + " ");

            m_theApp.Dispatcher.Invoke(() =>
                    m_theApp.label_hint.Text = $"{hintDisp} ({hint.Length} chars)"
            );

            Log.Debug($"Hint Created: {hintDisp}");
        }
    }
}