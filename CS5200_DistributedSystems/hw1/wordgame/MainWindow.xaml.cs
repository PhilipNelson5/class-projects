using System.ComponentModel;
using System.Reflection;
using System.Windows;
using System.Windows.Controls;
using log4net;
using log4net.Config;

namespace wordgame
{
    /// <inheritdoc>
    ///     <cref></cref>
    /// </inheritdoc>
    /// <summary>
    ///     Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow
    {
        private static readonly ILog Log =
                LogManager.GetLogger(
                        MethodBase.GetCurrentMethod().DeclaringType);

        private readonly Control m_control;

        /// <inheritdoc />
        /// <summary>
        /// Initialize the main window and configure log4net
        /// </summary>
        public MainWindow()
        {
            XmlConfigurator.Configure();
            Log.Fatal("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓");
            Log.Fatal("┃                       Start                       ┃");
            Log.Fatal("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛");
            InitializeComponent();
            logTextBox.Text = "";
            logTextBox.AppendText("Initializing\n");
            m_control = new Control(this);

            EnableControls(false);
            label_score.Content = "";
            label_message.Content = "";
            label_definition.Content = "";
            label_hint.Text = "";
        }

        /// <summary>
        /// When the 'X' on the window is clicked, stops all background threads
        /// through the control object. Does not request server for exit conversation.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void MainWindow_Closing(object sender, CancelEventArgs e)
        {
            m_control.Stop();
        }

        /// <summary>
        /// Handles the new game button click and initiates the new game
        /// conversation through the control
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Btn_new_game_Click(object sender, RoutedEventArgs e)
        {
            EnableControls(true);
            label_message.Content = "";
            label_score.Content = "";
            txt_guess.Text = "";
            m_control.NewGame(txt_aNumber.Text,
                    txt_lastName.Text,
                    txt_firstName.Text,
                    txt_alias.Text);

            logTextBox.AppendText("new game initiated\n");
        }

        /// <summary>
        /// Handles the exit button click and initiates the exit conversation
        /// through the control
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Btn_exit_Click(object sender, RoutedEventArgs e)
        {
            EnableControls(false);
            m_control.Exit();
        }

        /// <summary>
        /// Disables user controls so the user must start a new game or
        /// exit the game
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Btn_giveUp_Click(object sender, RoutedEventArgs e)
        {
            EnableControls(false);
        }

        /// <summary>
        /// Handles the guess button click and initiates the guess conversation
        /// through the control
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Btn_guess_Click(object sender, RoutedEventArgs e)
        {
            m_control.MakeGuess(txt_guess.Text);
        }

        /// <summary>
        /// Handles the hint button click and initiates the get hint conversation
        /// through the control
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Btn_hint_Click(object sender, RoutedEventArgs e)
        {
            m_control.GetHint();
        }

        /// <summary>
        /// Handles the text changed even of the logTextBox in order to maintain the
        /// text box scrolled to the bottom
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void LogTextBox_TextChanged(object sender, TextChangedEventArgs e)
        {
            logTextBox.ScrollToEnd();
        }

        /// <summary>
        /// Enables or disables a set of user controls and resets some labels
        /// </summary>
        /// <param name="enable">true to enable, false to disable controls</param>
        public void EnableControls(bool enable)
        {
            btn_hint.IsEnabled = enable;
            btn_guess.IsEnabled = enable;
            btn_giveUp.IsEnabled = enable;

            txt_guess.IsEnabled = enable;

            if (enable)
            {
                label_definition.Content = "";
                label_hint.Text = "";
                label_hint.IsEnabled = enable;
                label_definition.IsEnabled = enable;
            }
            else
            {
                label_hint.IsEnabled = enable;
                label_definition.IsEnabled = enable;
            }
        }
    }
}