using Messages;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace MessagesTests
{
  [TestClass]
  public class MessagesTests
  {
    [TestMethod]
    public void DecodeAckMessage()
    {
      var correctAck = new AckMessage(1);
      var testAck = AckMessage.Decode(new byte[] {0, 8, 0, 1});

      Assert.AreEqual(correctAck.GameId, testAck.GameId);
    }

    [TestMethod]
    public void DecodeAnswerMessage()
    {
      var correctAnswer = new AnswerMessage(6, false, 0, "___t___");
      var testAnswer = AnswerMessage.Decode(new byte[]
          {0, 4, 0, 6, 0, 0, 0, 0, 14, 0, 95, 0, 95, 0, 95, 0, 116, 0, 95, 0, 95, 0, 95});
      Assert.AreEqual(correctAnswer.GameId, testAnswer.GameId);
      Assert.AreEqual(correctAnswer.Result, testAnswer.Result);
      Assert.AreEqual(correctAnswer.Score, testAnswer.Score);
      Assert.AreEqual(correctAnswer.Hint, testAnswer.Hint);
    }

    [TestMethod]
    public void DecodeBadMessage()
    {
      var badMessage = ErrorMessage.Decode(new byte[] {0, 0, 6, 0, 3, 4, 66, 5, 33, 4, 5});
      Assert.IsNull(badMessage);
    }

    [TestMethod]
    public void EncodeExitMessage()
    {
      var exitMessage = new ExitMessage(2);
      var correctBytes = new byte[] {0, 7, 0, 2};
      CollectionAssert.AreEqual(correctBytes, exitMessage.Encode());
    }

    [TestMethod]
    public void DecodeGameDefMessage()
    {
      var correctGameDef = new GameDefMessage(3, "_________", "shaped like a sac");
      var testGameDef = GameDefMessage.Decode(new byte[]
      {
          0, 2, 0, 3, 0, 18, 0, 95, 0, 95, 0, 95, 0, 95, 0, 95, 0, 95, 0, 95, 0, 95, 0, 95, 0, 34, 0, 115, 0, 104, 0,
          97, 0, 112, 0, 101, 0, 100, 0, 32, 0, 108, 0, 105, 0, 107, 0, 101, 0, 32, 0, 97, 0, 32, 0, 115, 0, 97, 0, 99
      });
      Assert.AreEqual(correctGameDef.GameId, testGameDef.GameId);
      Assert.AreEqual(correctGameDef.Hint, testGameDef.Hint);
      Assert.AreEqual(correctGameDef.Definition, testGameDef.Definition);
    }

    [TestMethod]
    public void EncodeGetHintMessage()
    {
      var getHintMessage = new GetHintMessage(4);
      var correctBytes = new byte[] {0, 5, 0, 4};
      CollectionAssert.AreEqual(correctBytes, getHintMessage.Encode());
    }

    [TestMethod]
    public void EncodeGuessMessage()
    {
      var guessMessage = new GuessMessage(5, "guess");
      var correctBytes = new byte[] {0, 3, 0, 5, 0, 10, 0, 103, 0, 117, 0, 101, 0, 115, 0, 115};
      CollectionAssert.AreEqual(correctBytes, guessMessage.Encode());
    }

    [TestMethod]
    public void DecodeHeartbeat()
    {
      var correctHeartbeat = new HeartbeatMessage(4);
      var testHeartbeat = HeartbeatMessage.Decode(new byte[] {0, 10, 0, 4});
      Assert.AreEqual(correctHeartbeat.GameId, testHeartbeat.GameId);
    }

    [TestMethod]
    public void DecodeHintMessage()
    {
      var correctHint = new HintMessage(4, "go_tis_");
      var testHint = HintMessage.Decode(new byte[]
          {0, 6, 0, 4, 0, 14, 0, 103, 0, 111, 0, 95, 0, 116, 0, 105, 0, 115, 0, 95});
      Assert.AreEqual(correctHint.GameId, testHint.GameId);
      Assert.AreEqual(correctHint.Hint, testHint.Hint);
    }

    [TestMethod]
    public void EncodeNewGameMessage()
    {
      var newGameMessage = new NewGameMessage("A01234567", "Last", "First", "Alias");
      var correctBytes = new byte[]
      {
          0, 1, 0, 18, 0, 65, 0, 48, 0, 49, 0, 50, 0, 51, 0, 52, 0, 53, 0, 54, 0, 55, 0, 8, 0, 76, 0, 97, 0, 115, 0,
          116, 0, 10, 0, 70, 0, 105, 0, 114, 0, 115, 0, 116, 0, 10, 0, 65, 0, 108, 0, 105, 0, 97, 0, 115
      };
      CollectionAssert.AreEqual(correctBytes, newGameMessage.Encode());
    }
  }
}