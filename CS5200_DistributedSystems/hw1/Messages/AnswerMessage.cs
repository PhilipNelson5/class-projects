using System;
using System.IO;

namespace Messages
{
  public class AnswerMessage : Message
  {
    public AnswerMessage(short gameId, bool result, short score, string hint) : base(4)
    {
      GameId = gameId;
      Result = result;
      Score = score;
      Hint = hint;
    }

    public short GameId { get; }
    public bool Result { get; }
    public short Score { get; }
    public string Hint { get; }

    public new static AnswerMessage Decode(byte[] bytes)
    {
      var ms = new MemoryStream(bytes);

      try
      {
        ReadShort(ms);
        var gameId = ReadShort(ms);
        var result = ReadBool(ms);
        var score = ReadShort(ms);
        var hint = ReadString(ms);
        return new AnswerMessage(gameId, result, score, hint);
      }
      catch (Exception)
      {
        return null;
      }
    }

    public override byte[] Encode()
    {
      // purposely unimplemented.
      // Client will not send an Answer message
      throw new NotImplementedException();
    }
  }
}