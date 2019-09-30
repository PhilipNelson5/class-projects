using System;
using System.IO;

namespace Messages
{
  public class HintMessage : Message
  {
    public HintMessage(short gameId, string hint) : base(6)
    {
      GameId = gameId;
      Hint = hint;
    }

    public short GameId { get; }
    public string Hint { get; }

    public new static HintMessage Decode(byte[] bytes)
    {
      var ms = new MemoryStream(bytes);

      try
      {
        ReadShort(ms);
        var gameId = ReadShort(ms);
        var hint = ReadString(ms);

        return new HintMessage(gameId, hint);
      }
      catch
      {
        return null;
      }
    }

    public override byte[] Encode()
    {
      // purposely unimplemented.
      // Client will not send a Hint message
      throw new NotImplementedException();
    }
  }
}