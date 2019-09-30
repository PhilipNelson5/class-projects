using System;
using System.IO;

namespace Messages
{
  public class ExitMessage : Message
  {
    public ExitMessage(short gameId) : base(7)
    {
      GameId = gameId;
    }

    public short GameId { get; }

    public new static ExitMessage Decode(byte[] bytes)
    {
      // purposely unimplemented.
      // Client will not receive an Exit message
      throw new NotImplementedException();
    }

    public override byte[] Encode()
    {
      var ms = new MemoryStream();

      Write(ms, MessageType);
      Write(ms, GameId);

      return ms.ToArray();
    }
  }
}