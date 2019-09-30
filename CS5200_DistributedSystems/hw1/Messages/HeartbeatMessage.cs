using System;
using System.IO;

namespace Messages
{
  public class HeartbeatMessage : Message
  {
    public HeartbeatMessage(short gameId) : base(10)
    {
      GameId = gameId;
    }

    public short GameId { get; }

    public new static HeartbeatMessage Decode(byte[] bytes)
    {
      var ms = new MemoryStream(bytes);

      try
      {
        ReadShort(ms);
        var gameId = ReadShort(ms);

        return new HeartbeatMessage(gameId);
      }
      catch
      {
        return null;
      }
    }

    public override byte[] Encode()
    {
      // purposely unimplemented.
      // Client will not send a Heartbeat message
      throw new NotImplementedException();
    }
  }
}