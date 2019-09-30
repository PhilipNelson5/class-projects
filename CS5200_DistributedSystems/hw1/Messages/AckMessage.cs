using System;
using System.IO;

namespace Messages
{
  public class AckMessage : Message
  {
    public AckMessage(short id) : base(8)
    {
      GameId = id;
    }

    public short GameId { get; }

    public new static AckMessage Decode(byte[] bytes)
    {
      var ms = new MemoryStream(bytes);

      try
      {
        ReadShort(ms);
        var id = ReadShort(ms);
        return new AckMessage(id);
      }
      catch (Exception)
      {
        return null;
      }
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