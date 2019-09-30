using System;
using System.IO;

namespace Messages
{
  public class ErrorMessage : Message
  {
    public ErrorMessage(short gameId, string error) : base(9)
    {
      GameId = gameId;
      Error = error;
    }

    public short GameId { get; }
    public string Error { get; }

    public new static ErrorMessage Decode(byte[] bytes)
    {
      var ms = new MemoryStream(bytes);

      try
      {
        ReadShort(ms);
        var gameId = ReadShort(ms);
        var error = ReadString(ms);

        return new ErrorMessage(gameId, error);
      }
      catch
      {
        return null;
      }
    }

    public override byte[] Encode()
    {
      // purposely unimplemented.
      // Client will not send an Error message
      throw new NotImplementedException();
    }
  }
}