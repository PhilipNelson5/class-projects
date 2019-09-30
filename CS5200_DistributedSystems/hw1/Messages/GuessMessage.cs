using System;
using System.IO;

namespace Messages
{
  public class GuessMessage : Message
  {
    public GuessMessage(short gameId, string guess) : base(3)
    {
      GameId = gameId;
      Guess = guess;
    }

    public short GameId { get; }
    public string Guess { get; }

    public new static GuessMessage Decode(byte[] bytes)
    {
      // purposely unimplemented.
      // Client will not receive a Guess message
      throw new NotImplementedException();
    }

    public override byte[] Encode()
    {
      var ms = new MemoryStream();

      Write(ms, MessageType);
      Write(ms, GameId);
      Write(ms, Guess);

      return ms.ToArray();
    }
  }
}