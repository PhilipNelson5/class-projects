using System;
using System.IO;

namespace Messages
{
    public class GameDefMessage : Message
    {
        public GameDefMessage(short gameId, string hint, string definition) : base(2)
        {
            GameId = gameId;
            Hint = hint;
            Definition = definition;
        }

        public short GameId { get; }
        public string Hint { get; }
        public string Definition { get; }

        public new static GameDefMessage Decode(byte[] bytes)
        {
            var ms = new MemoryStream(bytes);

            try
            {
                ReadShort(ms);
                var gameId = ReadShort(ms);
                var hint = ReadString(ms);
                var definition = ReadString(ms);

                return new GameDefMessage(gameId, hint, definition);
            }
            catch
            {
                return null;
            }
        }

        public override byte[] Encode()
        {
            var ms = new MemoryStream();

            Write(ms, MessageType);
            Write(ms, GameId);
            Write(ms, Hint);
            Write(ms, Definition);

            return ms.ToArray();
        }
    }
}