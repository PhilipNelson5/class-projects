using System;
using System.IO;

namespace Messages
{
    public class GetHintMessage : Message
    {
        public short GameId { get; private set; }
        public GetHintMessage(short gameId) : base(5)
        {
            GameId = gameId;
        }
        public new static GetHintMessage Decode(byte[] bytes)
        {
            // purposely unimplemented.
            // Client will not receive a Get Hint message
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
