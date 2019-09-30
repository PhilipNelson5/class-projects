using System;
using System.IO;

namespace Messages
{
  public class NewGameMessage : Message
  {
    public NewGameMessage(string aNumber,
        string lastName,
        string firstName,
        string alias) : base(1)
    {
      ANumber = aNumber;
      LastName = lastName;
      FirstName = firstName;
      Alias = alias;
    }

    public string ANumber { get; }
    public string LastName { get; }
    public string FirstName { get; }
    public string Alias { get; }

    public new static NewGameMessage Decode(byte[] bytes)
    {
      var ms = new MemoryStream(bytes);

      ReadShort(ms);
      var aNumber = ReadString(ms);
      var lastName = ReadString(ms);
      var firstName = ReadString(ms);
      var alias = ReadString(ms);

      return new NewGameMessage(aNumber, lastName, firstName, alias);
    }

    public override byte[] Encode()
    {
      var ms = new MemoryStream();

      Write(ms, MessageType);
      Write(ms, ANumber);
      Write(ms, LastName);
      Write(ms, FirstName);
      Write(ms, Alias);

      return ms.ToArray();
    }
  }
}