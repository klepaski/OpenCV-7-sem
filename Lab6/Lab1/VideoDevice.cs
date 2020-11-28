using System;


namespace Lab1
{
    struct VideoDevice
    {
        public string Device_Name;
        public int Device_ID;
        public Guid Identifier;

        public VideoDevice(int ID, string Name, Guid Identity = new Guid())
        {
            Device_ID = ID;
            Device_Name = Name;
            Identifier = Identity;
        }
        
        public override string ToString()
        {
            return String.Format("[{0}] {1}: {2}", Device_ID, Device_Name, Identifier);
        }
    }
}
