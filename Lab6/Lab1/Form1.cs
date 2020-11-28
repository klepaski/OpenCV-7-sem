using System;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;

using Emgu.CV;
using Emgu.CV.Structure;

using DirectShowLib;
namespace Lab1
{
    public partial class Form1 : Form
    {
        CascadeClassifier faceCascade = new CascadeClassifier(@"D:\4\ПММП\Lab6\frontalface.xml");
        CascadeClassifier eyeCascade = new CascadeClassifier(@"D:\4\ПММП\Lab6\eye.xml");
        CascadeClassifier bodyCascade = new CascadeClassifier(@"D:\4\ПММП\Lab6\fullbody.xml");

        #region Variables

        #region Camera Capture Variables
        private Emgu.CV.VideoCapture _capture = null;
        private bool _captureInProgress = false;
        int CameraDevice = 0;
        VideoDevice[] WebCams;
        #endregion

        #region Camera Settings
        int Brightness_Store = 0;
        int Contrast_Store = 0;
        int Sharpness_Store = 0;
        #endregion

        #endregion

        public Form1()
        {
            InitializeComponent();

            DsDevice[] _SystemCamereas = DsDevice.GetDevicesOfCat(FilterCategory.VideoInputDevice);
            WebCams = new VideoDevice[_SystemCamereas.Length];
            for (int i = 0; i < _SystemCamereas.Length; i++)
            {
                WebCams[i] = new VideoDevice(i, _SystemCamereas[i].Name, _SystemCamereas[i].ClassID);
                CamSelect.Items.Add(WebCams[i].ToString());
            }
            if (CamSelect.Items.Count > 0)
            {
                CamSelect.SelectedIndex = 0;
                Shot.Enabled = true;
            }
        }


        private void Shot_Click(object sender, EventArgs e)
        {
            if (_capture != null)
            {
                if (_captureInProgress)
                {
                    Shot.Text = "Start";
                    _capture.Pause();
                    _captureInProgress = false;
                }
                else
                {
                    Shot.Text = "Stop";
                    _capture.Start();
                    _captureInProgress = true;
                }

            }
            else
            {
                SetupCapture(CamSelect.SelectedIndex);
                Shot_Click(null, null);
            }
        }
        private void SetupCapture(int Camera_Identifier)
        {
            CameraDevice = Camera_Identifier;

            if (_capture != null) _capture.Dispose();
            try
            {
                _capture = new VideoCapture(CameraDevice);
                _capture.ImageGrabbed += ProcessFrame;
            }
            catch (NullReferenceException excpt)
            {
                MessageBox.Show(excpt.Message);
            }
        }

        private void ProcessFrame(object sender, EventArgs arg)
        {
            Mat frame = new Mat();
            Mat frame1 = new Mat();
            _capture.Retrieve(frame);
            _capture.Retrieve(frame1);
            CvInvoke.MedianBlur(frame1, frame, 5);

            #region detection
            var imageToDisplay = frame.ToImage<Bgr, byte>();
            var faces = faceCascade.DetectMultiScale(frame.ToImage<Gray, byte>(), 1.1, 10, Size.Empty);


            foreach (var face in faces)
            {
                imageToDisplay.Draw(face, new Bgr(Color.BurlyWood), 3);
            }

            var eyes = eyeCascade.DetectMultiScale(frame.ToImage<Gray, byte>(), 1.1, 10, Size.Empty);

            foreach (var eye in eyes)
            {
                imageToDisplay.Draw(eye, new Bgr(Color.AliceBlue), 3);
            }

            var des = new HOGDescriptor();
            des.SetSVMDetector(HOGDescriptor.GetDefaultPeopleDetector());

            foreach (var pedestrain in des.DetectMultiScale(imageToDisplay, 0, new Size(8, 8), new Size(0, 0)).AsEnumerable())
            {
                imageToDisplay.Draw(pedestrain.Rect, new Bgr(Color.Red), 1);
            }
            #endregion


            DisplayImage(imageToDisplay.ToBitmap());
        }

        private delegate void DisplayImageDelegate(Bitmap Image);
        private void DisplayImage(Bitmap Image)
        {
            if (pictureBox1.InvokeRequired)
            {
                try
                {
                    DisplayImageDelegate DI = new DisplayImageDelegate(DisplayImage);
                    this.BeginInvoke(DI, new object[] { Image });
                }
                catch (Exception ex)
                {
                }
            }
            else
            {
                pictureBox1.Image = Image;
            }
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }
    }
}
