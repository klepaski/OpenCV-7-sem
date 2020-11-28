using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Cvb;

//DiresctShow
using DirectShowLib;
using Emgu.CV.Dpm;

namespace PMMP_Lab06
{
    public partial class Form1 : Form
    {
        #region Variables
        #region Camera Capture Variables
        private Emgu.CV.VideoCapture _capture = null; //Camera
        private bool _captureInProgress = false; //Variable to track camera state
        int CameraDevice = 0; //Variable to track camera device selected
        VideoDevice[] WebCams; //List containing all the camera available
        #endregion
        #region Camera Settings
        int Brightness_Store = 0;
        int Contrast_Store = 0;
        int Sharpness_Store = 0;

        CascadeClassifier faceCascade = new CascadeClassifier("haarcascade_frontalface_default.xml");
        //for eye
        CascadeClassifier eyeCascade = new CascadeClassifier("haarcascade_eye.xml");

        CascadeClassifier bodyCascade = new CascadeClassifier("haarcascade_fullbody.xml");
        #endregion
        #endregion
        public Form1()
        {
            InitializeComponent();
            Slider_Enable(false); 

            DsDevice[] _SystemCamereas = DsDevice.GetDevicesOfCat(FilterCategory.VideoInputDevice);
            WebCams = new VideoDevice[_SystemCamereas.Length];
            for (int i = 0; i < _SystemCamereas.Length; i++)
            {
                WebCams[i] = new VideoDevice(i, _SystemCamereas[i].Name, _SystemCamereas[i].ClassID); //fill web cam array
                Camera_Selection.Items.Add(WebCams[i].ToString());
            }
            if (Camera_Selection.Items.Count > 0)
            {
                Camera_Selection.SelectedIndex = 0; //Set the selected device the default
                captureButton.Enabled = true; //Enable the start
            }
        }

        private void Slider_Enable(bool State)
        {
            /*Brigtness_SLD.Enabled = State;
            Contrast_SLD.Enabled = State;
            Sharpness_SLD.Enabled = State;*/
        }

        private void CaptureButton_Click(object sender, EventArgs e)
        {
            if (_capture != null)
            {
                if (_captureInProgress)
                {
                    //stop the capture
                    captureButton.Text = "Start Capture"; //Change text on button
                    Slider_Enable(false);
                    _capture.Pause(); //Pause the capture
                    _captureInProgress = false; //Flag the state of the camera
                }
                else
                {
                    captureButton.Text = "Stop"; //Change text on button
                    Slider_Enable(true);  //Enable User Controls
                    _capture.Start(); //Start the capture
                    _captureInProgress = true; //Flag the state of the camera
                }

            }
            else
            {
                //set up capture with selected device
                SetupCapture(Camera_Selection.SelectedIndex);
                //Be lazy and Recall this method to start camera
                CaptureButton_Click(null, null);
            }
        }

        private void SetupCapture(int Camera_Identifier)
        {
            //update the selected device
            CameraDevice = Camera_Identifier;

            //Dispose of Capture if it was created before
            if (_capture != null) _capture.Dispose();
            try
            {
                //Set up capture device
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
            Mat frame = new Mat();//;
            Mat frame1 = new Mat();//;
            _capture.Retrieve(frame);
            _capture.Retrieve(frame1);
            var imageToDisplay = frame.ToImage<Bgr, byte>();

            var faces = faceCascade.DetectMultiScale(frame.ToImage<Gray, byte>(), 1.1, 10, Size.Empty); //the actual face detection happens here
            foreach (var face in faces)
            {
                imageToDisplay.Draw(face, new Bgr(Color.BurlyWood), 3); //the detected face(s) is highlighted here using a box that is drawn around it/them
            }

            var eyes = eyeCascade.DetectMultiScale(frame.ToImage<Gray, byte>(), 1.1, 10, Size.Empty); //the actual eye detection happens here
            foreach (var eye in eyes)
            {
                imageToDisplay.Draw(eye, new Bgr(Color.AliceBlue), 3); //the detected face(s) is highlighted here using a box that is drawn around it/them
            }

            var des = new HOGDescriptor();
            des.SetSVMDetector(HOGDescriptor.GetDefaultPeopleDetector());
            foreach (var pedestrain in des.DetectMultiScale(imageToDisplay, 0, new Size(8, 8), new Size(0, 0)).AsEnumerable())
            {
                imageToDisplay.Draw(pedestrain.Rect, new Bgr(Color.Red), 1);
            }
            DisplayImage(imageToDisplay.ToBitmap());
        }

        private delegate void DisplayImageDelegate(Bitmap Image);

        private void DisplayImage(Bitmap Image)
        {
            if (captureBox.InvokeRequired)
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
                captureBox.Image = Image;
            }
        }
    }
}
