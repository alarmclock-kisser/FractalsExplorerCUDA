using System.Diagnostics;

namespace FractalsExplorerCUDA
{
	public partial class WindowMain : Form
	{
		// ----- ----- ----- ATTRIBUTES ----- ----- ----- \\
		public string Repopath;

		public CudaContextHandling CTXH;
		public GuiBuilder GUIB;
		public ImageRecorder REC;
		public ImageHandling IMGH;





		private Dictionary<NumericUpDown, int> previousNumericValues = [];
		private bool isProcessing;
		private Form? fullScreenForm;
		private bool isDragging;
		private Point mouseDownLocation;
		private Dictionary<string, object> currentOverlayArgs = [];
		private bool ctrlKeyPressed;
		private bool kernelExecutionRequired;
		private Single mandelbrotZoomFactor;
		private Stopwatch? stopwatch;

		// ----- ----- ----- CONSTRUCTORS ----- ----- ----- \\
		public WindowMain()
		{
			this.InitializeComponent();

			// Set repopath
			this.Repopath = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, @"..\..\..\"));

			// Init CUDA
			this.CTXH = new CudaContextHandling(this.Repopath, this.listBox_log, this.comboBox_devices, this.comboBox_kernels, this.progressBar_vram);

			// Init GUIB
			this.GUIB = new GuiBuilder(this.Repopath, this.listBox_log, this.CTXH, this.panel_kernelArgs, this.checkBox_silent);

			// Init REC
			this.REC = new ImageRecorder(this.Repopath, this.label_cached);

			// Init IMGH
			this.IMGH = new ImageHandling(this.Repopath, this.listBox_images, this.pictureBox_view, null, this.label_meta);

			// REGISTER EVENTS
			this.comboBox_kernels.SelectedIndexChanged += (s, e) => this.LoadKernel(this.comboBox_kernels.SelectedItem?.ToString() ?? "");
			this.pictureBox_view.DoubleClick += (s, e) => this.fullScreen_DoubleClick(s, e);
			this.pictureBox_view.MouseDown += this.pictureBox_view_MouseDown;
			this.pictureBox_view.MouseMove += this.pictureBox_view_MouseMove;
			this.pictureBox_view.MouseUp += this.pictureBox_view_MouseUp;
			this.pictureBox_view.MouseWheel += this.pictureBox_view_MouseWheel;
			this.pictureBox_view.Paint += this.PictureBox_view_Paint;

			// Select first device if available & toggle dark mode on
			if (this.comboBox_devices.Items.Count > 0)
			{
				this.comboBox_devices.SelectedIndex = 0;
			}
			this.checkBox_darkMode.Checked = true;
		}







		// ----- ----- ----- METHODS ----- ----- ----- \\
		public void RenderArgsIntoPicturebox()
		{
			object[] argValues = this.GUIB.GetArgumentValues();
			string[] argNames = this.GUIB.GetArgumentNames();
			Dictionary<string, object> args = argValues
				.Select((value, index) => new { Name = argNames[index], Value = value })
				.ToDictionary(x => x.Name, x => x.Value);

			// Filter args (wie in deinem Code)
			args = args.Where(x => !x.Key.ToLower().Contains("input") &&
								  !x.Key.ToLower().Contains("output") &&
								  !x.Key.ToLower().Contains("pixel") &&
								  !x.Key.ToLower().Contains("width") &&
								  !x.Key.ToLower().Contains("height") &&
								  !x.Key.EndsWith("R") &&
								  !x.Key.EndsWith("G") &&
								  !x.Key.EndsWith("B"))
				.ToDictionary(x => x.Key, x => x.Value);

			// Optional: Add recording and frame id
			if (this.REC.CachedImages.Count > 0)
			{
				args.Add("Recording", this.REC.CachedImages.Count);
			}



			// Speichere args für Overlay-Zeichnung im Paint-Event
			this.currentOverlayArgs = args;

			// PictureBox neu zeichnen (ruft PictureBox_view_Paint auf)
			this.pictureBox_view.Invalidate();
		}

		public void MoveImage(int index = -1, bool refresh = true)
		{
			// Abort if CTRL down
			if (ModifierKeys == Keys.Control)
			{
				return;
			}

			if (index == -1 && this.IMGH.CurrentObject != null)
			{
				index = this.IMGH.Images.IndexOf(this.IMGH.CurrentObject);
			}

			if (index < 0 && index >= this.IMGH.Images.Count)
			{
				MessageBox.Show("Invalid index", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}

			ImageObject image = this.IMGH.Images[index];

			// Move image Host <-> CUDA
			if (image.OnHost)
			{
				// Move to CUDA: Get bytes
				byte[] bytes = image.GetPixelsAsBytes(false);

				// STOPWATCH
				Stopwatch sw = Stopwatch.StartNew();

				// Create buffer
				IntPtr pointer = this.CTXH.MemoryH?.PushData(bytes, this.checkBox_silent.Checked) ?? 0;

				// STOPWATCH
				sw.Stop();
				//this.label_pushTime.Text = $"Push time: {sw.ElapsedMilliseconds} ms";

				// Check pointer
				if (pointer == 0)
				{
					MessageBox.Show("Failed to push data to CUDA", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
					return;
				}

				// Set pointer, void image
				image.Pointer = pointer;
				image.Img = new Bitmap(1, 1);
			}
			else if (image.OnDevice)
			{
				// STOPWATCH
				Stopwatch sw = Stopwatch.StartNew();

				// Move to Host
				byte[] bytes = this.CTXH.MemoryH?.PullData<byte>(image.Pointer, true, this.checkBox_silent.Checked) ?? [];
				if (bytes.Length == 0)
				{
					MessageBox.Show("Failed to pull data from CUDA", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
					return;
				}

				// STOPWATCH
				sw.Stop();
				//this.label_pullTime.Text = $"Pull time: {sw.ElapsedMilliseconds} ms";

				// Create image
				image.SetImageFromBytes(bytes, true);
			}

			// Refill list
			if (refresh)
			{
				this.IMGH.FillImagesListBox();
			}
		}

		public void LoadKernel(string kernelName = "")
		{
			// Load kernel
			this.CTXH.KernelH?.LoadKernel(kernelName, this.checkBox_silent.Checked);
			if (this.CTXH.KernelH?.Kernel == null)
			{
				MessageBox.Show("Failed to load kernel", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}

			// Get arguments
			this.GUIB.BuildPanel(0.48f, true);

		}

		public void ExecuteKernelOOP(int index = -1, string kernelName = "")
		{
			// If index is -1, use current object
			if (index == -1 && this.IMGH.CurrentObject != null)
			{
				index = this.IMGH.Images.IndexOf(this.IMGH.CurrentObject);
			}

			if (index < 0 || index >= this.IMGH.Images.Count)
			{
				MessageBox.Show("Invalid index", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}

			// Get image
			ImageObject? image = this.IMGH.Images[index];
			if (image == null)
			{
				MessageBox.Show("Invalid image", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}

			// Verify image on device
			bool moved = false;
			if (image.OnHost)
			{
				this.MoveImage(index);
				if (image.OnHost)
				{
					MessageBox.Show("Could not move image to device", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
					return;
				}

				moved = true;
			}

			// STOPWATCH
			Stopwatch sw = Stopwatch.StartNew();

			// Load kernel
			this.CTXH.KernelH?.LoadKernel(kernelName, this.checkBox_silent.Checked);
			if (this.CTXH.KernelH?.Kernel == null)
			{
				MessageBox.Show("Failed to load kernel", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}

			// Get image attributes for kernel call
			IntPtr pointer = image.Pointer;
			int width = image.Width;
			int height = image.Height;
			int channels = 4;
			int bitdepth = image.BitsPerPixel / channels;

			// Get variable arguments
			object[] args = this.GUIB.GetArgumentValues();

			// Call exec kernel -> pointer
			image.Pointer = this.CTXH.KernelH?.ExecuteKernel(pointer, width, height, channels, bitdepth, args, this.checkBox_silent.Checked) ?? image.Pointer;

			// STOPWATCH
			sw.Stop();
			//this.label_execTime.Text = $"Exec. time: {sw.ElapsedMilliseconds} ms";

			// Optional: Move back to host
			if (moved)
			{
				this.MoveImage(index);
			}

			// If kernel is Mandelbrot, cache image with interval
			if (image.Img != null && (this.checkBox_record.Checked || IsKeyLocked(Keys.CapsLock)))
			{
				this.REC.AddImage(image.Img, sw.ElapsedMilliseconds);
			}

			// Reset cache if checkbox is unchecked || cache isnt empty || not CAPS locked
			if (!this.checkBox_record.Checked && this.REC.CachedImages.Count != 0 && !IsKeyLocked(Keys.CapsLock))
			{
				this.REC.CachedImages.Clear();
				this.REC.CountLabel.Text = $"Images: -";
			}

			// Refill list
			this.IMGH.FillImagesListBox();
		}




		// ----- ----- ----- PRIVATES ----- ----- ----- \\
		private void pictureBox_view_MouseDown(object? sender, MouseEventArgs e)
		{
			if (e.Button == MouseButtons.Left)
			{
				this.isDragging = true;
				this.mouseDownLocation = e.Location;
			}
		}

		private void pictureBox_view_MouseMove(object? sender, MouseEventArgs e)
		{
			if (!this.isDragging)
			{
				return;
			}

			try
			{
				// 1. Find NumericUpDown controls more efficiently
				NumericUpDown? numericX = this.panel_kernelArgs.Controls.OfType<NumericUpDown>().FirstOrDefault(x => x.Name.ToLower().Contains("offsetx"));
				NumericUpDown? numericY = this.panel_kernelArgs.Controls.OfType<NumericUpDown>().FirstOrDefault(x => x.Name.ToLower().Contains("offsety"));
				NumericUpDown? numericZ = this.panel_kernelArgs.Controls.OfType<NumericUpDown>().FirstOrDefault(x => x.Name.ToLower().Contains("zoom"));
				NumericUpDown? numericMouseX = this.panel_kernelArgs.Controls.OfType<NumericUpDown>().FirstOrDefault(x => x.Name.ToLower().Contains("mousex"));
				NumericUpDown? numericMouseY = this.panel_kernelArgs.Controls.OfType<NumericUpDown>().FirstOrDefault(x => x.Name.ToLower().Contains("mousey"));

				if (!(numericX == null || numericY == null || numericZ == null))
				{
					// 2. Calculate smooth delta with sensitivity factor and zoom
					float sensitivity = 0.001f * (float) (1 / numericZ.Value);
					decimal deltaX = (decimal) ((e.Location.X - this.mouseDownLocation.X) * -sensitivity);
					decimal deltaY = (decimal) ((e.Location.Y - this.mouseDownLocation.Y) * -sensitivity);

					// 3. Update values with boundary checks
					this.UpdateNumericValue(numericX, deltaX);
					this.UpdateNumericValue(numericY, deltaY);
				}

				// 4. Update mouse position for smoother continuous dragging
				this.mouseDownLocation = e.Location;

				// 5. Update mouse coordinates in NumericUpDown controls
				if (!(numericMouseX == null || numericMouseY == null))
				{
					this.UpdateNumericValue(numericMouseX, e.Location.X);
					this.UpdateNumericValue(numericMouseY, e.Location.Y);
				}
			}
			catch (Exception ex)
			{
				Debug.WriteLine($"MouseMove error: {ex.Message}");
			}
		}

		private void pictureBox_view_MouseUp(object? sender, MouseEventArgs e)
		{
			if (e.Button == MouseButtons.Left)
			{
				this.isDragging = false;

				// Re-execute kernel
				this.button_exec_Click(sender, e);

				this.RenderArgsIntoPicturebox();
			}
		}

		private void pictureBox_view_MouseWheel(object? sender, MouseEventArgs e)
		{
			// Check for CTRL key press
			if (Control.ModifierKeys == Keys.Control)
			{
				this.ctrlKeyPressed = true;
				this.kernelExecutionRequired = true; // Set the flag
				NumericUpDown? numericI = this.panel_kernelArgs.Controls.OfType<NumericUpDown>().FirstOrDefault(x => x.Name.ToLower().Contains("iter"));
				if (numericI == null)
				{
					this.CTXH.KernelH?.Log("MaxIter control not found!", "", 3);
					return;
				}

				// Increase/Decrease maxIter
				if (e.Delta > 0)
				{
					numericI.Value += 2;
				}
				else if (e.Delta < 0)
				{
					if (numericI.Value > 0)
					{
						numericI.Value -= 2;
					}
				}
				return;
			}

			// Check if CTRL key was previously pressed
			if (this.ctrlKeyPressed)
			{
				this.ctrlKeyPressed = false; // Reset the flag
				this.kernelExecutionRequired = true;
			}

			// 1. Find NumericUpDown controls more efficiently
			NumericUpDown? numericZ = this.panel_kernelArgs.Controls.OfType<NumericUpDown>().FirstOrDefault(x => x.Name.ToLower().Contains("zoom"));
			if (numericZ == null)
			{
				this.CTXH.KernelH?.Log("Zoom control not found!", "", 3);
				return;
			}

			// 2. Calculate zoom factor
			if (e.Delta > 0)
			{
				this.mandelbrotZoomFactor *= 1.1f;
			}
			else
			{
				this.mandelbrotZoomFactor /= 1.1f;
			}

			// 3. Update zoom value with boundary checks
			decimal newValue = (decimal) this.mandelbrotZoomFactor;
			if (newValue < numericZ.Minimum)
			{
				newValue = numericZ.Minimum;
			}
			if (newValue > numericZ.Maximum)
			{
				newValue = numericZ.Maximum;
			}
			numericZ.Value = newValue;

			// Call re-exec kernel
			this.kernelExecutionRequired = true;

			if (!this.ctrlKeyPressed && this.kernelExecutionRequired)
			{
				this.kernelExecutionRequired = false;
				this.button_exec_Click(sender, e);
				this.RenderArgsIntoPicturebox();
			}
		}

		private void PictureBox_view_Paint(object? sender, PaintEventArgs e)
		{
			PictureBox? pbox = sender as PictureBox;
			if (pbox == null)
			{
				return;
			}

			// Erstens: Basis-Image zeichnen (falls noch nicht automatisch)
			if (pbox.Image != null)
			{
				e.Graphics.DrawImage(pbox.Image, new Point(0, 0));
			}

			// Overlay nur zeichnen, wenn MandelbrotMode aktiv ist und Overlay-Daten vorhanden sind
			if (currentOverlayArgs != null && currentOverlayArgs.Count > 0)
			{
				// Overlay erstellen - Größe auf PictureBox-Größe oder nach Wunsch
				Size overlaySize = new Size(pbox.Width / 8, pbox.Height / 8);

				// Overlay vom GuiBuilder holen (erwartet: CreateOverlayBitmap(Size, Dictionary, ...))
				Bitmap overlay = this.GUIB.CreateOverlayBitmap(overlaySize, currentOverlayArgs, fontSize: 12, color: Color.White, imageSize: pbox.Size);

				// Overlay transparent zeichnen an gewünschter Position (z.B. oben links)
				e.Graphics.DrawImageUnscaled(overlay, new Point(10, 10));

				overlay.Dispose();
			}
		}



		private void RegisterNumericToSecondPow(NumericUpDown numeric)
		{
			// Initialwert speichern
			this.previousNumericValues.Add(numeric, (int) numeric.Value);

			numeric.ValueChanged += (s, e) =>
			{
				// Verhindere rekursive Aufrufe
				if (this.isProcessing)
				{
					return;
				}

				this.isProcessing = true;

				try
				{
					int newValue = (int) numeric.Value;
					int oldValue = this.previousNumericValues[numeric];
					int max = (int) numeric.Maximum;
					int min = (int) numeric.Minimum;

					// Nur verarbeiten, wenn sich der Wert tats chlich ge ndert hat
					if (newValue != oldValue)
					{
						int calculatedValue;

						if (newValue > oldValue)
						{
							// Verdoppeln, aber nicht  ber Maximum
							calculatedValue = Math.Min(oldValue * 2, max);
						}
						else if (newValue < oldValue)
						{
							// Halbieren, aber nicht unter Minimum
							calculatedValue = Math.Max(oldValue / 2, min);
						}
						else
						{
							calculatedValue = oldValue;
						}

						// Nur aktualisieren wenn notwendig
						if (calculatedValue != newValue)
						{
							numeric.Value = calculatedValue;
						}

						this.previousNumericValues[numeric] = calculatedValue;
					}
				}
				finally
				{
					this.isProcessing = false;
				}
			};
		}

		private void UpdateNumericValue(NumericUpDown numeric, decimal delta)
		{
			decimal newValue = numeric.Value + delta;

			// Ensure value stays within allowed range
			if (newValue < numeric.Minimum)
			{
				newValue = numeric.Minimum;
			}

			if (newValue > numeric.Maximum)
			{
				newValue = numeric.Maximum;
			}

			numeric.Value = newValue;
		}

		private void fullScreen_DoubleClick(object? sender, EventArgs e)
		{
			if (this.fullScreenForm != null)
			{
				// Bereits aktiv, also schließen
				this.fullScreenForm.Close();
				this.fullScreenForm = null;
				return;
			}

			// Neues Fullscreen-Form erstellen
			this.fullScreenForm = new Form
			{
				FormBorderStyle = FormBorderStyle.None,
				TopMost = true,
				WindowState = FormWindowState.Maximized,
				BackColor = Color.Black,
				StartPosition = FormStartPosition.Manual,
				KeyPreview = true
			};

			// PictureBox in Originalgröße oder gestreckt anzeigen
			PictureBox pb = new()
			{
				SizeMode = PictureBoxSizeMode.AutoSize,
				Image = this.pictureBox_view.Image,
				BackColor = Color.Black
			};

			// Reuse all existing event handlers
			pb.MouseDown += this.pictureBox_view_MouseDown!;
			pb.MouseMove += this.pictureBox_view_MouseMove!;
			pb.MouseUp += this.pictureBox_view_MouseUp!;
			pb.MouseWheel += this.pictureBox_view_MouseWheel!;
			pb.Paint += this.PictureBox_view_Paint!;
			pb.Focus(); // Damit MouseWheel funktioniert



			this.fullScreenForm.Controls.Add(pb);

			// ESC beenden
			this.fullScreenForm.KeyDown += (s, args) =>
			{
				if (args.KeyCode == Keys.Escape)
				{
					this.fullScreenForm?.Close();
					this.fullScreenForm = null;
				}
			};


			this.fullScreenForm.Show();
			this.fullScreenForm.Focus(); // wichtig für ESC
		}



		// ----- ----- ----- EVENT HANDLERS ----- ----- ----- \\
		private void checkBox_darkMode_CheckedChanged(object sender, EventArgs e)
		{
			DarkModeToggle.ToggleDarkMode(this, this.checkBox_darkMode.Checked);
		}

		private void button_exec_Click(object? sender, EventArgs e)
		{
			string kernelName = this.comboBox_kernels.SelectedItem?.ToString() ?? "";

			// If CTRL down: Execute on all images
			if (ModifierKeys == Keys.Control)
			{
				// this.ExecuteKernelOOPAll(kernelName);
				return;
			}

			this.ExecuteKernelOOP(-1, this.comboBox_kernels.SelectedItem?.ToString() ?? "");
		}

		private void button_create_Click(object sender, EventArgs e)
		{

		}

		private void button_import_Click(object sender, EventArgs e)
		{

		}

		private void button_export_Click(object sender, EventArgs e)
		{

		}
	}
}
