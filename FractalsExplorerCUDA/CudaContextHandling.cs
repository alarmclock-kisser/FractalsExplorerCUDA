
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace FractalsExplorerCUDA
{
	public class CudaContextHandling
	{
		// ----- ----- ATTRIBUTES ----- ----- \\
		private string Repopath;
		private ListBox LogList;
		private ComboBox DevicesCombo;
		public ComboBox KernelsCombo;
		public ProgressBar VramBar;

		public int Index = -1;
		public CUdevice? Device = null;
		public PrimaryContext? Context = null;

		public CudaMemoryHandling? MemoryH;
		public CudaKernelHandling? KernelH;

		// ----- ----- CONSTRUCTORS ----- ----- \\
		public CudaContextHandling(string repopath, ListBox listBox_log, ComboBox comboBox_devices, ComboBox comboBox_kernels, ProgressBar? progressBar_Vram = null)
		{
			this.Repopath = repopath;
			this.LogList = listBox_log;
			this.DevicesCombo = comboBox_devices;
			this.KernelsCombo = comboBox_kernels;
			this.VramBar = progressBar_Vram ?? new ProgressBar();

			// Register events
			this.DevicesCombo.SelectedIndexChanged += (s, e) => this.InitDevice(this.DevicesCombo.SelectedIndex);

			// Fill devices combobox
			this.FillDevicesCombobox();
		}




		// ----- ----- METHODS ----- ----- \\
		public string Log(string message = "", string inner = "", int indent = 0)
		{
			string indentString = new string('~', indent);
			string logMessage = $"[Ctx] {indentString}{message} ({inner})";
			this.LogList.Items.Add(logMessage);
			this.LogList.TopIndex = this.LogList.Items.Count - 1;
			return logMessage;
		}


		public int GetDeviceCount()
		{
			// Trycatch

			int deviceCount = 0;

			try
			{
				deviceCount = CudaContext.GetDeviceCount();
			}
			catch (CudaException ex)
			{
				this.Log("Couldn't get device count", ex.Message, 1);
			}

			return deviceCount;
		}

		public List<CUdevice> GetDevices()
		{
			List<CUdevice> devices = [];
			int deviceCount = this.GetDeviceCount();

			for (int i = 0; i < deviceCount; i++)
			{
				// Trycatch
				try
				{
					CUdevice device = new(i);
					devices.Add(device);
				}
				catch (CudaException ex)
				{
					this.Log("Couldn't get device # " + i, ex.Message, 1);
				}
				catch (Exception ex)
				{
					this.Log("Couldn't get device # " + i, ex.Message, 1);
				}
				finally
				{
					if (devices.Count == 0)
					{
						this.Log("No devices found", "", 1);
					}
				}
			}

			return devices;
		}

		public Version GetCapability(int index = -1)
		{
			index = index == -1 ? this.Index : index;

			Version ver = new(0, 0);

			try
			{
				ver = CudaContext.GetDeviceComputeCapability(index);
			}
			catch (CudaException ex)
			{
				this.Log("Couldn't get device capability", ex.Message, 1);
			}

			return ver;
		}

		public string GetName(int index = -1)
		{
			index = index == -1 ? this.Index : index;

			string name = "N/A";

			try
			{
				name = CudaContext.GetDeviceName(index);
			}
			catch (CudaException ex)
			{
				this.Log("Couldn't get device name", ex.Message, 1);
			}

			return name;
		}

		public void FillDevicesCombobox(ComboBox? comboBox = null)
		{
			comboBox ??= this.DevicesCombo;
			comboBox.Items.Clear();

			List<CUdevice> devices = this.GetDevices();
			for (int i = 0; i < devices.Count; i++)
			{
				CUdevice device = devices[i];
				string deviceName = CudaContext.GetDeviceName(i);
				Version capability = this.GetCapability(i);
				comboBox.Items.Add($"{deviceName} ({capability.Major}.{capability.Minor})");
			}
		}

		public void InitDevice(int index = -1)
		{
			this.Dispose();

			index = index == -1 ? this.Index : index;
			if (index < 0 || index >= this.GetDeviceCount())
			{
				this.Log("Invalid device id", "Out of range");
				return;
			}			

			this.Index = index;
			this.Device = new CUdevice(index);
			this.Context = new PrimaryContext(this.Device.Value);
			this.Context.SetCurrent();
			this.MemoryH = new CudaMemoryHandling(this.Repopath, this.LogList, this.Context, this.VramBar);
			this.KernelH = new CudaKernelHandling(this.Repopath, this.LogList, this.Context, this.MemoryH, this.KernelsCombo);

			this.Log($"Initialized #{index}", "'" + this.GetName() + "'");

		}

		public void Dispose()
		{
			this.Context?.Dispose();
			this.Context = null;
			this.Device = null;
			this.MemoryH?.Dispose();
			this.MemoryH = null;
			this.KernelH?.Dispose();
			this.KernelH = null;
		}

	}
}