using System;
using System.Collections.Generic;
using System.Drawing.Imaging;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace FractalsExplorerCUDA
{
	public class ImageRecorder
	{
		// ----- ----- ATTRIBUTES ----- ----- \\
		public string Repopath;
		public Label CountLabel;


		public List<Image> CachedImages = [];
		public List<long> CachedIntervalls = [];



		// ----- ----- CONSTRUCTORS ----- ----- \\
		public ImageRecorder(string repopath, Label? countLabel = null)
		{
			// Set attributes
			this.Repopath = repopath;
			this.CountLabel = countLabel ?? new Label();

			// Reset cache
			this.ResetCache();
		}




		// ----- ----- METHODS ----- ----- \\
		public void ResetCache()
		{
			this.CachedImages.Clear();
			this.CachedIntervalls.Clear();

			// GC
			GC.Collect();
			GC.WaitForPendingFinalizers();

			// Update label
			this.CountLabel.Text = $"Images: -";
		}


		public void AddImage(Image image, long interval)
		{
			this.CachedImages.Add(image);
			this.CachedIntervalls.Add(interval);

			// Update label
			this.CountLabel.Text = $"Images: {this.CachedImages.Count}";
		}

		public string CreateGif(string folder = "", string name = "cudabitmaps", int frameRate = 5, bool doLoop = false)
		{
			if (this.CachedImages.Count == 0)
			{
				return "";
			}

			if (string.IsNullOrEmpty(folder))
			{
				folder = Path.Combine(this.Repopath, "Resources", "ExportedGifs");
			}
			if (!Directory.Exists(folder))
			{
				Directory.CreateDirectory(folder);
			}

			string baseFile = Path.Combine(folder, name + ".gif");
			string file = baseFile;
			int fileIndex = 1;
			while (File.Exists(file))
			{
				file = Path.Combine(folder, $"{name}_{fileIndex:D3}.gif");
				fileIndex++;
			}

			// Delay in 1/100 Sekunden
			int delay = 100 / frameRate;
			byte[] delayBytes = new byte[this.CachedImages.Count * 4];
			for (int i = 0; i < this.CachedImages.Count; i++)
			{
				delayBytes[i * 4 + 0] = (byte) (delay & 0xFF);
				delayBytes[i * 4 + 1] = (byte) ((delay >> 8) & 0xFF);
				delayBytes[i * 4 + 2] = 0;
				delayBytes[i * 4 + 3] = 0;
			}

			// Delay PropertyItem
			PropertyItem delayItem = (PropertyItem) FormatterServices.GetUninitializedObject(typeof(PropertyItem));
			delayItem.Id = 0x5100;
			delayItem.Type = 4;
			delayItem.Len = delayBytes.Length;
			delayItem.Value = delayBytes;

			// Loop PropertyItem
			PropertyItem loopItem = (PropertyItem) FormatterServices.GetUninitializedObject(typeof(PropertyItem));
			loopItem.Id = 0x5101;
			loopItem.Type = 1;
			loopItem.Len = 4;
			loopItem.Value = doLoop ? [0, 0, 0, 0] : [1, 0, 0, 0];

			ImageCodecInfo gifEncoder = ImageCodecInfo.GetImageEncoders().First(e => e.MimeType == "image/gif");
			System.Drawing.Imaging.Encoder encoder = System.Drawing.Imaging.Encoder.SaveFlag;
			EncoderParameters encParams = new EncoderParameters(1);

			using (Bitmap firstFrame = new(this.CachedImages[0]))
			{
				firstFrame.SetPropertyItem(delayItem);
				firstFrame.SetPropertyItem(loopItem);

				encParams.Param[0] = new EncoderParameter(encoder, (long) EncoderValue.MultiFrame);
				firstFrame.Save(file, gifEncoder, encParams);

				encParams.Param[0] = new EncoderParameter(encoder, (long) EncoderValue.FrameDimensionTime);
				for (int i = 1; i < this.CachedImages.Count; i++)
				{
					using Bitmap frame = new(this.CachedImages[i]);
					frame.SetPropertyItem(delayItem);
					firstFrame.SaveAdd(frame, encParams);
				}

				encParams.Param[0] = new EncoderParameter(encoder, (long) EncoderValue.Flush);
				firstFrame.SaveAdd(encParams);
			}

			// Dispose of the images
			foreach (Image img in this.CachedImages)
			{
				img.Dispose();
			}
			this.CachedImages.Clear();
			this.CachedIntervalls.Clear();
			this.CountLabel.Text = $"Images: -";

			return file;
		}

		public async Task<string> CreateGifAsync(string folder = "", string name = "animated_", int frameRate = 5, bool doLoop = false, Size? resize = null)
		{
			// Sicherheitsprüfung
			if (this.CachedImages.Count == 0)
			{
				return "";
			}

			// Zielordner festlegen
			string baseFolder = string.IsNullOrWhiteSpace(folder)
				? Path.Combine(this.Repopath, "Resources", "ExportedGifs")
				: folder;

			Directory.CreateDirectory(baseFolder);

			// Einzigartigen Dateinamen erzeugen
			string baseName = Path.Combine(baseFolder, name + ".gif");
			string file = baseName;
			int counter = 1;
			while (File.Exists(file))
			{
				file = Path.Combine(baseFolder, $"{name}_{counter:D3}.gif");
				counter++;
			}

			// Delay berechnen (in 1/100 Sek)
			int delay = 100 / Math.Max(1, frameRate);

			// Parallel resizen (wenn notwendig)
			Bitmap[] frames = await Task.Run(() =>
			{
				return this.CachedImages
					.AsParallel()
					.AsOrdered() // Reihenfolge beibehalten
					.Select(img =>
					{
						if (resize.HasValue)
						{
							Bitmap bmp = new Bitmap(resize.Value.Width, resize.Value.Height);
							using (Graphics g = Graphics.FromImage(bmp))
							{
								g.DrawImage(img, 0, 0, resize.Value.Width, resize.Value.Height);
							}
							return bmp;
						}
						else
						{
							return new Bitmap(img);
						}
					})
					.ToArray();
			});

			// Delay PropertyItem erzeugen
			PropertyItem delayItem = (PropertyItem) FormatterServices.GetUninitializedObject(typeof(PropertyItem));
			delayItem.Id = 0x5100;
			delayItem.Type = 4; // LONG
			delayItem.Len = 4;
			delayItem.Value = BitConverter.GetBytes(delay);

			// Looping PropertyItem
			PropertyItem loopItem = (PropertyItem) FormatterServices.GetUninitializedObject(typeof(PropertyItem));
			loopItem.Id = 0x5101;
			loopItem.Type = 1; // BYTE
			loopItem.Len = 4;
			loopItem.Value = doLoop ? [0, 0, 0, 0] : [1, 0, 0, 0];

			// Encoder setup
			ImageCodecInfo gifEncoder = ImageCodecInfo.GetImageEncoders().First(e => e.MimeType == "image/gif");
			System.Drawing.Imaging.Encoder encoder = System.Drawing.Imaging.Encoder.SaveFlag;
			EncoderParameters encParams = new EncoderParameters(1);

			await Task.Run(() =>
			{
				using Bitmap first = frames[0];
				first.SetPropertyItem(delayItem);
				first.SetPropertyItem(loopItem);

				encParams.Param[0] = new EncoderParameter(encoder, (long) EncoderValue.MultiFrame);
				first.Save(file, gifEncoder, encParams);

				encParams.Param[0] = new EncoderParameter(encoder, (long) EncoderValue.FrameDimensionTime);
				for (int i = 1; i < frames.Length; i++)
				{
					using Bitmap frame = frames[i];
					frame.SetPropertyItem(delayItem);
					first.SaveAdd(frame, encParams);
				}

				encParams.Param[0] = new EncoderParameter(encoder, (long) EncoderValue.Flush);
				first.SaveAdd(encParams);
			});

			// Dispose of the images
			this.CachedImages.ForEach(img => img.Dispose());
			this.CachedImages.Clear();
			this.CachedIntervalls.Clear();
			this.CountLabel.Text = $"Images: -";

			return file;
		}

		public async Task<string> CreateGifAsync(string folder = "", string name = "animated_", int frameRate = 5, bool doLoop = false, Size? resize = null, ProgressBar? pBar = null)
		{
			// Sicherheitsprüfung
			if (this.CachedImages.Count == 0)
			{
				return "";
			}

			// ProgressBar initialisieren
			if (pBar != null)
			{
				pBar.Minimum = 0;
				pBar.Maximum = 100;
				pBar.Value = 0;
			}

			// Zielordner festlegen
			string baseFolder = string.IsNullOrWhiteSpace(folder)
				? Path.Combine(this.Repopath, "Resources", "ExportedGifs")
				: folder;

			Directory.CreateDirectory(baseFolder);

			// Einzigartigen Dateinamen erzeugen
			string baseName = Path.Combine(baseFolder, name + ".gif");
			string file = baseName;
			int counter = 1;
			while (File.Exists(file))
			{
				file = Path.Combine(baseFolder, $"{name}_{counter:D3}.gif");
				counter++;
			}

			// Delay berechnen (in 1/100 Sek)
			int delay = 100 / Math.Max(1, frameRate);

			// Parallel resizen (wenn notwendig)
			Bitmap[] frames = await Task.Run(() =>
			{
				var result = new Bitmap[this.CachedImages.Count];
				int totalImages = this.CachedImages.Count;
				int processedCount = 0;
				object progressLock = new object();

				if (pBar != null && !resize.HasValue)
				{
					pBar.Invoke((MethodInvoker) (() => pBar.Value = 33));
				}

				Parallel.For(0, totalImages, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, i =>
				{
					Bitmap bmp;
					if (resize.HasValue)
					{
						bmp = new Bitmap(resize.Value.Width, resize.Value.Height);
						using Graphics g = Graphics.FromImage(bmp);
						g.DrawImage(this.CachedImages[i], 0, 0, resize.Value.Width, resize.Value.Height);
					}
					else
					{
						bmp = new Bitmap(this.CachedImages[i]);
					}

					result[i] = bmp;

					if (pBar != null && resize.HasValue)
					{
						lock (progressLock)
						{
							processedCount++;
							int progress = (int) (33 * processedCount / (double) totalImages);
							pBar.Invoke((MethodInvoker) (() => pBar.Value = progress));
						}
					}
				});

				return result;
			});

			// Delay PropertyItem erzeugen
			PropertyItem delayItem = (PropertyItem) FormatterServices.GetUninitializedObject(typeof(PropertyItem));
			delayItem.Id = 0x5100;
			delayItem.Type = 4; // LONG
			delayItem.Len = 4;
			delayItem.Value = BitConverter.GetBytes(delay);

			// Looping PropertyItem
			PropertyItem loopItem = (PropertyItem) FormatterServices.GetUninitializedObject(typeof(PropertyItem));
			loopItem.Id = 0x5101;
			loopItem.Type = 1; // BYTE
			loopItem.Len = 4;
			loopItem.Value = doLoop ? [0, 0, 0, 0] : [1, 0, 0, 0];

			// Encoder setup
			ImageCodecInfo gifEncoder = ImageCodecInfo.GetImageEncoders().First(e => e.MimeType == "image/gif");
			System.Drawing.Imaging.Encoder encoder = System.Drawing.Imaging.Encoder.SaveFlag;
			EncoderParameters encParams = new EncoderParameters(1);

			await Task.Run(() =>
			{
				using Bitmap first = frames[0];
				first.SetPropertyItem(delayItem);
				first.SetPropertyItem(loopItem);

				encParams.Param[0] = new EncoderParameter(encoder, (long) EncoderValue.MultiFrame);
				first.Save(file, gifEncoder, encParams);

				encParams.Param[0] = new EncoderParameter(encoder, (long) EncoderValue.FrameDimensionTime);
				for (int i = 1; i < frames.Length; i++)
				{
					using Bitmap frame = frames[i];
					frame.SetPropertyItem(delayItem);
					first.SaveAdd(frame, encParams);

					// ProgressBar von 33% bis 100% erhöhen
					if (pBar != null)
					{
						int progress = 33 + (int) (67 * i / (double) frames.Length);
						pBar.Invoke((MethodInvoker) (() => pBar.Value = progress));
					}
				}

				encParams.Param[0] = new EncoderParameter(encoder, (long) EncoderValue.Flush);
				first.SaveAdd(encParams);
			});

			// Dispose of the images
			this.CachedImages.ForEach(img => img.Dispose());
			this.CachedImages.Clear();
			this.CachedIntervalls.Clear();
			this.CountLabel.Text = $"Images: -";

			// ProgressBar auf 100% belassen
			if (pBar != null)
			{
				pBar.Invoke((MethodInvoker) (() => pBar.Value = 100));
			}

			return file;
		}









	}
}
