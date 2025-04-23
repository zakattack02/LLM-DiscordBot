# Discord Bot Project

This project is a versatile Discord bot that can read messages from a specified channel, respond to user commands, and perform various AI-powered tasks such as image captioning, object detection, and OCR. Additionally, it supports remote training of AI models using platforms like DigitalOcean.

## Table of Contents

- [Installation](#installation)  
- [Configuration](#configuration)  
- [Usage](#usage)  
- [Features](#features)  
- [Technologies Used](#technologies-used)  
- [Dataset](#dataset)  
- [Remote Training on DigitalOcean](#remote-training-on-digitalocean)  
- [License](#license)  

## Installation

1. Install the required dependencies:  
   ```sh
   pip install -r requirements.txt
   ```

## Configuration

1. Rename `config/config.example.json` to `config.json`.  
2. Open `config.json` and update the bot token and other necessary settings.

## Usage

To run the bot, execute the following command:  

```sh
python src/bot-gpt.py
```
or 

```sh
python restart.py
```

Ensure your bot is added to a Discord server and has the necessary permissions.

## Features

- Reads messages from a specified channel  
- Supports command execution  
- Can integrate with external AI models  
- Image captioning using BLIP-2  
- Object detection using YOLOv8  
- OCR (Optical Character Recognition) using Tesseract  
- GIF handling and analysis  
- Video thumbnail extraction using FFmpeg  
- Asynchronous processing to prevent blocking  

## Technologies Used
 
- Transformers (Hugging Face)  
- Torch (PyTorch)  
- PIL (Pillow)  
- OpenCV  
- Tesseract OCR  
- FFmpeg  
- YOLOv8  
- asyncio  

## Dataset

### Raiders of the Lost Kek: 3.5 Years of Augmented 4chan Posts from the Politically Incorrect Board

This project uses the dataset released with the paper titled: **"Raiders of the Lost Kek: 3.5 Years of Augmented 4chan Posts from the Politically Incorrect Board"**. The dataset is available on [Zenodo](https://zenodo.org/records/3606810).

The dataset is a single newline-delimited JSON file. Each line in the file consists of a JSON object representing a full 4chan `/pol/` thread. The JSON objects contain all the key/values returned by the 4chan API, along with three additional keys:

- **entities**: A list of named entities detected for each post using the spaCy Python library.
- **perspectives**: Scores returned by Google’s Perspective API, including seven scores in the `[0; 1]` interval.
- **extracted_poster_id**: A unique identifier for each poster.

This dataset provides a rich source of information for analyzing 4chan `/pol/` threads, including named entity recognition and toxicity scoring.

## Remote Training on DigitalOcean

If you're planning to use DigitalOcean for training the model, we can still achieve remote training. DigitalOcean provides scalable cloud computing, and we can set up a virtual machine (VM) with a GPU (or CPU, if GPU is not required) to train your model. Here's how you can adapt the setup:

### Steps for Training on DigitalOcean:

1. **Set Up a DigitalOcean Droplet**:
   - Create a Droplet on DigitalOcean with a suitable GPU (if needed) or CPU machine.
   - Ensure you choose a machine with enough memory and compute power for training, such as a Droplet with 8GB or more of RAM.
   
2. **Prepare Your Droplet**:
   - SSH into the Droplet once it's set up.
   - Install necessary dependencies such as Python, PyTorch, transformers, and any other libraries required for your model training.

3. **Train the Model Remotely**:
   - Upload the training data and model code to the Droplet.
   - Run the training script on the Droplet directly.

4. **Download the Trained Model Back**:
   - Once the training is complete, save the trained model and download it to your local machine or directly use it in your bot system.

### Detailed Instructions for Remote Training:

#### 1. Create a DigitalOcean Droplet:
   - Go to [DigitalOcean](https://www.digitalocean.com/) and create an account if you don't already have one.
   - Create a new droplet, choosing a suitable size (for example, an 8GB RAM droplet).
   - Make sure to choose the right data center region, select a Linux distribution (e.g., Ubuntu), and set up SSH keys for access.

#### 2. Install Dependencies on the Droplet:

Once your droplet is up and running, SSH into it:

```bash
ssh root@your_droplet_ip
```

Then, install the required dependencies:

```bash
# Update package list and install Python dependencies
sudo apt update
sudo apt install python3-pip python3-dev
sudo apt install git

# Install PyTorch and Hugging Face transformers
pip3 install torch transformers
```

If you need a GPU-enabled droplet, ensure that you also install the necessary CUDA libraries for PyTorch, which can be done by following the [PyTorch installation guide for CUDA](https://pytorch.org/get-started/locally/).

#### 3. Upload Training Data to DigitalOcean Droplet:

You can use `scp` (secure copy) or `rsync` to upload files to the Droplet. Here’s an example of uploading your training file:

```bash
scp /path/to/your/training_data.txt root@your_droplet_ip:/path/to/remote/directory
```

Alternatively, you can upload your code and other necessary files in the same way.

#### 4. Run the Training Script on the Droplet:

After uploading the files, SSH into the Droplet and run the training script.

```bash
python3 train_script.py
```

This will execute the model training on your DigitalOcean Droplet. You can configure the script as necessary (for example, by adjusting batch size, learning rate, etc.).

#### 5. Download the Trained Model to Your Local Machine:

Once the training is complete, save the model to a file (for example, `model.pt` or in the Hugging Face format). Then, you can use `scp` again to download it to your local machine:

```bash
scp root@your_droplet_ip:/path/to/trained_model.pt /local/path/to/save/model
```

#### 6. Use the Trained Model in Your Bot:

Once you’ve downloaded the trained model, you can integrate it back into your bot system as follows:

- Ensure that the model is loaded from the location where it was saved (either from the cloud or your local machine).
- Modify your bot code to use the newly trained model for inference.

### Conclusion:

- **DigitalOcean Droplets** are suitable for remote training. The process involves creating a droplet, SSHing into it, and uploading the training data.
- You can **train remotely** on the droplet, then **download the trained model** to your local system for use with your bot.
- **Dependencies** (like PyTorch and Hugging Face Transformers) need to be installed on the droplet, but once set up, the process is very similar to training locally.

## License

This project is licensed under the Apache License 2.0 License. See the `LICENSE` file for details.
