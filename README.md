# Discord Bot Project

This project is a versatile Discord bot that can read messages from a specified channel, respond to user commands, and perform various AI-powered tasks such as image captioning, object detection, OCR.

## Table of Contents

- [Installation](#installation)  
- [Configuration](#configuration)  
- [Usage](#usage)  
- [Features](#features)  
- [Technologies Used](#technologies-used)  
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

## License

This project is licensed under the Apache License 2.0 License. See the `LICENSE` file for details.
