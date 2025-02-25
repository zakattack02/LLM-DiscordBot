# Discord Bot Project

This project is a simple Discord bot that can read messages from a specified channel and respond to user commands.

## Table of Contents

- [Installation](#installation)  
- [Configuration](#configuration)  
- [Usage](#usage)  
- [Features](#features)  
- [License](#license)  

## Installation

1. Clone the repository:  
   ```sh
   git clone <repository-url>
   cd Discord-Bot
   ```

2. Install the required dependencies:  
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

## License

This project is licensed under the Apache License 2.0 License. See the `LICENSE` file for details.
