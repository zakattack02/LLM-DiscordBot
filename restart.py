import os
import time
import subprocess
import json
from enum import Enum, auto

# Load BOT_SCRIPT dynamically from the configuration file
with open('config/config.json', 'r') as config_file:
    config = json.load(config_file)
    BOT_SCRIPT = config.get("BOT_SCRIPT")  # Default to 'bot/FUCKery.py' if not found

# Define states for the state machine
class State(Enum):
    MENU = auto()
    START_BOT = auto()
    STOP_BOT = auto()
    RESTART_BOT = auto()
    RUN_TRAINING = auto()
    CONFIGURE_PATHS = auto()
    STATUS = auto()  # New state for monitoring tasks
    EXIT = auto()

# Function to start the bot
def start_bot():
    print("🔄 Starting bot...")
    return subprocess.Popen(["python", BOT_SCRIPT])

# Function to stop the bot
def stop_bot(process):
    if process and process.poll() is None:  # Check if the process is running
        print("❌ Stopping bot...")
        process.terminate()
        process.wait()
    else:
        print("⚠️ Bot is not running.")

# Function to restart the bot
def restart_bot(process):
    stop_bot(process)
    time.sleep(3)  # Wait before restarting
    return start_bot()

# Function to run the training script
def run_training_script():
    print("🧠 Starting training script...")
    subprocess.run(["python", "scripts/Train.py"], check=True)

# Function to configure TrainingData and Dataset_path
def configure_training_paths():
    with open('config/config.json', 'r') as config_file:
        config = json.load(config_file)
    with open('config/DefaultPATH.json', 'r') as default_file:
        default_config = json.load(default_file)

    print("Current TrainingData path:", config["TrainingData"])
    print("Do you want to use the default TrainingData path? (yes/no): ")
    user_input = input().strip().lower()
    if user_input in ["no", "n"]:
        new_training_data = input("Enter the new TrainingData path: ").strip()
        config["TrainingData"] = new_training_data
    if user_input in ["yes", "y"]:
        config["TrainingData"] = default_config["TrainingData"]

    print("Current Dataset_path:", config["Dataset_path"])
    print("Do you want to use the default Dataset_path? (yes/no): ")
    user_input = input().strip().lower()
    if user_input in ["no", "n"]:
        new_dataset_path = input("Enter the new Dataset_path: ").strip()
        config["Dataset_path"] = new_dataset_path
    if user_input in ["yes", "y"]:
        config["Dataset_path"] = default_config["Dataset_path"]

    with open('config/config.json', 'w') as config_file:
        json.dump(config, config_file, indent=4)

    print("Training paths updated successfully.")
    print(f"TrainingData: {config['TrainingData']}")
    print(f"Dataset_path: {config['Dataset_path']}")

# Function to display the menu
def display_menu():
    print("\033[1;36m\n=== LLM-DiscordBot Manager ===\033[0m")
    print("\033[1;33m1. Start Bot\033[0m")
    print("\033[1;33m2. Stop Bot\033[0m")
    print("\033[1;33m3. Restart Bot\033[0m")
    print("\033[1;33m4. Run Training Script\033[0m")
    print("\033[1;33m5. Configure Training Paths\033[0m")
    print("\033[1;33m6. Exit\033[0m")
    print("\033[1;36m==============================\033[0m")

# Function to clear the terminal and display the current state
def display_state(state):
    os.system('cls' if os.name == 'nt' else 'clear')  # Clear the terminal
    print("\033[1;36m=== LLM-DiscordBot Manager ===\033[0m")  # Cyan header
    print(f"\033[1;33mCurrent State: {state.name}\033[0m")  # Yellow state name
    print("\033[1;36m==============================\033[0m")  # Cyan footer

# Function to monitor the bot or task status
def monitor_status(process):
    with open('config/config.json', 'r') as config_file:
        config = json.load(config_file)
    use_discord = config.get("Use_Discord", False)  # Default to False if not found

    #os.system('cls' if os.name == 'nt' else 'clear')  # Clear the terminal
    print("\033[1;36m=== LLM-DiscordBot Manager ===\033[0m")  # Cyan header
    if process and process.poll() is None:
        print("\033[1;32mTask is running...\033[0m")  # Green status
    else:
        print("\033[1;31mTask has stopped.\033[0m")  # Red status
        return
    if use_discord:
        print("\033[1;33mPress 's' to stop, 'r' to restart, or 'q' to return to the menu.\033[0m")
        print("\033[1;36m==============================\033[0m")  # Cyan footer
        time.sleep(1) 

    if not use_discord:
        print("\033[1;33mType 'exit' in the bot to return to the menu.\033[0m")
        print("\033[1;36m==============================\033[0m")  # Cyan footer
        time.sleep(1)  # Delay to avoid excessive terminal refresh

    # Wait for user input
    user_input = input("Enter your choice: ").strip().lower()
    if user_input == "s":
        return "stop"
    elif user_input == "r":
        return "restart"
    elif user_input == "q":
        return "menu"
    time.sleep(1)  # Delay to avoid excessive terminal refresh

def check_exit_signal():
    """Check for an exit signal file to transition to EXIT state."""
    if os.path.exists("exit_signal.flag"):
        os.remove("exit_signal.flag")
        return True
    return False

# Main function to handle the state machine
def main():
    bot_process = None
    current_state = State.MENU

    while current_state != State.EXIT:
        display_state(current_state)  # Clear terminal and display current state

        if current_state == State.MENU:
            display_menu()
            choice = input("Enter your choice: ").strip()
            if choice == "1":
                current_state = State.START_BOT
            elif choice == "2":
                current_state = State.STOP_BOT
            elif choice == "3":
                current_state = State.RESTART_BOT
            elif choice == "4":
                current_state = State.RUN_TRAINING
            elif choice == "5":
                current_state = State.CONFIGURE_PATHS
            elif choice == "6":
                current_state = State.EXIT
            else:
                print("❌ Invalid choice. Please try again.")

        elif current_state == State.START_BOT:
            if bot_process and bot_process.poll() is None:
                print("⚠️ Bot is already running.")
                while not check_exit_signal(): 
                    time.sleep(1)  # Wait for the exit signal
                current_state = State.MENU  # Explicitly transition to MENU after the loop
            else:
                # Ask if the user wants to use Discord
                print("Do you want to use Discord for this session? (yes/no): ")
                user_input = input().strip().lower()
                with open('config/config.json', 'r') as config_file:
                    config = json.load(config_file)
                if user_input in ["yes", "y"]:
                    config["Use_Discord"] = True
                else:
                    config["Use_Discord"] = False
                with open('config/config.json', 'w') as config_file:
                    json.dump(config, config_file, indent=4)
                print(f"Use_Discord set to: {config['Use_Discord']}")

                # Start the bot
                bot_process = start_bot()
            if check_exit_signal():
                current_state = State.MENU

        elif current_state == State.STOP_BOT:
            stop_bot(bot_process)
            bot_process = None
            current_state = State.MENU

        elif current_state == State.RESTART_BOT:
            bot_process = restart_bot(bot_process)
            current_state = State.STATUS

        elif current_state == State.RUN_TRAINING:
            run_training_script()
            current_state = State.STATUS

        elif current_state == State.CONFIGURE_PATHS:
            configure_training_paths()
            current_state = State.MENU

        elif current_state == State.STATUS:
            if bot_process and bot_process.poll() is None:
                action = monitor_status(bot_process)
                if action == "stop":
                    current_state = State.STOP_BOT
                elif action == "restart":
                    current_state = State.RESTART_BOT
                elif action == "menu":
                    current_state = State.MENU
            else:
                print("⚠️ Task is no longer running. Returning to menu.")
                current_state = State.MENU

    display_state(current_state)  # Display exit state
    print("👋 Exiting manager...")
    stop_bot(bot_process)

if __name__ == "__main__":
    main()
