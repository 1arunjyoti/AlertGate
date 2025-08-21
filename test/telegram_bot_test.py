from typing import Final
from telegram import Update
from telegram.ext import ContextTypes, CommandHandler, MessageHandler, Application, filters
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

Token: Final = os.getenv("TELEGRAM_BOT_TOKEN")
Bot_username: Final = os.getenv("BOT_USERNAME")

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Hello, ! I am your bot {Bot_username}."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Hello, ! I am your bot {Bot_username}. How can I assist you today?"
    )

async def custom_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Hello, ! I am your bot {Bot_username}. This is a custom command response."
    )

#handle responses to text messages
def handle_response(text: str) -> str:
    processed: str = text.lower()

    if "hello" in processed:
        return "Hello! How can I help you today?"
    elif "help" in processed:
        return "Sure! What do you need help with?"
    else:
        return "I'm not sure how to respond to that."

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_type: str = update.message.chat.type
    text: str = update.message.text

    print(f'User ({update.message.chat.id}) in chat type: {message_type}: "{text}"')
    if message_type == "group":
        if Bot_username in text:
            new_text: str = text.replace(Bot_username, "").strip()
            response: str = handle_response(new_text)
        else:
            return
    else:
        response: str = handle_response(text)
    
    print('Bot response:', response)
    await update.message.reply_text(response)

async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'Update "{update}" caused error "{context.error}"')

if __name__ == "__main__":
    print("Starting bot...")
    app = Application.builder().token(Token).build()

    #command handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("custom", custom_command))

    #Message handler
    app.add_handler(MessageHandler(filters.TEXT, handle_message))

    #Error handler
    app.add_error_handler(error)

    print("Polling...!")
    app.run_polling(poll_interval=3)