from config import token
from model import StyleTransferModel
from io import BytesIO

TELEGRAM_TOKEN = token
PORT = 8443
HEROKU_APP_NAME = "taraborinstyleflow"

model = StyleTransferModel()
first_image_file = {}

def send_msg(update, context, text):
    context.bot.send_message(chat_id=update.effective_chat.id, text=text)

def start(update, context):
    send_msg(update, context, "Приветсвую! Я умею изменять стиль изображения.")
    send_msg(update, context, "Пришлите изображение, стиль которого необходимо изменить.")

def get_photo(update, context):
    chat_id = update.message.chat_id

    image_info = update.message.photo[-1]
    image_file = context.bot.get_file(image_info)
    
    if chat_id in first_image_file:
        content_image_file = first_image_file[chat_id]
        style_image_file = image_file
        del first_image_file[chat_id]

        send_photo(update, context, content_image_file, style_image_file)
    else:
        first_image_file[chat_id] = image_file
        send_msg(update, context, "Пришлите изображение со стилем.")

def send_photo(update, context, content_image_file, style_image_file):
    send_msg(update, context, "Ожидайте, идет обработка...")

    content_image_stream = BytesIO()
    content_image_file.download(out=content_image_stream)

    style_image_stream = BytesIO()
    style_image_file.download(out=style_image_stream)

    output = model.transfer_style(content_image_stream, style_image_stream)

    output_stream = BytesIO()
    output.save(output_stream, format='PNG')
    output_stream.seek(0)
    context.bot.send_photo(update.message.chat_id, photo=output_stream)

    send_msg(update, context, "Выполнено.")
    send_msg(update, context, "Чтобы изменить стиль другого изображения, пришлите его.")

if __name__ == '__main__':
    from telegram.ext import Updater, MessageHandler, Filters, CommandHandler

    """updater = Updater(token=token)
    dispatcher = updater.dispatcher
    start_handler = CommandHandler('start', start)
    dispatcher.add_handler(start_handler)
    dispatcher.add_handler(MessageHandler(Filters.photo, get_photo))
    updater.start_polling()
    updater.idle()"""
    
    
    updater = Updater(token=token)
    # add handlers
    dispatcher = updater.dispatcher
    start_handler = CommandHandler('start', start)
    dispatcher.add_handler(start_handler)
    dispatcher.add_handler(MessageHandler(Filters.photo, get_photo))
    # Start the Bot
    if not HEROKU_APP_NAME:  # pooling mode
        print("Can't detect 'HEROKU_APP_NAME' env. Running bot in pooling mode.")
        print("Note: this is not a great way to deploy the bot in Heroku.")
        updater.start_polling()
        updater.idle()

    else:  # webhook mode
        print(f"Running bot in webhook mode. Make sure that this url is correct: https://{HEROKU_APP_NAME}.herokuapp.com/")
        updater.start_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=TELEGRAM_TOKEN,
            webhook_url=f"https://{HEROKU_APP_NAME}.herokuapp.com/{TELEGRAM_TOKEN}"
        )
        updater.idle()
