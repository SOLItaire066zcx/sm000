import asyncio
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from playwright.async_api import async_playwright
import os

# Configuration
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "7975255175:AAEeOzj64js1-vnRlSAW5bMqv92aR5cl7a8")
CHAT_ID = os.getenv("CHAT_ID", "7569017578")
SCRAPING_INTERVAL_SECONDS = 60

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

scraping_task = None
stop_scraping_event = None

async def scrape_fifa_all_matches():
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            url = "https://1xbet.com/fr/live/fifa/2648571-fc-24-5x5-superleague"
            await page.goto(url, timeout=60000)
            await page.wait_for_selector("div.c-events__item", timeout=15000)
            await asyncio.sleep(3)
            matchs = await page.query_selector_all("div.c-events__item")
            if not matchs:
                await browser.close()
                return "❌ Aucun match trouvé sur la page."
            match_results = []
            for match in matchs:
                try:
                    teams_el = await match.query_selector(".c-events__teams")
                    teams = await teams_el.inner_text() if teams_el else "Équipes inconnues"
                    score_el = await match.query_selector(".c-events-scoreboard__value")
                    score = await score_el.inner_text() if score_el else "Pas de score"
                    odds_els = await match.query_selector_all(".c-bets__bet")
                    cotes = [await o.inner_text() for o in odds_els[:3]]
                    formatted = f"⚽ {teams}\n🔢 Score : {score}\n💸 Cotes : {', '.join(cotes)}"
                    match_results.append(formatted)
                except Exception as e:
                    logger.error(f"Erreur lors du parsing d'un match: {e}")
                    continue
            await browser.close()
            return "\n\n".join(match_results)
        except Exception as e:
            logger.error(f"Erreur critique lors du scraping: {e}")
            return f"❌ Erreur lors du scraping: {e}"

async def scraping_loop(context: ContextTypes.DEFAULT_TYPE):
    global stop_scraping_event
    while not stop_scraping_event.is_set():
        message = await scrape_fifa_all_matches()
        try:
            await context.bot.send_message(chat_id=CHAT_ID, text=f"📊 FIFA 5x5 LIVE :\n\n{message}")
        except Exception as e:
            logger.error(f"Erreur lors de l'envoi Telegram: {e}")
        await asyncio.wait([stop_scraping_event.wait()], timeout=SCRAPING_INTERVAL_SECONDS)

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global scraping_task, stop_scraping_event
    user_id = str(update.effective_user.id)
    if user_id != str(CHAT_ID):
        await update.message.reply_text("Désolé, vous n'êtes pas autorisé à utiliser ce bot.")
        return
    if scraping_task and not scraping_task.done():
        await update.message.reply_text("La surveillance est déjà en cours.")
        return
    stop_scraping_event = asyncio.Event()
    scraping_task = asyncio.create_task(scraping_loop(context))
    await update.message.reply_text("Surveillance des matchs FIFA 5x5 démarrée !")

async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global scraping_task, stop_scraping_event
    user_id = str(update.effective_user.id)
    if user_id != str(CHAT_ID):
        await update.message.reply_text("Désolé, vous n'êtes pas autorisé à utiliser ce bot.")
        return
    if scraping_task and not scraping_task.done():
        stop_scraping_event.set()
        await scraping_task
        await update.message.reply_text("Surveillance arrêtée.")
    else:
        await update.message.reply_text("Aucune surveillance n'est en cours.")

async def restart_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await stop_command(update, context)
    await start_command(update, context)

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global scraping_task
    user_id = str(update.effective_user.id)
    if user_id != str(CHAT_ID):
        await update.message.reply_text("Désolé, vous n'êtes pas autorisé à utiliser ce bot.")
        return
    if scraping_task and not scraping_task.done():
        await update.message.reply_text("La surveillance des matchs FIFA 5x5 est en cours.")
    else:
        await update.message.reply_text("La surveillance des matchs FIFA 5x5 n'est pas en cours.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    aide = (
        "\U0001F916 <b>Menu d'aide du bot FIFA 5x5</b>\n\n"
        "<b>/start</b> : Démarrer la surveillance des matchs FIFA 5x5\n"
        "<b>/stop</b> : Arrêter la surveillance\n"
        "<b>/restart</b> : Redémarrer la surveillance\n"
        "<b>/status</b> : Afficher l'état de la surveillance\n"
        "<b>/help</b> : Afficher ce menu d'aide\n\n"
        "Seul l'utilisateur autorisé peut contrôler le bot."
    )
    await update.message.reply_text(aide, parse_mode="HTML")

async def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("stop", stop_command))
    application.add_handler(CommandHandler("restart", restart_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("help", help_command))
    print("\n🚀 Bot Telegram de contrôle lancé. Envoie /start, /stop, /restart, /status, /help à ton bot pour piloter le scraping.\n")
    await application.run_polling(poll_interval=3)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nArrêt du bot.\n") 