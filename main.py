import asyncio
from playwright.async_api import async_playwright
import pandas as pd
import re
from telegram import Bot, Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes
import os
import sys
import logging

# Configure basic logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


# Ensure the correct asyncio event loop policy is set for Windows early
# This helps with subprocess support required by libraries like Playwright
if sys.platform.startswith("win"):
    # Attempt to get the current policy or set it if not set
    try:
        # Check if a policy is already set
        existing_policy = asyncio.get_event_loop_policy()
        # If policy is ProactorEventLoop (default on some Windows), try to switch
        # Check if it's safe to set a new policy (no running loop)
        try:
            asyncio.get_running_loop()
            # If we reach here, a loop is running, cannot change policy safely
        except RuntimeError:
             # No loop is running, it's safe to set the policy
             if not isinstance(existing_policy, asyncio.WindowsSelectorEventLoopPolicy):
                 if sys.version_info >= (3, 8):
                     try:
                         asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
                         logger.info("Set WindowsSelectorEventLoopPolicy")
                     except ValueError:
                         logger.warning("Could not set WindowsSelectorEventLoopPolicy, policy might be immutable.")
                 else:
                     logger.warning("Python version < 3.8, cannot reliably set WindowsSelectorEventLoopPolicy.")
        except Exception as e:
             logger.warning(f"Error checking/setting asyncio policy: {e}")

    except Exception as e:
        logger.warning(f"Error accessing or setting asyncio event loop policy: {e}")


# =============== CONFIG ===============
BOT_TOKEN = "7975255175:AAEeOzj64js1-vnRlSAW5bMqv92aR5cl7a8"
CHAT_ID = "7569017578"
MATCH_LIMIT = 10  # Augmenté la limite par défaut pour avoir plus de matchs
SCRAPING_INTERVAL_SECONDS = 60 # Intervalle entre les scrapings (1 minute)
HISTO_CSV = "historique_fifa5x5.csv"

# Global variables for managing the scraping task
scraping_task = None
stop_scraping_event = None

# Helper to run an async coroutine from a synchronous context within an executor
# This is needed to call Playwright (async) from the synchronous executor thread
def run_async_in_sync_executor(coro):
    # asyncio.run creates and manages its own event loop for the coroutine
    # This is often needed when calling async code from a non-async context or executor thread
    try:
        return asyncio.run(coro)
    except Exception as e:
        logger.error(f"Error running async coroutine in executor: {e}")
        return None # Return None or raise a specific error


# =============== SCRAPING 1XBET (Playwright - using selectors) ===============
async def scrape_1xbet():
    logger.info("[Playwright] Lancement du navigateur...")
    browser = None # Initialize browser to None
    try:
        async with async_playwright() as p:
            # Keep headless=False for now for easier debugging if needed, switch to True for production
            browser = await p.chromium.launch(headless=True) # Set to True for normal operation
            page = await browser.new_page()
            url = "https://1xbet.com/fr/live/esports"
            logger.info(f"[Playwright] Navigation vers {url}...")
            try:
                # Increased timeout again, sites like this can be slow or add delays
                # Waiting until 'domcontentloaded' is often faster than 'load'
                await page.goto(url, timeout=90000, wait_until='domcontentloaded') # 90 seconds, wait for DOM
                # Wait for individual match items - using first_selector=True as we only need one to appear
                await page.wait_for_selector("div.c-events__item", timeout=30000) # 30 seconds for first item
                logger.info("[Playwright] Match items trouvés, attente supplémentaire pour le rendu...")
            except Exception as e:
                logger.warning(f"[Playwright] Timeout ou erreur lors de l'attente des matchs : {e}")
                return [] # Return empty list if matches not found

            # Add a small fixed delay after selectors are likely present, sometimes needed for dynamic content
            await asyncio.sleep(5)

            logger.info("[Playwright] Extraction des données des matchs...")
            match_elements = await page.query_selector_all("div.c-events__item")
            data = []

            # Use enumerate with a slice to respect MATCH_LIMIT
            for idx, match_el in enumerate(match_elements[:MATCH_LIMIT]):
                try:
                    # Extract data using selectors within the match element
                    teams_el = await match_el.query_selector(".c-events__teams")
                    teams_text = await teams_el.inner_text() if teams_el else "Équipe Inconnue vs Inconnue"
                    # Split teams text carefully, handle potential extra spaces or format issues
                    if " vs " in teams_text:
                         team1, team2 = [t.strip() for t in teams_text.split(" vs ", 1)] # Split only once
                    else:
                         team1, team2 = teams_text.strip(), "Équipe Inconnue"

                    # Extract score (can be None for live matches or if structure differs)
                    score_text = None
                    # Look for the main score element
                    score_el = await match_el.query_selector(".c-events-scoreboard__value")
                    if score_el:
                        score_text = (await score_el.inner_text()).strip()
                    # You might need to add other selectors here if score appears differently elsewhere

                    # Extract cotes for V1, Nul, V2
                    # Assuming first 3 .c-bets__bet are V1, Nul, V2. Check if they exist and are numbers.
                    cote_els = await match_el.query_selector_all(".c-bets__bet")
                    cotes_values = []
                    for cote_el in cote_els:
                        try:
                            cote_val_text = (await cote_el.inner_text()).strip()
                            cotes_values.append(float(cote_val_text))
                        except ValueError:
                             cotes_values.append(None) # Append None if not a valid float
                        except Exception:
                             cotes_values.append(None) # Catch other errors

                    cote1 = cotes_values[0] if len(cotes_values) > 0 else None
                    cN = cotes_values[1] if len(cotes_values) > 1 else None
                    cote2 = cotes_values[2] if len(cotes_values) > 2 else None


                    # Extract time (adjust selector if needed)
                    time_el = await match_el.query_selector(".c-events__time")
                    time_text = (await time_el.inner_text()).strip() if time_el else "Inconnue"
                    # Basic format check for HH:MM
                    if re.match(r'^\d{2}:\d{2}$', time_text):
                        heure = time_text
                    else:
                        heure = "Inconnue" # Keep as Inconnue if format is unexpected


                    data.append({
                        'team1': team1,
                        'team2': team2,
                        'score': score_text, # Include score in the scraped data
                        'cote1': cote1,
                        'cote2': cote2,
                        'cote_nul': cN,
                        'heure': heure
                    })
                except Exception as e:
                    logger.warning(f"[Playwright] Erreur lors du parsing de l'élément match {idx}: {e}")
                    # Continue to the next match element even if one fails
                    continue

            logger.info(f"[Playwright] Extraction terminée. {len(data)} matchs collectés.")
            return data
    except Exception as e:
        logger.error(f"[Playwright] Erreur critique lors du scraping : {e}")
        return [] # Return empty list on critical failure
    finally:
        if browser:
            await browser.close() # Ensure browser is closed even if errors occur


# =============== SCRAPING HISTORIQUE (structure à adapter selon la page) ===============
# This function is a placeholder. If a dedicated history page exists,
# implement scraping logic here similar to scrape_1xbet.
# Ensure the returned list of dicts includes 'team1', 'team2', 'score', 'verdict'.
# Cotes and heure are not strictly needed for historical stats but can be included.
async def scrape_historique():
    # Example placeholder implementation:
    # await asyncio.sleep(1) # Simulate async work
    # logger.info("Scraping historique (placeholder) exécuté.")
    return []


# =============== STOCKAGE CSV ===============
# Function to save a list of dicts to the history CSV.
# Designed primarily to save finished matches with score and verdict.
def save_history_to_csv(data_to_add, filename=HISTO_CSV):
    # Ensure data_to_add is a list of dicts
    if not isinstance(data_to_add, list) or (data_to_add and not isinstance(data_to_add[0], dict)):
        logger.warning(f"save_history_to_csv received invalid data type: {type(data_to_add)}")
        return

    # Define expected keys for history entries
    history_keys = ["team1", "team2", "score", "verdict"]
    valid_history_entries = []

    for item in data_to_add:
        # Check if essential keys for history are present and not None
        if all(item.get(key) is not None for key in history_keys):
            # Create a new dict with only the history keys
            valid_history_entries.append({key: item.get(key) for key in history_keys})
        else:
            # Log a warning if a potential history entry is missing data
            if item.get('team1') and item.get('team2'): # Only log if it looks like a match attempt
                 logger.debug(f"Skipping adding match to history due to missing data: {item.keys()}")


    if not valid_history_entries:
        # logger.info("No valid history data to save.") # Can be noisy
        return

    df_to_add = pd.DataFrame(valid_history_entries)

    try:
        # Ensure the CSV file exists with headers if it's new
        if not os.path.exists(filename):
             df_to_add.to_csv(filename, mode='w', header=True, index=False)
             logger.info(f"Created new history CSV: {filename}")
        else:
            # Check existing columns to prevent issues with append
            try:
                 existing_df_cols = pd.read_csv(filename, nrows=0).columns.tolist()
                 if existing_df_cols != history_keys:
                     logger.warning(f"Existing history CSV columns {existing_df_cols} do not match expected {history_keys}. Appending may cause issues.")

            except pd.errors.EmptyDataError:
                 # File exists but is empty, write with header
                 df_to_add.to_csv(filename, mode='w', header=True, index=False)
                 logger.info(f"History CSV {filename} was empty, wrote headers.")
                 return # Exit after writing the first data


            # Append without header
            df_to_add.to_csv(filename, mode='a', header=False, index=False)
            # logger.info(f"Appended {len(valid_history_entries)} entries to history CSV: {filename}") # Can be noisy


    except Exception as e:
        logger.error(f"Error writing to history CSV {filename}: {e}")


# Function to get team statistics from the history CSV
def get_team_stats(team, filename=HISTO_CSV):
    if not os.path.exists(filename):
        # logger.debug(f"History file not found for stats: {filename}") # Can be noisy
        return {'played': 0, 'won': 0, 'draw': 0, 'lost': 0, 'win_rate': 0.0}
    try:
        df = pd.read_csv(filename)
        # Ensure columns exist before accessing
        if not all(col in df.columns for col in ['team1', 'team2', 'verdict']):
             logger.warning(f"History file {filename} is missing required columns for stats.")
             return {'played': 0, 'won': 0, 'draw': 0, 'lost': 0, 'win_rate': 0.0}

        # Filter for rows where the team played
        # Use .copy() to avoid SettingWithCopyWarning with future operations (though not expected here)
        team_matches = df[(df['team1'] == team) | (df['team2'] == team)].copy()
        played = len(team_matches)

        if played == 0:
            return {'played': 0, 'won': 0, 'draw': 0, 'lost': 0, 'win_rate': 0.0}

        # Calculate wins, draws, losses using .shape[0] for row count
        won = team_matches[(team_matches['team1'] == team) & (team_matches['verdict'] == 'V1') |
                           (team_matches['team2'] == team) & (team_matches['verdict'] == 'V2')].shape[0]

        draw = team_matches[team_matches['verdict'] == 'Nul'].shape[0]
        lost = played - won - draw
        win_rate = won / played if played > 0 else 0.0 # Avoid division by zero

        return {'played': played, 'won': won, 'draw': draw, 'lost': lost, 'win_rate': win_rate}
    except pd.errors.EmptyDataError:
        # Handle empty CSV file gracefully
        # logger.debug(f"History file {filename} is empty for stats.") # Can be noisy
        return {'played': 0, 'won': 0, 'draw': 0, 'lost': 0, 'win_rate': 0.0}
    except Exception as e:
        logger.error(f"Error reading or processing history CSV {filename} for stats: {e}")
        return {'played': 0, 'won': 0, 'draw': 0, 'lost': 0, 'win_rate': 0.0}


# =============== PREDICTION ENRICHIE ===============
# Function to generate enriched predictions based on cotes and history stats
def enriched_predict(match):
    # Ensure match dict has all necessary keys and valid data types, provide defaults
    team1 = match.get('team1', 'Inconnue')
    team2 = match.get('team2', 'Inconnue')
    heure = match.get('heure', 'Inconnue')
    # Attempt to get cotes as floats, default to None if not available or not valid numbers
    try:
        c1 = float(match['cote1']) if match.get('cote1') is not None else None
        c2 = float(match['cote2']) if match.get('cote2') is not None else None
        cN = float(match['cote_nul']) if match.get('cote_nul') is not None else None
    except (ValueError, TypeError):
        c1, c2, cN = None, None, None # Set all to None if any cote is invalid

    # Cannot predict without valid cotes
    if c1 is None or c2 is None or cN is None or c1 <= 0 or c2 <= 0 or cN <= 0:
         return {
            'team1': team1, 'team2': team2, 'heure': heure,
            'cote1': c1, 'cote2': c2, 'cote_nul': cN, 'cote_1N': None, 'cote_2N': None,
            'pronostic': 'N/A', 'cote_pronostic': None, 'confiance_score': 'N/A', 'conseil': 'N/A',
            'details': f"Match: {team1} vs {team2} à {heure} → Impossible de prédire (cotes manquantes ou invalides)"
         }

    # Calculate double chance cotes - Handle potential division by zero if cote is 0
    c1N = 1 / (1/c1 + 1/cN) if c1 > 0 and cN > 0 else None
    c2N = 1 / (1/c2 + 1/cN) if c2 > 0 and cN > 0 else None

    # Probabilities (handle division by zero if total is 0)
    p1 = 1 / c1 if c1 > 0 else 0
    p2 = 1 / c2 if c2 > 0 else 0
    pN = 1 / cN if cN > 0 else 0
    total_proba = p1 + p2 + pN
    proba1 = p1 / total_proba if total_proba > 0 else 0
    proba2 = p2 / total_proba if total_proba > 0 else 0
    probaN = pN / total_proba if total_proba > 0 else 0

    # Historique (handle errors getting stats and ensure win_rate is float)
    stats1 = get_team_stats(team1)
    stats2 = get_team_stats(team2)
    win_rate1 = float(stats1.get('win_rate', 0.0))
    win_rate2 = float(stats2.get('win_rate', 0.0))

    # Score combiné : proba + historique (simple sum) - This is the score used for ranking options
    score1 = proba1 + win_rate1
    score2 = proba2 + win_rate2
    scoreN = probaN

    options = [
        ('V1', c1, score1),
        ('V2', c2, score2),
        ('Nul', cN, scoreN),
        # Calculate combined score for 1N/2N - using simple average of V/N scores
        ('1N', c1N, (score1 + scoreN)/2 if c1N is not None else -1), # Use -1 as a score if option not available
        ('2N', c2N, (score2 + scoreN)/2 if c2N is not None else -1)
    ]
    # Filter out options with None cote or score < 0 (for N/A options)
    options = [(name, cote, score) for name, cote, score in options if cote is not None and score >= 0]

    if not options:
         return {
            'team1': team1, 'team2': team2, 'heure': heure,
            'cote1': c1, 'cote2': c2, 'cote_nul': cN, 'cote_1N': c1N, 'cote_2N': c2N,
            'pronostic': 'N/A', 'cote_pronostic': None, 'confiance_score': 'N/A', 'conseil': 'N/A',
            'details': f"Match: {team1} vs {team2} à {heure} → Impossible de prédire (options valides manquantes après calculs)"
         }

    # On choisit l'option avec le score combiné le plus élevé
    best_bet = max(options, key=lambda x: x[2])

    # Calculate confidence based on the difference between the best and second best *scores*
    confidence_score_diff = 0.0 # Default confidence difference
    conseil = "Indécis" # Default conseil

    if len(options) > 1:
        # Sort by score descending to find the second best
        options_sorted_by_score = sorted(options, key=lambda x: x[2], reverse=True)
        best_score = options_sorted_by_score[0][2]
        second_best_score = options_sorted_by_score[1][2]
        confidence_score_diff = best_score - second_best_score

        # Map score difference to conseil - Thresholds need tuning based on observed score differences
        if confidence_score_diff > 0.3:
             conseil = "Très Sûr"
        elif confidence_score_diff > 0.15:
            conseil = "Sûr"
        elif confidence_score_diff > 0.05:
            conseil = "Prudent"
        else:
            conseil = "Indécis"

    # Format confidence display
    confidence_display = f"{confidence_score_diff:.2f}"


    return {
        'team1': team1,
        'team2': team2,
        'heure': heure,
        'cote1': c1,
        'cote2': c2,
        'cote_nul': cN,
        'cote_1N': c1N,
        'cote_2N': c2N,
        'pronostic': best_bet[0],
        'cote_pronostic': best_bet[1],
        'confiance_score': confidence_display, # Display the score difference
        'conseil': conseil,
        # Ensure details string format is correct for Telegram markdown and newlines
        'details': (
            f"Match: {team1} vs {team2} à {heure}\n"
            f"→ Pronostic: *{best_bet[0]}* (Cote: {best_bet[1]})\n"
            f"Indice de confiance (Score Diff): {confidence_display} | Conseil: {conseil}\n"
            f"Stats: {team1} [{stats1.get('played', 0)}] ({win_rate1*100:.1f}% W) | {team2} [{stats2.get('played', 0)}] ({win_rate2*100:.1f}% W)"
        )
    }

# =============== ENVOI AU BOT TELEGRAM (Async) ===============
# Function to send prediction results to Telegram, handles message splitting
async def send_to_telegram_async(context: ContextTypes.DEFAULT_TYPE, results):
    bot = context.bot
    base_text = "⚽ Prédictions MCFIFA5x5SPR :\n\n"
    message_limit = 4000 # Keep a margin below Telegram's 4096 limit

    if not results:
        # Send a message indicating no matches found or predicted
        try:
            await bot.send_message(chat_id=CHAT_ID, text=base_text + "Aucun match trouvé ou prédit pour l'instant.", parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
             logger.error(f"Error sending 'no results' Telegram message: {e}")
        return # Exit if no results

    # Build messages, splitting if necessary
    messages = []
    current_message = base_text

    for result in results:
        detail_text = result.get('details', 'Erreur: Détails manquants') + "\n\n"
        # Check if adding the next detail will exceed the limit
        if len(current_message) + len(detail_text) > message_limit and current_message != base_text:
            # Add current message to list (strip trailing newlines)
            messages.append(current_message.strip())
            # Start a new message
            current_message = base_text + detail_text
        else:
            # Add detail to current message
            current_message += detail_text

    # Add the last current message if it's not empty (and not just the base text)
    if current_message.strip() != base_text.strip():
        messages.append(current_message.strip())

    # Send the messages
    for i, msg in enumerate(messages):
        try:
            # Add pagination info if there's more than one message part
            final_msg_text = msg
            if len(messages) > 1:
                 final_msg_text += f"\n\n_(Part {i+1}/{len(messages)})_"

            await bot.send_message(chat_id=CHAT_ID, text=final_msg_text, parse_mode=ParseMode.MARKDOWN)
            await asyncio.sleep(0.5) # Small delay between sending parts if split
        except Exception as e:
            logger.error(f"Error sending Telegram message part {i+1}: {e}")
            # Attempt to send a basic error message if sending fails
            try:
                await bot.send_message(chat_id=CHAT_ID, text=f"❌ Erreur lors de l'envoi des prédictions (partie {i+1}): {e}")
            except:
                pass # Ignore if even error message fails


# =============== AJOUT À L'HISTORIQUE AU FIL DU TEMPS ===============
# Function to add a finished match entry (score and verdict) to the history CSV.
# Allows duplicate entries if the same match/score appears multiple times.
def add_match_to_history(match_data, filename=HISTO_CSV):
    # Ensure match_data dict has necessary keys and valid data, provide defaults
    team1 = match_data.get('team1', 'Inconnue')
    team2 = match_data.get('team2', 'Inconnue')
    score = match_data.get('score') # Score is critical for history
    verdict = match_data.get('verdict') # Verdict is critical for history

    # Validate essential data types and presence
    if not isinstance(team1, str) or not isinstance(team2, str) or team1 == 'Inconnue' or team2 == 'Inconnue':
        # logger.debug(f"Skipping history add: Invalid team names in {match_data}")
        return
    if score is None or verdict is None or not isinstance(score, str) or not isinstance(verdict, str):
        # logger.debug(f"Skipping history add: Missing or invalid score/verdict in {match_data}")
        return


    try:
        # Data to append - create a DataFrame from a single row
        data_to_append = pd.DataFrame([{
            "team1": team1,
            "team2": team2,
            "score": score,
            "verdict": verdict
        }])

        # Use pandas to_csv with mode='a' for append. Handle initial file creation.
        if not os.path.exists(filename):
             # Write with header if the file doesn't exist
             data_to_append.to_csv(filename, mode='w', header=True, index=False)
             logger.info(f"Created new history CSV and added 1 entry: {filename}")
        else:
            # Append without header if the file exists
            data_to_append.to_csv(filename, mode='a', header=False, index=False)
            # logger.debug(f"Appended 1 entry to history CSV: {filename}")

    except Exception as e:
        logger.error(f"Error adding match to history CSV {filename}: {e}")


# =============== SCRAPING LOOP FOR THE BOT (Async) ===============
# This loop runs the scraping, processing, and sending at intervals, controlled by Telegram commands.
async def scraping_loop(context: ContextTypes.DEFAULT_TYPE):
    global stop_scraping_event
    # loop = asyncio.get_running_loop() # Not needed directly when using run_in_executor with default loop

    logger.info("Scraping loop started.")

    while not stop_scraping_event.is_set():
        matches = [] # Initialize matches list for each loop iteration

        try:
            # --- STEP 1: SCRAPE MATCHES ---
            # Run the async scrape_1xbet within a separate thread via executor
            # This is the fix attempt for NotImplementedError with subprocesses
            # The run_async_in_sync_executor helper will call asyncio.run(scrape_1xbet()) in a thread
            # We get the current running loop from the context to use its executor
            loop = asyncio.get_running_loop() # Get the event loop managed by run_polling

            logger.info("Running scrape_1xbet in executor...")
            matches = await loop.run_in_executor(
                None, # Use the default thread pool executor
                run_async_in_sync_executor, # The synchronous callable that runs the async coroutine
                scrape_1xbet() # The async coroutine function call
            )
            # Note: scrape_1xbet logs its own progress and errors now

            # --- STEP 2: PROCESS SCRAPED MATCHES ---
            if not matches:
                logger.info("No matches collected by the scraper.")
                # Optionally send a message to Telegram if no matches were found
                # await context.bot.send_message(chat_id=CHAT_ID, text="Aucun match trouvé par le scraper.")
            else:
                logger.info(f"Processing {len(matches)} matches collected by the scraper.")

                # Add finished matches to history CSV
                added_to_history_count = 0
                for m in matches:
                    # Determine verdict from score if score is available and valid
                    verdict = None
                    score = m.get('score')
                    if score and isinstance(score, str) and '-' in score:
                        try:
                            s1, s2 = map(int, score.split('-', 1)) # Split only once
                            verdict = "V1" if s1 > s2 else ("V2" if s2 > s1 else "Nul")
                            # Add to history using the dedicated function
                            add_match_to_history({
                                "team1": m.get('team1'),
                                "team2": m.get('team2'),
                                "score": score,
                                "verdict": verdict
                            })
                            added_to_history_count += 1
                        except ValueError:
                            logger.debug(f"Could not parse score '{score}' for history.")
                            pass # Ignore matches with invalid score format for history
                        except Exception as e:
                            logger.warning(f"Error processing match for history: {e}")
                            pass # Catch other errors

                if added_to_history_count > 0:
                     logger.info(f"Added {added_to_history_count} finished matches to history CSV.")


                # Scraping historique from dedicated page (placeholder)
                # If scrape_historique is implemented and returns data, process it similarly
                # historique = await loop.run_in_executor(None, run_async_in_sync_executor, scrape_historique())
                # if historique:
                #     # Process historique data, add to history CSV
                #     # This would involve similar logic to processing 'matches' for history


                # --- STEP 3: GENERATE PREDICTIONS ---
                # Generate predictions only for matches with valid cotes
                predictable_matches = [m for m in matches if m.get('cote1') is not None and m.get('cote2') is not None and m.get('cote_nul') is not None]

                if not predictable_matches:
                     logger.info("No matches with valid cotes found for prediction.")
                     results = [] # No results to send if no predictable matches
                else:
                     logger.info(f"Generating predictions for {len(predictable_matches)} matches.")
                     results = [enriched_predict(m) for m in predictable_matches]
                     # The enriched_predict function logs errors internally now


                # --- STEP 4: SEND PREDICTIONS TO TELEGRAM ---
                logger.info(f"Sending {len(results)} predictions to Telegram...")
                await send_to_telegram_async(context, results)


        except Exception as e:
             logger.error(f"Erreur générale dans la boucle de scraping : {e}", exc_info=True) # Log exception info
             # Attempt to send a general error message to Telegram
             try:
                 await context.bot.send_message(chat_id=CHAT_ID, text=f"❌ Erreur lors de l'exécution de la boucle de scraping: {e}")
             except Exception as telegram_error:
                 logger.error(f"Erreur lors de l'envoi de l'erreur générale Telegram: {telegram_error}")


        # --- STEP 5: WAIT BEFORE NEXT CYCLE ---
        logger.info(f"😴 Attente avant le prochain scraping ({SCRAPING_INTERVAL_SECONDS} secondes)...")
        # Wait for the specified interval, respecting the stop_scraping_event
        try:
            await asyncio.wait_for(stop_scraping_event.wait(), timeout=SCRAPING_INTERVAL_SECONDS)
            # If wait_for completes without timeout, it means the event was set, so the loop should exit
            break # Exit the while loop

        except asyncio.TimeoutError:
            # Timeout occurred, means the event was NOT set, so we continue the loop
            pass

        except Exception as e:
            logger.error(f"Error during sleep interval: {e}", exc_info=True)
            # Continue loop despite sleep error, but maybe with a short delay
            await asyncio.sleep(5)


    logger.info("Scraping loop stopped.") # Log when the loop finishes


# =============== HANDLERS TELEGRAM ===============
# Command handlers for the Telegram bot

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global scraping_task, stop_scraping_event
    user_id = str(update.effective_user.id)

    # Check if the user is authorized
    if user_id != str(CHAT_ID):
        logger.warning(f"Unauthorized user {user_id} attempted /start")
        await update.message.reply_text("Désolé, vous n'êtes pas autorisé à utiliser ce bot.")
        return

    # Check if scraping is already running
    if scraping_task and not scraping_task.done():
        logger.info("Attempted /start, but scraping is already running.")
        await update.message.reply_text("La surveillance est déjà en cours.")
        return

    # Start the scraping loop task
    logger.info("Received /start command. Starting scraping loop...")
    stop_scraping_event = asyncio.Event() # Reset the stop event
    # Pass the context to the scraping loop
    scraping_task = asyncio.create_task(scraping_loop(context))
    await update.message.reply_text("Surveillance des matchs FIFA 5x5 démarrée !")


async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global scraping_task, stop_scraping_event
    user_id = str(update.effective_user.id)

    # Check if the user is authorized
    if user_id != str(CHAT_ID):
        logger.warning(f"Unauthorized user {user_id} attempted /stop")
        await update.message.reply_text("Désolé, vous n'êtes pas autorisé à utiliser ce bot.")
        return

    # Stop the scraping loop task
    if scraping_task and not scraping_task.done():
        logger.info("Received /stop command. Stopping scraping loop...")
        stop_scraping_event.set() # Signal the loop to stop
        # Wait for the scraping task to finish cleanly
        try:
            await asyncio.wait_for(scraping_task, timeout=SCRAPING_INTERVAL_SECONDS + 10) # Wait slightly longer than interval
            logger.info("Scraping task finished after stop signal.")
            await update.message.reply_text("Surveillance arrêtée.")
        except asyncio.TimeoutError:
            logger.warning("Scraping task did not finish cleanly after stop signal within timeout. It might be stuck.")
            await update.message.reply_text("Signal d'arrêt envoyé, mais la tâche de surveillance semble prendre du temps à s'arrêter.")
        except Exception as e:
             logger.error(f"Error while waiting for scraping task to stop: {e}")
             await update.message.reply_text(f"Erreur lors de l'arrêt de la surveillance: {e}")

        scraping_task = None # Reset task variable
        stop_scraping_event = None # Reset event variable
    else:
        logger.info("Attempted /stop, but no scraping is running.")
        await update.message.reply_text("Aucune surveillance n'est en cours.")


async def restart_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Received /restart command.")
    # Stop if running, then start
    await stop_command(update, context)
    # Add a small delay before starting to ensure resources are freed
    await asyncio.sleep(2)
    await start_command(update, context)


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global scraping_task
    user_id = str(update.effective_user.id)

    # Check if the user is authorized
    if user_id != str(CHAT_ID):
        logger.warning(f"Unauthorized user {user_id} attempted /status")
        await update.message.reply_text("Désolé, vous n'êtes pas autorisé à utiliser ce bot.")
        return

    # Report status
    if scraping_task and not scraping_task.done():
        logger.info("Received /status command. Reporting running.")
        await update.message.reply_text("La surveillance des matchs FIFA 5x5 est en cours.")
    else:
        logger.info("Received /status command. Reporting not running.")
        await update.message.reply_text("La surveillance des matchs FIFA 5x5 n'est pas en cours.")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
     # Check if the user is authorized - Help can be shown to anyone? Or only authorized?
     # Let's show help to authorized users to keep it simple based on previous checks
    if user_id != str(CHAT_ID):
        logger.warning(f"Unauthorized user {user_id} attempted /help")
        await update.message.reply_text("Désolé, vous n'êtes pas autorisé à utiliser ce bot.")
        return

    logger.info("Received /help command. Sending help message.")
    aide = (
        "🤖 <b>Menu d'aide du bot FIFA 5x5</b>\n\n"
        "<b>/start</b> : Démarrer la surveillance des matchs FIFA 5x5 (envoie des pronos toutes les minutes)\n"
        "<b>/stop</b> : Arrêter la surveillance\n"
        "<b>/restart</b> : Redémarrer la surveillance\n"
        "<b>/status</b> : Afficher l'état de la surveillance\n"
        "<b>/help</b> : Afficher ce menu d'aide\n\n"
        f"Seul l'utilisateur avec l'ID {CHAT_ID} peut contrôler le bot."
    )
    # Use update.message.reply_html for parse_mode="HTML"
    await update.message.reply_html(aide)


# =============== MAIN RUNNER (Synchronous Entry Point) ===============
# This is the main entry point that sets up the Telegram bot Application
# and starts the polling mechanism. It should NOT be async.
def main():
    # Create the Telegram Application builder
    application = Application.builder().token(BOT_TOKEN).build()

    # Add command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("stop", stop_command))
    application.add_handler(CommandHandler("restart", restart_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("help", help_command))

    # Start the bot
    logger.info("🚀 Bot Telegram de contrôle lancé. Envoie /start, /stop, /restart, /status, /help à ton bot pour piloter le scraping.")
    # run_polling is a synchronous method that will block and run the bot.
    # It manages its own asyncio event loop internally.
    application.run_polling(poll_interval=1.0) # Polling interval for Telegram updates

# This block executes when the script is run directly
if __name__ == "__main__":
    # Note: The Windows event loop policy is attempted to be set at the very top now.
    # asyncio.run() is NOT used here because application.run_polling() manages the loop.
    try:
        main() # Call the synchronous main function that starts the bot
    except KeyboardInterrupt:
        logger.info("\nArrêt du bot (interruption clavier).\n")
    except Exception as e:
        logger.error(f"Une erreur inattendue a provoqué l'arrêt du script : {e}", exc_info=True)
