import streamlit as st
import pandas as pd
import numpy as np
import fastf1
from datetime import datetime
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key from .env
load_dotenv()
GEMINI_API_KEY = os.getenv("F1_GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Enable FastF1 cache
fastf1.Cache.enable_cache('cache_dir')

# Title
st.title("🏁 F1 Fantasy 2025: AI Team Optimizer")

st.sidebar.header("💸 Fantasy Prices (Editable Tables)")

# Editable Driver Prices (Data Editor)
default_driver_prices = {
    "Lando Norris": 29.0, "Max Verstappen": 28.4, "Charles Leclerc": 25.9, "Lewis Hamilton": 24.2,
    "Oscar Piastri": 23.0, "George Russell": 21.0, "Kimi Antonelli": 18.4, "Liam Lawson": 18.0,
    "Carlos Sainz": 13.1, "Alex Albon": 12.0, "Pierre Gasly": 11.8, "Yuki Tsunoda": 9.6,
    "Fernando Alonso": 8.8, "Lance Stroll": 8.1, "Esteban Ocon": 7.3, "Jack Doohan": 7.2,
    "Oliver Bearman": 6.7, "Nico Hulkenberg": 6.4, "Isack Hadjar": 6.2, "Gabriel Bortoleto": 6.0
}
driver_df = pd.DataFrame({
    "Driver": list(default_driver_prices.keys()),
    "Price": list(default_driver_prices.values())
})
st.sidebar.subheader("📝 Edit Driver Prices")
edited_driver_df = st.sidebar.data_editor(driver_df, use_container_width=True, num_rows="fixed")
driver_prices = dict(zip(edited_driver_df["Driver"], edited_driver_df["Price"]))

# Editable Constructor Prices (Data Editor)
default_constructor_prices = {
    "McLaren": 30.0, "Ferrari": 27.1, "Red Bull Racing": 25.2, "Mercedes": 22.7,
    "Williams": 13.1, "Alpine": 9.5, "Aston Martin": 8.5, "Racing Bulls": 8.0,
    "Haas": 7.0, "Sauber": 6.2
}
constructor_df = pd.DataFrame({
    "Constructor": list(default_constructor_prices.keys()),
    "Price": list(default_constructor_prices.values())
})
st.sidebar.subheader("🛠️ Edit Constructor Prices")
edited_constructor_df = st.sidebar.data_editor(constructor_df, use_container_width=True, num_rows="fixed")
constructor_prices = dict(zip(edited_constructor_df["Constructor"], edited_constructor_df["Price"]))

# Input current team
st.subheader("🧑‍💻 Enter Your Current Team")
driver_options = list(driver_prices.keys())
constructor_options = list(constructor_prices.keys())

current_drivers = st.multiselect("Select Your 5 Drivers", driver_options, max_selections=5)
current_teams = st.multiselect("Select Your 2 Constructors", constructor_options, max_selections=2)

# Free transfers input
free_transfers = st.number_input("🔁 Number of Free Transfers Available", min_value=0, max_value=3, value=2)

# Load past race results
@st.cache_data
def get_completed_race_data(year):
    schedule = fastf1.get_event_schedule(year)
    today = pd.Timestamp(datetime.today())
    past_events = schedule[schedule['EventDate'] < today]
    results = []

    for _, event in past_events.iterrows():
        try:
            session = fastf1.get_session(year, event['RoundNumber'], 'R')
            session.load()
            race_data = session.results[['FullName', 'TeamName', 'Position', 'Points']]
            race_data['Round'] = event['RoundNumber']
            results.append(race_data)
        except:
            continue

    return pd.concat(results) if results else pd.DataFrame()

# Prepare data for Gemini prompt
# Prepare data for Gemini prompt
def generate_prompt(driver_prices, constructor_prices, current_team, free_transfers, results_df):
    rules_2025 = """
📌 What's New for F1 Fantasy 2025:

**Transfers:**
- 2 free transfers per race week (1 transfer = 1 in and 1 out). Additional transfers = -10 points each.
- 1 unused transfer can carry over to the next race (max 2).
- If a driver races in Sprint but is replaced before GP Qualifying, the game may suggest a swap (counts toward your transfers).

**Chips:**
- 🟢 Autopilot: Automatically assigns DRS Boost (2x) to the highest scoring driver in your team.
- 🔴 Extra DRS Boost: Gives one driver a 3x multiplier. You can still assign 2x to another driver.
- 🛡️ No Negative: Negative points are nullified per scoring category.
- ♻️ Wildcard: Unlimited transfers, still within $100M budget.
- 💰 Limitless: Unlimited transfers with no budget limit for one race week.
- 🔄 Final Fix: One driver swap after Qualifying (2x DRS carries over to the new driver).

**DRS Boost (2x):**
- Can be assigned weekly to **one driver in your team**.
- If "Extra DRS Boost" chip is active, assign both a 2x and a 3x DRS Boost to two separate drivers.

**Pitstops:**
- Constructors earn points for fastest pitstop time:
  - <2.0s = 20 pts
  - 2.00–2.19s = 10 pts
  - 2.20–2.49s = 5 pts
  - 2.50–2.99s = 2 pts
  - >3.0s = 0 pts
- 5 pt bonus for fastest pitstop in race
- 15 pt bonus for breaking the world record (<1.8s)

**Disqualification Penalties:**
- Drivers receive -20 points for DNF/Not Classified
- Disqualified drivers only get -20
- Constructors receive an **additional** penalty:
  - Qualifying: -5
  - Race/Sprint: -10

**Qualifying Scoring:**
- Drivers score from 10 pts (P1) to 1 pt (P10), then 0, NC = -5
- Constructors total both drivers’ points + Q2/Q3 bonuses

**Sprint & Race Scoring:**
- Position gains/losses, overtakes, Fastest Lap, Driver of the Day all add points
- Constructors total both drivers’ scores
- DNF = -20, DSQ = -20 + constructor penalty
"""
# Budget calculations
    driver_cost = sum(driver_prices.get(d, 0) for d in current_team['drivers'])
    constructor_cost = sum(constructor_prices.get(c, 0) for c in current_team['constructors'])
    team_budget = driver_cost + constructor_cost

    prompt = f"""
You are an F1 Fantasy strategist. Based on 2025 rules, real data, and my current selections, generate the **best team strategy** for this week.

This is the current 2025 driver line up: {', '.join(driver_prices.keys())}.
This is the current 2025 constructor line up: {', '.join(constructor_prices.keys())}.

🎯 OBJECTIVE:
- Suggest exactly **2 transfers** (2 OUT and 2 IN) — drivers or constructors.
  - OUT must be from my current team
  - IN must be new players not already in the team
- Use **at most ${100 - team_budget:.1f}M** for the 2 IN players (unless using Limitless chip).
- Recommend the best chip to use: Autopilot, Extra DRS Boost, No Negative, Wildcard, Limitless, Final Fix.
- Always assign **2x DRS Boost** to one of my current drivers: {', '.join(current_team['drivers'])}
- If using "Extra DRS Boost" chip:
  - Apply **3x DRS** to one current driver. The best one you think. Preferably a top scoring driver and in a top team
  - Apply **2x DRS** to a different current driver. The best one you think. Preferably a top scoring driver and in a top team
- 🚫 Never assign DRS to drivers not in the current team.
- 🚫 Never exceed $100M team cost unless using the Limitless chip.

💰 Current Team Budget: ${team_budget:.1f}M
💸 Budget Remaining for Transfers: ${100 - team_budget:.1f}M

👥 My Current Team:
Drivers: {', '.join(current_team['drivers'])}
Constructors: {', '.join(current_team['constructors'])}

📊 Driver Prices:
{pd.Series(driver_prices).to_string()}

🏎️ Constructor Prices:
{pd.Series(constructor_prices).to_string()}

📈 Driver Points So Far:
{results_df.groupby('FullName')['Points'].sum().to_string()}

📉 Constructor Points So Far:
{results_df.groupby('TeamName')['Points'].sum().to_string()}

📚 F1 Fantasy 2025 Rules:
{rules_2025}

🧠 Return your response in this exact format (nothing else):

- 2 OUT: [Player/Team Name 1, Player/Team Name 2]
- 2 IN: [Player/Team Name 1, Player/Team Name 2]
- CHIP: (Autopilot, Extra DRS Boost, No Negative, Wildcard, Limitless, Final Fix)
- BOOST:
    - 2x: (Driver name from current team)
    - 3x: (Driver name from current team — only if using Extra DRS Boost)
- REASON: (Short explanation)
"""
    return prompt




# Generate AI Strategy Button
if st.button("🔮 Get AI Fantasy Strategy"):
    if len(current_drivers) != 5 or len(current_teams) != 2:
        st.error("Please select exactly 5 drivers and 2 constructors.")
    else:
        st.info("Generating strategy using real data and AI analysis...")
        results_df = get_completed_race_data(2025)

        if results_df.empty:
            st.warning("No race data available yet.")
        else:
            prompt = generate_prompt(
                driver_prices,
                constructor_prices,
                {"drivers": current_drivers, "constructors": current_teams},
                free_transfers,
                results_df
            )
            try:
                model = genai.GenerativeModel("gemini-2.0-flash")
                response = model.generate_content(prompt)
                st.success("🚀 Here's your AI-optimized F1 Fantasy Strategy:")
                st.markdown(response.text)
            except Exception as e:
                st.error(f"❌ Gemini API Error: {e}")
