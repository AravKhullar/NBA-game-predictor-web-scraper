import streamlit as st
import pandas as pd
import joblib

# --- 1. Load model + player data ---
model = joblib.load("model.pkl")
players = pd.read_csv("players.csv", index_col=0)

# --- 2. Abbreviation ↔ Full-Name Mapping ---
abbr_to_full = {
    "ATL":"Atlanta Hawks", "BOS":"Boston Celtics", "BRK":"Brooklyn Nets",
    "CHA":"Charlotte Hornets", "CHI":"Chicago Bulls", "CLE":"Cleveland Cavaliers",
    "DAL":"Dallas Mavericks","DEN":"Denver Nuggets","DET":"Detroit Pistons",
    "GSW":"Golden State Warriors","HOU":"Houston Rockets","IND":"Indiana Pacers",
    "LAC":"Los Angeles Clippers","LAL":"Los Angeles Lakers","MEM":"Memphis Grizzlies",
    "MIA":"Miami Heat","MIL":"Milwaukee Bucks","MIN":"Minnesota Timberwolves",
    "NOP":"New Orleans Pelicans","NYK":"New York Knicks","OKC":"Oklahoma City Thunder",
    "ORL":"Orlando Magic","PHI":"Philadelphia 76ers","PHX":"Phoenix Suns",
    "POR":"Portland Trail Blazers","SAC":"Sacramento Kings","SAS":"San Antonio Spurs",
    "TOR":"Toronto Raptors","UTA":"Utah Jazz","WAS":"Washington Wizards"
}
# Invert for lookup
full_to_abbr = {full:abbr for abbr,full in abbr_to_full.items()}

# Build the same opp_code mapping your model saw:
abbrs_sorted = sorted(abbr_to_full.keys()) 
# these codes must match matches["Opponent"].astype("category").cat.codes 
abbr_to_code = {abbr:i for i,abbr in enumerate(abbrs_sorted)}

# --- 3. Clean + Prepare player data ---
def height_to_inches(ht):
    if isinstance(ht, str) and "-" in ht:
        f,i = ht.split("-")
        return int(f)*12 + int(i)
    return None

players["Ht"] = players["Ht"].apply(height_to_inches)
players["Exp"] = pd.to_numeric(players["Exp"], errors="coerce")
players["Season"] = players["Season"].astype(int)

# --- 4. Team aggregate function (weighted by MP) ---
def get_team_stats(abbr, season, df=players, top_n=8):
    tp = df[(df["Team"]==abbr) & (df["Season"]==season)]
    tp = tp.sort_values("MP", ascending=False).head(top_n)
    cols = ["MP","FG","FGA","3P","3PA","FT","FTA","TRB","AST","STL","BLK","TOV","PF","PTS"]
    w = tp["MP"] / tp["MP"].sum()
    return (tp[cols].T * w).T.sum()

# --- 5. Streamlit UI ---
st.title("🏀 NBA Match Predictor")

teams = sorted(full_to_abbr.keys())
your_team = st.selectbox("Your Team", teams)
opponent  = st.selectbox("Opponent",  teams, index=teams.index("Boston Celtics"))
venue     = st.radio("Venue", ["Home","Away"])
hour      = st.slider("Start Hour (0–23)", 0, 23, 19)
day_code  = st.slider("Day of Week (0=Mon,6=Sun)", 0, 6, 2)
streak    = st.slider("Win Streak (neg=losing)", -10, 10, 1)

if st.button("Predict Game Result"):
    season = 2024
    # map to abbr
    your_abbr = full_to_abbr[your_team]
    opp_abbr  = full_to_abbr[opponent]
    # pull real stats
    your_stats = get_team_stats(your_abbr, season)
    opp_stats  = get_team_stats(opp_abbr,  season)
    # assign home/away based on venue
    if venue=="Home":
        home_stats = your_stats.add_prefix("home_")
        away_stats = opp_stats.add_prefix("away_")
    else:
        home_stats = opp_stats.add_prefix("home_")
        away_stats = your_stats.add_prefix("away_")
    # build the input vector
    base = {
        "venue_code": 0 if venue=="Home" else 1,
        "opp_code":   abbr_to_code[opp_abbr],
        "hour":       hour,
        "day_code":   day_code,
        "streak_value": streak,
        # placeholder rolling features
        "point_diff_rolling": 3.0,
        "Tm_rolling":         112.0,
        "Opp_rolling":        109.0,
        "win_rolling":        0.6
    }
    # merge in the team stats
    row = pd.DataFrame([base])
    row = pd.concat([row, home_stats.to_frame().T, away_stats.to_frame().T], axis=1)
    row["point_diff_player_avg"] = row["home_PTS"] - row["away_PTS"]
    # reorder columns exactly
    row = row[model.feature_names_in_]
    # predict!
    pred = model.predict(row)[0]
    st.success("✅ Win" if pred==1 else "❌ Loss")
