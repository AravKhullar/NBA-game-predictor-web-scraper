import streamlit as st
import pandas as pd
import joblib

# --- 1. Load model + data ---
model = joblib.load("model.pkl")
matches = pd.read_csv("matches.csv", index_col=0)
players = pd.read_csv("players.csv", index_col=0)

# --- 2. Notebook‐style feature engineering on matches ---
matches["Date"] = pd.to_datetime(matches["Date"])
matches["venue_code"]   = matches["Venue"].astype("category").cat.codes
matches["opp_code"]     = matches["Opponent"].astype("category").cat.codes
matches["hour"]         = matches["Start (ET)"].str.replace(":.+", "", regex=True).astype(int)
matches["day_code"]     = matches["Date"].dt.dayofweek
matches["target"]       = (matches["Result"] == "W").astype(int)
matches["point_diff"]   = matches["Tm"] - matches["Opp"]
matches["win_flag"]     = (matches["Result"] == "W").astype(int)
matches["streak_value"] = matches["Streak"]\
                             .str.replace("W ", "")\
                             .str.replace("L ", "-")\
                             .astype(int)
matches["streak_value"] = matches.groupby("Team")["streak_value"].shift(1)

# --- 3. Clean & prepare player data ---
def height_to_inches(ht):
    if isinstance(ht, str) and "-" in ht:
        f,i = ht.split("-")
        return int(f)*12 + int(i)
    return None

players["Ht"]     = players["Ht"].apply(height_to_inches)
players["Exp"]    = pd.to_numeric(players["Exp"], errors="coerce")
players["Season"] = players["Season"].astype(int)

# --- 4. Weighted team‐stats function from notebook ---
def get_team_stats(abbr, season, df=players, top_n=8):
    tp = df[(df["Team"]==abbr) & (df["Season"]==season)]
    tp = tp.sort_values("MP", ascending=False).head(top_n)
    cols = ["MP","FG","FGA","3P","3PA","FT","FTA","TRB","AST","STL","BLK","TOV","PF","PTS"]
    w = tp["MP"] / tp["MP"].sum()
    return (tp[cols].T * w).T.sum()

# --- 5. Rolling stats helper (last 4 games) ---
def get_recent_stats(abbr):
    recent = matches[matches["Team"]==abbr].sort_values("Date").tail(4)
    return {
      "point_diff_rolling": recent["point_diff"].mean(),
      "Tm_rolling":         recent["Tm"].mean(),
      "Opp_rolling":        recent["Opp"].mean(),
      "win_rolling":        recent["win_flag"].mean()
    }

# --- 6. Mapping full names ↔ abbreviations & codes ---
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
full_to_abbr = {full:abbr for abbr,full in abbr_to_full.items()}
abbrs_sorted = sorted(abbr_to_full.keys())
abbr_to_code = {abbr:i for i,abbr in enumerate(abbrs_sorted)}

# --- 7. Streamlit UI ---
st.title("🏀 NBA Match Predictor")

teams = sorted(full_to_abbr.keys())
your_team = st.selectbox("Your Team", teams)
opponent  = st.selectbox("Opponent",  teams, index=teams.index("Boston Celtics"))
venue     = st.radio("Venue", ["Home","Away"])
hour      = st.slider("Start Hour (1 - 12 PM)", 1, 12, 1)
day_code  = st.slider("Day of Week (0=Mon,6=Sun)",    0, 6, 2)
streak    = st.slider("Win Streak (neg = losing)", -10,10,1)

if st.button("Predict Game Result"):
    season   = 2024
    your_abbr= full_to_abbr[your_team]
    opp_abbr = full_to_abbr[opponent]

    # real rolling stats
    recent = get_recent_stats(your_abbr)
    
    # team‐stats from players.csv
    your_stats = get_team_stats(your_abbr, season)
    opp_stats  = get_team_stats(opp_abbr, season)

    # assign home_/away_ prefixes
    home_stats = your_stats.add_prefix("home_")
    away_stats = opp_stats.add_prefix("away_")

    # base features + rolling
    base = {
        "venue_code":     1 if venue=="Home" else 0,
        "opp_code":       abbr_to_code[opp_abbr],
        "hour":           hour,
        "day_code":       day_code,
        "streak_value":   streak,
        **recent
    }

    # build full input row
    row = pd.DataFrame([base])
    row = pd.concat([row, home_stats.to_frame().T, away_stats.to_frame().T], axis=1)
    row["point_diff_player_avg"] = row["home_PTS"] - row["away_PTS"]

    # debug: inspect what the model sees
    prob = model.predict_proba(row[model.feature_names_in_])[0]
    st.write(f"🔍 Probabilities → Loss: {prob[0]:.2f}, Win: {prob[1]:.2f}")

    # final predict
    pred = model.predict(row[model.feature_names_in_])[0]
    st.success("✅ Win" if pred==1 else "❌ Loss")
