# TfL Public Transport Performance Analytics — RQ1 Pipeline

## Folder structure
```
collector/
  tfl_collector.py         <- run this on your own machine to start collecting real TfL data
scripts/
  build_trip_records.py    <- turns raw collected data into clean trip_records.csv
  generate_synthetic_data.py<- makes a realistic test dataset (already run, see data/)
  rq1_analysis.py           <- the full RQ1 statistical + visual analysis
data/
  synthetic_trip_records.csv <- test dataset (38,416 rows) so you can run RQ1 analysis today
outputs/
  rq1_analysis_report.txt   <- full text output: summary tables + Kruskal-Wallis/Dunn's results
  rq1_summary_by_*.csv      <- summary tables (route/time band/day/stop)
  rq1_*.png                 <- box plots, heatmaps, black-spot chart
```

## What to do next

### 1. Start real data collection NOW (takes weeks to accumulate — start early!)
On your own machine (this can't run in this sandbox — needs open internet access):
```
cd collector
pip install requests
python tfl_collector.py --routes 24,38,73 --once      # test it works
python tfl_collector.py --routes 24,38,73 --interval 60 # or use cron, see script docstring
```
Pick real routes relevant to your dissertation (mix of high/low frequency per
your methodology's sampling strategy, section 1.5.6).

Optional: get a free TfL API key (https://api-portal.tfl.gov.uk/signup) and
set `TFL_APP_KEY` as an environment variable for higher rate limits over a
multi-week collection period.

### 2. Once you have a few weeks of data
```
cd scripts
python build_trip_records.py --raw ../data/raw_snapshots.csv --out ../data/trip_records.csv
python rq1_analysis.py --data ../data/trip_records.csv --outdir ../outputs
```
Same schema as the synthetic data, so no code changes needed.

### 3. Read the important methodological note
`build_trip_records.py`'s docstring explains that TfL's live API only gives
real-time predictions, not scheduled timetables — so "delay" here is a proxy
(actual vs. first-predicted arrival), not strict schedule-adherence. For true
OTP, add a one-off join against `/Line/{id}/Timetable/{fromStopId}`. Mention
this proxy explicitly in your Limitations section (1.5.13) if you don't add
that join — it's a legitimate, literature-consistent workaround, just be
transparent about it.

## What RQ1 analysis currently gives you (on synthetic data — swap in real data later)
- Descriptive summary tables (OTP%, mean/median delay, headway CoV) by route,
  time band, day of week, and stop
- Kruskal-Wallis tests for each dimension (as specified in Table 1.5.2)
- Dunn's post-hoc pairwise comparisons where the KW test is significant
- Mann-Whitney U test for weekday vs weekend
- Box plots (delay by route, weekday vs weekend)
- Heatmaps (delay by day×time, OTP% by route×time band)
- Ranked "black spot" stops chart

## Extending to RQ2 / RQ3
Once RQ1 is solid on real data, RQ2 (diagnostic regression — needs weather/
traffic/calendar data joined on date+hour+stop) and RQ3 (predictive models)
can reuse this same trip_records.csv as their base. Ask me when you're ready
and I'll build those next.
