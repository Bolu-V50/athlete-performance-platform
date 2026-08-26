-- Analytics layer — Phase 4
--
-- These live as views rather than Python so the metrics have one definition
-- that the dashboard, the pipeline and any ad-hoc SQL all share. If ACWR is
-- computed in three places it will eventually mean three different things.

-- =====================================================================
-- v_daily_load — dense daily load series
-- ---------------------------------------------------------------------
-- Rest days must appear as zero-load rows. If you only EWMA the days an
-- athlete actually trained, acute load never decays across a rest week and
-- ACWR reads high for someone who has been resting -- the exact opposite of
-- the truth. generate_series fills the gaps.
-- =====================================================================
create or replace view v_daily_load as
with bounds as (
    select athlete_id, min(date) as d0, max(date) as d1
    from training_load
    group by athlete_id
)
select
    b.athlete_id,
    g.day::date                              as date,
    coalesce(t.session_load, 0)::numeric     as session_load,
    row_number() over (partition by b.athlete_id order by g.day) as rn
from bounds b
cross join lateral generate_series(b.d0, b.d1, interval '1 day') g(day)
left join training_load t
       on t.athlete_id = b.athlete_id
      and t.date = g.day::date;

-- =====================================================================
-- v_acwr — acute:chronic workload ratio, EWMA form
-- ---------------------------------------------------------------------
-- Rolling-average ACWR weights a session from 27 days ago exactly as heavily
-- as yesterday's, which is not how fitness or fatigue decay. The EWMA form
-- (Williams et al. 2017) discounts the past geometrically.
--   lambda_acute   = 2/(7+1)  = 0.25
--   lambda_chronic = 2/(28+1) = 0.068966
-- A recursive CTE is genuinely required here: each day's EWMA depends on the
-- previous day's EWMA, which no plain window function can express.
-- =====================================================================
create or replace view v_acwr as
with recursive ewma as (
    select
        athlete_id, date, session_load, rn,
        session_load as acute,
        session_load as chronic
    from v_daily_load
    where rn = 1

    union all

    select
        d.athlete_id, d.date, d.session_load, d.rn,
        d.session_load * 0.250000 + e.acute   * 0.750000,
        d.session_load * 0.068966 + e.chronic * 0.931034
    from ewma e
    join v_daily_load d
      on d.athlete_id = e.athlete_id
     and d.rn = e.rn + 1
)
select
    athlete_id,
    date,
    round(session_load, 1)                      as session_load,
    round(acute, 1)                             as acute_load,
    round(chronic, 1)                           as chronic_load,
    case when chronic > 0 then round(acute / chronic, 3) end as acwr,
    -- The 0.80-1.30 "sweet spot" is a heuristic from the team-sport
    -- literature, not a law. It is a conversation starter for a coach, not an
    -- automatic instruction, and the README says so.
    case
        when rn < 28      then 'insufficient_history'
        when chronic <= 0 then 'insufficient_history'
        when acute / chronic <  0.80 then 'undertrained'
        when acute / chronic <= 1.30 then 'sweet_spot'
        when acute / chronic <= 1.50 then 'caution'
        else 'high_risk'
    end                                          as acwr_zone,
    rn                                           as days_of_history
from ewma;

-- =====================================================================
-- v_cmj_baseline — individual 28-day rolling baseline
-- ---------------------------------------------------------------------
-- The window is RANGE over an interval, not ROWS. With CMJ testing twice a
-- week, "27 preceding rows" is about thirteen weeks of history, not 28 days --
-- a silent and serious error. RANGE '28 days' means 28 days regardless of how
-- often the athlete was tested.
--
-- The window excludes the current day (`1 day preceding`): a baseline that
-- contains today's value is partly explained by it, which shrinks the very
-- deviation the flag exists to detect.
-- =====================================================================
create or replace view v_cmj_baseline as
select
    a.athlete_code,
    a.squad,
    a.sport,
    s.athlete_id,
    s.session_date,
    m.metric_value                    as jump_height_m,
    avg(m.metric_value)          over w as baseline_mean,
    stddev_samp(m.metric_value)  over w as baseline_sd,
    count(*)                     over w as baseline_n
from performance_metrics m
join sessions s using (session_id)
join athletes a using (athlete_id)
where m.metric_name = 'jump_height_m'
  and m.source = 'force_plate'
window w as (
    partition by s.athlete_id
    order by s.session_date
    range between interval '28 days' preceding and interval '1 day' preceding
);

-- =====================================================================
-- v_cmj_flags — neuromuscular fatigue flag
-- ---------------------------------------------------------------------
-- A z-score needs a defensible denominator. Fewer than four prior trials in
-- the window gives an SD that is mostly noise, so those days return no flag
-- rather than a confident-looking wrong one.
-- =====================================================================
create or replace view v_cmj_flags as
select
    b.*,
    case
        when baseline_n >= 4 and baseline_sd > 0
        then round(((jump_height_m - baseline_mean) / baseline_sd)::numeric, 2)
    end as z_score,
    case
        when baseline_n < 4 or baseline_sd is null or baseline_sd = 0 then 'no_baseline'
        when (jump_height_m - baseline_mean) / baseline_sd <= -1.5 then 'flag'
        when (jump_height_m - baseline_mean) / baseline_sd <= -1.0 then 'watch'
        else 'normal'
    end as baseline_status
from v_cmj_baseline b;

-- =====================================================================
-- v_athlete_status — one row per athlete: the squad overview
-- ---------------------------------------------------------------------
-- Latest CMJ and latest ACWR side by side. This is what the dashboard's
-- alert table reads, and what the daily briefing summarises.
-- =====================================================================
create or replace view v_athlete_status as
with last_cmj as (
    select distinct on (athlete_id)
        athlete_id, athlete_code, squad, sport, session_date,
        jump_height_m, baseline_mean, baseline_sd, baseline_n,
        z_score, baseline_status
    from v_cmj_flags
    order by athlete_id, session_date desc
),
last_load as (
    select distinct on (athlete_id)
        athlete_id, date as load_date, acute_load, chronic_load, acwr, acwr_zone
    from v_acwr
    order by athlete_id, date desc
)
select
    a.athlete_id,
    a.athlete_code,
    a.squad,
    a.sport,
    c.session_date          as last_cmj_date,
    round(c.jump_height_m::numeric, 3)  as jump_height_m,
    round(c.baseline_mean::numeric, 3)  as baseline_mean_m,
    round(c.baseline_sd::numeric, 4)    as baseline_sd_m,
    c.baseline_n,
    c.z_score,
    coalesce(c.baseline_status, 'no_data') as baseline_status,
    l.load_date,
    l.acute_load,
    l.chronic_load,
    l.acwr,
    coalesce(l.acwr_zone, 'no_data')       as acwr_zone,
    -- Priority for the coach's attention queue. A neuromuscular flag outranks
    -- a load-ratio warning: the jump is a measurement of the athlete, whereas
    -- ACWR is an inference about accumulated exposure.
    case
        when c.baseline_status = 'flag'  then 1
        when l.acwr_zone = 'high_risk'   then 2
        when c.baseline_status = 'watch' then 3
        when l.acwr_zone = 'caution'     then 4
        else 5
    end as attention_rank
from athletes a
left join last_cmj  c on c.athlete_id = a.athlete_id
left join last_load l on l.athlete_id = a.athlete_id;
