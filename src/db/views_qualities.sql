-- Physical-quality analytics — catalogue-driven, polarity-aware
--
-- Nothing below names a specific test. Every metric is described by
-- metric_catalog, so a new battery becomes visible everywhere by inserting
-- catalogue rows. The one thing these views must get right is direction: a 505
-- time falling is an improvement, a jump height falling is not, and a single
-- unsigned "percent change" column would be wrong for seven of the metrics.

-- CREATE OR REPLACE VIEW can only append columns, never insert one in the
-- middle, so a column added to an existing view fails with "cannot change name
-- of view column". Dropping in dependency order keeps this file re-runnable.
-- Dependency order matters and it spans files: the report views in
-- views_report.sql are built on v_metric_history, so they have to go first even
-- though they are defined elsewhere. Listing them here rather than reaching for
-- DROP ... CASCADE keeps the teardown explicit -- CASCADE would happily remove
-- something nobody remembered was downstream.
drop view if exists v_squad_comparison;
drop view if exists v_metric_reliability;
drop view if exists v_recent_vs_prior;
drop view if exists v_test_day;
drop view if exists v_quality_profile;
drop view if exists v_metric_trend;
drop view if exists v_metric_history;

-- =====================================================================
-- v_metric_history — every measurement, with what it means attached
-- =====================================================================
create or replace view v_metric_history as
select
    a.athlete_id,
    a.athlete_code,
    a.squad,
    a.sport,
    s.session_date,
    s.session_type,
    m.metric_name,
    m.metric_value,
    m.source,
    mc.display_name,
    mc.unit,
    mc.quality,
    mc.higher_is_better,
    mc.is_headline,
    qc.display_name as quality_name,
    qc.sort_order   as quality_order
from performance_metrics m
join sessions s        using (session_id)
join athletes a        using (athlete_id)
join metric_catalog mc on mc.metric_name = m.metric_name
join quality_catalog qc on qc.quality = mc.quality;

-- =====================================================================
-- v_metric_trend — first vs latest, per athlete per metric
-- ---------------------------------------------------------------------
-- `pct_change` is the raw arithmetic change. `pct_improvement` applies the
-- metric's polarity, so positive always means the athlete got better. Only the
-- second one is safe to sort by, average, or hand to a language model.
--
-- Change is also expressed in the athlete's OWN standard deviation for that
-- metric. A 2% shift means something different for a 10 m sprint (where 2% is
-- several typical errors) than for an IMTP RFD (where it is noise), and a
-- single percentage threshold across a mixed battery would flag the wrong
-- things.
--
-- Direction is taken from a least-squares slope over ALL tests, not from first
-- versus latest. Comparing two endpoints throws away every measurement in
-- between and inherits the full test-retest noise of both: a Yo-Yo IR1 with 6%
-- typical error read as a 5% decline for an athlete who was in fact improving,
-- purely because of which two days happened to be first and last. regr_slope
-- uses the whole series; regr_r2 says how much of the variation the line
-- actually explains, so a confident-looking trend through scattered points can
-- be recognised as one.
-- =====================================================================
create or replace view v_metric_trend as
with ranked as (
    select
        athlete_id, athlete_code, squad, metric_name, display_name, unit,
        quality, quality_name, quality_order, higher_is_better, is_headline, source,
        first_value(metric_value) over w_asc  as first_value,
        first_value(session_date) over w_asc  as first_date,
        last_value(metric_value)  over w_full as latest_value,
        last_value(session_date)  over w_full as latest_date,
        count(*)                  over w_part as n_tests,
        stddev_samp(metric_value) over w_part as own_sd,
        avg(metric_value)         over w_part as own_mean,
        regr_slope(metric_value, extract(epoch from session_date::timestamp) / 86400.0)
                                  over w_part as slope_per_day,
        regr_r2(metric_value, extract(epoch from session_date::timestamp) / 86400.0)
                                  over w_part as trend_r2
    from v_metric_history
    window
        w_part as (partition by athlete_id, metric_name, source),
        w_asc  as (partition by athlete_id, metric_name, source order by session_date),
        w_full as (partition by athlete_id, metric_name, source order by session_date
                   rows between unbounded preceding and unbounded following)
)
select distinct
    athlete_id, athlete_code, squad,
    metric_name, display_name, unit, quality, quality_name, quality_order,
    higher_is_better, is_headline, source, n_tests,
    first_date,  round(first_value::numeric, 3)  as first_value,
    latest_date, round(latest_value::numeric, 3) as latest_value,
    round(own_sd::numeric, 4)                    as own_sd,
    round(((latest_value - first_value) / nullif(first_value, 0) * 100)::numeric, 1)
        as pct_change,
    round(((case higher_is_better when true then 1 when false then -1 else null end)
           * (latest_value - first_value) / nullif(first_value, 0) * 100)::numeric, 1)
        as pct_improvement,
    round(((case higher_is_better when true then 1 when false then -1 else null end)
           * (latest_value - first_value) / nullif(own_sd, 0))::numeric, 2)
        as change_in_sd,
    -- fitted change across the whole series, which is what direction uses
    round((slope_per_day * (latest_date - first_date))::numeric, 4)
        as fitted_change,
    round(((case higher_is_better when true then 1 when false then -1 else null end)
           * slope_per_day * (latest_date - first_date)
           / nullif(first_value, 0) * 100)::numeric, 1)
        as pct_improvement_fitted,
    round(((case higher_is_better when true then 1 when false then -1 else null end)
           * slope_per_day * (latest_date - first_date)
           / nullif(own_sd, 0))::numeric, 2)
        as fitted_change_in_sd,
    round(trend_r2::numeric, 2) as trend_r2
from ranked;

-- =====================================================================
-- v_quality_profile — one row per athlete per physical quality
-- ---------------------------------------------------------------------
-- The headline metric for each quality, plus a direction. The threshold is one
-- of the athlete's own standard deviations: a change smaller than the variation
-- they already show between tests is not evidence of anything.
-- =====================================================================
create or replace view v_quality_profile as
select
    t.athlete_id,
    t.athlete_code,
    t.squad,
    t.quality,
    t.quality_name,
    t.quality_order,
    t.metric_name,
    t.display_name,
    t.unit,
    t.n_tests,
    t.first_date,
    t.first_value,
    t.latest_date,
    t.latest_value,
    t.pct_improvement,
    t.pct_improvement_fitted,
    t.change_in_sd,
    t.fitted_change_in_sd,
    t.trend_r2,
    case
        when t.n_tests < 3 or t.own_sd is null or t.own_sd = 0 then 'insufficient_data'
        when t.fitted_change_in_sd >=  1.0 then 'improving'
        when t.fitted_change_in_sd <= -1.0 then 'declining'
        else 'stable'
    end as direction
from v_metric_trend t
where t.is_headline;

-- =====================================================================
-- v_test_day — everything measured on one athlete-day
-- ---------------------------------------------------------------------
-- What a coach opens when they ask "what did we do with this athlete on
-- Tuesday". Includes the athlete's own mean for context, so a single number
-- is readable without going and finding the history.
-- =====================================================================
create or replace view v_test_day as
select
    h.athlete_id,
    h.athlete_code,
    h.squad,
    h.session_date,
    h.session_type,
    h.quality,
    h.quality_name,
    h.quality_order,
    h.metric_name,
    h.display_name,
    h.unit,
    h.source,
    h.higher_is_better,
    h.is_headline,
    round(h.metric_value::numeric, 3) as value,
    round(t.own_mean_r::numeric, 3)   as athlete_mean,
    round((case h.higher_is_better when true then 1 when false then -1 else null end
           * (h.metric_value - t.own_mean_r) / nullif(t.own_sd_r, 0))::numeric, 2)
        as z_vs_own_mean
from v_metric_history h
left join (
    select athlete_id, metric_name, source,
           avg(metric_value)         as own_mean_r,
           stddev_samp(metric_value) as own_sd_r
    from v_metric_history
    group by athlete_id, metric_name, source
) t on t.athlete_id = h.athlete_id
   and t.metric_name = h.metric_name
   and t.source = h.source;
