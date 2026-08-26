-- Analytics that exist for the athlete report
--
-- The capability profile answers "which way is this quality going". A review
-- report has to answer four more questions that a trend line cannot:
--   * is this athlete a repeatable tester, or is the metric too noisy to read?
--   * where do they sit against the people they train with?
--   * is the trend recent, or is it old news the coach has already acted on?
--   * how much testing is this conclusion actually resting on?

drop view if exists v_squad_comparison;
drop view if exists v_metric_reliability;
drop view if exists v_recent_vs_prior;

-- =====================================================================
-- v_metric_reliability — how repeatable is this athlete on this metric
-- ---------------------------------------------------------------------
-- Coefficient of variation across the athlete's own tests. A 4% change means
-- something when the athlete's CV is 1.2% and nothing when it is 9%. Reporting
-- a trend without this is how a monitoring system manufactures findings.
--
-- CV is only meaningful on a ratio scale with a positive mean, so metrics that
-- can be negative (countermovement depth) return NULL rather than a number that
-- looks fine and means nothing.
-- =====================================================================
create or replace view v_metric_reliability as
select
    athlete_id,
    athlete_code,
    metric_name,
    display_name,
    unit,
    quality,
    quality_name,
    quality_order,
    is_headline,
    count(*)                                as n_tests,
    round(avg(metric_value)::numeric, 3)    as mean_value,
    round(stddev_samp(metric_value)::numeric, 4) as sd_value,
    case when min(metric_value) > 0 then
        round((stddev_samp(metric_value) / nullif(avg(metric_value), 0) * 100)::numeric, 1)
    end                                     as cv_pct
from v_metric_history
group by athlete_id, athlete_code, metric_name, display_name, unit,
         quality, quality_name, quality_order, is_headline;

-- =====================================================================
-- v_squad_comparison — where they sit among the people they train with
-- ---------------------------------------------------------------------
-- Compared within SQUAD, never across the programme. A swimmer and a
-- basketballer share no normative band for anything, and a z-score computed
-- across both is a number about the roster rather than about the athlete.
-- Rank 1 is always the best performer, whichever direction the metric runs.
-- =====================================================================
create or replace view v_squad_comparison as
with latest as (
    select distinct on (athlete_id, metric_name, source)
        athlete_id, athlete_code, squad, metric_name, display_name, unit,
        quality, quality_name, quality_order, higher_is_better,
        session_date, metric_value
    from v_metric_history
    where is_headline
    order by athlete_id, metric_name, source, session_date desc
)
select
    l.*,
    round(avg(metric_value)         over w::numeric, 3) as squad_mean,
    round(stddev_samp(metric_value) over w::numeric, 4) as squad_sd,
    count(*) over w                                     as squad_n,
    round((
        (case higher_is_better when true then 1 when false then -1 else null end)
        * (metric_value - avg(metric_value) over w)
        / nullif(stddev_samp(metric_value) over w, 0)
    )::numeric, 2)                                      as z_vs_squad,
    rank() over (
        partition by squad, metric_name
        order by case higher_is_better when true then -metric_value else metric_value end
    )                                                   as squad_rank
from latest l
window w as (partition by squad, metric_name);

-- =====================================================================
-- v_recent_vs_prior — is the trend recent, or old news?
-- ---------------------------------------------------------------------
-- A season-long slope cannot tell a coach whether something is happening now or
-- happened in March and has already been dealt with. This compares the mean of
-- the last six weeks against the six weeks before, anchored on the athlete's own
-- most recent test rather than on today, so an athlete who has not been tested
-- for a month is not silently compared against an empty window.
-- =====================================================================
create or replace view v_recent_vs_prior as
with anchor as (
    select athlete_id, max(session_date) as ref_date
    from v_metric_history group by athlete_id
),
windowed as (
    select
        h.athlete_id, h.athlete_code, h.metric_name, h.display_name, h.unit,
        h.quality, h.quality_name, h.quality_order, h.higher_is_better, h.is_headline,
        case
            when h.session_date >  a.ref_date - interval '42 days' then 'recent'
            when h.session_date >  a.ref_date - interval '84 days' then 'prior'
        end as bucket,
        h.metric_value
    from v_metric_history h
    join anchor a using (athlete_id)
)
select
    athlete_id, athlete_code, metric_name, display_name, unit,
    quality, quality_name, quality_order, higher_is_better, is_headline,
    count(*) filter (where bucket = 'recent')                     as n_recent,
    count(*) filter (where bucket = 'prior')                      as n_prior,
    round(avg(metric_value) filter (where bucket = 'recent')::numeric, 3) as recent_mean,
    round(avg(metric_value) filter (where bucket = 'prior')::numeric, 3)  as prior_mean,
    round((
        (case higher_is_better when true then 1 when false then -1 else null end)
        * (avg(metric_value) filter (where bucket = 'recent')
           - avg(metric_value) filter (where bucket = 'prior'))
        / nullif(avg(metric_value) filter (where bucket = 'prior'), 0) * 100
    )::numeric, 1)                                                as recent_pct_change
from windowed
where bucket is not null
group by athlete_id, athlete_code, metric_name, display_name, unit,
         quality, quality_name, quality_order, higher_is_better, is_headline;
