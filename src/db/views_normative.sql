-- Athlete values placed against published normative data
--
-- The comparison is only meaningful when the reference population resembles the
-- athlete, so a row is matched on sport AND sex. An unmatched sex would compare
-- a woman against a male reference band, which is not a hard question to get
-- wrong and is deeply misleading when it happens.
--
-- Where a paper reports a 95% confidence interval or a plain range instead of a
-- standard deviation, no z-score is produced. A CI is uncertainty about the
-- mean, not the spread of athletes: dividing by it would make an athlete look
-- four standard deviations from normal when they are half of one.

drop view if exists v_normative_comparison;

create or replace view v_normative_comparison as
with latest as (
    select distinct on (h.athlete_id, h.metric_name, h.source)
        h.athlete_id, h.athlete_code, h.squad, h.sport,
        h.metric_name, h.display_name, h.unit, h.quality, h.quality_name, h.quality_order,
        h.higher_is_better, h.session_date, h.metric_value
    from v_metric_history h
    order by h.athlete_id, h.metric_name, h.source, h.session_date desc
)
select
    l.athlete_id,
    l.athlete_code,
    l.squad,
    l.sport,
    l.metric_name,
    l.display_name,
    l.unit,
    l.quality,
    l.quality_name,
    l.quality_order,
    l.higher_is_better,
    l.session_date,
    round(l.metric_value::numeric, 3)      as athlete_value,
    n.population,
    n.level                                as reference_level,
    n.n                                    as reference_n,
    n.mean_value                           as reference_mean,
    n.spread_type,
    n.sd_value                             as reference_sd,
    n.low_value                            as reference_low,
    n.high_value                           as reference_high,
    n.protocol_note,
    s.study_key,
    s.citation,
    s.doi,
    s.pmid,
    s.year                                 as reference_year,
    s.publication,
    s.source_url,
    s.note                                 as study_note,
    -- Difference from the reference mean, expressed so that positive always
    -- means the athlete is on the better side of it.
    round((
        (case l.higher_is_better when true then 1 when false then -1 else null end)
        * (l.metric_value - n.mean_value) / nullif(n.mean_value, 0) * 100
    )::numeric, 1)                          as pct_vs_reference,
    -- Only computed where the paper published a standard deviation.
    case when n.spread_type = 'sd' then
        round((
            (case l.higher_is_better when true then 1 when false then -1 else null end)
            * (l.metric_value - n.mean_value) / nullif(n.sd_value, 0)
        )::numeric, 2)
    end                                     as z_vs_reference,
    case
        when n.spread_type <> 'sd' then 'no_sd_published'
        when (case l.higher_is_better when true then 1 when false then -1 else null end)
             * (l.metric_value - n.mean_value) / nullif(n.sd_value, 0) >=  1.0 then 'above_reference'
        when (case l.higher_is_better when true then 1 when false then -1 else null end)
             * (l.metric_value - n.mean_value) / nullif(n.sd_value, 0) <= -1.0 then 'below_reference'
        else 'within_reference'
    end                                     as standing
from latest l
join normative_values n
  on n.metric_name = l.metric_name
 and n.sport = l.sport
join athletes a on a.athlete_id = l.athlete_id
 and (n.sex is null or n.sex = a.sex)
join reference_studies s on s.study_key = n.study_key;
