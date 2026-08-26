-- Metric catalogue — the dictionary that makes a long-format metrics table usable
--
-- A long table absorbs any metric as data. The cost is that nothing in the table
-- itself says what a metric means, what unit it is in, or -- critically -- which
-- direction is an improvement. A 505 agility time falling from 2.42 s to 2.35 s
-- is progress; a jump height falling is not. Without this table every z-score,
-- trend arrow and "most improved" statement is a coin flip on sign.
--
-- This is also what lets the dashboard render a test it has never seen before:
-- add a row here and the metric appears with the right unit, grouping and
-- polarity, with no code change.

create table if not exists metric_catalog (
    metric_name      text primary key,
    display_name     text not null,
    unit             text not null,
    quality          text not null,     -- the physical quality it belongs to
    -- A metric can have more than one producer: body mass falls out of a CMJ
    -- weighing period as well as a formal anthropometry session, and the two
    -- coexist in performance_metrics because the unique key includes `source`.
    -- These two columns therefore name the PRIMARY producer, used to route a
    -- wide export to the right session type. Display metadata below (unit,
    -- quality, polarity) is source-independent and is what the analytics join on.
    session_type     text not null,     -- primary test that produces it
    source           text not null,     -- primary device / method
    -- NULL means the metric has no meaningful direction: a phase duration is
    -- context, not an achievement, and forcing a polarity onto it would make
    -- every trend and z-score computed from it a fabrication. Views return NULL
    -- for improvement and z on these rather than guessing.
    higher_is_better boolean,
    is_headline      boolean not null default false,
    typical_min      numeric,           -- physiological acceptance range
    typical_max      numeric,
    description      text
);

alter table metric_catalog alter column higher_is_better drop not null;

create index if not exists idx_catalog_quality on metric_catalog(quality);
create index if not exists idx_catalog_session on metric_catalog(session_type);

insert into metric_catalog (metric_name, display_name, unit, quality, session_type, source,
                            higher_is_better, is_headline, typical_min, typical_max, description) values
-- ---- neuromuscular power: countermovement jump (force plate) ----
('jump_height_m',             'CMJ jump height',        'm',      'power',        'CMJ_test',    'force_plate',      true,  true,  0.05,  1.20, 'Impulse-momentum jump height from the vertical GRF trace'),
('rsi_mod',                   'RSI-modified',           'm/s',    'power',        'CMJ_test',    'force_plate',      true,  true,  0.05,  1.20, 'Jump height divided by time to take-off; jump strategy, not just output'),
('peak_power_w_kg',           'CMJ peak power',         'W/kg',   'power',        'CMJ_test',    'force_plate',      true,  false, 15.0,  90.0, 'Peak concentric power per kilogram'),
('peak_force_bw',            'CMJ peak force',          'xBW',    'power',        'CMJ_test',    'force_plate',      true,  false, 1.20,  5.00, 'Peak vertical force in body weights'),
('contraction_time_s',        'CMJ contraction time',   's',      'power',        'CMJ_test',    'force_plate',      false, false, 0.20,  1.60, 'Onset to take-off; longer under fatigue'),
('countermovement_depth_m',   'Countermovement depth',  'm',      'power',        'CMJ_test',    'force_plate',      true,  false, -0.70, 0.00, 'Lowest centre-of-mass displacement; strategy marker'),
-- Remaining CMJ outputs. These were being stored by the pipeline and, because
-- every view joins the catalogue, silently dropped from the entire system --
-- present in the table, invisible everywhere else. The scheduled health check
-- found them on its first run.
('jump_height_flight_time_m',  'CMJ height (flight time)','m',     'power',        'CMJ_test',    'force_plate',      true,  false, 0.05,  1.20, 'Flight-time estimate, reported as a cross-check on the impulse-momentum value; they diverge when landing posture differs from take-off'),
('takeoff_velocity_ms',        'Take-off velocity',      'm/s',    'power',        'CMJ_test',    'force_plate',      true,  false, 0.90,  5.00, 'Vertical velocity at take-off, from net impulse'),
('peak_force_n',               'CMJ peak force',         'N',      'power',        'CMJ_test',    'force_plate',      true,  false, 400,   6000, 'Absolute peak vertical force; the body-weight multiple is the comparable form'),
('peak_power_w',               'CMJ peak power',         'W',      'power',        'CMJ_test',    'force_plate',      true,  false, 800,   9000, 'Absolute peak concentric power'),
('net_impulse_ns',             'Net vertical impulse',   'N.s',    'power',        'CMJ_test',    'force_plate',      true,  false, 40,    600,  'Integral of force above body weight from onset to take-off; jump height is derived from this'),
('flight_time_s',              'Flight time',            's',      'power',        'CMJ_test',    'force_plate',      true,  false, 0.20,  1.00, 'Airborne duration'),
('body_weight_n',              'Body weight',            'N',      'body_comp',    'CMJ_test',    'force_plate',      null,  false, 350,   1600, 'From the quiet-standing period; a calibration output, not a performance measure'),
-- Phase durations carry NULL polarity. Longer is not simply worse: a longer
-- eccentric phase can mean a deeper countermovement rather than a slower
-- athlete, and asserting a direction would manufacture trends.
('unweighting_duration_s',     'Unweighting duration',   's',      'power',        'CMJ_test',    'force_plate',      null,  false, 0.05,  1.20, 'Onset to peak downward velocity; jump strategy'),
('ecc_duration_s',             'Braking duration',       's',      'power',        'CMJ_test',    'force_plate',      null,  false, 0.02,  0.80, 'Peak downward velocity to zero velocity'),
('con_duration_s',             'Propulsion duration',    's',      'power',        'CMJ_test',    'force_plate',      null,  false, 0.05,  0.80, 'Zero velocity to take-off'),
-- ---- maximal strength: isometric mid-thigh pull ----
('imtp_peak_force_n',         'IMTP peak force',        'N',      'max_strength', 'IMTP_test',   'force_plate',      true,  false, 800,   6000, 'Peak isometric force in the mid-thigh pull position'),
('imtp_relative_force_nkg',   'IMTP relative force',    'N/kg',   'max_strength', 'IMTP_test',   'force_plate',      true,  true,  15.0,  70.0, 'Peak force per kilogram; the comparable strength number'),
('imtp_rfd_0_250ms_ns',       'IMTP RFD 0-250 ms',      'N/s',    'max_strength', 'IMTP_test',   'force_plate',      true,  false, 1000,  20000,'Rate of force development; explosive strength'),
-- ---- anaerobic capacity: 30 s Wingate ----
('wingate_peak_power_w_kg',   'Wingate peak power',     'W/kg',   'anaerobic',    'wingate_test','cycle_ergometer',  true,  true,  5.0,   20.0, 'Highest 1 s power in the 30 s test'),
('wingate_mean_power_w_kg',   'Wingate mean power',     'W/kg',   'anaerobic',    'wingate_test','cycle_ergometer',  true,  false, 3.0,   14.0, 'Average power across the 30 s'),
('wingate_fatigue_index_pct', 'Wingate fatigue index',  '%',      'anaerobic',    'wingate_test','cycle_ergometer',  false, false, 10.0,  80.0, 'Percentage drop from peak to minimum power; lower is better'),
-- ---- speed: timing gates ----
('sprint_10m_s',              '10 m sprint',            's',      'speed',        'sprint_test', 'timing_gates',     false, true,  1.30,  2.60, 'Acceleration over the first 10 m; lower is better'),
('sprint_30m_s',              '30 m sprint',            's',      'speed',        'sprint_test', 'timing_gates',     false, false, 3.50,  6.50, 'Lower is better'),
('max_velocity_ms',           'Maximum velocity',       'm/s',    'speed',        'sprint_test', 'timing_gates',     true,  false, 5.00,  12.0, 'Peak running velocity over the flying section'),
-- ---- change of direction ----
('agility_505_s',             '505 change of direction','s',      'cod',          'agility_test','timing_gates',     false, true,  2.00,  3.60, '180-degree turn test; lower is better'),
('cod_deficit_s',             'COD deficit',            's',      'cod',          'agility_test','timing_gates',     false, false, 0.00,  1.50, '505 time minus 10 m sprint time; isolates turning ability from straight speed'),
-- ---- aerobic endurance ----
('yoyo_ir1_distance_m',       'Yo-Yo IR1 distance',     'm',      'aerobic',      'aerobic_test','field_test',       true,  true,  200,   3200, 'Intermittent-recovery running capacity'),
('vo2max_mlkgmin',            'Estimated VO2max',       'ml/kg/min','aerobic',    'aerobic_test','field_test',       true,  false, 30.0,  75.0, 'Estimated from Yo-Yo IR1 distance'),
('vift_kmh',                  '30-15 IFT final speed',  'km/h',   'aerobic',      'aerobic_test','field_test',       true,  false, 13.0,  24.0, 'Final velocity in the 30-15 Intermittent Fitness Test'),
-- ---- sport-specific: swimming ----
-- A land-based 10 m sprint and a Yo-Yo IR1 tell you very little about a
-- swimmer, so swimming carries its own speed and endurance measures. This is
-- the reason `quality` and `is_headline` are per-metric rather than per-test:
-- two sports can fill the same physical quality with entirely different
-- measurements and the profile still lines up.
('swim_100m_free_s',          '100 m freestyle',        's',      'speed',        'swim_test',   'pool_timing',      false, true,  45.0,  90.0, 'Time-trial from a push start; lower is better'),
('css_ms',                    'Critical swim speed',    'm/s',    'aerobic',      'swim_test',   'pool_timing',      true,  true,  0.90,  2.10, 'Speed at the aerobic-anaerobic transition, from 200 m and 400 m trials'),
-- ---- body composition ----
('body_mass_kg',              'Body mass',              'kg',     'body_comp',    'anthropometry','anthropometry',   true,  false, 35.0,  160.0,'Neutral direction; carried for normalisation, not judged'),
('sum7_skinfolds_mm',         'Sum of 7 skinfolds',     'mm',     'body_comp',    'anthropometry','anthropometry',   false, true,  25.0,  200.0,'ISAK sum of seven sites. Headline for body composition because body mass has no universal direction -- a thrower gaining mass and a distance runner gaining mass are not the same event. Skinfolds still are not universally lower-is-better either; treat this direction as a default, not a judgement.')
on conflict (metric_name) do update set
    display_name     = excluded.display_name,
    unit             = excluded.unit,
    quality          = excluded.quality,
    session_type     = excluded.session_type,
    source           = excluded.source,
    higher_is_better = excluded.higher_is_better,
    is_headline      = excluded.is_headline,
    typical_min      = excluded.typical_min,
    typical_max      = excluded.typical_max,
    description      = excluded.description;

-- Human-readable names for the qualities, used as chart and section headings.
create table if not exists quality_catalog (
    quality      text primary key,
    display_name text not null,
    sort_order   int not null
);

insert into quality_catalog (quality, display_name, sort_order) values
('power',        'Neuromuscular power',  1),
('max_strength', 'Maximal strength',     2),
('speed',        'Speed',                3),
('cod',          'Change of direction',  4),
('anaerobic',    'Anaerobic capacity',   5),
('aerobic',      'Aerobic endurance',    6),
('body_comp',    'Body composition',     7)
on conflict (quality) do update set
    display_name = excluded.display_name,
    sort_order   = excluded.sort_order;
