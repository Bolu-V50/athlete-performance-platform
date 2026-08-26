-- Published normative data
--
-- A number on its own means nothing to a coach who does not test that quality
-- every week. "IMTP relative peak force 21.9 N/kg" is only interpretable next to
-- what the published literature reports for a comparable population.
--
-- Every row in normative_values traces to a real, verifiable paper recorded in
-- reference_studies with a DOI or PMID. Nothing here is remembered, estimated or
-- inferred: each was retrieved and read before being entered, and a value that
-- could not be verified was left out rather than approximated.
--
-- Three things the schema insists on, because getting them wrong is how a
-- comparison misleads:
--
--   * `spread_type` is explicit. Papers report SD, a 95%% confidence interval, or
--     a plain range, and they are not interchangeable. A CI describes uncertainty
--     about the mean; an SD describes the spread of athletes. Presenting one as
--     the other makes an elite band look four times tighter than it is.
--   * `protocol_note` travels with the value. A countermovement jump without arm
--     swing is several centimetres lower than one with; a 10 m sprint time
--     depends on the starting position and gate height. Comparing across
--     protocols is the most common way these tables get misused.
--   * `population` keeps the paper's own wording. "Professional" means different
--     things in different leagues, and paraphrasing it loses that.

create table if not exists reference_studies (
    study_key     text primary key,
    citation      text not null,
    doi           text,
    pmid          text,
    year          int  not null,
    publication   text not null,
    verified_on   date not null,
    source_url    text,
    note          text
);

create table if not exists normative_values (
    norm_id       bigserial primary key,
    study_key     text not null references reference_studies(study_key) on delete cascade,
    metric_name   text not null references metric_catalog(metric_name),
    population    text not null,          -- the paper's own description
    sport         text,                   -- mapped onto this platform's sport names
    sex           text check (sex in ('M', 'F') or sex is null),
    level         text,
    n             int,
    mean_value    numeric not null,
    spread_type   text not null check (spread_type in ('sd', 'ci95', 'range', 'none')),
    sd_value      numeric,
    low_value     numeric,                -- CI lower bound, or range minimum
    high_value    numeric,
    unit          text not null,
    protocol_note text,
    unique (study_key, metric_name, population)
);

create index if not exists idx_norm_metric on normative_values(metric_name);
create index if not exists idx_norm_sport on normative_values(sport, sex);

-- =====================================================================
-- Studies
-- ---------------------------------------------------------------------
-- READ AND EXCLUDED, recorded here so the decision is visible rather than
-- silent: Bangsbo J, Iaia FM, Krustrup P. The Yo-Yo Intermittent Recovery Test.
-- Sports Med 2008;38(1):37-51 (PMID 18081366). It is the definitive review of
-- the test and was retrieved and read, but almost all of its group values are
-- presented only as bar charts. Reading a mean off a chart to two decimal
-- places produces a number with the appearance of precision and none of the
-- substance, and the few figures stated in its text cover rugby and hockey --
-- neither of which is in this programme. Nothing from it is loaded.
-- =====================================================================
insert into reference_studies (study_key, citation, doi, pmid, year, publication, verified_on, source_url, note) values
('seraphin2025',
 'Seraphin A, Edson C, Price C, Jones B, Jagielo A, Mullner J. Normative Performance Test Metrics in Professional Female Club Soccer. Journal of Sports Medicine and Allied Health Sciences 2025;10(3).',
 '10.25035/jsmahs.10.03.01', null, 2025, 'J Sports Med Allied Health Sci', date '2026-08-26',
 'https://scholarworks.bgsu.edu/jsmahs/vol10/iss3/1/',
 'Conference abstract rather than a full paper: the methods are summarised only, so the protocol detail behind these numbers is thinner than for the others here.'),

('krustrup2005',
 'Krustrup P, Mohr M, Ellingsgaard H, Bangsbo J. Physical demands during an elite female soccer game: importance of training status. Med Sci Sports Exerc 2005;37(7):1242-8.',
 '10.1249/01.mss.0000170062.73981.94', '16015145', 2005, 'Med Sci Sports Exerc', date '2026-08-26',
 'https://pubmed.ncbi.nlm.nih.gov/16015145/',
 'Reports a mean with the observed range, not a standard deviation.'),

('suarez2025',
 'Suarez-Balsera C, Ferioli D, Marin-Cascales E, Rago V, Spyrou K, Martinez-Serrano A, Di Mauro D, Marin JM, Alcaraz PE, Freitas TT. Profiling the Countermovement Jump Characteristics of Basketball Players across Competitive Levels and Playing Positions. J Hum Kinet 2025;96.',
 '10.5114/jhk/196138', null, 2025, 'Journal of Human Kinetics', date '2026-08-26',
 'https://pmc.ncbi.nlm.nih.gov/articles/PMC12121892/',
 'Male players only. Reports 95% confidence intervals, not standard deviations.'),

('dossantos2018',
 'Dos Santos T, Thomas C, Comfort P, Jones PA. Comparison of Change of Direction Speed Performance and Asymmetries between Team-Sport Athletes: Application of Change of Direction Deficit. Sports (Basel) 2018;6(4):174.',
 '10.3390/sports6040174', null, 2018, 'Sports (Basel)', date '2026-08-26',
 'https://pmc.ncbi.nlm.nih.gov/articles/PMC6315619/',
 'Reports the 505 separately for left and right turns; both are entered, because the difference between them is itself a finding.')
on conflict (study_key) do update set
    citation = excluded.citation, doi = excluded.doi, pmid = excluded.pmid,
    year = excluded.year, publication = excluded.publication,
    verified_on = excluded.verified_on, source_url = excluded.source_url, note = excluded.note;

-- =====================================================================
-- Values
-- =====================================================================
insert into normative_values (study_key, metric_name, population, sport, sex, level, n,
                              mean_value, spread_type, sd_value, low_value, high_value,
                              unit, protocol_note) values
-- Seraphin 2025 — professional women's soccer, USA first division
('seraphin2025', 'imtp_relative_force_nkg',
 'Professional women''s soccer players, 1st division club, USA', 'Football', 'F', 'professional', 28,
 21.6, 'sd', 2.04, null, null, 'N/kg', null),
('seraphin2025', 'jump_height_m',
 'Professional women''s soccer players, 1st division club, USA', 'Football', 'F', 'professional', 28,
 0.289, 'sd', 0.042, null, null, 'm', 'Reported in centimetres (28.9 +/- 4.2 cm); converted to metres for comparison.'),

-- Krustrup 2005 — elite female soccer
('krustrup2005', 'yoyo_ir1_distance_m',
 'Elite female soccer players, Danish top division', 'Football', 'F', 'elite', 14,
 1379, 'range', null, 600, 1960, 'm',
 'Mean with observed range; the paper does not report a standard deviation.'),

-- Suarez-Balsera 2025 — male basketball
('suarez2025', 'jump_height_m',
 'Professional male basketball players', 'Basketball', 'M', 'professional', 39,
 0.392, 'ci95', null, 0.373, 0.411, 'm',
 'Hands on hips throughout: no arm swing. An arm-swing jump is several centimetres higher and is not comparable.'),
('suarez2025', 'jump_height_m',
 'Semi-professional male basketball players', 'Basketball', 'M', 'semi-professional', 39,
 0.360, 'ci95', null, 0.340, 0.379, 'm',
 'Hands on hips throughout: no arm swing.'),

-- Dos Santos 2018 — team-sport athletes, 505 and 10 m
('dossantos2018', 'agility_505_s',
 'Female soccer players (left turn)', 'Football', 'F', 'university/club', 15,
 2.672, 'sd', 0.197, null, null, 's', 'Turning off the left limb.'),
('dossantos2018', 'agility_505_s',
 'Female soccer players (right turn)', 'Football', 'F', 'university/club', 15,
 2.668, 'sd', 0.235, null, null, 's', 'Turning off the right limb.'),
('dossantos2018', 'agility_505_s',
 'Male basketball players (left turn)', 'Basketball', 'M', 'university/club', 17,
 2.492, 'sd', 0.158, null, null, 's', 'Turning off the left limb.'),
('dossantos2018', 'agility_505_s',
 'Male basketball players (right turn)', 'Basketball', 'M', 'university/club', 17,
 2.433, 'sd', 0.130, null, null, 's', 'Turning off the right limb.'),
('dossantos2018', 'sprint_10m_s',
 'Female soccer players', 'Football', 'F', 'university/club', 15,
 2.139, 'sd', 0.141, null, null, 's',
 'Sprint times depend heavily on starting position and timing-gate height; compare protocols before comparing numbers.'),
('dossantos2018', 'sprint_10m_s',
 'Male basketball players', 'Basketball', 'M', 'university/club', 17,
 1.854, 'sd', 0.074, null, null, 's',
 'Sprint times depend heavily on starting position and timing-gate height.'),
('dossantos2018', 'sprint_10m_s',
 'Male soccer players', 'Football', 'M', 'university/club', 16,
 1.932, 'sd', 0.092, null, null, 's', null),
('dossantos2018', 'agility_505_s',
 'Male soccer players (right turn)', 'Football', 'M', 'university/club', 16,
 2.401, 'sd', 0.135, null, null, 's', 'Turning off the right limb.')
on conflict (study_key, metric_name, population) do update set
    mean_value = excluded.mean_value, spread_type = excluded.spread_type,
    sd_value = excluded.sd_value, low_value = excluded.low_value,
    high_value = excluded.high_value, n = excluded.n,
    protocol_note = excluded.protocol_note;
