-- Athlete Performance Platform — Phase 1 schema
-- Target: PostgreSQL 15+ (Supabase). Idempotent: safe to re-run.
--
-- Design decisions are documented inline; see README "Design decisions".

-- =====================================================================
-- athletes
-- ---------------------------------------------------------------------
-- De-identified by design: we store an opaque athlete_code, never a name.
-- Re-identification lives outside this system, in the club's own records.
-- =====================================================================
create table if not exists athletes (
    athlete_id   serial primary key,
    athlete_code text unique not null,
    sport        text not null,
    sex          text check (sex in ('M', 'F', 'X') or sex is null),
    squad        text,
    created_at   timestamptz not null default now()
);

-- =====================================================================
-- sessions
-- ---------------------------------------------------------------------
-- One row per athlete per day per session type. The unique constraint is what
-- makes the ingest pipeline idempotent (ON CONFLICT ... DO UPDATE).
-- =====================================================================
create table if not exists sessions (
    session_id   serial primary key,
    athlete_id   int not null references athletes(athlete_id) on delete cascade,
    session_date date not null,
    session_type text not null,          -- 'CMJ_test' | 'training' | 'match'
    created_at   timestamptz not null default now(),
    unique (athlete_id, session_date, session_type)
);

-- =====================================================================
-- performance_metrics  — LONG format (name/value), not wide
-- ---------------------------------------------------------------------
-- Different sports and different devices emit different metric sets. A wide
-- table would be mostly NULL and would need a DDL migration for every new
-- metric. Long format absorbs new metrics as data, not as schema change.
-- `source` is part of the unique key so the same metric measured by two
-- devices (e.g. jump height from force plate vs. GymAware) coexists rather
-- than one silently overwriting the other.
-- =====================================================================
create table if not exists performance_metrics (
    metric_id    bigserial primary key,
    session_id   int not null references sessions(session_id) on delete cascade,
    metric_name  text not null,          -- 'jump_height_m' | 'rsi_mod' | 'peak_force_n'
    metric_value numeric,
    source       text not null,          -- 'force_plate' | 'gymaware' | 'gps'
    ingested_at  timestamptz not null default now(),
    unique (session_id, metric_name, source)
);

-- =====================================================================
-- training_load
-- ---------------------------------------------------------------------
-- session_load is a GENERATED column: duration x sRPE can never drift out of
-- sync with its inputs, and the pipeline cannot write an inconsistent value.
-- =====================================================================
create table if not exists training_load (
    load_id      bigserial primary key,
    athlete_id   int not null references athletes(athlete_id) on delete cascade,
    date         date not null,
    duration_min numeric check (duration_min >= 0),
    srpe         numeric check (srpe between 0 and 10),
    session_load numeric generated always as (duration_min * srpe) stored,
    unique (athlete_id, date)
);

-- =====================================================================
-- Indexes
-- ---------------------------------------------------------------------
-- The access patterns that matter: (a) fetch all metrics for a session,
-- (b) walk one athlete's history in date order for ACWR / rolling baselines,
-- (c) pull one metric across the whole squad for a given day.
-- =====================================================================
create index if not exists idx_metrics_session      on performance_metrics(session_id);
create index if not exists idx_load_athlete_date    on training_load(athlete_id, date);
create index if not exists idx_sessions_athlete_date on sessions(athlete_id, session_date);
create index if not exists idx_metrics_name         on performance_metrics(metric_name);
