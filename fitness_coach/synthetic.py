"""Synthetic athletes + session logs for the demo.

Four athletes, two per population, with varied archetypes so the demo shows
the architecture handling different patterns:

  - pl_001 "Sam"    — early novice powerlifter, mostly smooth progression
  - pl_002 "Diane"  — stalling intermediate, hits plateau + injury
  - rn_001 "Marco"  — beginner half-marathon trainee, life-event interruption
  - rn_002 "Priya"  — returning runner with ITB history, conservative ramp

Each athlete carries an initial state + 6 SessionLog records. Two sessions
per athlete contain a "surprise" — pain, missed sleep, plateau, breakthrough,
form regression — so the comparison vs. the rigid straw coach has visible
moments where the agentic architecture should pull ahead.

Synthetic data is grounded in published methodology:
  - Powerlifters: Rippetoe Starting Strength linear progression
  - Runners: Daniels' Running Formula (VDOT, 80/20, 10% mileage rule)
"""

from __future__ import annotations

from datetime import date

from fitness_coach.schemas import (
    EvidenceItem,
    LiftActivity,
    PowerlifterState,
    RunActivity,
    RunnerState,
    SafetySignal,
    SessionLog,
)


# ====================================================================
# pl_001 — Sam, early novice powerlifter (mostly smooth)
# Surprises: session 2 (bad sleep → high RPE), session 5 (deadlift PR breakthrough)
# ====================================================================

PL_001_INITIAL = PowerlifterState(
    athlete_id="pl_001",
    name="Sam",
    training_age_months=4,
    bodyweight_lb=178.0,
    current_lifts={"squat": 220.0, "bench": 150.0, "deadlift": 270.0, "ohp": 95.0},
)

PL_001_SESSIONS = [
    # Session 0 — baseline, all RPE 7
    SessionLog(
        athlete_id="pl_001", session_index=0, session_date=date(2026, 4, 1),
        activity_log=[
            LiftActivity(exercise="squat", sets=3, reps_per_set=[5, 5, 5], load_lb=220.0, rpe=7.0),
            LiftActivity(exercise="bench", sets=3, reps_per_set=[5, 5, 5], load_lb=150.0, rpe=7.0),
            LiftActivity(exercise="deadlift", sets=1, reps_per_set=[5], load_lb=270.0, rpe=7.5),
        ],
        evidence=[
            EvidenceItem(source="athlete_self_report", trust_weight="high",
                         content="Felt strong; slept 8 hours; warmups went smoothly."),
            EvidenceItem(source="objective_metric", trust_weight="medium",
                         content="Bar speed on top sets consistent with previous week."),
        ],
    ),
    # Session 1 — clean progression. OHP dropped from program here onward
    # (see evidence) — Sam preferred a tighter 3-lift split.
    SessionLog(
        athlete_id="pl_001", session_index=1, session_date=date(2026, 4, 3),
        activity_log=[
            LiftActivity(exercise="squat", sets=3, reps_per_set=[5, 5, 5], load_lb=225.0, rpe=7.5),
            LiftActivity(exercise="bench", sets=3, reps_per_set=[5, 5, 5], load_lb=152.5, rpe=7.5),
            LiftActivity(exercise="deadlift", sets=1, reps_per_set=[5], load_lb=275.0, rpe=8.0),
        ],
        evidence=[
            EvidenceItem(source="athlete_self_report", trust_weight="high",
                         content="Clean progression; no issues."),
            EvidenceItem(source="athlete_self_report", trust_weight="medium",
                         content="Dropped OHP from the program — preferring a 3-lift split focused on squat/bench/deadlift; will reassess in 4 weeks."),
        ],
    ),
    # Session 2 — SURPRISE: bad sleep, RPE jumps
    SessionLog(
        athlete_id="pl_001", session_index=2, session_date=date(2026, 4, 5),
        activity_log=[
            LiftActivity(exercise="squat", sets=3, reps_per_set=[5, 5, 5], load_lb=230.0, rpe=8.5,
                         notes="Felt heavy from set 1"),
            LiftActivity(exercise="bench", sets=3, reps_per_set=[5, 5, 4], load_lb=155.0, rpe=9.0,
                         notes="Missed last rep of set 3"),
            LiftActivity(exercise="deadlift", sets=1, reps_per_set=[5], load_lb=280.0, rpe=8.5),
        ],
        evidence=[
            EvidenceItem(source="athlete_self_report", trust_weight="high",
                         content="Only got ~4 hours of sleep last night — bedroom too hot."),
            EvidenceItem(source="objective_metric", trust_weight="high",
                         content="Bar speed visibly slower on all top sets."),
        ],
    ),
    # Session 3 — recovery; coach should hold rather than progress
    SessionLog(
        athlete_id="pl_001", session_index=3, session_date=date(2026, 4, 8),
        activity_log=[
            LiftActivity(exercise="squat", sets=3, reps_per_set=[5, 5, 5], load_lb=230.0, rpe=7.5),
            LiftActivity(exercise="bench", sets=3, reps_per_set=[5, 5, 5], load_lb=155.0, rpe=7.5),
            LiftActivity(exercise="deadlift", sets=1, reps_per_set=[5], load_lb=285.0, rpe=8.0),
        ],
        evidence=[
            EvidenceItem(source="athlete_self_report", trust_weight="high",
                         content="Slept well, RPE back down. Held loads from last session."),
        ],
    ),
    # Session 4 — clean progression resumes
    SessionLog(
        athlete_id="pl_001", session_index=4, session_date=date(2026, 4, 10),
        activity_log=[
            LiftActivity(exercise="squat", sets=3, reps_per_set=[5, 5, 5], load_lb=235.0, rpe=8.0),
            LiftActivity(exercise="bench", sets=3, reps_per_set=[5, 5, 5], load_lb=157.5, rpe=8.0),
            LiftActivity(exercise="deadlift", sets=1, reps_per_set=[5], load_lb=290.0, rpe=8.0),
        ],
        evidence=[
            EvidenceItem(source="athlete_self_report", trust_weight="high",
                         content="Solid session; everything moved well."),
        ],
    ),
    # Session 5 — SURPRISE: deadlift breakthrough (PR, low RPE)
    SessionLog(
        athlete_id="pl_001", session_index=5, session_date=date(2026, 4, 12),
        activity_log=[
            LiftActivity(exercise="squat", sets=3, reps_per_set=[5, 5, 5], load_lb=240.0, rpe=8.0),
            LiftActivity(exercise="bench", sets=3, reps_per_set=[5, 5, 5], load_lb=160.0, rpe=8.0),
            LiftActivity(exercise="deadlift", sets=1, reps_per_set=[5], load_lb=305.0, rpe=7.5,
                         notes="Felt unexpectedly easy"),
        ],
        evidence=[
            EvidenceItem(source="athlete_self_report", trust_weight="high",
                         content="Deadlift came up fast — barely felt the +15."),
            EvidenceItem(source="objective_metric", trust_weight="high",
                         content="Bar speed on deadlift top set faster than at previous lower load."),
        ],
    ),
]


# ====================================================================
# pl_002 — Diane, stalling intermediate (plateau + injury)
# Surprises: session 3 (bench plateau confirmed), session 4 (shoulder pain)
# ====================================================================

PL_002_INITIAL = PowerlifterState(
    athlete_id="pl_002",
    name="Diane",
    training_age_months=14,
    bodyweight_lb=155.0,
    current_lifts={"squat": 295.0, "bench": 195.0, "deadlift": 360.0, "ohp": 130.0},
    consecutive_failed_sessions={"bench": 0},
)

PL_002_SESSIONS = [
    SessionLog(
        athlete_id="pl_002", session_index=0, session_date=date(2026, 4, 1),
        activity_log=[
            LiftActivity(exercise="squat", sets=3, reps_per_set=[5, 5, 5], load_lb=295.0, rpe=8.0),
            LiftActivity(exercise="bench", sets=3, reps_per_set=[5, 5, 5], load_lb=195.0, rpe=8.5),
            LiftActivity(exercise="deadlift", sets=1, reps_per_set=[5], load_lb=360.0, rpe=8.0),
        ],
        evidence=[
            EvidenceItem(source="athlete_self_report", trust_weight="high",
                         content="Bench felt grindy as usual; squat and deadlift solid."),
        ],
    ),
    # Session 1 — first bench failure
    SessionLog(
        athlete_id="pl_002", session_index=1, session_date=date(2026, 4, 3),
        activity_log=[
            LiftActivity(exercise="squat", sets=3, reps_per_set=[5, 5, 5], load_lb=300.0, rpe=8.5),
            LiftActivity(exercise="bench", sets=3, reps_per_set=[5, 5, 3], load_lb=200.0, rpe=9.5,
                         notes="Failed last 2 reps of set 3"),
            LiftActivity(exercise="deadlift", sets=1, reps_per_set=[5], load_lb=365.0, rpe=8.5),
        ],
        evidence=[
            EvidenceItem(source="athlete_self_report", trust_weight="high",
                         content="Bench is stalling — couldn't lock out final reps."),
        ],
    ),
    # Session 2 — second bench failure (consecutive_failed_sessions['bench'] now 2)
    SessionLog(
        athlete_id="pl_002", session_index=2, session_date=date(2026, 4, 5),
        activity_log=[
            LiftActivity(exercise="squat", sets=3, reps_per_set=[5, 5, 5], load_lb=305.0, rpe=8.5),
            LiftActivity(exercise="bench", sets=3, reps_per_set=[5, 4, 3], load_lb=200.0, rpe=9.5,
                         notes="Bench plateau — same load, worse reps"),
            LiftActivity(exercise="deadlift", sets=1, reps_per_set=[5], load_lb=370.0, rpe=8.5),
        ],
        evidence=[
            EvidenceItem(source="athlete_self_report", trust_weight="high",
                         content="Bench got worse despite holding load."),
        ],
    ),
    # Session 3 — SURPRISE: third bench failure → plateau confirmed
    SessionLog(
        athlete_id="pl_002", session_index=3, session_date=date(2026, 4, 8),
        activity_log=[
            LiftActivity(exercise="squat", sets=3, reps_per_set=[5, 5, 5], load_lb=310.0, rpe=8.5),
            LiftActivity(exercise="bench", sets=3, reps_per_set=[4, 3, 2], load_lb=200.0, rpe=10.0,
                         notes="Third consecutive failure — clear plateau"),
            LiftActivity(exercise="deadlift", sets=1, reps_per_set=[5], load_lb=370.0, rpe=9.0),
        ],
        evidence=[
            EvidenceItem(source="athlete_self_report", trust_weight="high",
                         content="Bench is dead-stuck. Need to deload."),
            EvidenceItem(source="form_check", trust_weight="high",
                         content="Form looks fine on video — this is a strength plateau, not a technique issue."),
        ],
    ),
    # Session 4 — SURPRISE: shoulder pain
    SessionLog(
        athlete_id="pl_002", session_index=4, session_date=date(2026, 4, 10),
        activity_log=[
            LiftActivity(exercise="squat", sets=3, reps_per_set=[5, 5, 5], load_lb=310.0, rpe=8.5),
            LiftActivity(exercise="bench", sets=2, reps_per_set=[3, 0], load_lb=180.0, rpe=10.0,
                         notes="Stopped mid-second-set due to left anterior shoulder pain"),
            LiftActivity(exercise="deadlift", sets=1, reps_per_set=[5], load_lb=370.0, rpe=8.5),
        ],
        evidence=[
            EvidenceItem(source="athlete_self_report", trust_weight="high",
                         content="Sharp pain in left front delt during bench, ~5/10."),
            EvidenceItem(source="form_check", trust_weight="medium",
                         content="Slight asymmetry visible in setup — possibly compensating."),
        ],
        safety_signals=[
            SafetySignal(
                severity="moderate", signal_type="pain",
                location_or_movement="bench",
                description="Left anterior shoulder pain ~5/10 mid-set; stopped lift",
            ),
        ],
    ),
    # Session 5 — recovered; deload bench, hold others
    SessionLog(
        athlete_id="pl_002", session_index=5, session_date=date(2026, 4, 13),
        activity_log=[
            LiftActivity(exercise="squat", sets=3, reps_per_set=[5, 5, 5], load_lb=310.0, rpe=8.0),
            LiftActivity(exercise="bench", sets=3, reps_per_set=[5, 5, 5], load_lb=180.0, rpe=7.5,
                         notes="Deloaded bench; shoulder pain-free at lighter load"),
            LiftActivity(exercise="deadlift", sets=1, reps_per_set=[5], load_lb=370.0, rpe=8.5),
        ],
        evidence=[
            EvidenceItem(source="athlete_self_report", trust_weight="high",
                         content="Shoulder OK at lighter load; saw PT, cleared to continue."),
        ],
    ),
]


# ====================================================================
# rn_001 — Marco, beginner half-marathon trainee
# Surprises: session 3 (life event - skipped most of week), session 5 (knee twinge)
# ====================================================================

RN_001_INITIAL = RunnerState(
    athlete_id="rn_001",
    name="Marco",
    training_age_months=8,
    vdot=42.0,
    current_weekly_mileage=17.0,
    target_event="half_marathon",
    target_event_date=date(2026, 9, 15),
    weeks_at_current_mileage=2,
)

RN_001_SESSIONS = [
    # Each "session" here represents one week of training
    SessionLog(
        athlete_id="rn_001", session_index=0, session_date=date(2026, 4, 5),
        activity_log=[
            RunActivity(run_type="easy", distance_mi=4.0, duration_min=42.0),
            RunActivity(run_type="easy", distance_mi=3.0, duration_min=31.0),
            RunActivity(run_type="easy", distance_mi=4.0, duration_min=42.0),
            RunActivity(run_type="long", distance_mi=6.0, duration_min=66.0),
        ],
        evidence=[
            EvidenceItem(source="athlete_self_report", trust_weight="high",
                         content="Solid week, easy paces felt easy."),
        ],
    ),
    SessionLog(
        athlete_id="rn_001", session_index=1, session_date=date(2026, 4, 12),
        activity_log=[
            RunActivity(run_type="easy", distance_mi=5.0, duration_min=52.0),
            RunActivity(run_type="easy", distance_mi=4.0, duration_min=42.0),
            RunActivity(run_type="easy", distance_mi=5.0, duration_min=52.0),
            RunActivity(run_type="long", distance_mi=7.0, duration_min=77.0),
        ],
        evidence=[
            EvidenceItem(source="athlete_self_report", trust_weight="high",
                         content="Bumped to 21 mi; long run felt fine."),
        ],
    ),
    SessionLog(
        athlete_id="rn_001", session_index=2, session_date=date(2026, 4, 19),
        activity_log=[
            RunActivity(run_type="easy", distance_mi=5.0, duration_min=51.0),
            RunActivity(run_type="easy", distance_mi=4.0, duration_min=42.0),
            RunActivity(run_type="tempo", distance_mi=4.0, duration_min=30.0,
                        average_pace_min_per_mi=7.5, perceived_effort=7.5),
            RunActivity(run_type="easy", distance_mi=4.0, duration_min=41.0),
            RunActivity(run_type="long", distance_mi=8.0, duration_min=86.0),
        ],
        evidence=[
            EvidenceItem(source="athlete_self_report", trust_weight="high",
                         content="First tempo this cycle — paced it conservatively."),
        ],
    ),
    # Session 3 — SURPRISE: work travel, only one run logged
    SessionLog(
        athlete_id="rn_001", session_index=3, session_date=date(2026, 4, 26),
        activity_log=[
            RunActivity(run_type="easy", distance_mi=4.0, duration_min=43.0,
                        notes="Hotel treadmill; only run of the week"),
        ],
        evidence=[
            EvidenceItem(source="athlete_self_report", trust_weight="high",
                         content="Work travel ate the week — only got one run in. Back next week."),
        ],
    ),
    SessionLog(
        athlete_id="rn_001", session_index=4, session_date=date(2026, 5, 3),
        activity_log=[
            RunActivity(run_type="easy", distance_mi=5.0, duration_min=53.0),
            RunActivity(run_type="easy", distance_mi=4.0, duration_min=42.0),
            RunActivity(run_type="easy", distance_mi=5.0, duration_min=53.0),
            RunActivity(run_type="long", distance_mi=7.0, duration_min=77.0),
        ],
        evidence=[
            EvidenceItem(source="athlete_self_report", trust_weight="high",
                         content="Back to pre-skip baseline; legs felt fresh from the unintended rest."),
        ],
    ),
    # Session 5 — SURPRISE: knee twinge on long run
    SessionLog(
        athlete_id="rn_001", session_index=5, session_date=date(2026, 5, 10),
        activity_log=[
            RunActivity(run_type="easy", distance_mi=5.0, duration_min=52.0),
            RunActivity(run_type="easy", distance_mi=4.0, duration_min=42.0),
            RunActivity(run_type="easy", distance_mi=5.0, duration_min=53.0),
            RunActivity(run_type="long", distance_mi=8.0, duration_min=92.0,
                        notes="Cut short by ~2 mi due to right knee twinge in mile 6"),
        ],
        evidence=[
            EvidenceItem(source="athlete_self_report", trust_weight="high",
                         content="Right knee felt off mid-long-run — cut it short. ~3/10 sharpness."),
        ],
        safety_signals=[
            SafetySignal(
                severity="low", signal_type="pain",
                location_or_movement="right knee",
                description="Sharp twinge mid-long-run, ~3/10; resolved on stop",
            ),
        ],
    ),
]


# ====================================================================
# rn_002 — Priya, returning runner with ITB history (conservative)
# Surprises: session 2 (form regression), session 5 (ITB warning)
# ====================================================================

RN_002_INITIAL = RunnerState(
    athlete_id="rn_002",
    name="Priya",
    training_age_months=36,
    vdot=38.0,
    current_weekly_mileage=10.0,
    target_event="10k",
    target_event_date=date(2026, 8, 1),
    weeks_at_current_mileage=2,
    injury_history=["ITB syndrome (2025-12, resolved)"],
)

RN_002_SESSIONS = [
    SessionLog(
        athlete_id="rn_002", session_index=0, session_date=date(2026, 4, 5),
        activity_log=[
            RunActivity(run_type="easy", distance_mi=3.0, duration_min=33.0),
            RunActivity(run_type="easy", distance_mi=3.0, duration_min=33.0),
            RunActivity(run_type="easy", distance_mi=4.0, duration_min=44.0),
        ],
        evidence=[
            EvidenceItem(source="athlete_self_report", trust_weight="high",
                         content="Conservative return; left ITB feels OK."),
        ],
    ),
    SessionLog(
        athlete_id="rn_002", session_index=1, session_date=date(2026, 4, 12),
        activity_log=[
            RunActivity(run_type="easy", distance_mi=3.0, duration_min=33.0),
            RunActivity(run_type="easy", distance_mi=3.0, duration_min=33.0),
            RunActivity(run_type="easy", distance_mi=4.0, duration_min=44.0),
            RunActivity(run_type="easy", distance_mi=3.0, duration_min=33.0),
        ],
        evidence=[
            EvidenceItem(source="athlete_self_report", trust_weight="high",
                         content="13 mi week, all easy; no ITB issues."),
        ],
    ),
    # Session 2 — SURPRISE: form regression
    SessionLog(
        athlete_id="rn_002", session_index=2, session_date=date(2026, 4, 19),
        activity_log=[
            RunActivity(run_type="easy", distance_mi=4.0, duration_min=46.0,
                        average_pace_min_per_mi=11.5,
                        notes="Pace slower than usual"),
            RunActivity(run_type="easy", distance_mi=3.0, duration_min=35.0),
            RunActivity(run_type="easy", distance_mi=4.0, duration_min=46.0),
        ],
        evidence=[
            EvidenceItem(source="athlete_self_report", trust_weight="high",
                         content="Felt off — left side felt heavier."),
            EvidenceItem(source="form_check", trust_weight="high",
                         content="Coach observed compensatory gait on left, possibly favoring ITB. Pace consistent with this."),
            EvidenceItem(source="objective_metric", trust_weight="high",
                         content="Average pace ~11:30/mi vs usual ~11:00/mi at same effort."),
        ],
    ),
    SessionLog(
        athlete_id="rn_002", session_index=3, session_date=date(2026, 4, 26),
        activity_log=[
            RunActivity(run_type="easy", distance_mi=3.0, duration_min=33.0),
            RunActivity(run_type="easy", distance_mi=4.0, duration_min=43.0),
            RunActivity(run_type="easy", distance_mi=3.0, duration_min=33.0),
            RunActivity(run_type="easy", distance_mi=4.0, duration_min=43.0),
        ],
        evidence=[
            EvidenceItem(source="athlete_self_report", trust_weight="high",
                         content="Form felt normal again; pace back to usual."),
        ],
    ),
    SessionLog(
        athlete_id="rn_002", session_index=4, session_date=date(2026, 5, 3),
        activity_log=[
            RunActivity(run_type="easy", distance_mi=4.0, duration_min=43.0),
            RunActivity(run_type="easy", distance_mi=4.0, duration_min=43.0),
            RunActivity(run_type="easy", distance_mi=4.0, duration_min=43.0),
            RunActivity(run_type="easy", distance_mi=3.0, duration_min=33.0),
        ],
        evidence=[
            EvidenceItem(source="athlete_self_report", trust_weight="high",
                         content="15 mi week, smooth; ITB silent."),
        ],
    ),
    # Session 5 — SURPRISE: ITB warning, halted final long run
    SessionLog(
        athlete_id="rn_002", session_index=5, session_date=date(2026, 5, 10),
        activity_log=[
            RunActivity(run_type="easy", distance_mi=4.0, duration_min=44.0),
            RunActivity(run_type="easy", distance_mi=4.0, duration_min=44.0),
            RunActivity(run_type="easy", distance_mi=3.0, duration_min=34.0),
            RunActivity(run_type="easy", distance_mi=2.0, duration_min=26.0,
                        notes="Cut planned 4-miler short due to ITB tightness"),
        ],
        evidence=[
            EvidenceItem(source="athlete_self_report", trust_weight="high",
                         content="ITB tightness returning at end of week — pulled the plug on last run."),
        ],
        safety_signals=[
            SafetySignal(
                severity="low", signal_type="injury_recurrence",
                location_or_movement="left ITB",
                description="Familiar tightness pattern returning; preventive halt of final run",
            ),
        ],
    ),
]


# ====================================================================
# Registry — used by run.py and tests
# ====================================================================

ATHLETES: dict[str, tuple] = {
    "pl_001": (PL_001_INITIAL, PL_001_SESSIONS),
    "pl_002": (PL_002_INITIAL, PL_002_SESSIONS),
    "rn_001": (RN_001_INITIAL, RN_001_SESSIONS),
    "rn_002": (RN_002_INITIAL, RN_002_SESSIONS),
}


def all_athletes() -> list[tuple]:
    """Return list of (initial_state, session_logs) tuples for all athletes."""
    return list(ATHLETES.values())
