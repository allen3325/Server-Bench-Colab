"""Final validation is the closing stage of an evolve, invoked by the harness.

It is NOT called by the coding agent; it is called by the HARNESS to verify the
evolved system and report the final outcome. The driver invokes it when a run
ends: the agent stops, the accepted head is selected under the existing policy,
the private validator measures it against the held-out partition, and the run
reports a final outcome in its own journal.

WHERE, AND WHY THERE
---------------------
Inside `preserve()`, before `release()`. Two constraints fix the position: it
needs the card, which `release()` gives back; and it must run after the agent
and its container are gone, which by `preserve()` they are.

WHAT IT MAY NOT DO
-------------------
The coding agent never invokes it and never sees its result. Private final
outcomes stay out of the optimisation loop, so the output goes to the
operator-facing store and nothing reaches the workspace.

A REFUSAL IS AN OUTCOME
------------------------
A run whose candidates were all rejected has no accepted head and therefore no
final candidate. Recording that is the honest end state; inventing one would put
a candidate the lineage never accepted in front of the held-out partition, so the
driver records it as a final outcome rather than as an error.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "serverbench" / "harness"))

from lifecycle.run import FINAL_VALIDATION_TIMEOUT_S, Lifecycle  # noqa: E402

PRESERVE = inspect.getsource(Lifecycle.preserve)
STAGE = inspect.getsource(Lifecycle._final_validation)


def test_the_driver_calls_it_at_the_end_of_a_run() -> None:
    assert "self._final_validation()" in PRESERVE, (
        "the driver never invokes final validation, so it is a script an "
        "operator runs rather than the closing stage of an evolve")


def test_it_runs_while_the_card_is_still_held() -> None:
    """`release()` gives the device back; final validation needs it."""
    release = inspect.getsource(Lifecycle.release)
    assert "release_all" in release
    assert "_final_validation" not in release, (
        "final validation must not run after the device is released")
    # `SUMMARY.json` must exist before the stage starts: it is the file
    # `validate_evolve.lineage_of` resolves the lineage from, so without it the
    # validator refuses on a missing file rather than on the run's eligibility.
    assert PRESERVE.index("SUMMARY.json") < PRESERVE.index(
        "_final_validation()"), (
        "final validation runs before SUMMARY.json is written; it will refuse "
        "on a missing file rather than on the run's eligibility")


def test_it_delegates_to_the_launcher() -> None:
    assert "validate_evolve.py" in STAGE
    for banned in ("MeasurementReport", "run_block", "accepted_head",
                   "nearest_rank", "goodput"):
        assert banned not in STAGE, (
            f"the driver stage references {banned!r}; it decides WHEN, and "
            f"everything else lives behind the launcher")


def test_it_runs_as_a_subprocess_to_keep_the_boundary_structural() -> None:
    """Importing it would put the final validator and the grading path on one
    interpreter, making the independence a matter of discipline."""
    assert "subprocess.Popen" in STAGE
    assert "sys.executable" in STAGE


def test_it_is_never_fatal_to_the_run() -> None:
    """A run's evidence is written by this point; losing the summary because a
    final validation could not start would destroy the record of what happened."""
    assert "except (OSError, subprocess.SubprocessError)" in STAGE
    assert "final_validation_failed" in STAGE
    # `Popen` + `communicate`, not `run(check=True)`: a non-zero exit is an
    # OUTCOME here -- a refusal or a VALID_FAIL -- not an exception. The handle
    # is kept so a second caller waits for an in-flight validation instead of
    # starting a rival that would be refused the card and record the refusal
    # as the head's verdict.
    assert "returncode" in STAGE and "check=True" not in STAGE, (
        "a non-zero exit is an OUTCOME here -- a refusal or a VALID_FAIL -- not "
        "an exception")
    assert "final_validation_awaited" in STAGE, (
        "a second caller must wait for the validation already running")


def test_the_outcome_is_journalled_with_its_evidence_path() -> None:
    assert '"final_validation"' in STAGE
    assert "outcome=" in STAGE and "evidence=" in STAGE, (
        "the run's own record must say what happened without anyone reading a "
        "separate directory")
    assert "REFUSED" in STAGE, (
        "a refusal must be recorded as an outcome, not lost as an error")


def test_the_agent_is_named_as_excluded() -> None:
    assert "coding agent neither invoked this nor sees its result" in STAGE, (
        "the constraint that matters most should be stated where someone "
        "editing this will read it")


def test_a_dry_run_does_not_final_validate() -> None:
    assert "self.plan.dry_run" in STAGE, (
        "a --dry-run launches no agent and produces no candidate; taking a card "
        "to validate nothing would make the cheap smoke expensive")


def test_the_stage_is_bounded() -> None:
    """Bounded, but by the TASK's own cost rather than by a flat constant.

    A scenario verdict is an 1800 s AgentX replay plus a model load, a session
    rebuild and the correctness stage, so a flat ceiling either kills it
    part-way (NO_RESULT for a candidate that was measuring correctly) or
    over-reserves every task. `final_validation_timeout` derives it per task.

    What still matters is that it is bounded at all -- unbounded, a stuck
    validation holds the card forever and the run never gives it back.
    """
    assert 0 < FINAL_VALIDATION_TIMEOUT_S <= 3 * 3600
    assert "timeout=final_validation_timeout(" in STAGE, (
        "unbounded, a stuck final validation holds the card forever and the run "
        "never gives it back")


def test_the_run_hands_its_own_devices_to_the_validator() -> None:
    """The run's card is what the verdict measures on, so it is HANDED IN.

    The verdict runs at this point precisely BECAUSE the run still holds the
    device. Passing the cards as `--deny` would leave the verdict with no card;
    it would fall back to taking a lease, the lease respects every live claim
    including this run's own, and the verdict would wait forever for a card
    the run is holding against it.

    So: `--gpu` with the run's own devices, and no `--deny`.
    """
    assert '"--gpu"' in STAGE, (
        "the verdict is given no device at all; it will take a lease and "
        "block on this run's own claim")
    assert "self.plan.gpus" in STAGE
    assert '"--deny"' not in STAGE, (
        "the run's cards are being fenced off from the verdict that has to "
        "measure on them -- see this test's docstring")


def test_the_run_hands_its_exact_build_inputs_to_the_validator() -> None:
    """The verdict must not re-derive a same-named path in another checkout."""
    assert '"--env-input"' in STAGE
    assert "self.plan.env_input" in STAGE


def test_the_agents_container_is_released_before_the_verdict_asks_for_the_card(
) -> None:
    """A foreign compute process on the card makes the verdict refuse it.

    The agent's container is still up at this point, and the verdict refuses a
    contended device -- correctly. It is stopped by EXACT RECORDED NAME: a
    pattern would match a neighbouring run's container as readily as ours.
    """
    assert "backend.remove(self.container)" in STAGE, (
        "the agent's container is not stopped before the verdict, so the card "
        "carries a foreign process when the verdict asks for it")
    assert "container_released" in STAGE

    # On Modal the verdict stands up its OWN sandbox, read-only to the measured
    # party, so the agent's sandbox is released like any other container. The
    # branch exists because the reason differs, and the reason is what the
    # run's record has to carry.
    assert "backend.measures_from_inside(self.container)" in STAGE, (
        "the release does not distinguish the backends, so nothing in the "
        "run's record can say why a sandbox went when it did")
    assert 'backend="modal"' in STAGE, (
        "the modal release is not labelled in the journal")
    assert "container_retained" not in STAGE, (
        "a Modal sandbox is held open across final validation. The verdict "
        "does not run in it -- it builds a fresh read-only one -- so holding "
        "this one bills the task's GPU allocation a second time")


def test_the_verdict_follows_the_backend_the_run_actually_used() -> None:
    """A run on Modal cannot be measured on this host's cards.

    `validate_evolve.py` reads `container.json` -- the same `modal:` prefix
    `lifecycle/backend.py` dispatches on -- and switches the candidate arm to
    the Modal verdict path instead of a local container, and local card
    indices are not forwarded for a run that had none.
    """
    from pathlib import Path

    body = (Path(__file__).resolve().parents[3] / "serverbench"
            / "harness" / "final_validate" / "tools" / "validate_evolve.py").read_text("utf-8")
    assert "def backend_of(" in body, (
        "the validator cannot tell where the run happened")
    assert '"--backend", "modal"' in body, (
        "the validator never tells measure_published which backend to serve "
        "the candidate on")
    # Docker is the only backend that takes card indices, a deny list or a
    # host port: Modal's cards are remote, and the local (namespace) backend
    # serves the verdict in its own sandboxes on the card it already has.
    assert 'if args.gpu' in body and 'backend == "docker"' in body, (
        "local card indices are forwarded for a run that had none")
    assert '"--backend", "local"' in body, (
        "a local-sandbox run is sent to the docker verdict")

    measure = (Path(__file__).resolve().parents[3] / "serverbench"
               / "harness" / "final_validate" / "tools"
               / "measure_published.py").read_text("utf-8")
    assert '"-m", "final_validate.modal_verdict"' in measure
    assert 'modal_python()' in measure
    assert 'UNSUPPORTED_REASON' not in measure
    assert "deposit_into_lineage" in measure, (
        "the deposit is not shared between the backends; a verdict that lands "
        "somewhere different depending on where it ran is two verdicts")

    assert 'if args.backend != "modal":\n            model_identity.check(' in measure
