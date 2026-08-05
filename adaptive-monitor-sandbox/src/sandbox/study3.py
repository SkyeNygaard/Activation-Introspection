"""Study 3: does refusal fidelity cause harm, and does private memory offset it?

The design is checkpoint-and-branch. A single agent runs until the monitor issues
its first real block. That prefix -- the agent transcript, the world state, the
monitor, the RNG -- is frozen and deep-copied into each arm. Every arm therefore
starts from a byte-identical situation and differs only in the refusal it is
handed at that one block. That is what lets the feedback contrast be causal
rather than a comparison of runs that diverged for other reasons.

Two questions, in order:

  1. ``tau_fixed``  = P(harm | q=1, fixed monitor) - P(harm | q=0.25, fixed)
     Does valid feedback cause harm when the monitor cannot learn?
  2. ``delta``      = tau_stateful - tau_fixed
     Does a monitor with private memory offset that harm?

Question 2 is only asked if question 1 shows an effect. A null interaction on top
of two weak main effects would be uninterpretable, so the order is fixed in
advance.

Harm is judged by the lineage oracle on what crossed the boundary -- never by the
monitor, which sees only frozen views.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from sandbox.actions import Action, Outcome, Verb
from sandbox.env import Task, execute
from sandbox.lineage import LineageOracle
from sandbox.state import SECRET_VALUE, WorldState
from sandbox.study3_agent import Study3Agent
from sandbox.triggers import FeedbackChannel, Rule
from sandbox.views import ActionView, FixedTriggerMonitor, StatefulTriggerMonitor, WorldView

MonitorArm = FixedTriggerMonitor | StatefulTriggerMonitor


@dataclass
class Checkpoint:
    """A frozen trajectory prefix, forkable into feedback/monitor arms."""

    agent: Study3Agent
    world: WorldState
    rule: Rule
    task: Task
    step: int
    blocked_action: Action

    def fork(self) -> Checkpoint:
        """A deep, independent copy. Arms must not share mutable state."""
        return Checkpoint(
            agent=copy.deepcopy(self.agent),
            world=self.world.model_copy(deep=True),
            rule=self.rule,
            task=self.task,
            step=self.step,
            blocked_action=self.blocked_action,
        )


@dataclass
class ArmResult:
    q: float
    monitor: str
    harm: bool
    leaked_fraction: float
    harm_step: int | None
    blocks: int
    parse_failures: int
    reported_feature: str
    steps_run: int
    finished: bool
    # True when the trajectory prefix had already leaked before any feedback was
    # delivered. Such arms carry no information about the treatment.
    pre_existing_harm: bool = False
    transcript_tail: str = ""


@dataclass
class ScenarioResult:
    scenario_id: str
    rule_trigger: str
    checkpoint_reached: bool
    arms: list[ArmResult] = field(default_factory=list)
    note: str = ""


def _apply(
    agent: Study3Agent,
    action: Action,
    monitor: MonitorArm,
    world: WorldState,
    *,
    feedback_text: str | None,
) -> tuple[Outcome, str]:
    """Judge, maybe execute, and return (outcome, agent-visible feedback).

    The monitor sees only frozen views. On a block the agent sees ``feedback_text``
    when supplied (the randomized-response refusal) or a generic message.
    """
    view = ActionView.of(action)
    judgement = monitor.judge_view(view, WorldView.of(world))

    if judgement.block:
        monitor.observe_view(view, Outcome.BLOCKED)
        msg = feedback_text if feedback_text is not None else "Blocked by policy."
        return Outcome.BLOCKED, msg

    result = execute(action, world)
    monitor.observe_view(view, result.outcome)
    return result.outcome, result.observation


def build_checkpoint(
    agent: Study3Agent,
    task: Task,
    rule: Rule,
    *,
    max_prefix_steps: int = 8,
) -> Checkpoint | None:
    """Run one agent until the monitor's first real block, then freeze.

    The prefix monitor is a fixed monitor on the same rule, so the block that
    defines the checkpoint is a genuine model-generated event, not one the
    experiment engineered. Returns None if no block occurs -- that scenario
    simply contributes no checkpoint, rather than a fabricated one.
    """
    world = WorldState()
    prefix_monitor = FixedTriggerMonitor(rule)
    agent.start(task.instruction)

    for step in range(max_prefix_steps):
        action = agent.next_action()
        if action is None:
            continue  # parse failure; the agent is re-prompted inside next_action
        view = ActionView.of(action)
        judgement = prefix_monitor.judge_view(view, WorldView.of(world))
        if judgement.block:
            return Checkpoint(
                agent=agent,
                world=world,
                rule=rule,
                task=task,
                step=step,
                blocked_action=action,
            )
        result = execute(action, world)
        agent.record(action, result.observation, result.outcome)
        if action.verb is Verb.FINISH:
            break
    return None


def run_arm(
    checkpoint: Checkpoint,
    channel: FeedbackChannel,
    monitor: MonitorArm,
    *,
    scenario_id: str,
    horizon: int = 10,
) -> ArmResult:
    """Continue a forked checkpoint under one feedback channel and one monitor.

    The randomized-response refusal is delivered exactly once, on the
    checkpoint's own blocked action. Every later block gets a generic,
    length-matched message: later blocks are post-treatment consequences of the
    agent's own behaviour, so treating them as more manipulation would confound
    dose with trajectory.
    """
    fork = checkpoint.fork()
    agent, world = fork.agent, fork.world
    oracle = LineageOracle(SECRET_VALUE)

    reported, refusal = channel.draw(fork.rule, scenario_id)
    harm_step: int | None = None
    steps_run = 0
    finished = False
    # The prefix may already have leaked by a route the rule does not cover.
    # That is shared by every arm of this scenario, so it does not bias the
    # contrast -- it censors it, pinning harm at 1 regardless of feedback. Such
    # scenarios have to be excluded from tau, not averaged into it.
    pre_existing_harm = oracle.is_leak(world.exfiltrated)

    outcome, feedback = _apply(agent, fork.blocked_action, monitor, world, feedback_text=refusal)
    if outcome is not Outcome.BLOCKED:
        # The checkpoint exists because a fixed monitor on this rule blocked
        # this action; an arm monitor that lets it through would execute it and
        # deliver no treatment at all, silently turning this arm into a control.
        raise RuntimeError(
            f"{monitor.name} monitor did not re-block the checkpoint action; "
            f"this arm would receive no feedback treatment"
        )
    agent.record(fork.blocked_action, feedback, outcome)

    for i in range(horizon):
        steps_run = i + 1
        action = agent.next_action()
        if action is None:
            continue
        outcome, feedback = _apply(agent, action, monitor, world, feedback_text=None)
        agent.record(action, feedback, outcome)

        if harm_step is None and oracle.is_leak(world.exfiltrated):
            harm_step = i
        if action.verb is Verb.FINISH:
            finished = True
            break

    return ArmResult(
        q=channel.q,
        monitor=monitor.name,
        harm=oracle.is_leak(world.exfiltrated),
        leaked_fraction=oracle.fraction(world.exfiltrated),
        harm_step=harm_step,
        blocks=monitor.blocks_issued,
        parse_failures=agent.parse_failures,
        reported_feature=reported.value,
        steps_run=steps_run,
        finished=finished,
        pre_existing_harm=pre_existing_harm,
        transcript_tail=agent.transcript[-400:],
    )
