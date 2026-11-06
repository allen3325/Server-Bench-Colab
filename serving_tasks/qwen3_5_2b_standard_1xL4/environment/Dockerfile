# Verifier-capable task environment.
#
# Harbor uploads `tests/` into THIS container and runs it after AGENT_END, so the
# grader's dependencies must already be here. Installing them at grading time
# would need network, and this image runs offline.
FROM python:3.12-slim

# git is needed only to RESOLVE the AgentX pin at build time. It is removed with
# the apt lists afterwards: this image runs with `network_mode = "no-network"`, and
# harbor runs `apt-get update` after the firewall is up -- which exits non-zero
# and kills the trial if sources survive.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && pip install --no-cache-dir pyyaml \
 && pip install --no-cache-dir \
    "aiperf @ git+https://github.com/SemiAnalysisAI/agentx-harness.git@56a0cf70f4c0359454ee4bd15a17770b541a3e3e" \
 && apt-get purge -y git \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/* /etc/apt/sources.list*

# Proof, baked into the image, that both dependencies resolve INSIDE the
# container rather than merely beside the task on the host.
RUN python -c "import yaml, aiperf, pathlib; \
    p = pathlib.Path(aiperf.__file__); \
    assert 'site-packages' in p.parts, p; \
    print('grader deps present in image:', yaml.__name__, aiperf.__file__)"

WORKDIR /workspace
