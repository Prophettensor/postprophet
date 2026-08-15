"""CLI entrypoint for the content-agent framework.

Commands:
    round   [--config PATH] [--top N]   produce a drafting brief (in-session, LLM step follows)
    measure [--config PATH]             read real outcomes + retrain (no_agent cron)
    ideas   [--config PATH]             show surfaced ideas only
    init    [--config PATH]             write an example config

The round command is the interactive (LLM-bearing) step. measure is the zero-LLM
cron step. Separating them keeps "building the product" (the framework) cleanly
apart from "using the product" (a builder instance running the loop).
"""

from __future__ import annotations

import argparse
import sys

from .pipeline import Pipeline
from .config import write_example_config


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(prog="content-agent", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_round = sub.add_parser("round", help="produce a drafting brief")
    p_round.add_argument("--config", default="content-agent.yaml")
    p_round.add_argument("--top", type=int, default=5)

    p_measure = sub.add_parser("measure", help="read outcomes + retrain (cron)")
    p_measure.add_argument("--config", default="content-agent.yaml")

    p_ideas = sub.add_parser("ideas", help="show surfaced ideas only")
    p_ideas.add_argument("--config", default="content-agent.yaml")

    p_init = sub.add_parser("init", help="write an example config")
    p_init.add_argument("--config", default="content-agent.yaml")

    args = parser.parse_args(argv)

    if args.command == "init":
        path = write_example_config(args.config)
        print(f"wrote example config to {path}")
        return

    # data dir sits next to the config file
    data_dir = _data_dir(args.config)
    pipe = Pipeline.from_config(args.config, data_dir)

    if args.command == "ideas":
        for idea in pipe.ideas():
            seed = idea["seed"]
            print(f"fit={idea['fit']:.2f} [{seed['kind']}] {seed['title']}")
            print(f"   angles: {', '.join(idea['angles'])} | {idea['why']}")
        return

    if args.command == "measure":
        n = pipe.measure()
        # Silent-when-nothing watchdog pattern: only emit when something was
        # recorded, so a no_agent cron delivers nothing on quiet ticks.
        if n > 0:
            print(f"recorded {n} new outcome(s)")
            print(pipe.state_summary())
        return

    if args.command == "round":
        print(pipe.round_brief(args.top))
        return


def _data_dir(config_path):
    import os
    base = os.path.dirname(os.path.abspath(config_path))
    return os.path.join(base, "data")


if __name__ == "__main__":
    main()
