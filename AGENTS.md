# SocialBehavior Agent Instructions

These instructions apply when working in this repository.

## Environment

- Use the project-local Python environment by default:
  - `./.conda/bin/python`
- Prefer running scripts and checks from repo root:
  - `/home/doesburg/Projects/SocialBehavior`

## Parameters

- Don't use CLI style parameters
- Use and define parameters inside script

## Communication Style

- Keep answers concise and technical.
- When the response is substantive (not a one-line factual reply), end with
  `1-3` concrete next-step suggestions as a numbered list.
- When comparing implementations, emphasize meaningful mechanism differences and
  avoid listing trivial incidental differences.
- If a meaninful chance in the code has been made, give a detailed and stepwise update in the accompanied README.md

## Project-Specific Modeling Preference

- For `/home/doesburg/Projects/SocialBehavior/predpreygrass_hamilton/predpreygrass_hamilton.py` the goal is
to mimic human evolution as close as possible to reality:
  - prdators ar humans
  - prey are deer
  - sexual reproduction
  - kin recognition cue
  - children are dependant on parent or related family for reproduction

## Validation Expectations

- After code edits, run minimal relevant validation where possible (for example
  syntax check and a short smoke run) using `./.conda/bin/python`.
- Report what was run and what could not be run.
