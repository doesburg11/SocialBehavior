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

## Project-Specific Modeling Preference

- For `predprey_public_goods/emerging_cooperation.py` vs MARL stag-hunt
  comparisons, treat this as an intentional core distinction:
  - `emerging_cooperation`: cooperation is trait-based (`nature` framing).
  - MARL stag-hunt: cooperation is action-based (`nurture` framing).
- Keep that distinction explicit, and focus comparisons on other remaining gaps.

## Validation Expectations

- After code edits, run minimal relevant validation where possible (for example
  syntax check and a short smoke run) using `./.conda/bin/python`.
- Report what was run and what could not be run.
