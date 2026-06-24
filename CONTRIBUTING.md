# Contributing to Waruka

Thanks for considering a contribution. Waruka is a side project that
happens to be useful for ultimate frisbee tactics review; bug reports,
patches, and feedback are all welcome.

## License

Waruka is licensed under **AGPL-3.0-or-later** (see [LICENSE](LICENSE)).
By contributing you agree that your contribution will be released under
the same license. This is constrained by Waruka's use of
[ultralytics](https://github.com/ultralytics/ultralytics) (also
AGPL-3.0); see [NOTICE.md](NOTICE.md) for the full third-party
attribution.

## Sign your commits (DCO)

Waruka follows the [Developer Certificate of Origin](https://developercertificate.org/)
(DCO) -- the same mechanism used by the Linux kernel, Docker, GitLab, and
many other open-source projects. The DCO is a short statement that you
have the right to submit the code you're contributing.

To sign off, add `-s` to your commit:

```bash
git commit -s -m "fix: handle empty tracks.json in classify"
```

This appends a line like:

```
Signed-off-by: Your Name <your.email@example.com>
```

The name and email must match your `git config user.name` and
`user.email`. By including the sign-off you certify that the
contribution is your own work (or you have the right to submit it) and
that you're aware it will be public and redistributed under Waruka's
license.

Pull requests without a sign-off on every commit will be blocked from
merging. If you forget, amend the last commit with:

```bash
git commit --amend -s --no-edit
```

For older commits in a branch, an interactive rebase can sign them all:

```bash
git rebase HEAD~N --signoff
```

(Substitute `N` for the number of commits you want to fix up.)

## Practical guidelines

- **Open an issue first** for non-trivial changes. A 30-second alignment
  saves a lot of rework, and some things are deliberately scoped out
  (see `_handover_v0_16.md` for the current direction).
- **One concern per PR.** Mixing a feature, a refactor, and a test fix
  in one PR makes review hard.
- **Prefer small visible changes** over big invisible ones. Waruka is a
  small enough codebase that aggressive refactors usually aren't worth
  the churn.
- **Don't commit large binary files** (test clips, model weights other
  than what's already vendored, broadcast outputs). They bloat the repo
  forever.

## Build + test

See [BUILDING.md](BUILDING.md) for how to produce a Windows bundle from
your branch. Bare minimum before sending a PR:

```powershell
python -m waruka gui          # GUI smoke -- main window opens
python -m waruka --help       # CLI dispatcher works
```

If your change touches the perception, classify, campath, or render
pipeline, also run the end-to-end CLI flow against a short clip and
inspect the resulting `broadcast.mp4`.

## What about a CLA?

Waruka uses DCO instead of a Contributor License Agreement -- friction
matters for a side project, and Waruka can't easily re-license away from
AGPL-3.0 anyway (ultralytics constrains us). If that ever changes (e.g.
a non-AGPL detector replaces ultralytics) we may revisit, but for now
DCO is the answer.

## Questions

Open an issue and ask. There's no separate chat or mailing list at this
stage.
