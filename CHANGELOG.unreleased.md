# Unreleased changes

Add here any changes made in a PR that are relevant to end users. Allowed sections:

- Added - for new features.
- Changed - for changes in existing functionality.
- Deprecated - for soon-to-be removed features.
- Removed - for now removed features.
- Fixed - for any bug fixes.
- Security - in case of vulnerabilities.

Section headings should be at level 3 (e.g. `### Added`).

## Unreleased

### Changed

- Paginated methods (and underlying paginators) that accept a `per_page` argument now only accept `int` values.  Default `per_page` values are set directly in method signatures, and explicitly passing `None` is no longer supported.

### Fixed

- Fix `wandb.Settings` update regression in `wandb.integration.metaflow` (@kptkin in https://github.com/wandb/wandb/pull/9211)
